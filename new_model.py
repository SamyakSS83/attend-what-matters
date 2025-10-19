# new_model.py
import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

# ---------- rotary (RoPE) helper ----------
class RotaryEmbeddingSmall:
    """
    Minimal RoPE utility for single-head or multi-head cases.
    This stores inv_freq as a buffer and can build cos/sin dynamically.
    This class avoids storing large cos/sin permanently to keep memory small.
    """
    def __init__(self, head_dim, base=10000):
        assert head_dim % 2 == 0, "head_dim for RoPE must be even"
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # not nn.Module to keep simple; will be moved to device manually when used
        self.register_buffer = lambda name, tensor, persistent=True: None
        self.inv_freq = inv_freq  # tensor on cpu; we'll move to device when needed

    def _build(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        inv = self.inv_freq.to(device=device, dtype=dtype)
        freqs = torch.einsum("n,d->n d", t, inv)  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x):
        # x: (..., head_dim)
        d = x.shape[-1]
        x1 = x[..., : d//2]
        x2 = x[..., d//2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, x):
        # x: (B, L, head_dim) or (B, heads, L, head_dim)
        L = x.shape[-2]
        device = x.device
        dtype = x.dtype
        cos, sin = self._build(L, device, dtype)  # (L, head_dim)
        # align dims: make cos, sin broadcastable to x
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)


# ---------- single-head attention with RoPE (keeps behaviour close to original) ----------
class SingleHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        # single-head attention: head_dim == embed_dim
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # head_dim must be even for RoPE
        assert embed_dim % 2 == 0, "embed_dim must be even for RoPE in single-head mode"
        self.rotary = RotaryEmbeddingSmall(head_dim=embed_dim)
        self.last_attn_weights = None

    def forward(self, x):
        # x: (B, L, E)
        B, L, E = x.shape
        q = self.q_proj(x)  # (B, L, E)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # apply rotary to q,k (as (B,L,E))
        q = self.rotary.apply(q)
        k = self.rotary.apply(k)

        # attention
        scale = 1.0 / math.sqrt(E)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, L, L)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        self.last_attn_weights = attn  # (B, L, L)
        out = torch.matmul(attn, v)  # (B, L, E)
        out = self.out_proj(out)
        return out


# ---------- Transformer block that mirrors your original structure ----------
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, dropout=0.25):
        super().__init__()
        # single-head attention to keep behaviour same as original (you used num_heads=1)
        self.attention_layer = SingleHeadSelfAttentionRoPE(embed_dim=embedding_size, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embedding_size)
        # original had a single linear MLP followed by ReLU and residual
        self.mlp_layer = nn.Linear(embedding_size, embedding_size)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_size)

    def forward(self, X):
        # X: (B, L, E)
        attention_out = self.attention_layer(X)
        attention_out = attention_out + X
        norm_attention_out = self.layer_norm1(attention_out)
        mlp_out = F.relu(self.mlp_layer(norm_attention_out))
        mlp_out = mlp_out + norm_attention_out
        final_out = self.layer_norm2(mlp_out)
        return final_out

    @property
    def last_attn_weights(self):
        return self.attention_layer.last_attn_weights


# ---------- Full model (close to your original, but with safe RoPE + safe projection) ----------
class MMBCDContrast(nn.Module):
    def __init__(self, embedding_dim=1024, position_dim=256, dropout=0.25):
        super().__init__()

        # load your dinov3/dinov2 line (keep user's local loading attempt)
        try:
            # Try to load local finetuned model if available
            local_ckpt = 'checkpointsv2/checkpoint_epoch5.pt'
            # local_ckpt = 'checkpointsv2_hr/best_checkpoint.pt'
            # if os.path.exists(local_ckpt):
            self.vision_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            # self.vision_model = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local', pretrained=False)
            
            state_dict = torch.load(local_ckpt, map_location='cpu')
            # If checkpoint is a dict with 'model' key, use that
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
                self.vision_model.load_state_dict(state_dict, strict=False)
            
        except Exception:
            # fallback to direct hub if user intended remote
            self.vision_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        # try to obtain backbone output dim
        vision_out_dim = getattr(self.vision_model, 'embed_dim', None)
        if vision_out_dim is None:
            vision_out_dim = getattr(self.vision_model, 'hidden_dim', None)
        if vision_out_dim is None:
            # if model returns a tensor of known size, we will assume embedding_dim later
            vision_out_dim = embedding_dim - position_dim  # best guess
        self.vision_out_dim = int(vision_out_dim)

        # position projection (keep original fc_positions)
        self.fc_positions = nn.Linear(position_dim, position_dim)

        # If concatenation already equals expected embedding_dim, don't project.
        # Otherwise create input_proj but initialize it to preserve vision features.
        concat_dim = self.vision_out_dim + position_dim
        self.need_input_proj = (concat_dim != embedding_dim)
        if self.need_input_proj:
            self.input_proj = nn.Linear(concat_dim, embedding_dim)
            # initialize to roughly preserve vision features: block identity for vision part,
            # small random for pos part, and bias zero.
            with torch.no_grad():
                # weight shape (embedding_dim, concat_dim)
                W = torch.zeros_like(self.input_proj.weight)
                # set left block (vision) to near identity on top-left of W if dims allow
                min_dim = min(self.vision_out_dim, embedding_dim)
                for i in range(min_dim):
                    W[i, i] = 1.0
                # small random for remaining columns (positions)
                pos_start = self.vision_out_dim
                if pos_start < concat_dim:
                    W[:, pos_start:] = 0.01 * torch.randn_like(W[:, pos_start:])
                self.input_proj.weight.copy_(W)
                if self.input_proj.bias is not None:
                    self.input_proj.bias.zero_()
        else:
            self.input_proj = None  # not used; pass concatenation through

        # transformer blocks (same small stack you had)
        self.transformer_block1 = TransformerBlock(embedding_size=embedding_dim, dropout=dropout)
        self.transformer_block2 = TransformerBlock(embedding_size=embedding_dim, dropout=dropout)
        self.transformer_block3 = TransformerBlock(embedding_size=embedding_dim, dropout=dropout)

        # classification head (same as your original)
        self.bn_vision = nn.BatchNorm1d(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)

        # attempt to freeze early layers as you did originally (best-effort)
        try:
            for param in self.vision_model.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, 3):
                for param in self.vision_model.blocks[i].parameters():
                    param.requires_grad = False
        except Exception:
            pass

    def _extract_vision_embedding(self, vision_out):
        """
        Accept a few possible outputs from self.vision_model:
        - a tensor (B, D)
        - a tuple/list where first element is tensor
        - a dict containing 'last_hidden_state' or 'pooled'
        """
        if isinstance(vision_out, torch.Tensor):
            return vision_out
        if isinstance(vision_out, (list, tuple)) and len(vision_out) > 0 and isinstance(vision_out[0], torch.Tensor):
            return vision_out[0]
        if isinstance(vision_out, dict):
            if 'pooled' in vision_out:
                return vision_out['pooled']
            if 'last_hidden_state' in vision_out:
                # try CLS pooling if present (assume last_hidden_state: (B, tokens, D))
                lh = vision_out['last_hidden_state']
                return lh[:, 0, :]
        # fallback: try to convert to tensor
        raise RuntimeError("Unexpected vision_model output; please adapt _extract_vision_embedding()")

    def forward(self, X_images, X_positions):
        # X_images: (B, S, 3, H, W)
        B = X_images.shape[0]
        S = X_images.shape[1]
        img_size = X_images.shape[3], X_images.shape[4]
        X_images_flat = X_images.view(-1, 3, *img_size)
        vision_out = self.vision_model(X_images_flat)
        vision_out = self._extract_vision_embedding(vision_out)  # (B*S, vision_out_dim) hopefully
        vision_out = vision_out.view(B, S, -1)  # (B, S, vision_out_dim)

        pos_proj = self.fc_positions(X_positions)  # (B,S,position_dim)

        vision_with_pos = torch.cat((vision_out, pos_proj), dim=2)  # (B,S, concat_dim)
        if self.need_input_proj:
            trans_in = self.input_proj(vision_with_pos)
        else:
            trans_in = vision_with_pos  # already matching embedding_dim

        b1 = self.transformer_block1(trans_in)
        b2 = self.transformer_block2(b1)
        b3 = self.transformer_block3(b2)
        roi_embeddings = b3  # (B, S, D)

        pooled = roi_embeddings[:, 0, :]
        pooled = self.bn_vision(pooled)
        pooled = self.dropout(pooled)
        pooled = F.relu(self.fc1(pooled))
        # return raw logits (IMPORTANT: do NOT apply sigmoid here)
        logits = self.fc2(pooled).view(-1)  # (B,)
        return logits, roi_embeddings

    def compute_contrastive_loss(self, roi_embeddings, rewards, temperature=0.1, iou_pos_thresh=0.2):
        device = roi_embeddings.device
        B, S, D = roi_embeddings.shape
        emb_norm = F.normalize(roi_embeddings, dim=2)
        anchor = emb_norm[:, 0, :].unsqueeze(1)
        others = emb_norm[:, 1:, :]
        rewards_others = rewards[:, 1:]
        cos_sim = torch.sum(anchor * others, dim=2)
        pos_mask = (rewards_others >= iou_pos_thresh).float()
        weights = rewards_others * pos_mask
        if weights.sum() == 0:
            return torch.tensor(0.0, device=device)
        loss = (weights * (1.0 - cos_sim)).sum() / (weights.sum() + 1e-8)
        return loss

    def get_attention_maps(self):
        return {
            'block1': self.transformer_block1.last_attn_weights,
            'block2': self.transformer_block2.last_attn_weights,
            'block3': self.transformer_block3.last_attn_weights,
        }

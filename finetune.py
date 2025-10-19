# finetune.py
import os
import argparse
from pathlib import Path
import math
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import timm
except Exception:
    timm = None

# import your dataset
from data import all_mammo


# def load_dinov2_backbone(device, pretrained=True, backbone_name=None, load_mode='timm', hub_model_name=None):
#     """
#     Try to load a DINOv2 backbone. We attempt several common methods:
#       1. timm.create_model(backbone_name, pretrained=pretrained)
#       2. torch.hub.load('facebookresearch/dinov2', hub_model_name, pretrained=pretrained) -- user local/remote
#     Returns model, emb_dim
#     If loading fails, raises an Exception.
#     """
#     model = None
#     emb_dim = None

#     if load_mode == 'timm' and timm is not None:
#         # try a few common timm model names (if installed)
#         candidates = [backbone_name] if backbone_name else [
#             'vit_base_patch16_224.dino',      # example names - may vary by timm/dino plugin
#             'vit_base_patch14_dinov2',       # hypothetical
#             'vit_base_patch16_224'
#         ]
#         for name in candidates:
#             if name is None:
#                 continue
#             try:
#                 model = timm.create_model(name, pretrained=pretrained)
#                 # try to get embedding dim
#                 if hasattr(model, 'embed_dim'):
#                     emb_dim = model.embed_dim
#                 elif hasattr(model, 'head') and hasattr(model.head, 'in_features'):
#                     emb_dim = model.head.in_features
#                 else:
#                     # try an attribute
#                     emb_dim = getattr(model, 'num_features', None)
#                 if emb_dim is None:
#                     emb_dim = 768
#                 print(f"[INFO] Loaded backbone via timm: {name} (emb_dim={emb_dim})")
#                 break
#             except Exception as e:
#                 # try next
#                 # print(f"[WARN] timm load {name} failed: {e}")
#                 model = None

#     if model is None and load_mode in ('hub', 'auto'):
#         # try torch.hub (remote or local repo)
#         hub_candidates = [hub_model_name] if hub_model_name else [
#             'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitb16', 'dinov2_vits14'
#         ]
#         for hub_name in hub_candidates:
#             if hub_name is None:
#                 continue
#             try:
#                 # choose repo depending on model name (dinov3 models live in a different repo)
#                 if 'dinov3' in str(hub_name).lower():
#                     repo = 'facebookresearch/dinov3'
#                 else:
#                     repo = 'facebookresearch/dinov2'
#                 model = torch.hub.load(repo, hub_name, pretrained=pretrained)
#                 # probing embedding dims
#                 emb_dim = getattr(model, 'embed_dim', None) or getattr(model, 'head', None) and getattr(model.head, 'in_features', None)
#                 if emb_dim is None:
#                     emb_dim = 768
#                 print(f"[INFO] Loaded backbone via torch.hub ({repo}): {hub_name} (emb_dim={emb_dim})")
#                 break
#             except Exception as e:
#                 # try the other repo as a fallback if initial attempt failed
#                 try:
#                     other_repo = './dinov3' if 'dinov3' not in str(hub_name).lower() else 'facebookresearch/dinov2'
#                     model = torch.hub.load(other_repo, hub_name, pretrained=pretrained)
#                     emb_dim = getattr(model, 'embed_dim', None) or getattr(model, 'head', None) and getattr(model.head, 'in_features', None)
#                     if emb_dim is None:
#                         emb_dim = 768
#                     print(f"[INFO] Loaded backbone via torch.hub ({other_repo}): {hub_name} (emb_dim={emb_dim})")
#                     break
#                 except Exception:
#                     model = None

#     if model is None:
#         raise RuntimeError("Failed to load DINOv2 backbone. Check timm/torch.hub availability and model names.")

#     # remove classification head (we'll use projection head)
#     try:
#         # many timm/torchhub models have .head
#         if hasattr(model, 'head'):
#             model.reset_classifier(0)
#     except Exception:
#         # fallback: set head to identity if present
#         if hasattr(model, 'head'):
#             model.head = nn.Identity()

#     model.to(device)
#     model.eval()  # backbone will be switched to train() when needed in training loop
#     return model, int(emb_dim)

def load_dinov2_backbone(device, pretrained=True, backbone_name=None, load_mode='timm', hub_model_name=None):
    """
    Load DINOv2 or DINOv3 backbone robustly via timm or torch.hub.
    Returns (model, embedding_dim)
    """
    model = None
    emb_dim = None

    # 1. Try timm first
    if load_mode == 'timm' and timm is not None:
        try:
            name = backbone_name or 'vit_base_patch14_dinov2'
            model = timm.create_model(name, pretrained=pretrained)
            emb_dim = getattr(model, 'embed_dim', getattr(model, 'num_features', 768))
            print(f"[INFO] Loaded DINOv2 via timm: {name} (dim={emb_dim})")
        except Exception as e:
            print(f"[WARN] timm load failed: {e}")
            model = None

    # 2. Fallback: torch.hub
    if model is None:
        hub_name = hub_model_name or 'dinov2_vitb14'
        repo = None

        if 'dinov3' in hub_name.lower():
            repo = './dinov3'
            # normalize known variants
            if 'pretrain' not in hub_name:
                hub_name = 'dinov3_vitb16'
        else:
            repo = 'facebookresearch/dinov2'
            if not hub_name.startswith('dinov2'):
                hub_name = 'dinov2_vitb14'

        print(f"[INFO] Trying torch.hub.load('{repo}', '{hub_name}')")
        if 'dinov3' in hub_name.lower() : model = torch.hub.load(repo, hub_name, source='local',pretrained = False)
        else: model = torch.hub.load(repo, hub_name, pretrained=pretrained)
        emb_dim = getattr(model, 'embed_dim', getattr(model, 'num_features', 768))
        print(f"[INFO] Loaded backbone from {repo}: {hub_name} (dim={emb_dim})")

    # 3. Clean classifier head
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(0)
    elif hasattr(model, 'head'):
        model.head = nn.Identity()

    model.to(device)
    model.eval()
    return model, int(emb_dim)

class FinetuneModel(nn.Module):
    def __init__(self, backbone, backbone_dim, proj_dim=256, num_classes=2, freeze_backbone_until_layer=None):
        """
        backbone: vision encoder (DINOv2)
        backbone_dim: output dim
        proj_dim: projection head dim (contrastive)
        num_classes: classification head output size
        freeze_backbone_until_layer: if given, try to freeze named parameters not containing this substring
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim

        # small projection head for contrastive loss
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, proj_dim)
        )

        # classifier head for supervised training (operates on pooled sample embedding)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(backbone_dim // 2, num_classes)
        )

        if freeze_backbone_until_layer is not None:
            for name, p in self.backbone.named_parameters():
                if freeze_backbone_until_layer not in name:
                    p.requires_grad = False
            print(f"[INFO] Freezing backbone parameters not containing '{freeze_backbone_until_layer}'")

    def forward_backbone(self, images):
        """
        images: (B*K, 3, H, W)
        returns: features (B*K, backbone_dim)
        """
        # Some DINO models accept images and return features directly (varies by implementation).
        # We'll try common patterns.
        self.backbone.eval()  # ensure backbone in eval for stable BN / statistics if it's pretrained
        with torch.no_grad():
            out = self.backbone(images)
        # extract tensor from possible dict/tuple
        if isinstance(out, dict):
            # try multiple keys
            for key in ('last_hidden_state', 'pooled', 'features'):
                if key in out:
                    out = out[key]
                    break
        if isinstance(out, (list, tuple)):
            out = out[0]
        # If ViT-style last_hidden_state returned (B, tokens, dim), try CLS token
        if out.dim() == 3:
            # assume CLS at token 0
            out = out[:, 0, :]

        return out

    def forward(self, crops):
        """
        crops: (B, K, 3, H, W)
        returns:
          sample_pooled: (B, backbone_dim)   -- mean of backbone features per-sample (no grad on backbone if frozen)
          crop_feats: (B, K, backbone_dim)   -- backbone features for each crop
          crop_proj: (B, K, proj_dim)        -- projection head outputs (for contrastive)
        """
        B, K, C, H, W = crops.shape
        images = crops.view(B * K, C, H, W)
        features = self.forward_backbone(images)  # (B*K, backbone_dim)
        features = features.view(B, K, -1)
        crop_feats = features
        sample_pooled = features.mean(dim=1)  # (B, backbone_dim)
        crop_proj = self.proj(features)  # (B, K, proj_dim)
        return sample_pooled, crop_feats, crop_proj


def nt_xent_loss_from_embeddings(emb, batch_size, K, temperature=0.1, eps=1e-8):
    """
    emb: (B*K, D) normalized embeddings
    batch_size: B
    K: crops per sample
    temperature: scalar
    We treat embeddings from same sample (same batch index // K) as positives.
    Loss = -log( sum_{j in positives(i)} exp(sim_ij/ T) / sum_{j != i} exp(sim_ij / T) )
    """
    device = emb.device
    N = emb.shape[0]  # B*K
    sim = torch.matmul(emb, emb.T)  # (N,N)
    sim = sim / temperature

    # mask to exclude self comparisons
    diag_mask = torch.eye(N, device=device).bool()
    # use a dtype-appropriate large negative value to avoid overflow when using float16
    neg_inf = torch.finfo(sim.dtype).min
    sim_masked = sim.masked_fill(diag_mask, neg_inf)

    # positive mask: same sample (index // K equal) and not self
    idx = torch.arange(N, device=device)
    sample_idx = idx // K
    pos_mask = sample_idx.unsqueeze(0) == sample_idx.unsqueeze(1)  # N x N
    pos_mask = pos_mask & (~diag_mask)

    # For each i, numerator is sum over positives, denominator is sum over all non-self
    exp_sim = sim_masked.exp()
    denom = exp_sim.sum(dim=1)  # (N,)
    numerator = (exp_sim * pos_mask.float()).sum(dim=1)  # (N,)

    # avoid zeros
    frac = numerator / (denom + eps)
    loss = -torch.log(frac + eps)
    return loss.mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch, args, scaler=None):
    model.train()
    losses = []
    pbar = tqdm(dataloader, desc=f"train epoch {epoch}", leave=False)
    for crops, box_enc, labels in pbar:
        # crops: (B, K, 3, H, W)
        crops = crops.to(device, dtype=torch.float32)
        labels = labels.view(-1).long().to(device)

        B, K = crops.shape[0], crops.shape[1]

        # forward
        # If backbone is expensive we used no-grad in forward_backbone; but we want to fine-tune optionally.
        # We'll temporarily enable grads for backbone if any parameters require grad
        backbone_requires_grad = any(p.requires_grad for p in model.backbone.parameters())

        if scaler is not None:
            # prefer torch.amp.autocast when available
            try:
                from torch import amp
                with amp.autocast(enabled=True, device_type='cuda'):
                    pooled, crop_feats, crop_proj = model(crops)
                    # classification logits
                    logits = model.classifier(pooled)
                    ce_loss = F.cross_entropy(logits, labels)

                    # contrastive: compute normalized embedding per crop (flattened)
                    emb = F.normalize(crop_proj.view(B * K, -1), dim=1)
                    contrastive_loss = nt_xent_loss_from_embeddings(emb, B, K, temperature=args.temperature)

                    loss = ce_loss * args.ce_weight + contrastive_loss * args.contrastive_weight
            except Exception:
                # fallback to older API
                with torch.cuda.amp.autocast():
                    pooled, crop_feats, crop_proj = model(crops)
                    # classification logits
                    logits = model.classifier(pooled)
                    ce_loss = F.cross_entropy(logits, labels)

                    # contrastive: compute normalized embedding per crop (flattened)
                    emb = F.normalize(crop_proj.view(B * K, -1), dim=1)
                    contrastive_loss = nt_xent_loss_from_embeddings(emb, B, K, temperature=args.temperature)

                    loss = ce_loss * args.ce_weight + contrastive_loss * args.contrastive_weight

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            pooled, crop_feats, crop_proj = model(crops)
            logits = model.classifier(pooled)
            ce_loss = F.cross_entropy(logits, labels)
            emb = F.normalize(crop_proj.view(B * K, -1), dim=1)
            contrastive_loss = nt_xent_loss_from_embeddings(emb, B, K, temperature=args.temperature)
            loss = ce_loss * args.ce_weight + contrastive_loss * args.contrastive_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({'loss': sum(losses)/len(losses)})

    return sum(losses)/len(losses)


@torch.no_grad()
def validate(model, dataloader, device, args):
    model.eval()
    total = 0
    correct = 0
    losses = []
    for crops, box_enc, labels in tqdm(dataloader, desc="validate", leave=False):
        crops = crops.to(device, dtype=torch.float32)
        labels = labels.view(-1).long().to(device)
        pooled, crop_feats, crop_proj = model(crops)
        logits = model.classifier(pooled)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    acc = correct / total if total > 0 else 0.0
    return sum(losses)/len(losses) if losses else 0.0, acc


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"[INFO] Saved checkpoint to {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', type=str, required=True)
    p.add_argument('--train_img_base', type=str, required=True)
    p.add_argument('--train_text_base', type=str, required=True)
    p.add_argument('--val_csv', type=str, required=True)
    p.add_argument('--val_img_base', type=str, required=True)
    p.add_argument('--val_text_base', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=5, help='early stopping patience on val loss')
    p.add_argument('--min_delta', type=float, default=1e-4, help='minimum change to qualify as improvement')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--img_size', type=int, default=448)
    p.add_argument('--topk', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone_lr', type=float, default=1e-5)
    p.add_argument('--proj_dim', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--contrastive_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--freeze_until_name', type=str, default=None, help='If set, freeze backbone params not containing this substring')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--backbone_load_mode', type=str, default='auto', choices=['timm','hub','auto'])
    p.add_argument('--backbone_name', type=str, default=None, help='timm backbone name (if using timm)')
    p.add_argument('--hub_model_name', type=str, default=None, help='torch.hub model name (if using hub)')
    p.add_argument('--backbone_version', type=str, choices=['v2', 'v3'], default=None,
                   help='shortcut to select backbone model: v2 -> dinov2_vitb14, v3 -> dinov3_vitb16')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # map shorthand backbone_version to hub_model_name when provided
    if args.backbone_version is not None:
        if args.backbone_version == 'v2':
            # prefer the DINOv2 base vit b14 hub name
            args.hub_model_name = args.hub_model_name or 'dinov2_vitb14'
        elif args.backbone_version == 'v3':
            # prefer the canonical dinov3 hub name used elsewhere in the repo
            args.hub_model_name = args.hub_model_name or 'dinov3-vitb16-pretrain-lvd1689m'

    # dataset
    train_dataset = all_mammo(args.train_csv, args.train_img_base, args.train_text_base, topk=args.topk, img_size=args.img_size, enable_augmentation=True)
    val_dataset = all_mammo(args.val_csv, args.val_img_base, args.val_text_base, topk=args.topk, img_size=args.img_size, enable_augmentation=False)
    # split into train/val simple split (you should use official splits)


   

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # load backbone
    backbone, backbone_dim = load_dinov2_backbone(device, pretrained=True,
                                                  backbone_name=args.backbone_name,
                                                  load_mode=args.backbone_load_mode,
                                                  hub_model_name=args.hub_model_name)

    model = FinetuneModel(backbone=backbone, backbone_dim=backbone_dim, proj_dim=args.proj_dim,
                          num_classes=2, freeze_backbone_until_layer=args.freeze_until_name)
    model.to(device)

    # prepare optimizer: separate params for backbone and heads
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    optim_params = []
    if len(backbone_params) > 0:
        optim_params.append({'params': backbone_params, 'lr': args.backbone_lr})
    optim_params.append({'params': head_params, 'lr': args.lr})
    optimizer = torch.optim.AdamW(optim_params, weight_decay=1e-2)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    # Prefer new torch.amp API (GradScaler accepts enabled flag)
    try:
        from torch import amp
        scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    except Exception:
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, device, args)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  time={elapsed:.1f}s")

        # save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': epoch,
            'args': vars(args),
        }
        ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt')
        save_checkpoint(ckpt, ckpt_path)

        # save best
        if val_loss + args.min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(ckpt, os.path.join(args.save_dir, 'best_checkpoint.pt'))
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (best={best_val_loss:.4f})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs (patience={args.patience})")
                break


if __name__ == '__main__':
    import random
    random.seed(42)
    main()

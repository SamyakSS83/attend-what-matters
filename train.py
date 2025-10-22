import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import json

from data import all_mammo
from new_model import MMBCDContrast


def repulsive_contrastive_loss(roi_embeddings, exclude_anchor=True, eps=1e-8):
    """
    Compute a simple repulsive contrastive loss over ROI embeddings.
    The goal is to make different proposals (ROIs) have distinct embeddings.

    roi_embeddings: Tensor of shape (B, S, D)
    exclude_anchor: if True, exclude index 0 (full-breast anchor) and compute
                    the loss only among the proposal ROIs.

    Loss (per-batch) is the mean squared off-diagonal cosine similarity:
      for each sample: sim_matrix = E E^T (E normalized)
      loss_sample = mean_{i != j} (sim_{ij}^2)
    Returns scalar tensor.
    """
    # roi_embeddings: (B, S, D)
    if roi_embeddings is None:
        return torch.tensor(0.0, device=roi_embeddings.device if hasattr(roi_embeddings, 'device') else 'cpu')
    B, S, D = roi_embeddings.shape
    if exclude_anchor:
        if S <= 1:
            return torch.tensor(0.0, device=roi_embeddings.device)
        emb = roi_embeddings[:, 1:, :]
    else:
        emb = roi_embeddings
    B, K, D = emb.shape
    # normalize embeddings along feature dim
    emb_norm = F.normalize(emb, dim=-1)
    # pairwise cosine similarity per sample: (B, K, K)
    sim = torch.einsum('bkd,bjd->bkj', emb_norm, emb_norm)
    # zero out diagonal
    if K <= 1:
        return torch.tensor(0.0, device=roi_embeddings.device)
    mask = torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(mask, 0.0)
    # mean squared off-diagonal similarity per sample
    denom = float(K * (K - 1))
    loss_per_sample = (sim * sim).sum(dim=(1, 2)) / (denom + eps)
    return loss_per_sample.mean()


def repulsive_contrastive_loss_batch(roi_embeddings, eps=1e-8):
    """
    Compute repulsive contrastive loss across the entire batch,
    excluding the first ROI (anchor) from each sample.

    roi_embeddings: Tensor of shape (B, S, D)
    Returns scalar tensor.
    """
    if roi_embeddings is None:
        return torch.tensor(0.0, device='cpu')
    
    B, S, D = roi_embeddings.shape
    if S <= 1:
        return torch.tensor(0.0, device=roi_embeddings.device)
    
    # Exclude the anchor (first ROI)
    emb = roi_embeddings[:, 1:, :]  # shape: (B, S-1, D)
    
    # Flatten batch and ROIs: (B*(S-1), D)
    emb_flat = emb.reshape(-1, D)
    
    K = emb_flat.shape[0]
    if K <= 1:
        return torch.tensor(0.0, device=roi_embeddings.device)
    
    # Normalize embeddings
    emb_norm = F.normalize(emb_flat, dim=-1)
    
    # Compute pairwise cosine similarity: (K, K)
    sim = torch.matmul(emb_norm, emb_norm.T)
    
    # Zero out diagonal (self-similarity)
    mask = torch.eye(K, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, 0.0)
    
    # Mean squared off-diagonal similarity
    loss = (sim * sim).sum() / (K * (K - 1) + eps)
    
    return loss



class WrappedDataset(torch.utils.data.Dataset):
    """Wrap all_mammo to expose image_path and raw proposals for IoU calculation."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        crops, pos, label = self.base[idx]
        # raw proposals and image_path available on base dataset
        proposals = self.base.all_proposals[idx]
        image_path = self.base.image_path_list[idx]
        return crops, pos, label, image_path, proposals


def collate_fn(batch):
    # batch is list of tuples: crops (S,3,H,W) tensors, pos (S,D), label (1,), image_path str, proposals np.array (S,5)
    crops_list = [b[0] for b in batch]
    pos_list = [b[1] for b in batch]
    labels = torch.cat([b[2] for b in batch], dim=0)
    image_paths = [b[3] for b in batch]
    proposals = [b[4] for b in batch]
    # stack crops -> (B, S, 3, H, W)
    crops_stack = torch.stack(crops_list, dim=0)
    pos_stack = torch.stack(pos_list, dim=0)
    return crops_stack, pos_stack, labels, image_paths, proposals


# NOTE: COCO json handling removed. Contrastive loss runs directly on ROI embeddings.


def train_epoch(model, dataloader, optimizer, device, contrastive_weight=1.0):
    model.train()
    total_loss = 0.0
    for X_images, X_positions, y, image_paths, proposals in tqdm(dataloader):
        X_images = X_images.to(device)
        X_positions = X_positions.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        logits, roi_embeddings = model(X_images, X_positions)

        # flatten logits and labels to 1D for stable loss computation
        logits = logits.view(-1)
        y = y.view(-1)
        # classification loss: logits are raw (not passed through sigmoid),
        # so use the numerically-stable BCE with logits variant.
        cls_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(10.0, device=device))

        # compute repulsive contrastive loss directly on ROI embeddings (exclude full-breast anchor)
        # this encourages distinct embeddings across proposals so only the correct ROI stands out
        contrastive_loss = repulsive_contrastive_loss(roi_embeddings, exclude_anchor=True)
        # contrastive_loss = repulsive_contrastive_loss_batch(roi_embeddings)

        loss = cls_loss + contrastive_weight * contrastive_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_epoch(model, dataloader, device, contrastive_weight=1.0):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_images, X_positions, y, image_paths, proposals in tqdm(dataloader):
            X_images = X_images.to(device)
            X_positions = X_positions.to(device)
            y = y.to(device).float()

            logits, roi_embeddings = model(X_images, X_positions)
            logits = logits.view(-1)
            y = y.view(-1)
            # use BCE with logits (logits are raw outputs)
            cls_loss = F.binary_cross_entropy_with_logits(logits, y)

            # compute repulsive contrastive loss on ROI embeddings (exclude anchor)
            contrastive_loss = repulsive_contrastive_loss(roi_embeddings, exclude_anchor=True)
            loss = cls_loss + contrastive_weight * contrastive_loss

            total_loss += loss.item()
            # convert logits to probabilities for metric computation
            probs = torch.sigmoid(logits)
            all_preds.append(probs.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_preds = np.array([])
        all_targets = np.array([])

    # compute simple metrics if possible
    metrics = {}
    try:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        preds_bin = (all_preds >= 0.5).astype(int) if all_preds.size else np.array([])
        metrics['accuracy'] = float(accuracy_score(all_targets, preds_bin)) if all_targets.size else None
        metrics['f1'] = float(f1_score(all_targets, preds_bin)) if all_targets.size else None
        metrics['roc_auc'] = float(roc_auc_score(all_targets, all_preds)) if all_targets.size else None
    except Exception:
        metrics = {}

    return total_loss / len(dataloader), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrastive_weight', type=float, default=0.4,
                    help='weight for contrastive loss (set lower than cls loss)')
    parser.add_argument('--lr', type=float, default= 1e-5, help='learning rate for AdamW')

    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--train_img_base', required=True)
    parser.add_argument('--train_text_base', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_csv', required=False, help='validation csv')
    parser.add_argument('--val_img_base', required=False, help='validation image base')
    parser.add_argument('--val_text_base', required=False, help='validation proposals base')
    parser.add_argument('--out_dir', default='./outputs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience on val loss')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='minimum change to qualify as improvement')
    parser.add_argument('--fp16', action='store_true', help='use AMP (mixed precision) for training')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze the vision backbone to speed up training')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for dataloaders')
    parser.add_argument('--cudnn_benchmark', action='store_true', help='set torch.backends.cudnn.benchmark = True')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--topk', type=int, default=4, help='number of ROIs per image (including full-breast)')
    parser.add_argument('--pool_mode', choices=['anchor', 'attn', 'avg', 'cls'], default='anchor', help="Pooling mode: 'anchor'=0th ROI, 'attn'=attention pooling, 'avg'=mean over ROIs, 'cls'=learned CLS token")
    parser.add_argument('--pool_attn_block', type=int, choices=[1,2,3], default=3, help='Transformer block to use for attention pooling when pool_mode=attn')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_dataset = all_mammo(args.train_csv, args.train_img_base, args.train_text_base, topk=args.topk, enable_augmentation=True, cache_dir='./cache_train')
    dataset = WrappedDataset(base_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    # no COCO json required; contrastive loss computed directly on ROI embeddings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MMBCDContrast(pool_mode=args.pool_mode, pool_attn_block=args.pool_attn_block)
    model = model.to(device)
    if args.freeze_backbone:
        try:
            for p in model.vision_model.parameters():
                p.requires_grad = False
        except Exception:
            pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == 'cuda')

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    n_epochs = args.epochs
    # optionally build validation dataloader
    val_loader = None
    if args.val_csv and args.val_img_base and args.val_text_base:
        val_base = all_mammo(args.val_csv, args.val_img_base, args.val_text_base, topk=args.topk, enable_augmentation=False, cache_dir='./cache_val')
        val_dataset = WrappedDataset(val_base)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, n_epochs + 1):
        # train using repulsive contrastive loss over ROI embeddings
        train_loss = train_epoch(model, dataloader, optimizer, device,
                         contrastive_weight=args.contrastive_weight)
        print(f'Epoch {epoch} train loss: {train_loss:.4f}')
        ckpt = os.path.join(args.out_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt)
        attn = model.get_attention_maps()
        # save attention if present
        for k, v in attn.items():
            if v is not None:
                np.save(os.path.join(args.out_dir, f'{k}_epoch{epoch}.npy'), v.detach().cpu().numpy())

        if val_loader is not None:
            # val_loss, metrics = evaluate_epoch(model, val_loader, device, coco_val_gt)
            val_loss, metrics = evaluate_epoch(model, val_loader, device,
                                   contrastive_weight=args.contrastive_weight)
            print(f'Epoch {epoch} val loss: {val_loss:.4f} metrics: {metrics}')
            # early stopping check
            if val_loss + args.min_delta < best_val:
                best_val = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f'No improvement for {epochs_no_improve} epochs')
                if epochs_no_improve >= args.patience:
                    print(f'Early stopping after {epoch} epochs')
                    break

        # save checkpoint and attention maps for inspection
        # ckpt = os.path.join(args.out_dir, f'model_epoch_{epoch}.pt')
        # torch.save(model.state_dict(), ckpt)
        # attn = model.get_attention_maps()
        # # save attention if present
        # for k, v in attn.items():
        #     if v is not None:
        #         np.save(os.path.join(args.out_dir, f'{k}_epoch{epoch}.npy'), v.detach().cpu().numpy())


if __name__ == '__main__':
    main()

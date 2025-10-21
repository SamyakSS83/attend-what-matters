import os
import argparse
import json

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    # already set
    pass

import torch
import numpy as np
from tqdm import tqdm

from data import all_mammo
from new_model import MMBCDContrast


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    # support checkpoints saved as {'model': state_dict, ...} or raw state_dict
    if isinstance(state, dict) and 'model' in state:
        state_dict = state['model']
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device,  out_dir=None):
    preds = []
    targets = []
    image_paths = []
    attn_maps = {}
    anchor_embs = []
    proposal_embs = []

    with torch.no_grad():
        for X_images, X_positions, y, img_paths, proposals in tqdm(dataloader):
            X_images = X_images.to(device)
            X_positions = X_positions.to(device)
            y = y.to(device).float()

            logits, roi_embeddings = model(X_images, X_positions)
            logits = logits.view(-1).detach().cpu().numpy()
            preds.append(logits)
            targets.append(y.view(-1).detach().cpu().numpy())
            image_paths.extend(img_paths)
            # collect embeddings: anchor (index 0) per image and proposal embeddings (1:)
            try:
                anc = roi_embeddings[:, 0, :].detach().cpu().numpy()
                anchor_embs.append(anc)
                # proposals excluding anchor
                if roi_embeddings.shape[1] > 1:
                    props = roi_embeddings[:, 1:, :].detach().cpu().numpy()
                    proposal_embs.append(props)
            except Exception:
                # if model doesn't return embeddings in expected shape, skip
                pass

            # collect attention maps (if any)
            atm = model.get_attention_maps()
            for k, v in atm.items():
                if v is not None:
                    attn_maps.setdefault(k, []).append(v.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0) if preds else np.array([])
    targets = np.concatenate(targets, axis=0) if targets else np.array([])

    # save predictions
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'preds.npy'), preds)
        np.save(os.path.join(out_dir, 'targets.npy'), targets)
        with open(os.path.join(out_dir, 'image_paths.txt'), 'w') as f:
            for p in image_paths:
                f.write(p + '\n')
        # save attention arrays per block
        for k, vlist in attn_maps.items():
            # stack along 0
            try:
                stacked = np.stack(vlist, axis=0)
                np.save(os.path.join(out_dir, f'attn_{k}.npy'), stacked)
            except Exception:
                pass

        # concatenate collected embeddings
        try:
            if anchor_embs:
                anchor_arr = np.concatenate(anchor_embs, axis=0)
                np.save(os.path.join(out_dir, 'anchor_embeddings.npy'), anchor_arr)
            else:
                anchor_arr = None
            if proposal_embs:
                # proposal_embs is list of (B, S-1, D) -> stack and reshape to (N, D)
                prop_stack = np.concatenate(proposal_embs, axis=0)
                prop_flat = prop_stack.reshape(-1, prop_stack.shape[-1])
                np.save(os.path.join(out_dir, 'proposal_embeddings.npy'), prop_flat)
            else:
                prop_flat = None
        except Exception:
            anchor_arr = None
            prop_flat = None

        # Run UMAP on anchor embeddings (per-image) and compute k-NN score on the 2D embedding
        umap_metrics = {}
        try:
            if anchor_arr is not None and anchor_arr.shape[0] >= 5:
                try:
                    import umap.umap_ as umap
                except Exception:
                    try:
                        import umap
                    except Exception:
                        umap = None
                if umap is not None:
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    emb2d = reducer.fit_transform(anchor_arr)
                    np.save(os.path.join(out_dir, 'umap_anchors_2d.npy'), emb2d)
                    # scatter plot colored by target
                    try:
                        import matplotlib.pyplot as plt
                        tgt = np.concatenate(targets, axis=0) if targets else np.array([])
                        plt.figure(figsize=(6, 6))
                        sc = plt.scatter(emb2d[:, 0], emb2d[:, 1], c=tgt, cmap='coolwarm', s=8, alpha=0.8)
                        plt.colorbar(sc)
                        plt.title('UMAP anchors')
                        plt.savefig(os.path.join(out_dir, 'umap_anchors.png'), dpi=150)
                        plt.close()
                    except Exception:
                        pass

                    # compute k-NN score on 2D embedding using cross-validation if sklearn available
                    try:
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.model_selection import cross_val_score
                        tgt = np.concatenate(targets, axis=0)
                        n_samples = emb2d.shape[0]
                        cv = 5 if n_samples >= 5 else max(2, n_samples)
                        knn = KNeighborsClassifier(n_neighbors=5)
                        scores = cross_val_score(knn, emb2d, tgt, cv=cv, scoring='accuracy')
                        scores_list = [float(s) for s in scores]
                        umap_metrics['umap_knn_accuracy_mean'] = float(np.mean(scores_list))
                        umap_metrics['umap_knn_accuracy_std'] = float(np.std(scores_list))
                        umap_metrics['umap_knn_scores'] = scores_list
                        umap_metrics['umap_knn_n_neighbors'] = int(knn.n_neighbors)
                        umap_metrics['umap_knn_cv_folds'] = int(cv)
                        # also print to stdout so the user sees these values immediately
                        print(f"UMAP k-NN scores (cv={cv}, k={knn.n_neighbors}): {scores_list}")
                    except Exception:
                        pass

        except Exception:
            pass

        # save umap metrics if collected
        if umap_metrics:
            try:
                with open(os.path.join(out_dir, 'umap_metrics.json'), 'w') as f:
                    json.dump(umap_metrics, f, indent=2)
            except Exception:
                pass

    # compute basic metrics if sklearn available
    metrics = {}
    try:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
        if preds.size and targets.size:
            # default threshold 0.5
            preds_bin_05 = (preds >= 0.5).astype(int)
            metrics['accuracy'] = float(accuracy_score(targets, preds_bin_05))
            metrics['f1'] = float(f1_score(targets, preds_bin_05))
            # precision / recall at 0.5
            metrics['precision_0.5'] = float(precision_score(targets, preds_bin_05, zero_division=0))
            metrics['recall_0.5'] = float(recall_score(targets, preds_bin_05, zero_division=0))
            # recall at multiple thresholds
            for thr in (0.1, 0.3, 0.5):
                key = f'recall@{thr}' if thr != 0.5 else 'recall@0.5'
                preds_bin = (preds >= thr).astype(int)
                try:
                    metrics[key] = float(recall_score(targets, preds_bin, zero_division=0))
                except Exception:
                    metrics[key] = None

            # false positive rate at 0.5: FP / (FP + TN)
            try:
                tn, fp, fn, tp = confusion_matrix(targets, preds_bin_05).ravel()
                denom = float(fp + tn)
                metrics['fpr_0.5'] = float(fp / denom) if denom > 0 else None
            except Exception:
                # fallback: compute from counts
                try:
                    cm = confusion_matrix(targets, preds_bin_05)
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                        denom = float(fp + tn)
                        metrics['fpr_0.5'] = float(fp / denom) if denom > 0 else None
                    else:
                        metrics['fpr_0.5'] = None
                except Exception:
                    metrics['fpr_0.5'] = None

            # ROC AUC using raw scores when possible
            try:
                metrics['roc_auc'] = float(roc_auc_score(targets, preds))
            except Exception:
                metrics['roc_auc'] = None

            # Precision-Recall curve and Average Precision (AUC of PR)
            try:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                if preds.size and targets.size:
                    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(targets, preds)
                    avg_prec = float(average_precision_score(targets, preds))
                    metrics['avg_precision'] = avg_prec
                    # Save arrays to out_dir if available
                    if out_dir:
                        try:
                            np.save(os.path.join(out_dir, 'pr_precision.npy'), precision_vals)
                            np.save(os.path.join(out_dir, 'pr_recall.npy'), recall_vals)
                            np.save(os.path.join(out_dir, 'pr_thresholds.npy'), pr_thresholds)
                        except Exception:
                            pass
                    # Plot PR curve
                    if out_dir:
                        try:
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(6, 6))
                            plt.plot(recall_vals, precision_vals, lw=2, color='b')
                            plt.xlabel('Recall')
                            plt.ylabel('Precision')
                            plt.title(f'Precision-Recall curve (AP={avg_prec:.4f})')
                            plt.grid(True)
                            plt.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
                            plt.close()
                        except Exception:
                            pass
                    # Also print average precision to stdout
                    try:
                        print(f"  Average Precision (AP / AUC-PR): {avg_prec}")
                    except Exception:
                        pass
            except Exception:
                # sklearn not available -- skip PR curve
                pass

            # Print concise metrics to stdout so user sees them immediately
            try:
                print('\nEvaluation metrics:')
                print(f"  Precision@0.5: {metrics.get('precision_0.5')}")
                print(f"  Recall@0.5:    {metrics.get('recall_0.5')}")
                print(f"  Recall@0.1:    {metrics.get('recall@0.1')}")
                print(f"  Recall@0.3:    {metrics.get('recall@0.3')}")
                print(f"  Recall@0.5:    {metrics.get('recall@0.5')}")
                print(f"  FPR@0.5:       {metrics.get('fpr_0.5')}")
                print(f"  ROC AUC:       {metrics.get('roc_auc')}")
            except Exception:
                pass

        # incorporate umap_metrics into returned metrics if available
        if out_dir:
            try:
                umf = os.path.join(out_dir, 'umap_metrics.json')
                if os.path.exists(umf):
                    with open(umf, 'r') as f:
                        um = json.load(f)
                    metrics.update(um)
            except Exception:
                pass
    except Exception:
        # sklearn not available or other error; leave metrics empty or with roc_auc if possible
        try:
            # try roc_auc only
            from sklearn.metrics import roc_auc_score
            if preds.size and targets.size:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(targets, preds))
                except Exception:
                    metrics['roc_auc'] = None
        except Exception:
            pass

    return metrics


# worker must be at module level so multiprocessing can pickle it
def eval_one_worker(args_tuple):
    ckpt_path, args_dict = args_tuple
    try:
        device_w = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = MMBCDContrast()
        load_checkpoint(m, ckpt_path, device_w)

        # recreate validation dataset/loader inside worker to avoid pickling
        val_base_w = all_mammo(args_dict['val_csv'], args_dict['val_img_base'], args_dict['val_text_base'], topk=4, enable_augmentation=False,cache_dir='./cache_eval')
        try:
            from train import WrappedDataset, collate_fn
        except Exception as e:
            return os.path.basename(ckpt_path), {'error': f'Failed importing train helpers in worker: {e}'}
        val_dataset_w = WrappedDataset(val_base_w)
        from torch.utils.data import DataLoader
        val_loader_w = DataLoader(val_dataset_w, batch_size=args_dict.get('batch_size', 32), shuffle=False, num_workers=args_dict.get('num_workers', 4), collate_fn=collate_fn)
        # coco_gt_w = load_coco_gt(args_dict['val_coco'])

        out_sub = os.path.join(args_dict.get('out_dir', './eval_outputs'), os.path.basename(ckpt_path).replace('.pt', ''))
        met = evaluate(m, val_loader_w, device_w,  out_dir=out_sub)
        return os.path.basename(ckpt_path), met
    except Exception as e:
        import traceback
        return os.path.basename(ckpt_path), {'error': str(e), 'trace': traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ckpt', help='model checkpoint .pt')
    group.add_argument('--ckpt_folder', help='folder containing multiple .pt checkpoints to evaluate')
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--val_img_base', required=True)
    parser.add_argument('--val_text_base', required=True)
    # parser.add_argument('--val_coco', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', default='./eval_outputs')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--topk', type=int, default=4, help='Number of proposals per image')
    parser.add_argument('--concurrent', type=int, default=4, help='Number of concurrent evaluations when evaluating multiple checkpoints')
    parser.add_argument('--pool_mode', choices=['anchor', 'attn', 'avg', 'cls'], default='anchor', help="Pooling mode used by model (anchor/attn/avg/cls)")
    parser.add_argument('--pool_attn_block', type=int, choices=[1,2,3], default=3, help='Block to use for attention pooling when pool_mode=attn')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Do not load any checkpoint yet. If a single --ckpt is specified we'll load it below.

    val_base = all_mammo(args.val_csv, args.val_img_base, args.val_text_base, topk=args.topk, enable_augmentation=False, cache_dir='./cache_eval')
    # reuse WrappedDataset / collate_fn from train.py by importing here to avoid duplication
    try:
        from train import WrappedDataset, collate_fn
    except Exception as e:
        raise RuntimeError('Could not import WrappedDataset/collate_fn from train.py. Make sure train.py is in the same folder and syntactically correct.') from e
    val_dataset = WrappedDataset(val_base)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # coco_gt = load_coco_gt(args.val_coco)
    # if single checkpoint requested
    if args.ckpt:
        model = MMBCDContrast(pool_mode=args.pool_mode, pool_attn_block=args.pool_attn_block)
        load_checkpoint(model, args.ckpt, device)
        metrics = evaluate(model, val_loader, device,  out_dir=args.out_dir)
        print(f'Checkpoint {args.ckpt} metrics:', metrics)
        # save summary
        with open(os.path.join(args.out_dir, 'metrics_summary.json'), 'w') as f:
            json.dump({os.path.basename(args.ckpt): metrics}, f, indent=2)
        return

    # otherwise evaluate all checkpoints in the folder, up to 4 in parallel
    ckpt_files = [os.path.join(args.ckpt_folder, p) for p in os.listdir(args.ckpt_folder) if p.endswith('.pt')]
    ckpt_files = sorted(ckpt_files)
    if not ckpt_files:
        print('No .pt files found in folder', args.ckpt_folder)
        return

    # we'll run up to 4 evaluations concurrently using multiprocessing
    from multiprocessing import Pool

    # Instead of forking workers that initialize CUDA, launch independent subprocesses
    # that run this script in single-checkpoint mode. This avoids CUDA re-init issues
    # caused by fork on Linux. We'll run up to 4 subprocesses concurrently.
    import sys
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_subproc(ckpt_path):
        out_sub = os.path.join(args.out_dir, os.path.basename(ckpt_path).replace('.pt', ''))
        os.makedirs(out_sub, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), '--ckpt', ckpt_path,
               '--val_csv', args.val_csv,
               '--val_img_base', args.val_img_base,
               '--val_text_base', args.val_text_base,
               '--pool_mode', args.pool_mode,
               '--pool_attn_block', str(args.pool_attn_block),
            #    '--val_coco', args.val_coco,
               '--batch_size', str(args.batch_size),
               '--out_dir', out_sub,
               '--num_workers', str(args.num_workers)]
        # run and capture output
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return os.path.basename(ckpt_path), {'error': f'Process failed', 'stdout': proc.stdout, 'stderr': proc.stderr}
            # read metrics_summary.json written inside out_sub
            met_file = os.path.join(out_sub, 'metrics_summary.json')
            if os.path.exists(met_file):
                try:
                    with open(met_file, 'r') as f:
                        d = json.load(f)
                    # metrics_summary.json contains {ckpt_basename: metrics}
                    key = os.path.basename(ckpt_path)
                    return key, d.get(key, d)
                except Exception as e:
                    return os.path.basename(ckpt_path), {'error': f'Failed reading metrics file: {e}'}
            else:
                return os.path.basename(ckpt_path), {'error': 'metrics file not produced', 'stdout': proc.stdout, 'stderr': proc.stderr}
        except Exception as e:
            return os.path.basename(ckpt_path), {'error': str(e)}

    summary = {}
    max_workers = min(args.concurrent, len(ckpt_files))
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_subproc, p): p for p in ckpt_files}
        # show a single tqdm progress bar for all subprocess evaluations
        for fut in tqdm(as_completed(futures), total=len(futures), desc='evaluating checkpoints'):
            name, met = fut.result()
            summary[name] = met

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Wrote metrics summary for', len(summary), 'checkpoints to', os.path.join(args.out_dir, 'metrics_summary.json'))


if __name__ == '__main__':
    main()

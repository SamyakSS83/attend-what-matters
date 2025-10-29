import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import json

# === Paths ===
# mammo_probs_path = "mammo_clip_vindr_test.npy"   # MammoCLIP probabilities
unibcd_probs_path = "preds.npy"                  # UniBCD probabilities
csv_path = "/home/samyak/scratch/my_vindr/model_specific_preprocessed_data/mmbcd_csvs/test.csv"
output_png = "precision_recall_overlay.png"

# === Load data ===
# mammo_probs = np.load(mammo_probs_path)
unibcd_probs = np.load(unibcd_probs_path)
df = pd.read_csv(csv_path)

# assert len(mammo_probs) == len(df), f"Length mismatch: mammo {len(mammo_probs)} vs {len(df)}"
assert len(unibcd_probs) == len(df), f"Length mismatch: unibcd {len(unibcd_probs)} vs {len(df)}"

y_true = df["cancer"].astype(int).values

# === Compute PR curves ===
def compute_pr_metrics(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f1 = f1_scores[best_idx]
    return precision, recall, ap, best_threshold, best_f1


def compute_roc_metrics(y_true, y_scores, fpr_targets=(0.1, 0.3, 0.5)):
    """Return AUROC and recall (TPR) at requested FPR thresholds.

    Uses interpolation on the ROC curve to get TPR at exact FPR values.
    If AUROC cannot be computed (e.g., only one class present), returns np.nan for auroc and for r_at_fprs.
    """
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except Exception:
        auroc = np.nan

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Ensure FPR is increasing for interpolation (it is by construction)
    r_at_fprs = {}
    for ft in fpr_targets:
        if ft <= 0:
            r_at_fprs[ft] = 0.0
        else:
            # If requested FPR is beyond max fpr, take last tpr
            if ft >= fpr.max():
                r_at_fprs[ft] = float(tpr[-1])
            else:
                r_at_fprs[ft] = float(np.interp(ft, fpr, tpr))

    return auroc, fpr, tpr, r_at_fprs


def compute_froc_metrics(y_true, y_scores, fppi_targets=(0.1, 0.3, 0.5)):
    """Compute a simple FROC curve from image-level scores.

    This converts ROC's FPR to FP per image (FPPI) using:
        FPPI = FPR * (N_neg / N_images)

    Returns: fppi (array), tpr (array), sens_at_fppi (dict)
    """
    n_images = len(y_true)
    if n_images == 0:
        return np.array([]), np.array([]), {t: float('nan') for t in fppi_targets}

    n_neg = int((y_true == 0).sum())
    n_pos = int((y_true == 1).sum())

    # If there are no negative samples, FPPI cannot be defined
    if n_neg == 0:
        return np.array([]), np.array([]), {t: float('nan') for t in fppi_targets}

    # ROC provides FPR and TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Convert to FP per image (FPPI)
    fppi = fpr * (n_neg / float(n_images))

    # Interpolate TPR at requested FPPI targets
    sens_at_fppi = {}
    for ft in fppi_targets:
        if ft <= 0:
            sens_at_fppi[ft] = 0.0
        else:
            if len(fppi) == 0:
                sens_at_fppi[ft] = float('nan')
            else:
                # If target beyond max FPPI, take last TPR
                if ft >= fppi.max():
                    sens_at_fppi[ft] = float(tpr[-1])
                else:
                    sens_at_fppi[ft] = float(np.interp(ft, fppi, tpr))

    return fppi, tpr, sens_at_fppi

# MammoCLIP
# p_mammo, r_mammo, ap_mammo, th_mammo, f1_mammo = compute_pr_metrics(y_true, mammo_probs)
# UniBCD
p_uni, r_uni, ap_uni, th_uni, f1_uni = compute_pr_metrics(y_true, unibcd_probs)
auroc, fpr, tpr, r_at_fprs = compute_roc_metrics(y_true, unibcd_probs)

# Print AUROC and recalls at specific FPR thresholds in a concise table-like format
print("\n=== UniBCD ROC ===")
if np.isnan(auroc):
    print("AUROC: nan (could not be computed)")
else:
    print(f"AUROC: {auroc:.4f}")

# Print Recall (TPR) at requested FPRs
for ft in (0.1, 0.3, 0.5):
    val = r_at_fprs.get(ft, float('nan'))
    if np.isnan(val):
        print(f"R@{ft:.1f} FPR:\tN/A")
    else:
        print(f"R@{ft:.1f} FPR:\t{val:.4f}")

# print("=== MammoCLIP ===")
# print(f"Average Precision (AP): {ap_mammo:.4f}")
# print(f"Best threshold (max F1): {th_mammo:.4f}")
# print(f"F1 score at best threshold: {f1_mammo:.4f}")

print("\n=== UniBCD ===")
print(f"Average Precision (AP): {ap_uni:.4f}")
print(f"Best threshold (max F1): {th_uni:.4f}")
print(f"F1 score at best threshold: {f1_uni:.4f}")

# === Plot overlay ===
plt.figure(figsize=(7,6))
plt.plot(r_uni, p_uni, label=f"UniBCD (AP={ap_uni:.4f}) ", lw=2)
# plt.plot(r_mammo, p_mammo, label=f"MammoCLIP (AP={ap_mammo:.4f})", lw=2, linestyle="--")

plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision–Recall Curve Comparison", fontsize=14)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.show()

print(f"Overlay plot saved to: {output_png}")

# === Compute and save FROC ===
output_froc_png = "froc_curve.png"
output_froc_npz = "froc_values.npz"

# Compute FROC (FP per image vs sensitivity)
fppi, froc_tpr, sens_at_fppi = compute_froc_metrics(y_true, unibcd_probs, fppi_targets=(0.1, 0.3, 0.5))

print("\n=== UniBCD FROC (FP per image) ===")
if fppi.size == 0:
    print("FROC could not be computed (insufficient positives/negatives).")
else:
    # Print sensitivities at requested FPPI points
    for ft in (0.1, 0.3, 0.5):
        val = sens_at_fppi.get(ft, float('nan'))
        if np.isnan(val):
            print(f"Sens@FPPI={ft:.1f}: N/A")
        else:
            print(f"Sens@FPPI={ft:.1f}: {val:.4f}")

    # Save numeric FROC data
    try:
        np.savez(output_froc_npz, fppi=fppi, tpr=froc_tpr, sens_at_fppi=json.dumps(sens_at_fppi))
        print(f"FROC numeric values saved to: {output_froc_npz}")
    except Exception as e:
        print(f"Warning: could not save FROC numeric values: {e}")

    # Plot FROC curve
    plt.figure(figsize=(7,6))
    plt.plot(fppi, froc_tpr, lw=2, label="UniBCD FROC")
    # mark requested FPPI targets
    xs = []
    ys = []
    for ft in (0.1, 0.3, 0.5):
        val = sens_at_fppi.get(ft, None)
        if val is not None and not np.isnan(val):
            xs.append(ft)
            ys.append(val)
    if xs:
        plt.scatter(xs, ys, c='red', zorder=5)
        for x,y in zip(xs, ys):
            plt.annotate(f"{y:.3f}@{x}", (x, y), textcoords="offset points", xytext=(5,-10))

    plt.xlabel("False Positives per Image (FPPI)", fontsize=12)
    plt.ylabel("Sensitivity (TPR)", fontsize=12)
    plt.title("FROC Curve", fontsize=14)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    try:
        plt.savefig(output_froc_png, dpi=300)
        print(f"FROC plot saved to: {output_froc_png}")
    except Exception as e:
        print(f"Warning: could not save FROC plot: {e}")
    plt.show()

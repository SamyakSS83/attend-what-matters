import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# === Paths ===
probs_path = "mammo_clip_vindr_test.npy"     # your npy file
csv_path   = "/home/samyak/scratch/my_vindr/model_specific_preprocessed_data/mmbcd_csvs/test.csv"      # your csv file
output_png = "precision_recall_curve.png"


# === Load data ===
probs = np.load(probs_path)
df = pd.read_csv(csv_path)

assert len(probs) == len(df), f"Length mismatch: {len(probs)} vs {len(df)}"

y_true = df["cancer"].astype(int).values
y_scores = probs

# === Compute PR curve ===
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

# === Compute best threshold based on F1 ===
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
best_f1 = f1_scores[best_idx]

print(f"Average Precision (AP): {ap:.4f}")
print(f"Best threshold (max F1): {best_threshold:.4f}")
print(f"F1 score at best threshold: {best_f1:.4f}")

# === Plot ===
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f'AP = {ap:.4f}')
plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f'Best F1 = {best_f1:.3f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.show()

print(f"Plot saved to: {output_png}")

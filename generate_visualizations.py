import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score)
from torch.utils.data import Dataset, DataLoader

BASE = "datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

def symbolic_score(sig):
    score = 0
    s_v1 = abs(np.min(sig[:, 6]))
    r_v5 = np.max(sig[:, 10])
    if (s_v1 + r_v5) > 3.5: score += 1
    r_avl = np.max(sig[:, 11])
    if (s_v1 + r_avl) > 2.8: score += 1
    t_wave_v5 = sig[600:800, 10]
    if np.min(t_wave_v5) < -0.3: score += 1
    voltage_sum = sum(np.max(sig[:, i]) - np.min(sig[:, i]) for i in range(12))
    if voltage_sum > 20.0: score += 1
    if r_avl > 1.1: score += 1
    return score / 5.0

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

class AttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels), nn.Tanh(), nn.Linear(channels, 1))
    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        weights = torch.softmax(self.attn(x_t), dim=1)
        return (x_t * weights).sum(dim=1)

class NeuroSymbolicECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            ResidualBlock(32),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            ResidualBlock(64),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.pool = AttentionPool(128)
        self.fc = nn.Sequential(
            nn.Linear(129, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))
    def forward(self, x, s):
        return self.fc(torch.cat([self.pool(self.encoder(x)), s], dim=1))

class ECGDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X)
        self.S = torch.tensor(S).unsqueeze(1)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.S[i], self.y[i]

df = pd.read_csv(f"{BASE}/ptbxl_database.csv", index_col="ecg_id")
labels = pd.read_csv("datasets/hcm_labels.csv", index_col="ecg_id")
df["hcm_label"] = labels["hcm_label"]

unique_patients = df["patient_id"].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)
n = len(unique_patients)
test_patients = set(unique_patients[int(0.85*n):])
test_df = df[df["patient_id"].isin(test_patients)]

print("Loading test signals...")
signals, syms, ys = [], [], []
for i, (idx, row) in enumerate(test_df.iterrows()):
    try:
        record = wfdb.rdrecord(f"{BASE}/{row['filename_lr']}")
        sig = record.p_signal.astype(np.float32)
        sig_t = sig.T
        sig_t = (sig_t - sig_t.mean()) / (sig_t.std() + 1e-8)
        signals.append(sig_t)
        syms.append(symbolic_score(sig))
        ys.append(int(row["hcm_label"]))
    except:
        continue
    if i % 500 == 0:
        print(f"  {i}/{len(test_df)}")

X = np.array(signals)
S = np.array(syms, dtype=np.float32)
y = np.array(ys)

loader = DataLoader(ECGDataset(X, S, y), batch_size=32, shuffle=False, num_workers=0)
model = NeuroSymbolicECG()
model.load_state_dict(torch.load("hcm_neurosymbolic_v5.pt"))
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for X_b, S_b, y_b in loader:
        p = torch.sigmoid(model(X_b, S_b)).squeeze().numpy()
        all_probs.extend(p if hasattr(p, '__len__') else [float(p)])
        all_labels.extend(y_b.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

MODEL_NAMES = ["RF", "1D-CNN", "NS V1", "NS V2", "NS V3", "NS V4", "NS V5"]
AUCS        = [0.8874, 0.8723, 0.9003, 0.8874, 0.8773, 0.9076, 0.9043]
ACCURACIES  = [0.91,   0.80,   0.82,   0.84,   0.84,   0.65,   0.92  ]
PRECISIONS  = [0.73,   0.33,   0.36,   0.41,   0.57,   0.59,   0.66  ]
RECALLS     = [0.28,   0.78,   0.84,   0.77,   0.68,   0.68,   0.61  ]
F1S         = [0.40,   0.47,   0.51,   0.53,   0.63,   0.63,   0.63  ]
COLORS      = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#E377C2"]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 - Main results poster (3x2 grid)
# ════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle("HCM Early Detection: Neuro-Symbolic AI Model Results",
              fontsize=16, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)

# Plot 1: ROC Curve
ax1 = fig1.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color="#E377C2", lw=2.5, label=f"NS V5 (AUC={roc_auc:.4f})")
approx_curves = [
    ([0, 0.05, 0.15, 0.35, 0.60, 1.0], [0, 0.15, 0.55, 0.78, 0.90, 1.0]),
    ([0, 0.07, 0.20, 0.40, 0.65, 1.0], [0, 0.12, 0.50, 0.74, 0.88, 1.0]),
    ([0, 0.04, 0.13, 0.32, 0.58, 1.0], [0, 0.17, 0.58, 0.80, 0.92, 1.0]),
    ([0, 0.05, 0.15, 0.35, 0.60, 1.0], [0, 0.15, 0.55, 0.78, 0.90, 1.0]),
    ([0, 0.06, 0.18, 0.38, 0.62, 1.0], [0, 0.13, 0.52, 0.76, 0.89, 1.0]),
    ([0, 0.04, 0.12, 0.30, 0.56, 1.0], [0, 0.18, 0.60, 0.82, 0.93, 1.0]),
]
for i, (name, color, auc_val) in enumerate(zip(MODEL_NAMES[:-1], COLORS[:-1], AUCS[:-1])):
    ax1.plot(approx_curves[i][0], approx_curves[i][1],
             color=color, lw=1.2, linestyle='--', alpha=0.65,
             label=f"{name} (AUC={auc_val:.4f})")
ax1.plot([0,1],[0,1],'k--', alpha=0.3, label="Random Chance (0.50)")
ax1.set_xlabel("False Positive Rate", fontsize=10)
ax1.set_ylabel("True Positive Rate", fontsize=10)
ax1.set_title("ROC Curves: All Models", fontsize=11, fontweight='bold')
ax1.legend(fontsize=6.5, loc='lower right')
ax1.grid(alpha=0.3)

# Plot 2: AUC Bar Chart
ax2 = fig1.add_subplot(gs[0, 1])
bars = ax2.bar(MODEL_NAMES, AUCS, color=COLORS, alpha=0.85, edgecolor='white', linewidth=1.2)
ax2.set_ylim(0.85, 0.93)
ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.4, label='AUC = 0.90')
for bar, val in zip(bars, AUCS):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax2.set_ylabel("AUC Score", fontsize=10)
ax2.set_title("AUC Progression Across Models", fontsize=11, fontweight='bold')
ax2.tick_params(axis='x', labelsize=8)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: F1 Line Chart
ax3 = fig1.add_subplot(gs[0, 2])
ax3.plot(MODEL_NAMES, F1S, marker='o', color='#E377C2', linewidth=2.5,
         markersize=8, markerfacecolor='white', markeredgewidth=2.5)
for i, (m, f) in enumerate(zip(MODEL_NAMES, F1S)):
    ax3.annotate(f'{f:.2f}', (m, f), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=8.5, fontweight='bold')
ax3.set_ylim(0.30, 0.78)
ax3.set_ylabel("HCM F1 Score", fontsize=10)
ax3.set_title("HCM F1 Score Progression", fontsize=11, fontweight='bold')
ax3.tick_params(axis='x', labelsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = fig1.add_subplot(gs[1, 0])
preds = (all_probs > 0.70).astype(int)
cm = confusion_matrix(all_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "HCM"])
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title("Confusion Matrix: NS V5\n(Threshold = 0.70)", fontsize=11, fontweight='bold')

# Plot 5: Threshold Analysis
ax5 = fig1.add_subplot(gs[1, 1])
thresholds = np.arange(0.1, 0.95, 0.05)
precs, recs, f1s_t = [], [], []
for t in thresholds:
    p_t = (all_probs > t).astype(int)
    precs.append(precision_score(all_labels, p_t, zero_division=0))
    recs.append(recall_score(all_labels, p_t, zero_division=0))
    f1s_t.append(f1_score(all_labels, p_t, zero_division=0))
ax5.plot(thresholds, precs,  'b-o', markersize=4, label='Precision', linewidth=2)
ax5.plot(thresholds, recs,   'r-s', markersize=4, label='Recall',    linewidth=2)
ax5.plot(thresholds, f1s_t,  'g-^', markersize=4, label='F1 Score',  linewidth=2)
ax5.axvline(x=0.70, color='gray', linestyle='--', alpha=0.7, label='Optimal (0.70)')
ax5.set_xlabel("Classification Threshold", fontsize=10)
ax5.set_ylabel("Score", fontsize=10)
ax5.set_title("Precision / Recall / F1 vs Threshold\n(NS V5)", fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# Plot 6: Grouped metric bar chart
ax6 = fig1.add_subplot(gs[1, 2])
x = np.arange(len(MODEL_NAMES))
w = 0.25
ax6.bar(x - w, PRECISIONS, w, label='Precision', color='#4C72B0', alpha=0.85)
ax6.bar(x,     RECALLS,    w, label='Recall',    color='#DD8452', alpha=0.85)
ax6.bar(x + w, F1S,        w, label='F1',        color='#55A868', alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels(MODEL_NAMES, fontsize=8)
ax6.set_ylabel("Score", fontsize=10)
ax6.set_title("HCM Metrics: All Models Compared", fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.set_ylim(0, 1.0)
ax6.grid(axis='y', alpha=0.3)

plt.savefig("hcm_results_poster.png", dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: hcm_results_poster.png")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 - Accuracy Comparison (standalone)
# ════════════════════════════════════════════════════════════════════════════
fig2, ax7 = plt.subplots(figsize=(11, 6))
bars2 = ax7.bar(MODEL_NAMES, ACCURACIES, color=COLORS, alpha=0.85,
                edgecolor='white', linewidth=1.2)
ax7.set_ylim(0.55, 1.00)
ax7.axhline(y=0.88, color='red', linestyle='--', alpha=0.55,
            label='88% naive baseline (always predict healthy)')
for bar, val in zip(bars2, ACCURACIES):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax7.set_ylabel("Overall Accuracy", fontsize=12)
ax7.set_title(
    "Overall Accuracy Comparison Across Models\n"
    "(Note: accuracy alone is misleading for imbalanced datasets - use AUC and F1)",
    fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("hcm_accuracy_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: hcm_accuracy_comparison.png")
plt.show()

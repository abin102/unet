import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# -----------------------------
# CSV paths
# -----------------------------
csv1 = "runs/uger/isic2018_experts/isic2018_mobilenetv2_freeze_lt_3_experts_repar-20251210-114622/metrics.csv"
csv2 = "runs/uger/isic2018_experts/isic2018_mobilenetv2_freeze_lt-20251210-020858/metrics.csv"

csv3 = "runs/uger/isic2018_experts/isic2018_mobilenetv2_freeze_lt_3_experts_repar-20251208-202722_pretrained/metrics.csv"
csv4 = "runs/uger/isic2018_experts/isic2018_mobilenetv2_freeze_lt-20251210-164929_pretrained/metrics.csv"

# Legend names
name1 = "UGER"
name2 = "Baseline"
name3 = "UGER (pretrained)"
name4 = "Baseline (pretrained)"

# -----------------------------
# Load CSVs
# -----------------------------
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)
df4 = pd.read_csv(csv4)

# -----------------------------
# Add suffixes and merge on epoch
# -----------------------------
def add_suffix(df, suffix):
    df = df.copy()
    df = df.add_suffix(suffix)
    df = df.rename(columns={f"epoch{suffix}": "epoch"})
    return df

df1_s = add_suffix(df1, "__A")
df2_s = add_suffix(df2, "__B")
df3_s = add_suffix(df3, "__C")
df4_s = add_suffix(df4, "__D")

merged = df1_s.merge(df2_s, on="epoch").merge(df3_s, on="epoch").merge(df4_s, on="epoch")

# 🔥 Keep only epochs <= 180
merged = merged[merged["epoch"] <= 180].reset_index(drop=True)

metrics = [
    "train_loss", "train_acc(%)",
    "val_loss", "val_acc(%)",
    "hard_acc(%)", "medium_acc(%)", "easy_acc(%)"
]

# -----------------------------
# Folder for saving plots
# -----------------------------
os.makedirs("comparison_plots", exist_ok=True)

def save_plot(metric):
    plt.figure(figsize=(7, 4))

    plt.plot(merged["epoch"], merged[f"{metric}__A"], label=name1)
    plt.plot(merged["epoch"], merged[f"{metric}__B"], label=name2)
    plt.plot(merged["epoch"], merged[f"{metric}__C"], label=name3)
    plt.plot(merged["epoch"], merged[f"{metric}__D"], label=name4)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ✅ Regex to remove %, (), spaces, etc. from filename
    # This keeps only letters, numbers, and underscores.
    safe_metric = re.sub(r"[^A-Za-z0-9_]+", "", metric)
    path = f"comparison_plots/{safe_metric}_upto180.png"

    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

# -----------------------------
# Generate plots
# -----------------------------
for m in metrics:
    save_plot(m)

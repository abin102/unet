import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
unet_csv = "runs/covid_segmentation/unet_ct_lung_infection-20251227-202240/metrics.csv"
repar_csv = "runs/covid_segmentation/unet_ct_lung_infection_reparm-20251230-181405/metrics.csv"

save_dir = "plots/unet_reparunet"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# Load CSVs
# -----------------------------
unet_df = pd.read_csv(unet_csv)
repar_df = pd.read_csv(repar_csv)

# -----------------------------
# Metrics to compare
# -----------------------------
metrics = [
    "train_loss",
    "val_loss",
    "dice_macro",
    "dice_class_1",
    "sen_class_1",
    "spec_class_1",
    "acc_pixel",
    "mae",
]

# -----------------------------
# Plot comparison
# -----------------------------
for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.plot(
        unet_df["epoch"],
        unet_df[metric],
        label="UNet",
        linewidth=2,
    )

    plt.plot(
        repar_df["epoch"],
        repar_df[metric],
        label="RePaR-UNet",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"UNet vs RePaR-UNet: {metric.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{metric}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"All comparison plots saved to: {save_dir}")

# -----------------------------
# Best-metric comparison
# -----------------------------

# Define whether higher or lower is better
metric_modes = {
    "dice_macro": "max",
    "dice_class_1": "max",
    "acc_pixel": "max",
    "sen_class_1": "max",
    "spec_class_1": "max",
    "val_loss": "min",
    "mae": "min",
}

pretty_names = {
    "dice_macro": "Dice (Macro)",
    "dice_class_1": "Dice (Class 1 – Infection)",
    "acc_pixel": "Pixel Accuracy",
    "sen_class_1": "Sensitivity (Class 1)",
    "spec_class_1": "Specificity (Class 1)",
    "val_loss": "Validation Loss",
    "mae": "MAE",
}

def get_best_metrics(df):
    results = {}
    for metric, mode in metric_modes.items():
        if mode == "max":
            idx = df[metric].idxmax()
        else:
            idx = df[metric].idxmin()

        results[metric] = {
            "value": df.loc[idx, metric],
            "epoch": int(df.loc[idx, "epoch"]),
        }
    return results


unet_best = get_best_metrics(unet_df)
repar_best = get_best_metrics(repar_df)

# -----------------------------
# Print comparison
# -----------------------------
print("\n==============================")
print("        BEST METRICS          ")
print("==============================\n")

print("UNet:")
print("Metric\t\t\t\tValue\t\tEpoch")
for m, info in unet_best.items():
    print(
        f"{pretty_names[m]:<28}"
        f"{info['value']:.4f}\t\t"
        f"{info['epoch']}"
    )

print("\nRePaR-UNet:")
print("Metric\t\t\t\tValue\t\tEpoch")
for m, info in repar_best.items():
    print(
        f"{pretty_names[m]:<28}"
        f"{info['value']:.4f}\t\t"
        f"{info['epoch']}"
    )

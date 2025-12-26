import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Path setup
# -------------------------------------------------
csv_path = "runs/covid_segmentation/unet_ct_lung_infection-20251224-102755/metrics.csv"
save_dir = os.path.dirname(csv_path)

# -------------------------------------------------
# Load CSV
# -------------------------------------------------
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# -------------------------------------------------
# Helper function to save figures
# -------------------------------------------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

# -------------------------------------------------
# 1. Training vs Validation Loss
# -------------------------------------------------
plt.figure()
plt.plot(epochs, df["train_loss"], label="Train Loss")
plt.plot(epochs, df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
save_plot("loss_curve.png")

# -------------------------------------------------
# 2. Training Accuracy
# -------------------------------------------------
plt.figure()
plt.plot(epochs, df["train_acc(%)"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.grid(True)
save_plot("train_accuracy.png")

# -------------------------------------------------
# 3. Pixel Accuracy
# -------------------------------------------------
plt.figure()
plt.plot(epochs, df["acc_pixel"])
plt.xlabel("Epoch")
plt.ylabel("Pixel Accuracy")
plt.title("Pixel Accuracy")
plt.grid(True)
save_plot("pixel_accuracy.png")

# -------------------------------------------------
# 4. Dice Scores
# -------------------------------------------------
plt.figure()
plt.plot(epochs, df["dice_macro"], label="Dice Macro")
plt.plot(epochs, df["dice_class_1"], label="Dice Class 1")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Dice Scores")
plt.legend()
plt.grid(True)
save_plot("dice_scores.png")

# -------------------------------------------------
# 5. Sensitivity & Specificity
# -------------------------------------------------
plt.figure()
plt.plot(epochs, df["sen_class_1"], label="Sensitivity (Class 1)")
plt.plot(epochs, df["spec_class_1"], label="Specificity (Class 1)")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Sensitivity & Specificity")
plt.legend()
plt.grid(True)
save_plot("sensitivity_specificity.png")

# -------------------------------------------------
# 6. MAE
# -------------------------------------------------
plt.figure()
plt.plot(epochs, df["mae"])
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.grid(True)
save_plot("mae.png")

print(f"All plots saved to: {save_dir}")

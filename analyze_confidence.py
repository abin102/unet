import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.config_utils import load_cfg
from utils.data_utils import make_dataloaders
from utils.builders import build_model_from_cfg
from data import REGISTRY as DATA_REG

HARD_THRESH = 0.60
EASY_THRESH = 0.90


@torch.no_grad()
def collect_confidences_and_labels(model, loader, device):
    """
    Collect confidences, predictions, targets AND, if possible, the image filenames.

    NOTE: assumes the DataLoader is NOT shuffled (e.g., val loader with shuffle=False),
    so that dataset indices follow the same order as iteration.
    """
    model.eval()
    all_conf = []
    all_preds = []
    all_targets = []
    all_filenames = []

    dataset = loader.dataset
    has_samples = hasattr(dataset, "samples")  # ManifestImageDataset has this

    if not has_samples:
        print("[WARN] Dataset has no 'samples' attribute; difficulty map by filename will not be possible.")

    global_idx = 0

    for batch in loader:
        # assume standard (inputs, targets) from your dataset
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            raise ValueError("Unexpected batch structure, got type: {}".format(type(batch)))

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        # ---- THIS is crucial: confidence must be max softmax probability over dim=1 ----
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

        all_conf.append(conf.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

        if has_samples:
            bsz = x.size(0)
            # assumes sequential sampling (shuffle=False)
            for i in range(bsz):
                img_file, _ = dataset.samples[global_idx + i]
                all_filenames.append(str(img_file))
            global_idx += bsz

    all_conf = np.concatenate(all_conf, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if has_samples:
        if len(all_filenames) != len(all_conf):
            raise RuntimeError(
                f"Mismatch between number of filenames ({len(all_filenames)}) "
                f"and confidences ({len(all_conf)}). Check DataLoader shuffle."
            )
    else:
        all_filenames = None

    return all_conf, all_preds, all_targets, all_filenames


def print_confidence_histogram(confidences, num_bins=20):
    """
    Print confidence distribution with fixed bins in [0, 1].
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    hist, _ = np.histogram(confidences, bins=bin_edges)

    print("Confidence distribution:")
    for i in range(num_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        # last bin closed on the right
        bracket = ")" if i < num_bins - 1 else "]"
        print(f"  [{left:.2f}, {right:.2f}{bracket}: {hist[i]}")


def plot_confidence_histogram(confidences, num_bins=20, out_dir=None):
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    plt.figure()
    plt.hist(confidences, bins=bin_edges, edgecolor="black")
    plt.xlabel("Confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "confidence_histogram.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved confidence histogram to: {path}")
    else:
        plt.show()
    plt.close()


def compute_reliability_stats(confidences, preds, targets, num_bins=20):
    """
    Compute per-bin accuracy and mean confidence, plus ECE.
    """
    n = len(confidences)
    correct = (preds == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_acc = np.zeros(num_bins, dtype=np.float32)
    bin_conf = np.zeros(num_bins, dtype=np.float32)
    bin_count = np.zeros(num_bins, dtype=np.int32)

    for i in range(num_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i < num_bins - 1:
            mask = (confidences >= left) & (confidences < right)
        else:
            # include right edge in last bin
            mask = (confidences >= left) & (confidences <= right)

        cnt = mask.sum()
        bin_count[i] = cnt
        if cnt > 0:
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = confidences[mask].mean()

    # Expected Calibration Error
    ece = 0.0
    for i in range(num_bins):
        if bin_count[i] == 0:
            continue
        weight = bin_count[i] / n
        ece += weight * abs(bin_acc[i] - bin_conf[i])

    return bin_edges, bin_acc, bin_conf, bin_count, ece


def plot_reliability_diagram(bin_edges, bin_acc, bin_conf, bin_count, ece, out_dir=None):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    width = bin_edges[1] - bin_edges[0]

    # Only plot bins that have samples
    non_empty = bin_count > 0
    bin_centers_plot = bin_centers[non_empty]
    bin_acc_plot = bin_acc[non_empty]

    plt.figure()
    # perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    # empirical accuracy per bin
    plt.bar(
        bin_centers_plot,
        bin_acc_plot,
        width=width,
        edgecolor="black",
        alpha=0.7,
        label="Empirical accuracy",
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
    plt.legend()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "reliability_diagram.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved reliability diagram to: {path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to your YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .ckpt")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Which split to evaluate on (using same construction as train.py).",
    )
    parser.add_argument(
        "--out_dir",
        default="analysis_outputs",
        help="Directory to save plots into (relative to project root).",
    )
    parser.add_argument(
        "--difficulty_map_path",
        type=str,
        default=None,
        help='If set, save JSON difficulty map: {"ISIC_0001.jpg": "hard"/"medium"/"easy"} or numeric.',
    )
    parser.add_argument(
        "--difficulty_numeric",
        action="store_true",
        help="If set, encode difficulty as 0/1/2 instead of strings.",
    )
    args = parser.parse_args()

    # -------- Load config --------
    cfg = load_cfg(args.cfg, overrides=[])

    # -------- Device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Data (same as train.py) --------
    data_args = cfg.get("data_args", {}) or {}
    train_set, val_set = DATA_REG[cfg["dataset"]](
        cfg["data_dir"],
        **data_args,
        image_size=cfg.get("image_size", 224),
        to_rgb=cfg.get("to_rgb", False),
        imagenet_norm=cfg.get("imagenet_norm", True),
        mean=cfg.get("mean", None),
        std=cfg.get("std", None),
    )

    train_loader, val_loader = make_dataloaders(
        train_set,
        val_set,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["persistent_workers"],
    )

    if args.split == "train":
        loader = train_loader
    else:
        loader = val_loader  # treat val as "test" here

    # -------- Model --------
    model_cfg = {"name": cfg["model"], "params": cfg.get("model_args", {})}
    model = build_model_from_cfg(model_cfg, device=device)

    # -------- Load checkpoint --------
    print(f"Loading checkpoint from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    # Your checkpoints look like: {"epoch": ..., "model": state_dict, ...}
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # Fallback if someone saved raw state_dict
        model.load_state_dict(ckpt)

    model.to(device)

    # -------- Collect confidences and labels (and filenames) --------
    confidences, preds, targets, filenames = collect_confidences_and_labels(model, loader, device)

    print(f"#samples: {len(confidences)}")
    print(f"conf min = {confidences.min():.4f}, max = {confidences.max():.4f}")

    # Sanity: they MUST be in [0, 1]
    if confidences.min() < 0.0 - 1e-5 or confidences.max() > 1.0 + 1e-5:
        print("WARNING: confidences are not in [0,1]. You are NOT using probabilities correctly.")

    # -------- Difficulty map & bucket counts --------
    if args.difficulty_map_path is not None:
        if filenames is None:
            raise RuntimeError(
                "Cannot build difficulty map by filename because dataset has no 'samples' attribute."
            )

        label_map = {}
        bucket_counts = {"hard": 0, "medium": 0, "easy": 0}

        for fname, conf in zip(filenames, confidences):
            if conf < HARD_THRESH:
                diff = "hard"
            elif conf < EASY_THRESH:
                diff = "medium"
            else:
                diff = "easy"

            bucket_counts[diff] += 1

            if args.difficulty_numeric:
                enc = {"hard": 0, "medium": 1, "easy": 2}[diff]
                label_map[fname] = enc
            else:
                label_map[fname] = diff

        # ensure directory exists
        diff_dir = os.path.dirname(args.difficulty_map_path)
        if diff_dir:
            os.makedirs(diff_dir, exist_ok=True)

        with open(args.difficulty_map_path, "w") as f:
            json.dump(label_map, f, indent=2)

        print(f"Saved difficulty map to: {args.difficulty_map_path}")
        print("Difficulty bucket counts (by max softmax confidence):")
        print(f"  hard   (< {HARD_THRESH:.2f}): {bucket_counts['hard']}")
        print(f"  medium ([{HARD_THRESH:.2f}, {EASY_THRESH:.2f})): {bucket_counts['medium']}")
        print(f"  easy   (>= {EASY_THRESH:.2f}): {bucket_counts['easy']}")

    # -------- Print and plot confidence histogram --------
    print_confidence_histogram(confidences, num_bins=20)
    plot_confidence_histogram(confidences, num_bins=20, out_dir=args.out_dir)

    # -------- Reliability diagram & ECE --------
    bin_edges, bin_acc, bin_conf, bin_count, ece = compute_reliability_stats(
        confidences, preds, targets, num_bins=20
    )

    print(f"ECE (Expected Calibration Error) = {ece:.6f}")

    plot_reliability_diagram(
        bin_edges, bin_acc, bin_conf, bin_count, ece, out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()

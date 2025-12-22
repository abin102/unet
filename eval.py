import argparse, yaml, os, csv, torch
from torch.utils.data import DataLoader
from data import REGISTRY as DATA_REG
from models import REGISTRY as MODEL_REG
from utils.io import load_checkpoint, ensure_dir
from train import _coerce_numbers, build_loss_and_splits  # reuse helpers from train.py
from tqdm import tqdm

@torch.no_grad()
def evaluate(cfg_path, ckpt_path, device="cuda", out_csv=None):
    # Load config
    cfg = yaml.safe_load(open(cfg_path))
    cfg = _coerce_numbers(cfg)

    # Dataset (test split only)
    _, test_set = DATA_REG[cfg["dataset"]](
        cfg["data_dir"],
        image_size=cfg.get("image_size", 32),
        to_rgb=cfg.get("to_rgb", False),
        imagenet_norm=cfg.get("imagenet_norm", False),
        **cfg.get("data_args", {})
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=cfg.get("pin_memory", True),
        persistent_workers=cfg.get("persistent_workers", True),
    )

    # Model
    model = MODEL_REG[cfg["model"]](**cfg["model_args"]).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()

    # Splits (for H/M/T) using same logic as train.py
    _, class_splits = build_loss_and_splits(cfg, test_set)
    num_classes = cfg["model_args"]["num_classes"]

    # Eval loop
    total = correct = 0
    class_correct = torch.zeros(num_classes, device=device)
    class_counts  = torch.zeros(num_classes, device=device)

    pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        outputs = model(x)

        # Sum logits if model returns (H, M, T)
        if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
            logits = outputs[0] + outputs[1] + outputs[2]
        else:
            logits = outputs

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        for c in range(num_classes):
            mask = (y == c)
            cnt = mask.sum()
            if cnt > 0:
                class_counts[c]  += cnt
                class_correct[c] += (preds[mask] == c).sum()

        pbar.set_postfix(acc=f"{correct/max(total,1):.4f}")

    overall_acc   = correct / max(total, 1)
    per_class_acc = (class_correct / class_counts.clamp(min=1)).cpu().tolist()

    # H/M/T
    head_acc = medium_acc = tail_acc = None
    if class_splits:
        def _mean_for(indices):
            if not indices: return None
            idx = torch.tensor([i for i in indices if 0 <= i < num_classes], device=device)
            if idx.numel() == 0: return None
            return (class_correct[idx] / class_counts[idx].clamp(min=1)).mean().item()
        head_acc   = _mean_for(class_splits.get("head", []))
        medium_acc = _mean_for(class_splits.get("medium", []))
        tail_acc   = _mean_for(class_splits.get("tail", []))

    # --- Class names with group info ---
    class_names = getattr(test_set, "classes", [f"class_{i}" for i in range(num_classes)])

    ordered_classes = []
    if class_splits:  # If we have H/M/T splits, order by them
        for group in ["head", "medium", "tail"]:
            for idx in class_splits.get(group, []):
                ordered_classes.append((idx, group))
        # Add any leftover classes (not in H/M/T)
        used = set(sum(class_splits.values(), []))
        for idx in range(num_classes):
            if idx not in used:
                ordered_classes.append((idx, "other"))
    else:  # fallback: natural order
        ordered_classes = [(i, "") for i in range(num_classes)]

    # --- Print results ---
    print(f"\n✅ Overall Test Accuracy: {overall_acc:.4f}")

    print("\nPer-class accuracy:")
    for idx, group in ordered_classes:
        label = class_names[idx]
        tag   = f" ({group})" if group else ""
        print(f"  {label}{tag}: {per_class_acc[idx]:.4f}")

    if class_splits:
        if head_acc   is not None: print(f"\nHead:   {head_acc:.4f}")
        if medium_acc is not None: print(f"Medium: {medium_acc:.4f}")
        if tail_acc   is not None: print(f"Tail:   {tail_acc:.4f}")

    # --- Save CSV ---
    ensure_dir("results")
    if out_csv is None:
        exp = cfg.get("exp_name", "experiment")
        out_csv = os.path.join("results", f"{exp}_eval.csv")

    # Build header with group info in brackets
    header = ["overall", "head", "medium", "tail"]
    header += [f"{class_names[idx]}({group})" if group else class_names[idx]
               for idx, group in ordered_classes]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([
            f"{overall_acc:.6f}",
            "" if head_acc   is None else f"{head_acc:.6f}",
            "" if medium_acc is None else f"{medium_acc:.6f}",
            "" if tail_acc   is None else f"{tail_acc:.6f}",
            *[f"{per_class_acc[idx]:.6f}" for idx, _ in ordered_classes],
        ])
    print(f"\n📁 Saved results to {out_csv}")

    return overall_acc, per_class_acc, {"head": head_acc, "medium": medium_acc, "tail": tail_acc}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", default=None, help="Optional custom output CSV path")
    args = ap.parse_args()

    evaluate(args.cfg, args.ckpt, out_csv=args.out_csv)

import argparse, yaml, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from data import REGISTRY as DATA_REG
from models import REGISTRY as MODEL_REG
from utils.io import load_checkpoint

# ---------- Helper: prediction entropy ----------
def prediction_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distributions per sample."""
    return -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)

# ---------- Helper: handle ResLT multiple outputs ----------
def get_logits(model, x):
    """Handle models that return multiple outputs (e.g., ResLT)."""
    outputs = model(x)
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        return outputs[0] + outputs[1] + outputs[2]  # sum ResLT outputs
    return outputs

# ---------- Compute class/group accuracies ----------
def compute_classwise_acc(preds, targets, num_classes=10):
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)
    for c in range(num_classes):
        mask = (targets == c)
        total[c] = mask.sum()
        correct[c] = (preds[mask] == c).sum()
    acc = correct.float() / total.clamp(min=1)
    return acc, correct, total

def compute_group_acc(class_acc, groups):
    return {gname: class_acc[cls_idx].mean().item() for gname, cls_idx in groups.items()}

# ---------- Main evaluation ----------
@torch.no_grad()
def evaluate(cfg1_path, ckpt1_path, cfg2_path, ckpt2_path,
             out_csv="ensemble_results.csv", device="cuda", entropy_thresh=0.6):
    # Load configs
    cfg1 = yaml.safe_load(open(cfg1_path))
    cfg2 = yaml.safe_load(open(cfg2_path))

    # Dataset (test split only, use cfg1 settings)
    _, test_set = DATA_REG[cfg1["dataset"]](
        cfg1["data_dir"],
        image_size=cfg1.get("image_size", 32),
        to_rgb=cfg1.get("to_rgb", False),
        imagenet_norm=cfg1.get("imagenet_norm", False),
        **cfg1.get("data_args", {})
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg1["batch_size"],
        shuffle=False,
        num_workers=cfg1["num_workers"],
        pin_memory=cfg1["pin_memory"],
        persistent_workers=cfg1["persistent_workers"],
    )

    # Build models
    model1 = MODEL_REG[cfg1["model"]](**cfg1["model_args"]).to(device)
    model2 = MODEL_REG[cfg2["model"]](**cfg2["model_args"]).to(device)
    load_checkpoint(ckpt1_path, model1)
    load_checkpoint(ckpt2_path, model2)
    model1.eval(), model2.eval()

    # Collect predictions + targets
    all_preds1, all_preds2, all_preds_ens, all_targets = [], [], [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        # Get logits
        logits1 = get_logits(model1, x)
        logits2 = get_logits(model2, x)

        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)

        # Entropy
        e1, e2 = prediction_entropy(p1), prediction_entropy(p2)
        diff = torch.abs(e1 - e2)

        # Weighted avg (fallback)
        w1, w2 = 1/(e1+1e-8), 1/(e2+1e-8)
        w_sum = w1 + w2
        p_ens = (w1.unsqueeze(1)*p1 + w2.unsqueeze(1)*p2) / w_sum.unsqueeze(1)

        # Final ensemble decision
        preds_ens = torch.empty_like(e1, dtype=torch.long)
        high_conf1 = (diff > entropy_thresh) & (e1 < e2)
        high_conf2 = (diff > entropy_thresh) & (e2 < e1)

        preds_ens[high_conf1] = p1[high_conf1].argmax(dim=1)
        preds_ens[high_conf2] = p2[high_conf2].argmax(dim=1)

        # For the rest, use weighted ensemble
        fallback = ~(high_conf1 | high_conf2)
        preds_ens[fallback] = p_ens[fallback].argmax(dim=1)

        # Save preds
        all_preds1.append(p1.argmax(dim=1).cpu())
        all_preds2.append(p2.argmax(dim=1).cpu())
        all_preds_ens.append(preds_ens.cpu())
        all_targets.append(y.cpu())

    # Concatenate all
    preds1 = torch.cat(all_preds1)
    preds2 = torch.cat(all_preds2)
    preds_ens = torch.cat(all_preds_ens)
    targets = torch.cat(all_targets)

    num_classes = 10
    groups = {
        "head": torch.tensor([0, 1]),
        "medium": torch.tensor([2, 3]),
        "tail": torch.tensor([4, 5, 6, 7, 8, 9]),
    }

    # Store results in dicts
    results = {}
    rows = []
    for name, preds in zip(["Model1", "Model2", "Ensemble"], [preds1, preds2, preds_ens]):
        class_acc, _, _ = compute_classwise_acc(preds, targets, num_classes=num_classes)
        group_acc = compute_group_acc(class_acc, groups)
        overall = (preds == targets).float().mean().item()

        results[name] = {
            "overall": overall,
            "per_class": class_acc.tolist(),
            "head": group_acc["head"],
            "medium": group_acc["medium"],
            "tail": group_acc["tail"],
        }

        # Prepare row for CSV
        row = {
            "model": name,
            "overall": overall,
            "head": group_acc["head"],
            "medium": group_acc["medium"],
            "tail": group_acc["tail"],
        }
        for c, acc in enumerate(class_acc.tolist()):
            row[f"class_{c}"] = acc
        rows.append(row)

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")
    print(df)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg1", required=True)
    ap.add_argument("--ckpt1", required=True)
    ap.add_argument("--cfg2", required=True)
    ap.add_argument("--ckpt2", required=True)
    ap.add_argument("--out_csv", default="ensemble_results.csv")
    ap.add_argument("--entropy_thresh", type=float, default=0.6,
                    help="Entropy difference threshold for preferring lower-entropy model")
    args = ap.parse_args()

    evaluate(args.cfg1, args.ckpt1, args.cfg2, args.ckpt2,
             out_csv=args.out_csv, device="cuda", entropy_thresh=args.entropy_thresh)

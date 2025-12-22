"""
utils/data_check.py

Utilities to inspect datasets (post-transform) for NaNs/Infs, label-range issues,
extreme pixel values, and empty classes. Also includes a small CLI to run checks
on datasets created via your registry (from data import REGISTRY).
"""

from typing import Optional, Tuple
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

# Assumes your project exposes REGISTRY in data/__init__.py
try:
    from data import REGISTRY
except Exception:
    REGISTRY = None  # allow import in environments where registry isn't on PATH

# -------------------------
# Basic utilities
# -------------------------

def inspect_batch(x: torch.Tensor, y: torch.Tensor, name: str = "batch") -> None:
    """Print quick diagnostics for a single (x,y) batch (post-transform)."""
    print(f"--- Inspecting {name} ---")
    print("x.shape, x.dtype:", tuple(x.shape), x.dtype)
    print("y.shape, y.dtype:", tuple(y.shape), y.dtype)
    print("x device:", x.device)
    finite = bool(torch.isfinite(x).all().item())
    print("x finite:", finite)
    try:
        vmin = float(x.min().item())
        vmax = float(x.max().item())
    except Exception:
        vmin = float("nan")
        vmax = float("nan")
    print("x min/max:", vmin, vmax)
    print("any extreme values >1e6?", bool((x.abs() > 1e6).any().item()))
    print("labels min/max:", int(y.min().item()), int(y.max().item()))
    print("unique labels:", torch.unique(y).cpu().numpy())
    # check for all-zero images (common corrupt)
    if x.ndim == 4:
        all_zero = bool((x.view(x.size(0), -1).abs().sum(dim=1) == 0).any().item())
        N, C, H, W = x.shape
        print("any all-zero images?", all_zero, "N,C,H,W:", (N, C, H, W))
    print("-------------------------\n")


def save_bad_batch(x: torch.Tensor, y: torch.Tensor, path: str = "bad_batch.pth", save_images: bool = False,
                   imagenet_norm: bool = False, out_dir: Optional[str] = None) -> None:
    """
    Save a problematic batch for later inspection.
    - saves a .pth with tensors by default.
    - if save_images=True, also saves per-sample PNGs (attempts to un-normalize if imagenet_norm=True).
    """
    path = os.path.abspath(path)
    out_dir = out_dir or os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"x": x.detach().cpu(), "y": y.detach().cpu()}, path)
    print(f"Saved bad batch to {path}")

    if save_images:
        # save first few images as PNG for quick visual check
        n = min(8, x.size(0))
        for i in range(n):
            img = x[i].clone()
            if imagenet_norm and img.ndim == 3 and img.size(0) == 3:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)
                img = img * std + mean
            # clamp to [0,1] for saving; if single-channel, save as 3-ch
            img = img.clamp(0.0, 1.0)
            out = os.path.join(out_dir, f"bad_img_{i}.png")
            try:
                save_image(img, out)
                print("Saved image:", out)
            except Exception as e:
                print("Failed to save image:", out, "error:", e)


# -------------------------
# Dataset scanning utilities
# -------------------------

def scan_ds(dataset: Dataset,
            num_classes: int,
            name: str = "dataset",
            max_batches: Optional[int] = None,
            batch_size: int = 64,
            num_workers: int = 4,
            pin_memory: bool = False,
            stop_on_error: bool = False) -> Tuple[bool, torch.Tensor]:
    """
    Scan a whole dataset (via DataLoader) for:
      - non-finite inputs
      - extreme input magnitudes (>1e6)
      - invalid label ranges
      - label distribution (counts)

    Returns: (ok: bool, label_counts: torch.Tensor)
    If stop_on_error is True, function returns immediately on first error.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    label_counts = torch.zeros(num_classes, dtype=torch.long)
    bad_batches = 0
    total_samples = 0
    for i, (x, y) in enumerate(loader):
        total_samples += x.size(0)
        # label range check
        if y.min().item() < 0 or y.max().item() >= num_classes:
            print(f"[{name}] Invalid label range in batch {i}: min {y.min().item()} max {y.max().item()}")
            bad_batches += 1
            if stop_on_error:
                return False, label_counts
        # non-finite inputs
        if not torch.isfinite(x).all():
            print(f"[{name}] Non-finite input found in batch {i}")
            bad_batches += 1
            if stop_on_error:
                return False, label_counts
        # extreme magnitude
        if (x.abs() > 1e6).any():
            print(f"[{name}] Extreme values >1e6 in batch {i}")
            bad_batches += 1
            if stop_on_error:
                return False, label_counts
        # accumulate label counts (handles multi-dim labels)
        try:
            binc = torch.bincount(y.view(-1), minlength=num_classes)
            label_counts += binc
        except Exception:
            # if labels are not integer tensors or irregular
            print(f"[{name}] Could not bincount labels in batch {i}; labels dtype: {y.dtype}")
            bad_batches += 1
            if stop_on_error:
                return False, label_counts

        if max_batches is not None and (i + 1) >= max_batches:
            break

    print(f"[{name}] scan done. batches checked: {(i+1) if 'i' in locals() else 0}, bad_batches: {bad_batches}, samples: {total_samples}")
    print("Label distribution (counts):", label_counts.tolist())
    zeros = (label_counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if zeros:
        print(f"[{name}] classes with zero samples: {zeros}")
    ok = (bad_batches == 0)
    return ok, label_counts


# -------------------------
# Helpers to build from registry + CLI
# -------------------------

def build_dataset_from_registry(name: str, data_dir: str, **kwargs):
    """
    Build (train_ds, val_ds) using your registry function.
    Expects REGISTRY[name] to return either (train_ds, val_ds) OR a wrapped object
    like ds.train.dataset, ds.test.dataset (we handle both common patterns).
    """
    if REGISTRY is None:
        raise RuntimeError("REGISTRY not found. Ensure `from data import REGISTRY` works in your project.")
    if name not in REGISTRY:
        raise KeyError(f"Dataset '{name}' not found in REGISTRY. Available: {list(REGISTRY.keys())}")
    ds = REGISTRY[name](data_dir, **kwargs)
    # common returns:
    # - (train_ds, val_ds)
    # - CIFAR10V2 -> returns ds.train.dataset, ds.test.dataset in registry helper (already handled upstream)
    if isinstance(ds, tuple) and len(ds) == 2:
        return ds
    # if registry returns an object with train/test attributes
    if hasattr(ds, "train") and hasattr(ds, "test"):
        tr = getattr(ds, "train")
        te = getattr(ds, "test")
        # if these are DataLoader-like wrappers with .dataset
        if hasattr(tr, "dataset"):
            tr = tr.dataset
        if hasattr(te, "dataset"):
            te = te.dataset
        return tr, te
    raise RuntimeError("Registry returned unexpected object. Expected (train, val) or an object with .train/.test.")


# -------------------------
# CLI entrypoint
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Dataset sanity checker using REGISTRY")
    p.add_argument("--name", type=str, required=True, help="dataset key in REGISTRY (e.g. 'cifar10','firerisk')")
    p.add_argument("--data-dir", type=str, required=True, help="root data directory")
    p.add_argument("--num-classes", type=int, default=None, help="number of classes (if omitted, tries to infer from dataset.classes)")
    p.add_argument("--max-batches", type=int, default=200, help="max batches to scan (useful for quick scans)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-example-images", action="store_true", help="save a few sample images from each split")
    p.add_argument("--imagenet-norm", action="store_true", help="if dataset uses ImageNet norm, attempt to un-normalize examples when saving images")
    args = p.parse_args()

    print("Building dataset from REGISTRY:", args.name)
    train_ds, val_ds = build_dataset_from_registry(args.name, args.data_dir)
    print("Train dataset:", type(train_ds), "len:", len(train_ds))
    print("Val   dataset:", type(val_ds),   "len:", len(val_ds))

    # infer num_classes if possible
    inferred_num_classes = None
    if hasattr(train_ds, "classes"):
        inferred_num_classes = len(getattr(train_ds, "classes"))
        print("Inferred num_classes from dataset.classes:", inferred_num_classes)
    if args.num_classes is None and inferred_num_classes is None:
        raise RuntimeError("num_classes not provided and could not be inferred. Provide --num-classes.")
    num_classes = args.num_classes or inferred_num_classes

    # quick inspect a few samples
    print("\n== Quick sample inspection (first few examples) ==")
    for i in range(min(4, len(train_ds))):
        x, y = train_ds[i]
        inspect_batch(x.unsqueeze(0) if x.ndim == 3 else x, torch.tensor([y]), f"train_sample_{i}")
        if args.save_example_images:
            # reuse save_bad_batch to save a few images
            save_bad_batch(x.unsqueeze(0) if x.ndim == 3 else x, torch.tensor([y]),
                           path=f"debug_train_sample_{i}.pth", save_images=True, imagenet_norm=args.imagenet_norm)

    for i in range(min(4, len(val_ds))):
        x, y = val_ds[i]
        inspect_batch(x.unsqueeze(0) if x.ndim == 3 else x, torch.tensor([y]), f"val_sample_{i}")
        if args.save_example_images:
            save_bad_batch(x.unsqueeze(0) if x.ndim == 3 else x, torch.tensor([y]),
                           path=f"debug_val_sample_{i}.pth", save_images=True, imagenet_norm=args.imagenet_norm)

    # full-scan (or partial if max_batches set)
    print("\n== Scanning train dataset ==")
    ok_train, train_counts = scan_ds(train_ds, num_classes=num_classes, name="train",
                                     max_batches=args.max_batches, batch_size=args.batch_size,
                                     num_workers=args.num_workers)
    print("\n== Scanning val dataset ==")
    ok_val, val_counts = scan_ds(val_ds, num_classes=num_classes, name="val",
                                 max_batches=args.max_batches, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    print("\nSummary:")
    print(" ok_train:", ok_train, "ok_val:", ok_val)
    print(" train label counts:", train_counts.tolist())
    print(" val   label counts:", val_counts.tolist())
    if not ok_train or not ok_val:
        print("One or more problems were found. Consider re-running with --max-batches 0 to scan full set or --save-example-images to save samples.")

if __name__ == "__main__":
    main()

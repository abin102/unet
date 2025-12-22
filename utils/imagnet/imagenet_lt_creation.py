#!/usr/bin/env python3
"""
build_imagenet_lt_from_txts.py

Create on-disk ImageNet-LT dataset structure (symlinks or copies) from provided .txt splits.

Usage:
  python build_imagenet_lt_from_txts.py \
    --imagenet-root /data/imagenet \
    --splits-root ./datasets/txts/ImageNet_LT \
    --out-root /data/imagenet_lt \
    --files ImageNet_LT_train.txt ImageNet_LT_val.txt ImageNet_LT_test.txt ImageNet_LT_open.txt \
    [--copy] [--remap]

Notes:
 - The .txt files should contain lines "REL_PATH LABEL", e.g. "train/n01440764/xxx.JPEG 0".
 - If REL_PATH already includes "train/" or "val/" the script will resolve correctly when imagenet-root is the parent that contains both train/ and val/.
 - Default behavior: create symlinks (fast, space-saving). Use --copy to physically copy files.
 - By default labels are used as-is. Use --remap to remap all distinct labels to contiguous 0..K-1.
"""

import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import os
import shutil
import sys

def read_txt(txt_path: Path):
    items = []
    with txt_path.open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 1:
                continue
            rel = parts[0]
            lbl = None
            if len(parts) > 1:
                try:
                    lbl = int(parts[1])
                except ValueError:
                    lbl = parts[1]
            items.append((rel, lbl))
    return items

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def make_entry_src(im_root: Path, rel: str):
    """Try to resolve the source path given relpath and imagenet root.
    Attempts:
      1) imagenet_root / rel
      2) if rel starts with train/val/test strip first component and try imagenet_root / stripped
    Returns resolved Path or raises FileNotFoundError
    """
    p1 = im_root / rel
    if p1.exists():
        return p1
    # if rel has leading component, try stripping first component
    if "/" in rel:
        stripped = rel.split("/", 1)[1]
        p2 = im_root / stripped
        if p2.exists():
            return p2
    # also try if imagenet_root is actually the train folder and rel includes train/... -> strip leading
    if rel.startswith("train/") or rel.startswith("val/") or rel.startswith("test/"):
        stripped = rel.split("/",1)[1]
        p3 = im_root / stripped
        if p3.exists():
            return p3
    raise FileNotFoundError(f"Could not find source file for rel='{rel}' (tried under {im_root})")

def populate(items, split_name, im_root: Path, out_root: Path, copy=False, label_map=None):
    out_split = out_root / split_name
    out_split.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    missing = []
    for rel, lbl in items:
        try:
            src = make_entry_src(im_root, rel)
        except FileNotFoundError as e:
            missing.append(rel)
            continue

        # compute label to use on disk
        out_label = lbl if label_map is None else label_map.get(lbl, None)
        if out_label is None:
            # if label_map used and label not present, map to -1
            out_label = -1

        class_dir = out_split / f"{out_label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / src.name

        if dst.exists():
            counts[out_label] += 1
            continue

        if copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(str(src.resolve()), str(dst))
        counts[out_label] += 1
    return counts, missing

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet-root", required=True, type=Path,
                   help="Path to your ImageNet root (parent of train/ and val/).")
    p.add_argument("--splits-root", required=True, type=Path, help="Folder containing the .txt split files.")
    p.add_argument("--out-root", required=True, type=Path, help="Where to create imagenet_lt/{train,val,test,open}/")
    p.add_argument("--files", nargs="+", required=True, help="List of split txt filenames (order matters).")
    p.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    p.add_argument("--remap", action="store_true", help="Remap labels to contiguous 0..K-1 based on union of labels in provided txts.")
    args = p.parse_args()

    if not args.splits_root.exists():
        print("splits-root not found:", args.splits_root, file=sys.stderr); sys.exit(2)
    if not args.imagenet_root.exists():
        print("imagenet-root not found:", args.imagenet_root, file=sys.stderr); sys.exit(2)

    # read all files, build union of labels
    all_items_by_file = OrderedDict()
    labels_seen = []
    for fname in args.files:
        fpath = args.splits_root / fname
        if not fpath.exists():
            print(f"Warning: {fpath} does not exist — skipping this split file.")
            continue
        items = read_txt(fpath)
        all_items_by_file[fname] = items
        for _, lbl in items:
            labels_seen.append(lbl)

    unique_labels = sorted(list({lbl for lbl in labels_seen if lbl is not None}))
    print("Unique labels found in txts (sample):", unique_labels[:10], " total:", len(unique_labels))

    label_map = None
    if args.remap:
        # remap labels to contiguous 0..K-1 with stable ordering
        remap_dict = {old: new for new, old in enumerate(unique_labels)}
        print("Remapping labels: sample:", list(remap_dict.items())[:10])
        label_map = remap_dict

    print("Populating dataset under:", args.out_root)
    summary = {}
    overall_missing = {}
    for fname, items in all_items_by_file.items():
        # choose split folder name based on filename heuristics
        key = "open" if "open" in fname.lower() else ("val" if "val" in fname.lower() else ("test" if "test" in fname.lower() else "train"))
        print(f"\nProcessing {fname} -> split '{key}' ({len(items)} entries)")
        counts, missing = populate(items, key, args.imagenet_root, args.out_root, copy=args.copy, label_map=label_map)
        summary[key] = counts
        overall_missing[key] = missing
        print(f"  Done. classes created: {len(counts)}, total files linked: {sum(counts.values())}, missing: {len(missing)}")
        if len(missing):
            print("  Missing examples (up to 10):", missing[:10])

    print("\n=== Summary per split ===")
    for k,counts in summary.items():
        total = sum(counts.values())
        maxc = max(counts.values()) if counts else 0
        minc = min(counts.values()) if counts else 0
        print(f"  {k}: classes={len(counts)} total={total} max={maxc} min={minc} missing={len(overall_missing.get(k,[]))}")

    print("\nDone. To use with LT_Dataset loader, point its root to:", args.out_root)

if __name__ == "__main__":
    main()

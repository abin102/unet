# utils/data_stats.py
import json
from collections import Counter
import numpy as np
import torch
import os
from torch.utils.data import Subset, DataLoader

def get_dataset_class_counts(dataset, max_samples_check=20000):
    """
    Return (counts_dict, total_count) for dataset.
    Handles:
      - torch.utils.data.Subset (respects subset.indices)
      - torch.utils.data.DataLoader (unwraps to .dataset)
      - torchvision datasets with .targets or .labels
      - custom datasets exposing get_cls_num_list / num_per_cls_dict
      - fallback: iterate up to max_samples_check
    """
    # If a DataLoader was passed, unwrap to the dataset
    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset

    # If Subset: use underlying dataset + indices
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = list(dataset.indices)
        # If underlying dataset stores targets/labels as arrays/lists, index into them
        if hasattr(base, "targets"):
            arr = np.array(base.targets)
            arr = arr[indices]
            cnt = Counter(int(x) for x in arr)
            return dict(sorted(cnt.items())), int(len(indices))
        if hasattr(base, "labels"):
            arr = np.array(base.labels)
            arr = arr[indices]
            cnt = Counter(int(x) for x in arr)
            return dict(sorted(cnt.items())), int(len(indices))
        # fallback: index into base via __getitem__
        cnt = Counter()
        for i in indices:
            _, y = base[i]
            cnt[int(y)] += 1
        return dict(sorted(cnt.items())), int(len(indices))

    # IMBALANCECIFAR10 style
    if hasattr(dataset, "get_cls_num_list"):
        lst = dataset.get_cls_num_list()
        return {i: int(n) for i, n in enumerate(lst)}, int(sum(lst))
    if hasattr(dataset, "num_per_cls_dict"):
        d = dataset.num_per_cls_dict
        return {int(k): int(v) for k, v in sorted(d.items())}, int(sum(d.values()))

    # torchvision datasets (targets / labels)
    if hasattr(dataset, "targets"):
        try:
            arr = np.array(dataset.targets)
            cnt = Counter(int(x) for x in arr)
            return dict(sorted(cnt.items())), int(len(arr))
        except Exception:
            pass
    if hasattr(dataset, "labels"):
        try:
            arr = np.array(dataset.labels)
            cnt = Counter(int(x) for x in arr)
            return dict(sorted(cnt.items())), int(len(arr))
        except Exception:
            pass

    # fallback: iterate up to max_samples_check
    cnt = Counter()
    total = 0
    for item in dataset:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            _, y = item[:2]
            cnt[int(y)] += 1
            total += 1
        else:
            # unknown return format; skip
            continue
        if total >= max_samples_check:
            break
    return dict(sorted(cnt.items())), int(total)


def save_dataset_stats(out_dir, train_set, val_set, cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    train_counts, train_total = get_dataset_class_counts(train_set)
    val_counts, val_total = get_dataset_class_counts(val_set)
    stats = {
        "train_counts": train_counts, "train_total": train_total,
        "val_counts": val_counts, "val_total": val_total,
    }
    if cfg is not None:
        # store relevant config bits (avoid dumping huge objects)
        stats["cfg_summary"] = {
            "dataset": cfg.get("dataset"),
            "data_args": cfg.get("data_args"),
            "imb_factor": (cfg.get("data_args") or {}).get("imb_factor"),
            "imbalance_ratio": (cfg.get("data_args") or {}).get("imbalance_ratio"),
        }
    with open(os.path.join(out_dir, "dataset_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    return stats

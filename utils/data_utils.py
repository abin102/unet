# utils/data_utils.py
from torch.utils.data import DataLoader
from typing import Tuple, Any, Optional
import torch
from loguru import logger

def make_dataloaders(train_set, val_set, batch_size, num_workers,
                     pin_memory=True, persistent_workers=False,
                     shuffle=True):
    drop_last = True  # explicitly declare

    logger.debug(
        "Initializing dataloaders with params: "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
        f"shuffle_train={shuffle}, drop_last={drop_last}"
    )

    logger.debug(
        f"Datasets — Train: {type(train_set).__name__} ({len(train_set)} samples), "
        f"Val: {type(val_set).__name__} ({len(val_set)} samples)"
    )

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last
    )
    logger.debug(f"DataLoader args: {dl_args}")

    train_loader = DataLoader(train_set, shuffle=shuffle, **dl_args)
    val_loader = DataLoader(val_set, shuffle=False, **dl_args)

    logger.debug(
        f"Train DataLoader: {len(train_loader)} batches "
        f"({len(train_set)} samples total, drop_last={drop_last})"
    )
    logger.debug(
        f"Val DataLoader: {len(val_loader)} batches "
        f"({len(val_set)} samples total, drop_last={drop_last})"
    )

    logger.debug("Dataloaders successfully initialized and returned.")
    return train_loader, val_loader


def infer_input_size_from_loader(loader, fallback=(1,3,64,64)):
    """Pull first batch from loader and infer (1,C,H,W). Robust to tuple/dict batches."""
    first_batch = None
    try:
        for b in loader:
            first_batch = b
            break
    except Exception:
        return fallback

    if first_batch is None:
        return fallback

    imgs = None
    if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 1:
        imgs = first_batch[0]
    elif isinstance(first_batch, dict):
        for k in ("image", "images", "img", "x"):
            if k in first_batch:
                imgs = first_batch[k]; break
    else:
        for attr in ("images", "image", "imgs", "x"):
            if hasattr(first_batch, attr):
                imgs = getattr(first_batch, attr); break

    if imgs is None or not torch.is_tensor(imgs):
        return fallback
    # imgs expected (N,C,H,W) or (N,H,W,C)
    if imgs.ndim == 4:
        N,a,b,c = imgs.shape
        if a in (1,3): return (1,int(a),int(b),int(c))
        if c in (1,3): return (1,int(c),int(a),int(b))
        return (1,int(a),int(b),int(c))
    if imgs.ndim == 3:
        C,H,W = imgs.shape
        return (1,int(C),int(H),int(W))
    return fallback

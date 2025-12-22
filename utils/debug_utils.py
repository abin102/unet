# utils/debug_utils.py
"""
Lightweight debugging helpers for model freezing / optimizer inspection.
Keeps train.py clean.
"""

import torch

def print_trainable_summary(model, prefix="[debug_utils]"):
    """Print total and trainable parameter counts, and ratio."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = 100.0 * trainable / max(total, 1)
    print(f"{prefix} Trainable params: {trainable:,}/{total:,} ({ratio:.2f}%)")
    return {"total": total, "trainable": trainable, "ratio": ratio}


def print_optimizer_summary(optimizer, model=None, prefix="[debug_utils]"):
    """Print optimizer param group details and optional model param check."""
    print(f"{prefix} Optimizer type: {optimizer.__class__.__name__}")
    for i, pg in enumerate(optimizer.param_groups):
        lr = pg.get("lr", None)
        wd = pg.get("weight_decay", None)
        pg_count = sum(p.numel() for p in pg["params"] if p.requires_grad)
        print(f"{prefix}   group[{i}]: {pg_count:,} params | lr={lr} | wd={wd}")
    if model is not None:
        print_trainable_summary(model, prefix=prefix + " [model check]")
    print()


def print_lr_after_unfreeze(trainer, prefix="[debug_utils]"):
    """For use inside Trainer.fit() immediately after unfreezing backbone."""
    try:
        lrs = [pg.get("lr") for pg in trainer.optimizer.param_groups]
        print(f"{prefix} Unfreeze complete → new optimizer LRs: {lrs}")
    except Exception as e:
        print(f"{prefix} Could not print LRs after unfreeze: {e}")

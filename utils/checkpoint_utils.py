from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
from loguru import logger  # <--- Loguru Version

def _pick(*dicts, keys=None):
    if keys is None: keys = []
    for d in dicts:
        if not isinstance(d, dict): continue
        for k in keys:
            if k in d: return d[k], k
    return None, None

def save_checkpoint(path: str, trainer, epoch: int, is_best: bool = False) -> None:
    """Save checkpoint, automatically unwrapping DDP models."""
    out_dir = os.path.dirname(path) if os.path.splitext(path)[1] else path
    os.makedirs(out_dir, exist_ok=True)

    # --- FIX: Unwrap DDP ---
    model_ref = getattr(trainer, "model", None)
    if isinstance(model_ref, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state = model_ref.module.state_dict()
    elif model_ref is not None:
        model_state = model_ref.state_dict()
    else:
        model_state = None
    # -----------------------

    ckpt = {
        "epoch": int(epoch),
        "model_state": model_state,
        "optim_state": getattr(trainer, "optimizer").state_dict() if getattr(trainer, "optimizer", None) else None,
        "scheduler_state": getattr(trainer, "scheduler").state_dict() if getattr(trainer, "scheduler", None) else None,
        "scaler_state": getattr(trainer, "scaler").state_dict() if getattr(trainer, "scaler", None) else None,
        "trainer_backbone_frozen": getattr(trainer, "_backbone_frozen", False),
        "latest_val_stats": getattr(trainer, "latest_val_stats", {}),
        "cfg": getattr(trainer, "cfg", None),
    }

    # Compat keys
    ckpt["model"] = ckpt["model_state"]
    ckpt["opt"] = ckpt["optim_state"]
    ckpt["sched"] = ckpt["scheduler_state"]

    torch.save(ckpt, os.path.join(out_dir, "last.ckpt"))
    if is_best:
        torch.save(ckpt, os.path.join(out_dir, "best.ckpt"))

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, trainer=None, map_location=None, strict=False):
    logger.info("Attempting to load checkpoint from: {}", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = "cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}"
    
    ckpt = torch.load(path, map_location=map_location)
    
    # Extract
    model_state, _ = _pick(ckpt, keys=["model_state", "model", "state_dict"])
    optim_state, _ = _pick(ckpt, keys=["optim_state", "opt", "optimizer"])
    sched_state, _ = _pick(ckpt, keys=["scheduler_state", "sched", "scheduler"])
    scaler_state, _ = _pick(ckpt, keys=["scaler_state", "scaler"])

    # --- FIX: Handle 'module.' prefix mismatch ---
    if model_state is not None:
        curr_keys = set(model.state_dict().keys())
        loaded_keys = list(model_state.keys())
        new_state = model_state

        if any(k.startswith("module.") for k in loaded_keys) and not any(k.startswith("module.") for k in curr_keys):
            logger.info("Stripping 'module.' prefix from checkpoint.")
            new_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        elif any(k.startswith("module.") for k in curr_keys) and not any(k.startswith("module.") for k in loaded_keys):
            logger.info("Adding 'module.' prefix to checkpoint.")
            new_state = {"module."+k: v for k, v in model_state.items()}
            
        model.load_state_dict(new_state, strict=strict)
        logger.info("Model weights loaded.")

    if optimizer and optim_state: optimizer.load_state_dict(optim_state)
    if scheduler and sched_state: scheduler.load_state_dict(sched_state)
    if trainer and scaler_state and hasattr(trainer, "scaler"): trainer.scaler.load_state_dict(scaler_state)

    epoch = ckpt.get("epoch") or ckpt.get("epo")
    start_epoch = int(epoch) + 1 if epoch is not None else 1
    
    return {"start_epoch": start_epoch, "raw": ckpt}
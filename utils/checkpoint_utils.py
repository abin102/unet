"""
Utility helpers for loading and saving training checkpoints.
Place this file in your project under `utils/checkpoint_utils.py` and import the
loader where you need it (e.g. in `train.py`).

Provides:
- save_checkpoint(path, trainer, is_best=False): saves a rich checkpoint from trainer and components
- load_checkpoint(path, model, optimizer=None, scheduler=None, trainer=None, map_location=None, strict=False):
    loads what it can, returns a dict with keys: start_epoch, scaler_state, trainer_state

The loader is intentionally permissive: it accepts several common key names
(`model`, `model_state`, `state_dict`) and likewise for optimizer/scheduler/scaler.
It will not raise on partial loads; instead it prints informative warnings.

Designed to keep `train.py` small by moving resume logic here.
"""

from typing import Optional, Tuple, Dict, Any
import os
import torch
from loguru import logger

def _pick(*dicts, keys=None):
    """Helper: find first available key in given dicts. Returns (value, key_name) or (None, None)."""
    if keys is None:
        keys = []
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d:
                return d[k], k
    return None, None


def save_checkpoint(path: str, trainer, epoch: int, is_best: bool = False) -> None:
    """Save a comprehensive checkpoint using objects available on `trainer`.

    `trainer` is expected to have attributes:
      - model
      - optimizer
      - scheduler (optional)
      - scaler (optional)
      - latest_val_stats (optional)
      - cfg (optional)
      - _backbone_frozen (optional)

    This function writes two files (like your current Checkpoint):
      - last.ckpt (path/last.ckpt)
      - best.ckpt (path/best.ckpt) if is_best True

    The saved dict uses explicit key names to be robust across code versions.
    """
    out_dir = os.path.dirname(path) if os.path.splitext(path)[1] else path
    os.makedirs(out_dir, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "model_state": getattr(trainer, "model").state_dict() if getattr(trainer, "model", None) is not None else None,
        "optim_state": getattr(trainer, "optimizer").state_dict() if getattr(trainer, "optimizer", None) is not None else None,
        "scheduler_state": getattr(trainer, "scheduler").state_dict() if getattr(trainer, "scheduler", None) is not None else None,
        "scaler_state": getattr(trainer, "scaler").state_dict() if getattr(trainer, "scaler", None) is not None else None,
        "trainer_backbone_frozen": getattr(trainer, "_backbone_frozen", False),
        "latest_val_stats": getattr(trainer, "latest_val_stats", {}),
        "cfg": getattr(trainer, "cfg", None),
    }

    # also keep old-style keys for backward compatibility
    ckpt["model"] = ckpt["model_state"]
    ckpt["opt"] = ckpt["optim_state"]
    ckpt["sched"] = ckpt["scheduler_state"]

    last_path = os.path.join(out_dir, "last.ckpt")
    torch.save(ckpt, last_path)

    if is_best:
        best_path = os.path.join(out_dir, "best.ckpt")
        torch.save(ckpt, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    trainer: Optional[object] = None,
    map_location: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load checkpoint and restore into provided components (model, optimizer, scheduler, trainer).

    Returns a dict with:
        - start_epoch
        - scaler_state
        - trainer_backbone_frozen
        - latest_val_stats
        - raw (full checkpoint)
    """

    logger.info("Attempting to load checkpoint from: {}", path)

    if not os.path.exists(path):
        logger.critical("Checkpoint not found at path: {}", path)
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Auto-detect map location
    if map_location is None:
        map_location = "cpu" if not torch.cuda.is_available() else f"cuda:{torch.cuda.current_device()}"
        logger.debug("Auto-selected map_location={}", map_location)
    else:
        logger.debug("Using provided map_location={}", map_location)

    # Load raw checkpoint
    try:
        ckpt = torch.load(path, map_location=map_location)
        logger.info("Successfully loaded checkpoint file: {} (keys=%d)", path, len(ckpt))
    except Exception as e:
        logger.critical("Failed to load checkpoint file: {}", e, exc_info=True)
        raise

    # --- Extract components ---
    model_state, model_key = _pick(ckpt, keys=["model_state", "model", "state_dict"])
    optim_state, opt_key = _pick(ckpt, keys=["optim_state", "opt", "optimizer", "optim_state_dict"])
    sched_state, sched_key = _pick(ckpt, keys=["scheduler_state", "sched", "scheduler"])
    scaler_state, scaler_key = _pick(ckpt, keys=["scaler_state", "scaler", "gradscaler"])

    logger.debug("Checkpoint keys detected -> model={}, optimizer={}, scheduler={}, scaler={}",
                 model_key, opt_key, sched_key, scaler_key)

    # --- Epoch ---
    epoch = ckpt.get("epoch") or ckpt.get("epo") or None
    start_epoch = int(epoch) + 1 if epoch is not None else 1
    logger.info("Restored training epoch from checkpoint: {} → starting at epoch %d", epoch, start_epoch)

    # --- Restore model ---
    if model_state is not None:
        try:
            model.load_state_dict(model_state, strict=strict)
            logger.info("Loaded model state from checkpoint (key='{}').", model_key)
        except Exception as e:
            logger.error("model.load_state_dict failed (key='{}'): {}", model_key, e, exc_info=True)
    else:
        logger.warning("No model state found in checkpoint; model weights not restored.")

    # --- Restore optimizer ---
    if optimizer is not None and optim_state is not None:
        try:
            optimizer.load_state_dict(optim_state)
            logger.info("Loaded optimizer state from checkpoint (key='{}').", opt_key)
        except Exception as e:
            logger.error("optimizer.load_state_dict failed (key='{}'): {}", opt_key, e, exc_info=True)
    elif optimizer is None:
        logger.debug("Optimizer not provided; skipping optimizer restore.")
    else:
        logger.warning("No optimizer state found in checkpoint; optimizer reset to initial state.")

    # --- Restore scheduler ---
    if scheduler is not None and sched_state is not None:
        try:
            scheduler.load_state_dict(sched_state)
            logger.info("Loaded scheduler state from checkpoint (key='{}').", sched_key)
        except Exception as e:
            logger.error("scheduler.load_state_dict failed (key='{}'): {}", sched_key, e, exc_info=True)
    elif scheduler is None:
        logger.debug("Scheduler not provided; skipping scheduler restore.")
    else:
        logger.warning("No scheduler state found in checkpoint; scheduler reset to defaults.")

    # --- Restore GradScaler ---
    if trainer is not None and scaler_state is not None:
        try:
            if hasattr(trainer, "scaler") and trainer.scaler is not None:
                trainer.scaler.load_state_dict(scaler_state)
                logger.info("Loaded GradScaler state (key='{}').", scaler_key)
            else:
                logger.warning("Trainer provided but missing `scaler` attribute; skipping scaler restore.")
        except Exception as e:
            logger.error("Failed to load GradScaler state: {}", e, exc_info=True)
    elif scaler_state is not None:
        logger.warning("GradScaler state present in checkpoint but no trainer provided; skipping scaler restore.")

    # --- Restore trainer bookkeeping ---
    trainer_backbone_frozen = ckpt.get("trainer_backbone_frozen", None)
    latest_val_stats = ckpt.get("latest_val_stats", None)

    if trainer is not None:
        if trainer_backbone_frozen is not None:
            try:
                trainer._backbone_frozen = bool(trainer_backbone_frozen)
                logger.debug("Restored trainer._backbone_frozen={} from checkpoint.", trainer_backbone_frozen)
            except Exception as e:
                logger.warning("Failed to set trainer._backbone_frozen: {}", e)
        if latest_val_stats is not None:
            try:
                trainer.latest_val_stats = latest_val_stats
                logger.debug("Restored trainer.latest_val_stats from checkpoint.")
            except Exception as e:
                logger.warning("Failed to set trainer.latest_val_stats: {}", e)
    else:
        if trainer_backbone_frozen or latest_val_stats:
            logger.debug("Trainer not provided; skipping trainer-specific restores.")

    logger.info("Checkpoint load complete: start_epoch=%d, has_scaler={}, has_val_stats={}",
                start_epoch, scaler_state is not None, latest_val_stats is not None)

    return {
        "start_epoch": start_epoch,
        "scaler_state": scaler_state,
        "trainer_backbone_frozen": trainer_backbone_frozen,
        "latest_val_stats": latest_val_stats,
        "raw": ckpt,
    }



# small usage example (copy into train.py):
# from utils.checkpoint_utils import load_checkpoint
# ckpt_info = load_checkpoint(args.resume, model, optimizer=optim, scheduler=sched, trainer=trainer)
# start_epoch = ckpt_info.get('start_epoch', 1)

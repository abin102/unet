"""
utils/wandb_utils.py
--------------------
Helper utilities for initializing and integrating Weights & Biases (W&B)
into training pipelines without bloating train.py.
"""

import os
import time
from loguru import logger


def init_wandb(cfg: dict, out_dir: str):
    """
    Initialize Weights & Biases logging safely with proper error handling.
    Returns (wandb, use_wandb) where `use_wandb=False` if init failed.

    Args:
        cfg (dict): full experiment config (expects keys "use_wandb" and optional "wandb")
        out_dir (str): directory where logs and wandb files are stored
    """
    use_wandb = bool(cfg.get("use_wandb", False))
    wandb_cfg = cfg.get("wandb", {}) or {}

    if not use_wandb:
        logger.info("W&B disabled in config.")
        return None, False

    try:
        import wandb

        # Initialize W&B
        wandb.init(
            project=wandb_cfg.get("project", cfg.get("exp_name", "default_project")),
            entity=wandb_cfg.get("entity", None),
            # name="-"+time.strftime("%Y%m%d-%H%M%S"),
            name=f"{cfg['exp_name']}-{time.strftime('%Y%m%d-%H%M%S')}",
            dir=out_dir,                  # store wandb files in run dir
            config=cfg,                   # log the entire config
            resume=wandb_cfg.get("resume", False),
        )

        logger.info(
            "WandB initialized: project={} | name={} | dir={}",
            wandb_cfg.get("project"),
            wandb.run.name if hasattr(wandb, "run") else "unknown",
            out_dir,
        )
        return wandb, True

    except Exception as e:
        logger.exception("❌ Failed to initialize W&B: {}", e)
        return None, False


def watch_model(wandb, model, log: str = "gradients", log_freq: int = 100, log_graph: bool = False):
    """
    Attach W&B model watcher to log gradients, parameters, and optionally graph.

    Args:
        wandb: the imported wandb module (from init_wandb)
        model: nn.Module or DDP-wrapped model
        log (str): what to log ('gradients', 'parameters', etc.)
        log_freq (int): logging frequency in steps
        log_graph (bool): whether to log computation graph (use False for dynamic models)
    """
    if wandb is None:
        logger.warning("watch_model called but wandb is None; skipping.")
        return

    try:
        core_model = getattr(model, "module", model)  # unwrap DDP/DataParallel
        wandb.watch(core_model, log=log, log_freq=log_freq, log_graph=log_graph)
        logger.info("🧠 WandB is now watching model (log={}, log_freq=%d, log_graph={}).",
                    log, log_freq, log_graph)
    except Exception as e:
        logger.exception("⚠️ wandb.watch() failed: {}", e)

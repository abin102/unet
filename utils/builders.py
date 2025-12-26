# utils/builders.py
from typing import Tuple, Optional, Dict, Any
import torch, torch.nn.functional as F
from torch import nn
import inspect
from typing import Dict, Any
import torch

from models import REGISTRY as MODEL_REG
# from callbacks.model_logger import ModelLogger
# from utils.model_summary import summarize_and_save_model  # if you kept this helper
from loguru import logger


def build_model_from_cfg(model_cfg: Dict[str, Any], device="cuda"):
    """
    Build model from a cfg dict.

    model_cfg may be either:
      - {"name": "<registry_key>", "params": {...}}
      - {"model": "<registry_key>", "model_args": {...}}  (legacy)

    This function will automatically inject `registry=MODEL_REG` into the constructor
    if the constructor accepts a `registry` parameter but the user didn't provide it.
    """
    logger.debug("build_model_from_cfg called with model_cfg={}, device={}", model_cfg, device)

    # normalize possible formats
    if "name" in model_cfg:
        model_name = model_cfg["name"]
        model_args = dict(model_cfg.get("params", {}) or {})
    else:
        model_name = model_cfg.get("model") or model_cfg.get("model_name")
        model_args = dict(model_cfg.get("model_args", {}) or {})

    logger.debug("Normalized model_cfg -> model_name={}, model_args={}", model_name, model_args)

    if model_name is None:
        logger.critical("build_model_from_cfg requires model_cfg with 'name' or 'model' key. Received: {}", model_cfg)
        raise ValueError("build_model_from_cfg requires model_cfg with 'name' or 'model' key")

    if model_name not in MODEL_REG:
        # show a short sample of keys to avoid flooding logs
        sample_keys = list(MODEL_REG.keys())[:20]
        logger.error("Unknown model '{}' (not found in MODEL_REG). Sample available keys: {}", model_name, sample_keys)
        raise KeyError(f"Unknown model: {model_name}")

    ctor = MODEL_REG[model_name]
    logger.info("Resolved model constructor for '{}': {}", model_name, getattr(ctor, "__name__", repr(ctor)))
    # inspect constructor signature and inject registry if needed and not provided
    try:
        sig = inspect.signature(ctor)
        params = sig.parameters
        if "registry" in params and "registry" not in model_args:
            # inject model registry for convenience
            model_args["registry"] = MODEL_REG
            logger.debug("Injected 'registry' into model args for constructor of '{}'.", model_name)
    except Exception as e:
        # if signature inspection fails, just proceed without injection
        logger.warning("Failed to inspect signature for constructor of '{}': {}. Proceeding without injection.", model_name, e)

    # finally build and move to device
    try:
        logger.debug("Instantiating model '{}' with args: {}", model_name, model_args)
        model = ctor(**model_args)
    except Exception as e:
        logger.critical("Failed to instantiate model '{}' with args {}: {}", model_name, model_args, e)
        raise

    # cheap parameter summary
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model '{}' instantiated. Total params: {}, Trainable params: {}",
                    model_name, total_params, trainable_params)
        # log top-10 parameter tensors by size at debug level
        try:
            named = [(n, p.numel()) for n, p in model.named_parameters()]
            topk = sorted(named, key=lambda x: x[1], reverse=True)[:10]
            logger.debug("Top parameter tensors for '{}' (name, numel): {}", model_name, topk)
        except Exception:
            # non-fatal; just skip detailed named-parameter logging if it fails
            logger.debug("Could not enumerate named parameters for detailed logging.")
    except Exception as e:
        logger.warning("Could not compute parameter counts for model '{}': {}", model_name, e)

    if device is not None:
        try:
            model = model.to(device)
            logger.debug("Moved model '{}' to device: {}", model_name, device)
        except Exception as e:
            # ignore device move errors here; caller can move later
            logger.warning("Failed to move model '{}' to device '{}': {}. Caller may move it later.", model_name, device, e)

    logger.debug("build_model_from_cfg returning model '{}'", model_name)
    return model




def build_loss(loss_name, loss_args, device):
    """
    Build segmentation loss from config.
    """
    import torch.nn as nn
    from losses import CEDiceLoss   # ← IMPORTANT: from losses/, not utils

    if loss_name is None:
        raise ValueError("loss_name must be specified in config")

    lname = loss_name.lower()
    logger.info("Building loss '{}' with args={}", lname, loss_args)

    if lname == "ce":
        loss = nn.CrossEntropyLoss(**loss_args)

    elif lname == "ce_dice":
        loss = CEDiceLoss(**loss_args)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return loss.to(device)




def get_optim(name, params, **kw):
    """
    Build optimizer by name. Logs useful information at DEBUG/INFO/ERROR levels.
    """
    name_orig = name
    try:
        name = name.lower()
    except Exception:
        logger.error("Optimizer name must be a string; got {}", name, exc_info=True)
        raise

    logger.debug("get_optim called: name={}, params_type={}, kw={}", name, type(params), {k: v for k, v in kw.items()})

    try:

        # --- ADDED ADAMW HERE ---
        if name == "adamw":
            logger.info("Creating AdamW optimizer with args: {}", {k: v for k, v in kw.items()})
            return torch.optim.AdamW(params, **kw)
        
        if name == "adam":
            logger.info("Creating Adam optimizer with args: {}", {k: v for k, v in kw.items()})
            return torch.optim.Adam(params, **kw)

        if name == "sgd":
            # pop defaults but log what we used
            momentum = kw.pop("momentum", 0.9)
            nesterov = kw.pop("nesterov", True)
            logger.info("Creating SGD optimizer (momentum={}, nesterov={}) with remaining args: {}",
                        momentum, nesterov, {k: v for k, v in kw.items()})
            return torch.optim.SGD(params, momentum=momentum, nesterov=nesterov, **kw)

        logger.error("Unknown optimizer requested: '{}' (original input: {})", name, name_orig)
        raise ValueError(f"Unknown optim: {name}")
    except Exception as e:
        logger.critical("Failed to construct optimizer '{}': {}", name, e, exc_info=True)
        raise


def _build_warmup_lambda(warmup_epochs: int):
    """
    Internal: builds warmup lambda. Logged by callers when used.
    """
    def warmup_lambda(epoch):
        factor = (epoch + 1) / float(warmup_epochs)
        # matches original repo behavior
        return (1.0 / 3.0) * (1.0 - factor) + factor
    return warmup_lambda


def get_scheduler(name, optim, **kw):
    """
    Build a scheduler given:
      name: "multistep" | "cosine" | None
      optim: optimizer
      **kw: scheduler args, e.g. for multistep: milestones, gamma, optional warmup_epochs

    Adds logging for creation choices and any warmup behavior.
    """
    logger.debug("get_scheduler called: name={}, optim={}, kw={}", name, type(optim), {k: v for k, v in kw.items()})

    if name is None:
        logger.debug("No scheduler requested (name is None). Returning None.")
        return None

    try:
        lname = name.lower()
    except Exception:
        logger.error("Scheduler name must be a string; got {}", name, exc_info=True)
        raise

    if lname in ("none", ""):
        logger.debug("Scheduler name indicates no scheduler ('{}'). Returning None.", name)
        return None

    logger.info("Building scheduler '{}' with args: {}", lname, {k: v for k, v in kw.items()})

    try:
        if lname == "cosine":
            # CosineAnnealingLR: expect 'T_max' or 'T_max' equivalent args in kw
            logger.debug("Creating CosineAnnealingLR with args: {}", kw)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **kw)
            logger.info("Created CosineAnnealingLR scheduler.")
            return scheduler

        if lname in ("multistep", "step"):
            # extract warmup if present
            warmup_epochs = kw.pop("warmup_epochs", None)
            logger.debug("Multistep/Step scheduler requested. warmup_epochs={}, remaining args={}", warmup_epochs, kw)

            # build main MultiStepLR with remaining args
            logger.debug("Creating MultiStepLR with args: {}", kw)
            main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **kw)
            logger.info("Created MultiStepLR as main scheduler.")

            if warmup_epochs is None or warmup_epochs <= 0:
                logger.debug("No warmup requested (warmup_epochs={}). Returning main scheduler.", warmup_epochs)
                return main_scheduler

            # Build exact warmup lambda to match original repo behavior
            logger.info("Building warmup LambdaLR for {} epochs using exact ResLT warmup schedule.", warmup_epochs)

            def _exact_reslt_warmup_lambda(warmup):
                def fn(epoch_idx):
                    factor = float(epoch_idx + 1) / float(warmup)
                    return (1.0/3.0) * (1.0 - factor) + factor
                return fn

            warmup_lambda = _exact_reslt_warmup_lambda(warmup_epochs)
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda)
            logger.debug("Created LambdaLR warmup scheduler.")

            # SequentialLR: use warmup_scheduler for warmup_epochs, then main_scheduler
            try:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optim,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs]
                )
                logger.info("Created SequentialLR combining warmup and main MultiStepLR. Warmup epochs: {}", warmup_epochs)
                return scheduler
            except AttributeError:
                # SequentialLR may not exist on older torch versions
                logger.error("torch.optim.lr_scheduler.SequentialLR not available in this PyTorch version.", exc_info=True)
                raise

        logger.error("Unknown scheduler requested: '{}'", name)
        raise ValueError(f"Unknown scheduler: {name}")

    except Exception as e:
        logger.critical("Failed to build scheduler '{}': {}", name, e, exc_info=True)
        raise


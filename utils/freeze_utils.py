# utils/freeze_utils.py
"""
Utilities to freeze a model backbone (keeps train.py slim).
Provides a single entrypoint `apply_freeze(model, cfg, verbose=True)`.
"""

from typing import Tuple, Optional, Dict
import torch
from loguru import logger

COMMON_BACKBONE_ATTRS = ("backbone", "features", "encoder", "base", "backbone_net", "body", "stem")

def _find_backbone_module(model):
    """
    Return (module, attr_name) if found by common attributes or heuristics.
    """
    logger.debug("Searching for backbone module using common attributes: {}", COMMON_BACKBONE_ATTRS)
    # 1) try common attribute names
    for attr in COMMON_BACKBONE_ATTRS:
        if hasattr(model, attr):
            m = getattr(model, attr)
            if m is not None:
                logger.debug("Found backbone via attribute '{}': {}", attr, m.__class__.__name__)
                return m, attr

    # 2) prefer a child module with the most params (likely the convolutional backbone)
    best = None
    best_name = None
    best_params = 0
    logger.debug("No direct backbone attribute found; falling back to largest child heuristic.")
    try:
        for name, sub in model.named_children():
            try:
                cnt = sum(p.numel() for p in sub.parameters())
                logger.debug("Child '{}' has %d params", name, cnt)
                if cnt > best_params:
                    best_params = cnt
                    best = sub
                    best_name = name
            except Exception as e:
                logger.debug("Skipping child '{}' due to exception while counting params: {}", name, e)
                continue
    except Exception as e:
        logger.error("Error while iterating named_children() to find backbone: {}", e)

    if best is not None:
        logger.debug("Selected child '{}' as backbone hint with %d params", best_name, best_params)
    else:
        logger.debug("No suitable child module found as backbone hint.")
    return best, best_name


def _set_bn_eval(mod):
    """
    Set all BatchNorm modules to eval mode in-place to avoid updating running stats.
    """
    logger.debug("Setting BatchNorm layers to eval() in module: {}", mod.__class__.__name__)
    bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    for m in mod.modules():
        if isinstance(m, bn_types):
            try:
                m.eval()
                logger.debug("Set {} to eval()", m.__class__.__name__)
            except Exception as e:
                logger.warning("Failed to set {} to eval(): {}", m.__class__.__name__, e)


def apply_freeze(model: torch.nn.Module, cfg: dict, verbose: bool = True) -> Dict:
    """
    Freeze backbone parameters according to config.
    Returns a dict with diagnostic info:
      {
        "trainer_backbone_frozen": bool,
        "frozen_params": int,
        "kept_params": int,
        "backbone_hint": str|None
      }
    """
    # defensive wrapper: ensure we don't crash the caller on unexpected errors
    try:
        freeze_epochs = int(cfg.get("freeze_backbone_epochs", 0) or 0)
    except Exception as e:
        logger.error("Invalid cfg for freeze_backbone_epochs: {}. Defaulting to 0. Error: {}", cfg.get("freeze_backbone_epochs"), e)
        freeze_epochs = 0

    info = {
        "trainer_backbone_frozen": False,
        "frozen_params": 0,
        "kept_params": 0,
        "backbone_hint": None,
    }

    if freeze_epochs <= 0:
        if verbose:
            logger.info("[freeze_utils] freeze_backbone_epochs not set or 0 => skipping freeze.")
        else:
            logger.debug("freeze_backbone_epochs <= 0, skipping freeze.")
        return info

    logger.info("freeze_backbone_epochs=%d -> attempting to freeze backbone.", freeze_epochs)

    try:
        backbone_mod, backbone_hint = _find_backbone_module(model)
    except Exception as e:
        logger.critical("Unexpected error while finding backbone module: {}", e, exc_info=True)
        return info

    try:
        total_params = sum(p.numel() for p in model.parameters())
    except Exception as e:
        logger.error("Failed to count total model parameters: {}", e)
        total_params = 0

    if backbone_mod is None:
        # fallback: use simple name-heuristic (old behavior), but log diagnostics
        logger.warning("Backbone submodule not found. Using name-heuristic to freeze parameters (fc/head/classifier kept).")
        frozen = 0
        kept = 0
        try:
            for name, p in model.named_parameters():
                if name.startswith("fc.") or name.startswith("head.") or name.startswith("classifier."):
                    p.requires_grad = True
                    kept += p.numel()
                    logger.debug("Keeping param trainable: {} (%d)", name, p.numel())
                else:
                    p.requires_grad = False
                    frozen += p.numel()
                    logger.debug("Freezing param: {} (%d)", name, p.numel())
        except Exception as e:
            logger.error("Error while applying name-heuristic freeze: {}", e, exc_info=True)

        info.update({"trainer_backbone_frozen": True, "frozen_params": frozen, "kept_params": kept, "backbone_hint": "name-heuristic"})
        # set BN eval across whole model
        try:
            _set_bn_eval(model)
        except Exception as e:
            logger.warning("Failed to set BatchNorm eval across whole model: {}", e)

        if verbose:
            logger.info("[freeze_utils] WARNING: backbone submodule not found. Used name-heuristic to freeze.")
            logger.info("[freeze_utils] Frozen params=%d/%d. Kept head params=%d.", frozen, total_params, kept)
        else:
            logger.debug("Fallback freeze completed: frozen=%d, kept=%d, total=%d", frozen, kept, total_params)
        return info

    # freeze backbone params by object identity
    try:
        backbone_param_ids = {id(p) for p in backbone_mod.parameters()}
    except Exception as e:
        logger.error("Failed to collect backbone parameter ids: {}", e, exc_info=True)
        backbone_param_ids = set()

    frozen = 0
    kept = 0

    # 1) freeze backbone params
    try:
        for p in backbone_mod.parameters():
            if p.requires_grad:
                p.requires_grad = False
            frozen += p.numel()
        logger.debug("Backbone module '{}' frozen param total: %d", backbone_hint, frozen)
    except Exception as e:
        logger.error("Error while freezing backbone parameters for '{}': {}", backbone_hint, e, exc_info=True)

    # 2) ensure non-backbone params are trainable
    try:
        for name, p in model.named_parameters():
            if id(p) in backbone_param_ids:
                continue
            if not p.requires_grad:
                p.requires_grad = True
            kept += p.numel()
            logger.debug("Ensuring trainable param: {} (%d)", name, p.numel())
    except Exception as e:
        logger.error("Error while ensuring non-backbone parameters are trainable: {}", e, exc_info=True)

    # set BatchNorm layers inside backbone to eval to avoid updating running stats
    try:
        _set_bn_eval(backbone_mod)
    except Exception as e:
        logger.warning("Failed to set BatchNorm to eval inside backbone '{}': {}", backbone_hint, e)

    info.update({"trainer_backbone_frozen": True, "frozen_params": frozen, "kept_params": kept, "backbone_hint": backbone_hint})

    if verbose:
        logger.info("[freeze_utils] Freezing backbone module '{}' -> frozen_params=%d/%d. Kept head params=%d.",
                    backbone_hint, frozen, total_params, kept)
        # show examples
        logger.info("[freeze_utils] Example frozen backbone param names (first 8):")
        cnt = 0
        try:
            for name, _ in backbone_mod.named_parameters():
                logger.info("   {}.{}", backbone_hint if backbone_hint else "<backbone>", name)
                cnt += 1
                if cnt >= 8:
                    break
        except Exception as e:
            logger.debug("Could not iterate backbone parameters to show examples: {}", e)

        logger.info("[freeze_utils] Example head param names (first 12):")
        shown = 0
        try:
            for name, p in model.named_parameters():
                if id(p) not in backbone_param_ids:
                    logger.info("   {}", name)
                    shown += 1
                    if shown >= 12:
                        break
        except Exception as e:
            logger.debug("Could not iterate head parameters to show examples: {}", e)
    else:
        logger.debug("Freeze applied; summary frozen=%d kept=%d total=%d backbone_hint={}",
                     frozen, kept, total_params, backbone_hint)

    return info
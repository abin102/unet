# utils/outputs.py
import torch
from typing import Any, Optional, Tuple
from torch import nn

def normalize_outputs(outputs: Any, device: torch.device, normalize_W: bool=True, logger=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Any]:
    W = None
    logits = None

    # dict case (keeps behavior)
    if isinstance(outputs, dict):
        logits = outputs.get("logits", None)
        if logits is None and "probs" in outputs:
            probs = outputs["probs"].detach().clamp(1e-6, 1 - 1e-6)
            logits = torch.log(probs / (1.0 - probs))
        W = outputs.get("W", None)
        full_outputs = outputs

    # tuple / list might be one of:
    #  - per-branch logits: (logits1, logits2, ...)  (all elements 2D) -> sum them
    #  - main logits + extra diagnostics: (logits, u, (t1,t2), assigned) -> take first as logits and pass through full_outputs
    elif isinstance(outputs, (tuple, list)):
        full_outputs = outputs
        # check if all elements are 2D tensors -> treat as per-branch logits
        all_2d = True

        for o in outputs:
            if not torch.is_tensor(o) or o.dim() != 2:
                all_2d = False
                break
        if all_2d:
            # sum per-branch logits
            logits = sum(outputs)
        else:
            # fallback: expect first element is logits (2D tensor)
            first = outputs[0]
            if torch.is_tensor(first) and first.dim() == 2:
                logits = first
            else:
                raise RuntimeError("Unsupported model outputs tuple: expected first element to be logits (N,C) or all elements to be logits.")
    elif torch.is_tensor(outputs):
        logits = outputs
        full_outputs = outputs
    else:
        raise RuntimeError(f"Unsupported model outputs type: {type(outputs)}")

    if logits is None:
        raise RuntimeError("Model outputs did not contain logits or probs")

    logits = logits.to(device).float()

    # W normalization (unchanged)
    if W is not None:
        W = W.to(device).float()
        if not torch.isfinite(W).all():
            if logger:
                logger.warning("Non-finite W -> set ones")
            W = torch.ones_like(logits)
        if normalize_W:
            W = W.clamp(min=0.0)
            row_sum = W.sum(dim=1, keepdim=True)
            zero_mask = row_sum == 0
            if zero_mask.any():
                W[zero_mask.expand_as(W)] = 1.0
            W = W / (row_sum + 1e-12)
        if not torch.isfinite(W).all() or (W.abs().max() < 1e-12):
            if logger:
                logger.warning("W invalid after normalize -> uniform fallback")
            W = torch.ones_like(logits) / float(logits.size(1))

    return logits, W, full_outputs

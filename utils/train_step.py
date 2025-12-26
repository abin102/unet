from asyncio.log import logger
from typing import Tuple, Dict, Any
import torch
from contextlib import nullcontext

def step_batch(trainer, x: torch.Tensor, y: torch.Tensor, scaler) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Performs forward, loss compute, backward, and optimizer step using standard PyTorch AMP.
    """
    # 1. Zero Gradients
    trainer.optimizer.zero_grad(set_to_none=True)

    amp_on = getattr(trainer, "amp", False)
    
    # --- FIX IS HERE: Use torch.amp.autocast for modern PyTorch ---
    if amp_on:
        # device_type="cuda" is required for the modern API
        amp_ctx = torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        amp_ctx = nullcontext()

    info: Dict[str, Any] = {}

    with amp_ctx:
        # 2. Forward Pass
        try:
            outputs = trainer.model(x)
        except TypeError:
            outputs = trainer.model(x, current_epoch=getattr(trainer, "current_epoch", -1))

        logits = None
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("out", None))
        elif isinstance(outputs, tuple):
             logits = outputs[0]
        else:
            logits = outputs

        # 3. Handle Target Shape (Segmentation requires B,H,W)
        if y.ndim == 4:
            y = y.squeeze(1)

        # 4. Compute Loss
        main_loss = trainer.loss_fn(logits, y)

    # 5. Backward Pass & Step (Standard PyTorch Logic)
    scaler.scale(main_loss).backward()

    if trainer.grad_clip > 0:
        scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)

    scaler.step(trainer.optimizer)
    scaler.update()
    
    # 6. Logging / Info
    step_idx = int(getattr(trainer, "_global_step", 0)) + 1
    setattr(trainer, "_global_step", step_idx)

    info['main_loss'] = main_loss.item()
    info["lr"] = float(trainer.optimizer.param_groups[0]["lr"])

    # 7. Callbacks
    for cb in trainer.cbs:
        if hasattr(cb, "on_batch_end"):
            try:
                cb.on_batch_end(trainer, step_idx, info)
            except Exception:
                pass
            
    return main_loss, logits, info
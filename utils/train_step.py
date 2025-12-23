# utils/train_step.py
from asyncio.log import logger
from typing import Tuple, Dict, Any
import torch
from contextlib import nullcontext

def step_batch(trainer, x: torch.Tensor, y: torch.Tensor, grad_stepper) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Performs forward, loss compute, backward+step (via grad_stepper).
    Specific for Segmentation (B, C, H, W).
    """
    # zero grads
    try:
        trainer.optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass

    amp_on = bool(getattr(grad_stepper, "amp", False))
    device_str = "cuda" if "cuda" in str(getattr(trainer, "device", "cpu")).lower() else "cpu"
    amp_ctx = (torch.cuda.amp.autocast(device_type=device_str) if amp_on else nullcontext())

    info: Dict[str, Any] = {}

    with amp_ctx:
        # ---------------------------------------------------------
        # 1. FORWARD PASS
        # ---------------------------------------------------------
        # U-Net usually doesn't need 'current_epoch', but we keep the try/except just in case
        try:
            outputs = trainer.model(x)
        except TypeError:
            outputs = trainer.model(x, current_epoch=getattr(trainer, "current_epoch", -1))

        logits = None

        # Handle Dict outputs (common in advanced models) or Raw Tensor
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("out", None))
        elif isinstance(outputs, tuple):
             logits = outputs[0]
        else:
            logits = outputs # Raw Tensor (B, C, H, W)

        # ---------------------------------------------------------
        # 2. HANDLE TARGET SHAPE
        # ---------------------------------------------------------
        # CrossEntropyLoss expects Target to be (B, H, W) LongTensor
        # If your loader outputs (B, 1, H, W), squeeze it.
        if y.ndim == 4:
            y = y.squeeze(1)

        # ---------------------------------------------------------
        # 3. COMPUTE LOSS
        # ---------------------------------------------------------
        # Standard CrossEntropy for Segmentation
        main_loss = trainer.loss_fn(logits, y)

    # ---------------------------------------------------------
    # 4. BACKWARD PASS
    # ---------------------------------------------------------
    diag = grad_stepper.backward_and_step(main_loss, grad_clip=trainer.grad_clip,
                                          debug=trainer.debug, debug_dir=trainer.debug_dir,
                                          logger=getattr(trainer, "logger", None))
    
    step_idx = int(getattr(trainer, "_global_step", 0)) + 1
    setattr(trainer, "_global_step", step_idx)

    info['main_loss'] = main_loss.item()
    info["lr"] = float(trainer.optimizer.param_groups[0]["lr"])

    # ---------------------------------------------------------
    # 5. CALLBACKS
    # ---------------------------------------------------------
    for cb in trainer.cbs:
        if hasattr(cb, "on_batch_end"):
            try:
                cb.on_batch_end(trainer, step_idx, info)
            except Exception:
                pass
            
    return main_loss, logits, info
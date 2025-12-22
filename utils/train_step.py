# utils/train_step.py
from asyncio.log import logger
from typing import Tuple, Dict, Any
import torch
from contextlib import nullcontext

def step_batch(trainer, x: torch.Tensor, y: torch.Tensor, grad_stepper) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Performs forward, loss compute, backward+step (via grad_stepper), and calls
    per-batch callbacks. Returns (loss, logits, info).
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
        current_epoch = getattr(trainer, "current_epoch", -1)
        try:
            # Prefer asking model for gate info; fallback if not supported
            outputs = trainer.model(x, current_epoch=current_epoch)
            # logger.debug(f"model {type(trainer.model).__name__} called from train_step")
        except TypeError:
            outputs = trainer.model(x)

        logits = None

        # ---------------------------------------------------------
        # 1. EXTRACT LOGITS
        # ---------------------------------------------------------
        if isinstance(outputs, torch.Tensor):
            logits = outputs
            outputs = None # Clear to save memory if not needed
            
        elif isinstance(outputs, tuple) and len(outputs) == 3 and all(torch.is_tensor(o) for o in outputs):
            # Legacy ResLT support
            logits =  outputs[0] + outputs[1] + outputs[2]

        elif isinstance(outputs, dict):
            logits = outputs.get("logits", None)

        # ---------------------------------------------------------
        # 2. COMPUTE MAIN LOSS
        # ---------------------------------------------------------
        res = trainer.loss_fn(logits, y)
        main_loss = res
        
        # ---------------------------------------------------------
        # 3. HANDLE AUXILIARY OUTPUTS (Curriculum Loss & Logging)
        # ---------------------------------------------------------
        if isinstance(outputs, dict):
            
            # --- Case A: Old "PlugAndPlayExperts" (Explicit name check) ---
            if "name" in outputs and outputs["name"] == "PlugAndPlayExperts":
                curriculum_loss_tensor = outputs["curriculum_loss"]
                curriculum_multiplier = float(trainer.cfg["model_args"].get("curriculum_loss_multiplier", 0.01))
                
                main_loss = main_loss + curriculum_multiplier * curriculum_loss_tensor
                info["curriculum_loss"] = curriculum_loss_tensor.item()

                if "to_plot" in outputs:
                    plot_data = outputs["to_plot"]
                    for k in plot_data:
                        if torch.is_tensor(plot_data[k]):
                            info[k] = plot_data[k].detach().cpu()
                        else:
                            info[k] = plot_data[k]

            # --- Case B: New "ReparamExpert" (Key check) ---
            # The new model returns "loss_aux" and "gating_weights" directly
            else:
                # 1. Handle Auxiliary Loss
                if "loss_aux" in outputs:
                    loss_aux = outputs["loss_aux"]
                    # Get multiplier from config, default to 1.0 if not found, 
                    # but typically we want the 0.01 logic from the config
                    mult = float(trainer.cfg.get("model_args", {}).get("curriculum_loss_multiplier", 1.0))
                    
                    main_loss = main_loss + (mult * loss_aux)
                    info["loss_aux"] = loss_aux.item()

                # 2. Handle Gating Weights (For WandB plotting)
                if "gating_weights" in outputs:
                    # Detach and move to CPU immediately to avoid GPU leaks in logger
                    info["gating_weights"] = outputs["gating_weights"].detach().cpu()


    # ---------------------------------------------------------
    # 4. BACKWARD PASS
    # ---------------------------------------------------------
    diag = grad_stepper.backward_and_step(main_loss, grad_clip=trainer.grad_clip,
                                          debug=trainer.debug, debug_dir=trainer.debug_dir,
                                          logger=getattr(trainer, "logger", None))
    
    step_idx = int(getattr(trainer, "_global_step", 0)) + 1
    setattr(trainer, "_global_step", step_idx)

    info['main_loss'] = main_loss.item()
    info["epoch"] = current_epoch
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
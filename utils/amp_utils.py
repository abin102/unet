# utils/amp_utils.py
import os
import torch
from typing import Dict, Any
from torch import nn

class GradStepper:
    """
    Encapsulate GradScaler and step logic so training loop stays small.
    """
    def __init__(self, amp_enabled: bool, optimizer: torch.optim.Optimizer, model: nn.Module, scaler_init_scale: float = 2**12):
        self.amp = bool(amp_enabled)
        self.optimizer = optimizer
        self.model = model
        # create a scaler even if amp is False (it will be disabled)
        self.scaler = torch.amp.GradScaler(enabled=self.amp, init_scale=scaler_init_scale)

    def backward_and_step(self, loss: torch.Tensor, grad_clip: float = 0.0, debug: bool = False, debug_dir: str = None, logger=None) -> Dict[str, Any]:
        """
        Do backward + optimizer step. Returns diagnostics dict.
        Raises RuntimeError on non-finite grads.
        """
        diag: Dict[str, Any] = {"pre_norm": None, "post_norm": None, "clipped": False, "current_scale": None}

        if self.amp:
            # AMP path
            self.scaler.scale(loss).backward()

            if grad_clip > 0:
                # try to unscale (safe; may raise for some optimizers, so catch)
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception:
                    pass

            # compute grad norm (L2)
            total_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    try:
                        g = p.grad.detach().float()
                        total_sq += float(g.norm(2).item()) ** 2
                    except Exception:
                        pass
            total_norm = total_sq ** 0.5
            diag["pre_norm"] = total_norm

            # clip if requested
            if grad_clip > 0:
                clipped = total_norm > grad_clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                diag["clipped"] = clipped

            # post norm
            post_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    try:
                        post_sq += float(p.grad.detach().float().norm(2).item()) ** 2
                    except Exception:
                        pass
            diag["post_norm"] = post_sq ** 0.5

            try:
                diag["current_scale"] = float(self.scaler.get_scale())
            except Exception:
                diag["current_scale"] = None

            # detect non-finite grads and optionally save debug info
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    if debug and debug_dir:
                        try:
                            os.makedirs(debug_dir, exist_ok=True)
                            g = p.grad.detach().cpu().view(-1)[:100].clone()
                            torch.save({"param": name, "grad_sample": g}, os.path.join(debug_dir, f"nan_grad_after_unscale_{name.replace('/','_')[:80]}.pth"))
                        except Exception:
                            pass
                    raise RuntimeError(f"Non-finite grad detected in {name}; aborting")

            # step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # non-AMP path
            loss.backward()

            total_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    try:
                        total_sq += float(p.grad.detach().float().norm(2).item()) ** 2
                    except Exception:
                        pass
            total_norm = total_sq ** 0.5
            diag["pre_norm"] = total_norm

            if grad_clip > 0:
                clipped = total_norm > grad_clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                diag["clipped"] = clipped

            post_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    try:
                        post_sq += float(p.grad.detach().float().norm(2).item()) ** 2
                    except Exception:
                        pass
            diag["post_norm"] = post_sq ** 0.5

            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    if debug and debug_dir:
                        try:
                            os.makedirs(debug_dir, exist_ok=True)
                            torch.save({"param": name}, os.path.join(debug_dir, f"nan_grad_{name.replace('/','_')[:80]}.pth"))
                        except Exception:
                            pass
                    raise RuntimeError(f"Non-finite grad detected in {name}; aborting")

            self.optimizer.step()

        return diag

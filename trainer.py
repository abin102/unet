# trainer.py (façade - tiny)
import torch
from torch import nn
from typing import Optional, Iterable, Dict, Any

from utils.eval_utils import evaluate
from utils.amp_utils import GradStepper
from utils.train_step import step_batch
from utils.io_utils import save_crash_debug
from utils.logging_utils import init_logger_and_tb


class Trainer:
    """
    Thin Trainer façade. Keeps same constructor signature and delegates
    work to small helpers in utils/.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer,
        scheduler=None,
        amp=False,
        grad_clip=0.0,
        device="cuda",
        callbacks: Iterable = (),
        progress_bar=True,
        class_splits: Optional[Dict[str, Iterable[int]]] = None,
        debug=False,
        debug_dir="debug",
        tb_logdir: Optional[str] = None,
        use_wandb: bool = False,
        normalize_W: bool = True,
        cfg: dict = None,
        logger=None,
        tb_writer=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = bool(amp)
        self.grad_clip = float(grad_clip)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.cbs = list(callbacks)
        self.progress_bar = progress_bar
        self.class_splits = class_splits or {}
        self.debug = bool(debug)
        self.debug_dir = debug_dir if self.debug else None
        self.tb_logdir = tb_logdir
        self.use_wandb = bool(use_wandb)
        self.normalize_W = bool(normalize_W)
        self.cfg = cfg or {}
        logger.info("Trainer loaded config keys:{}", list(self.cfg.keys()))

        # logger
        if logger is not None:
            self.logger = logger
            self.tb_writer = tb_writer
        else:
            self.logger, self.tb_writer = init_logger_and_tb(self.debug, self.debug_dir, self.tb_logdir)

        # AMP helper encapsulates GradScaler and step logic
        self._grad_stepper = GradStepper(self.amp, self.optimizer, self.model, scaler_init_scale=2 ** 12)

        # bookkeeping
        self.latest_val_stats = {}
        self._backbone_frozen = False

    def fit(self, train_loader, val_loader, epochs: int, start_epoch: int = 1):
        # call callbacks on_train_begin
        for cb in self.cbs:
            try:
                cb.on_train_begin(self)
            except Exception:
                self.logger.error("Callback on_train_begin failed")

        step_scheduler_when = str(self.cfg.get("scheduler_step", "epoch")).lower()

        self._global_step = 0
        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            self.current_epoch = epoch

            # unfreeze logic (kept simple — same heuristics as before)
            freeze_epochs = int(self.cfg.get("freeze_backbone_epochs", 0))
            # inside your training loop (e.g., Trainer.train_epoch or similar)
            if freeze_epochs > 0 and epoch == (freeze_epochs + 1) and self._backbone_frozen:
                try:
                    self.logger.info("Backbone freeze period ({} epochs) completed. Starting unfreeze at epoch {}.",
                                    freeze_epochs, epoch)

                    # try hint/backbone/features
                    hint = self.cfg.get("_backbone_name_hint", None)
                    if hint:
                        self.logger.debug("Backbone name hint found in cfg: {}", hint)

                    backbone = (
                        getattr(self.model, hint)
                        if hint and hasattr(self.model, hint)
                        else getattr(self.model, "backbone", None)
                        or getattr(self.model, "features", None)
                    )

                    if backbone is not None:
                        self.logger.info("Unfreezing backbone module: {}", backbone.__class__.__name__)
                        unfrozen_params = 0
                        for p in backbone.parameters():
                            if not p.requires_grad:
                                p.requires_grad = True
                                unfrozen_params += p.numel()

                        bn_layers = 0
                        for m in backbone.modules():
                            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                                m.train()
                                bn_layers += 1

                        self.logger.info("Backbone unfrozen successfully at epoch {}. Re-enabled gradients for {} params and set {} BatchNorm layers to train mode.",
                                        epoch, unfrozen_params, bn_layers)
                    else:
                        self.logger.warning("No explicit backbone module found (hint={}). Falling back to unfreezing all parameters.", hint)
                        total_unfrozen = 0
                        for name, p in self.model.named_parameters():
                            if not p.requires_grad:
                                p.requires_grad = True
                                total_unfrozen += p.numel()

                        bn_layers = 0
                        for m in self.model.modules():
                            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                                m.train()
                                bn_layers += 1

                        self.logger.info("Fallback unfreeze complete at epoch {}. Total params unfrozen={}, BatchNorm layers reactivated={}.",
                                        epoch, total_unfrozen, bn_layers)

                    # note: optimizer recreation optional
                    self.logger.debug("Keeping existing optimizer object. Caller may recreate optimizer externally if desired.")

                    # update state
                    self._backbone_frozen = False
                    self.logger.info("Trainer state updated: _backbone_frozen=False at epoch {}.", epoch)

                except Exception as e:
                    self.logger.error("Error during backbone unfreeze at epoch {}: {}", epoch, e, exc_info=True)

            total = correct = 0
            loss_sum = 0.0

            iterator = (train_loader if not self.progress_bar else __import__("tqdm").tqdm(train_loader, 
                                                                                           desc=f"Train {epoch}/{epochs}", 
                                                                                           leave=False))

            for batch in iterator:
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)

                try:
                    # single-batch step (forward, loss, backward, optimizer step, callbacks)
                    loss, logits, info = step_batch(self, x, y, grad_stepper=self._grad_stepper)
                    # self._global_step += 1

                except Exception as e:
                    tb = __import__("traceback").format_exc()
                    self.logger.exception("Exception during training step")
                    try:
                        save_crash_debug(self.debug_dir, "final", exc=str(e), 
                                         traceback=tb, model_state=self.model.state_dict(), 
                                         opt_state=self.optimizer.state_dict(), inputs=x.detach().cpu(), 
                                         targets=y.detach().cpu())
                    except Exception:
                        self.logger.exception("Failed to save crash debug")
                    raise

                bs = x.size(0)
                loss_sum += float(loss.item()) * bs

                # compute acc
                try:
                    preds = logits.argmax(1)
                    y_idx = y.argmax(dim=1) if y.dim() > 1 else y.view(-1)
                    correct += int((preds == y_idx).sum().item())
                except Exception:
                    self.logger.error("Could not compute batch accuracy")

                total += bs

                if self.progress_bar:
                    try:
                        iterator.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/total):.4f}")
                    except Exception:
                        pass

            train_loss = loss_sum / total
            train_acc = correct / total
            # evaluate
            self.model.eval()
            val_loss, val_acc, stats = evaluate(self, val_loader, epoch=epoch)
            self.latest_val_stats = stats

            # scheduler step (epoch-level)
            if getattr(self, "scheduler", None) is not None and step_scheduler_when == "epoch":
                try:
                    from torch.optim.lr_scheduler import ReduceLROnPlateau
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                except Exception:
                    self.logger.exception("Scheduler step failed at epoch end")

            #
            self.logger.info(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
                )


            for cb in self.cbs:
                try:
                    cb.on_epoch_end(self, epoch, train_loss, train_acc, val_loss, val_acc)
                except Exception:
                    self.logger.exception("Callback on_epoch_end failed")

        for cb in self.cbs:
            try:
                getattr(cb, "on_train_end", lambda *_: None)(self)
            except Exception:
                self.logger.exception("Callback on_train_end failed")

    def evaluate(self, loader, epoch: Optional[int] = None, epochs: Optional[int] = None):
        return evaluate(self, loader, epoch)

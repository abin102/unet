import torch
from torch import nn
from typing import Optional, Iterable, Dict, Any
from torch.amp import GradScaler

# Import the segmentation specific evaluate
from utils.eval_utils import evaluate
from utils.train_step import step_batch
from utils.io_utils import save_crash_debug
from utils.logging_utils import init_logger_and_tb

class Trainer:
    """
    Segmentation Trainer (Standard PyTorch Loop).
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
        debug=False,
        debug_dir="debug",
        tb_logdir: Optional[str] = None,
        use_wandb: bool = False,
        cfg: dict = None,
        logger=None,
        tb_writer=None,
        **kwargs
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
        self.debug = bool(debug)
        self.debug_dir = debug_dir if self.debug else None
        self.tb_logdir = tb_logdir
        self.use_wandb = bool(use_wandb)
        self.cfg = cfg or {}
        
        # logger
        if logger is not None:
            self.logger = logger
            self.tb_writer = tb_writer
        else:
            self.logger, self.tb_writer = init_logger_and_tb(self.debug, self.debug_dir, self.tb_logdir)

        # --- REPLACED GRAD STEPPER WITH STANDARD SCALER ---
        self.scaler = GradScaler(enabled=self.amp)
        self.latest_val_stats = {}

    def fit(self, train_loader, val_loader, epochs: int, start_epoch: int = 1):
        for cb in self.cbs:
            try: cb.on_train_begin(self)
            except Exception: self.logger.error("Callback on_train_begin failed")

        step_scheduler_when = str(self.cfg.get("scheduler_step", "epoch")).lower()
        self._global_step = 0

        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            self.current_epoch = epoch

            # --- Training Loop ---
            total_pixels = 0
            correct_pixels = 0
            loss_sum = 0.0

            iterator = (train_loader if not self.progress_bar else __import__("tqdm").tqdm(train_loader, 
                                                                                           desc=f"Train {epoch}/{epochs}", 
                                                                                           leave=False))

            for batch in iterator:
                x = batch[0].to(self.device, non_blocking=True)
                y = batch[1].to(self.device, non_blocking=True)

                try:
                    # Pass self.scaler instead of grad_stepper
                    loss, logits, info = step_batch(self, x, y, scaler=self.scaler)
                except Exception as e:
                    self.logger.exception("Exception during training step")
                    raise

                # --- Segmentation Metrics (Pixel Accuracy) ---
                bs = x.size(0)
                loss_sum += float(loss.item()) * bs

                try:
                    preds = logits.argmax(1) # (B, H, W)
                    if y.ndim == 4: y_target = y.squeeze(1)
                    else: y_target = y
                    
                    correct_pixels += int((preds == y_target).sum().item())
                    total_pixels += y_target.numel() 
                except Exception:
                    pass

                if self.progress_bar:
                    curr_acc = correct_pixels / max(total_pixels, 1)
                    iterator.set_postfix(loss=f"{loss.item():.4f}", pix_acc=f"{curr_acc:.4f}")

            # --- Epoch End Logic ---
            train_loss = loss_sum / len(train_loader.dataset)
            train_acc = correct_pixels / max(total_pixels, 1)

            # Evaluate 
            val_loss, main_score, stats = evaluate(self, val_loader, epoch=epoch)
            self.latest_val_stats = stats

            # Scheduler Step
            if getattr(self, "scheduler", None) is not None and step_scheduler_when == "epoch":
                try:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                except Exception:
                    pass

            # Logging
            log_msg = (f"Epoch {epoch:03d}/{epochs} | "
                       f"Tr.Loss={train_loss:.4f} Tr.PixAcc={train_acc:.4f} | "
                       f"Val.Loss={val_loss:.4f}")
            
            if "dice_infection" in stats:
                log_msg += f" Dice(Inf)={stats['dice_infection']:.4f}"
            if "mae" in stats:
                log_msg += f" MAE={stats['mae']:.4f}"
            
            self.logger.info(log_msg)

            for cb in self.cbs:
                try:
                    cb.on_epoch_end(self, epoch, train_loss, train_acc, val_loss, main_score)
                except Exception:
                    self.logger.exception("Callback on_epoch_end failed")

        for cb in self.cbs:
            try: getattr(cb, "on_train_end", lambda *_: None)(self)
            except Exception: pass

    def evaluate(self, loader, epoch=None):
        return evaluate(self, loader, epoch)
# callbacks/logging.py
import os
import math
from typing import Optional, Dict, Any, Iterable
from collections import deque
from torchvision.utils import make_grid

import torch
import numpy as np
from loguru import logger

# optional W&B
try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None
    _WANDB = False


class LoggingCallback:
    """
    Centralized logging for TensorBoard and optional W&B.

    Usage:
        cb = LoggingCallback(tb_logdir="runs/exp1", use_wandb=False, save_attention_dir="attn")
        trainer = Trainer(..., callbacks=[cb])

    Hooks expected on trainer:
      - cb.on_train_begin(trainer)
      - cb.on_epoch_end(trainer, epoch, train_loss, train_acc, val_loss, val_acc)
      - cb.on_batch_end(trainer, global_step, outputs, logits, targets, info)  # optional, we call internally
    """

    def __init__(
        self,
        tb_logdir: Optional[str] = None,
        use_wandb: bool = False,
        save_attention_dir: Optional[str] = None,
        max_attention_images: int = 8,
        log_w_histogram: bool = True,
        log_se_gates: bool = True,
        log_loss_per_class: bool = True,
    ):
        # prefer using the project-wide logger by default; will be overwritten in on_train_begin
        self._log = logger
 
        self.tb_logdir = tb_logdir
        self.use_wandb = use_wandb and _WANDB
        self.save_attention_dir = save_attention_dir
        self.max_attention_images = int(max_attention_images)
        self.log_w_histogram = bool(log_w_histogram)
        self.log_se_gates = bool(log_se_gates)
        self.log_loss_per_class = bool(log_loss_per_class)

        self.tb = None
        # sample routing buffers (optional for future image grids)
        self._sample_buf = {0: deque(maxlen=16), 1: deque(maxlen=16), 2: deque(maxlen=16)}
        self._examples_flush_every = 100  # flush image grids every N steps if you later enable images

        if tb_logdir is not None:
            # TensorBoard
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception:
                SummaryWriter = None

            if SummaryWriter is None:
                self._log.warning("TensorBoard requested but SummaryWriter not available.")
            else:
                os.makedirs(tb_logdir, exist_ok=True)
                self.tb = SummaryWriter(log_dir=tb_logdir)
                self._log.info("TensorBoard logging to {}", tb_logdir)

        if self.use_wandb and not _WANDB:
            self._log.warning("W&B requested but wandb not installed; disabling.")

        if save_attention_dir:
            os.makedirs(save_attention_dir, exist_ok=True)

        # internal: last LR logged to avoid repeated logs if unchanged
        self._last_logged_lr = None

    # -------------------------
    # Trainer hooks (minimal API)
    # -------------------------
    def on_train_begin(self, trainer):
        """
        Called once before training loop starts.
        trainer: Trainer instance — we may attach references if needed.
        """
        self.trainer = trainer
        # prefer trainer logger if present (keeps formatting/sinks consistent)
        try:
            self._log = getattr(trainer, "logger", self._log)
        except Exception:
            # fallback already set
            pass
        # optionally log model graph (first batch) outside; user can call cb.log_model_graph(...)
        self._log.debug("LoggingCallback attached to trainer.")
        return

    def _get_current_lr(self, trainer) -> float:
        """Return the (first) param_group LR or average if multiple groups."""
        try:
            lrs = [pg.get("lr", None) for pg in trainer.optimizer.param_groups]
            lrs = [float(x) for x in lrs if x is not None]
            if not lrs:
                return 0.0
            # if multiple groups, return the average to log a single scalar
            return float(sum(lrs) / len(lrs))
        except Exception:
            return 0.0

    def on_epoch_end(self, trainer, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        # top of on_epoch_end
        self._log.debug("on_epoch_end called: use_wandb={} _WANDB={} wandb_module={} wandb.run={}",
                        self.use_wandb, _WANDB, "present" if wandb is not None else "None", getattr(wandb, "run", None))

        # scalar logging to TensorBoard
        if self.tb:
            try:
                self.tb.add_scalar("epoch/train_loss", float(train_loss), epoch)
                self.tb.add_scalar("epoch/train_acc", float(train_acc), epoch)
                self.tb.add_scalar("epoch/val_loss", float(val_loss), epoch)
                self.tb.add_scalar("epoch/val_acc", float(val_acc), epoch)

                # learning rate
                lr = self._get_current_lr(trainer)
                self.tb.add_scalar("epoch/train_lr", float(lr), epoch)
            except Exception:
                self._log.exception("Failed TB epoch logging")

        # W&B logging for epoch-level scalars
        # inside LoggingCallback.on_epoch_end, replace existing W&B block with this
        if self.use_wandb and wandb is not None:
            try:
                lr = self._get_current_lr(trainer)
                step_idx = int(getattr(trainer, "_global_step", epoch))

                # flat, easy-to-find metric keys
                logd = {
                    "epoch/train_loss_epoch_end": float(train_loss),
                    "epoch/train_acc_on_epoch_end": float(train_acc),
                    "epoch/val_loss_epoch_end": float(val_loss),
                    "epoch/val_acc_epoch_end": float(val_acc),
                }

                # do the logging\
                # logger.warning("W&B: logging epoch scalars at epoch {}", epoch)
                wandb.log(logd)

                # debug: confirm success
                self._log.debug("W&B: logged epoch scalars successfully (epoch={})", epoch)
            except Exception:
                self._log.exception("Failed W&B epoch logging")

    def on_batch_end(
        self,
        trainer,
        global_step: int,
        info: Optional[Dict] = None,
    ):
        """
        Called by trainer each batch (optional). We expect:
          - logits: normalized logits Tensor [B,K]
          - targets: ground truth (tensor)
          - info: optional diagnostics dict (from step_batch or loss) containing keys like:
                W_mean_per_class (tensor), loss_per_class_mean (tensor), pos_per_class (tensor),
                gating_logits (tensor Nx3), assigned (tensor Nx1), curriculum_loss (float), main_loss (float)
        """
        step = int(global_step)

        # defensively ensure info exists
        if info is None:
            info = {}

        # -----------------
        # TensorBoard logs
        # -----------------
        try:
            if self.tb:
                # loss per class mean
                if self.log_loss_per_class and "loss_per_class_mean" in info:
                    v = info["loss_per_class_mean"]
                    if torch.is_tensor(v):
                        v_cpu = v.detach().cpu()
                        self.tb.add_scalar("train/loss_per_class_mean_avg", float(v_cpu.mean()), step)
                        # log first few classes as scalars
                        for i in range(min(8, v_cpu.numel())):
                            self.tb.add_scalar(f"train/loss_per_class_mean/class_{i}", float(v_cpu[i]), step)

                # curriculum and main losses (added)
                if "curriculum_loss" in info:
                    v = info["curriculum_loss"]
                    if isinstance(v, (int, float)):
                        self.tb.add_scalar("train/loss_curriculum", float(v), step)
                if "main_loss" in info:
                    v = info["main_loss"]
                    if isinstance(v, (int, float)):
                        self.tb.add_scalar("train/loss_main", float(v), step)

                # W stats
                if self.log_w_histogram and "W_mean_per_class" in info:
                    v = info["W_mean_per_class"]
                    if torch.is_tensor(v):
                        self.tb.add_histogram("train/W_mean_per_class", v.detach().cpu().numpy(), step)

                if "pos_per_class" in info:
                    v = info["pos_per_class"]
                    if torch.is_tensor(v):
                        self.tb.add_histogram("train/pos_per_class", v.detach().cpu().numpy(), step)

                # SE gating logging (if provided)
                if self.log_se_gates and "se_gates" in info and torch.is_tensor(info["se_gates"]):
                    g = info["se_gates"].detach().cpu()
                    self.tb.add_scalar("train/se_gates_mean", float(g.mean()), step)
                    self.tb.add_scalar("train/se_gates_std", float(g.std()), step)
                
                # log epoch 
                if "epoch" in info:
                    epoch_ = int(info["epoch"])
                    self.tb.add_scalar("train/epoch", epoch_, step)

            # W&B logs (per-batch)
            # W&B logs (per-batch)
            if self.use_wandb and wandb is not None:
                logd = {}
                if step == 1:
                    logger.debug("W&B: preparing per-batch logging at step {}", step)

                # W histograms
                if self.log_w_histogram and "W_mean_per_class" in info:
                    v = info["W_mean_per_class"]
                    if torch.is_tensor(v):
                        logd["train/W_mean_per_class_hist"] = wandb.Histogram(v.detach().cpu().numpy())
                        if step == 1:
                            logger.debug(
                                "W&B: logged W_mean_per_class_hist of value {} at step {}",
                                wandb.Histogram(v.detach().cpu().numpy()), step
                            )

                # loss per class average
                if "loss_per_class_mean" in info:
                    v = info["loss_per_class_mean"]
                    if torch.is_tensor(v):
                        # use .mean().item() safely
                        try:
                            logd["train/loss_per_class_mean_avg"] = float(v.detach().cpu().mean().item())
                        except Exception:
                            logd["train/loss_per_class_mean_avg"] = float(v.detach().cpu().mean())

                # pos per class histogram
                if "pos_per_class" in info:
                    v = info["pos_per_class"]
                    if torch.is_tensor(v):
                        logd["train/pos_per_class_hist"] = wandb.Histogram(v.detach().cpu().numpy())

                # curriculum/main losses
                if "curriculum_loss" in info:
                    v = info["curriculum_loss"]
                    # logger.warning("the curriculum loss instance is of type {}".format(type(v)))
                    if isinstance(v, (int, float)):
                        logd["train/loss_curriculum"] = float(v)
                        if step == 1:
                            logger.debug("W&B: logged curriculum_loss={} at step {}", float(v), step)
                if "main_loss" in info:
                    v = info["main_loss"]
                    if isinstance(v, (int, float)):
                        logd["train/loss_main"] = float(v)
                        if step == 1:
                            logger.debug("W&B: logged main_loss={} at step {}", float(v), step)

                # gating / routing info
                if "gating_logits" in info and torch.is_tensor(info["gating_logits"]):
                    gl = info["gating_logits"].detach().cpu().numpy()
                    logd["train/gating_logits_hist"] = wandb.Histogram(gl)
                    # also log softmax weights histogram and per-expert means
                    try:
                        weights = torch.nn.functional.softmax(info["gating_logits"], dim=1).detach().cpu().numpy()
                        logd["train/gating_weights_hist"] = wandb.Histogram(weights)
                        # mean weights per expert
                        logd["train/mean_weight_A"] = float(weights[:, 0].mean())
                        logd["train/mean_weight_B"] = float(weights[:, 1].mean())
                        logd["train/mean_weight_C"] = float(weights[:, 2].mean())
                        if step == 1:
                            logger.debug("W&B: logged gating weights and means at step {}", step)
                    except Exception:
                        # ignore if softmax or shape fails
                        pass

                if "gating_weights" in info and torch.is_tensor(info["gating_weights"]):
                    gw = info["gating_weights"].detach().cpu().numpy()
                    if step == 1:
                        logger.debug("W&b going to log gating weights with value {} at step {}", gw, step)
                        logger.debug("shape of gw is {} ", gw.shape)
                    logd["train/gating_weights_hist"] = wandb.Histogram(gw)
                    # mean weights per expert
                    logd["train/mean_weight_expertA"] = float(gw[:, 0].mean())
                    logd["train/mean_weight_expertB"] = float(gw[:, 1].mean())
                    logd["train/mean_weight_expertC"] = float(gw[:, 2].mean())
                    if step == 1:
                        logger.debug("W&B: going to log gating mean_weight_expertA={} at step {}", float(gw[:, 0].mean()), step)
                        logger.debug("W&B: going to log gating mean_weight_expertB={} at step {}", float(gw[:, 1].mean()), step)
                        logger.debug("W&B: going to log gating mean_weight_expertC={} at step {}", float(gw[:, 2].mean()), step)

                if "lr" in info:
                    lr_ = float(info["lr"])
                    logd["train/lr"] = lr_

                # assigned routing (counts)
                if "assigned" in info:
                    assigned = info["assigned"]
                    try:
                        assigned_np = assigned.detach().cpu().numpy() if torch.is_tensor(assigned) else np.array(assigned)
                        # counts per expert
                        counts = {f"assigned_count_expert_{i}": int((assigned_np == i).sum()) for i in (0, 1, 2)}
                        # attach counts to logd
                        logd.update({f"assigned_count_expert_{i}": counts[f"assigned_count_expert_{i}"] for i in counts})
                    except Exception:
                        pass
                
                # if "total_"

                if "epoch" in info:
                    epoch_ = int(info["epoch"])
                    logd["train/epoch"] = epoch_

                if "confidence" in info:
                    conf_  = info["confidence"].detach().cpu().numpy()
                    logd["train/confidence_hist"] = wandb.Histogram(conf_)
                    logd["train/confidence_mean"] = float(conf_.mean())
                    logd["train/confidence_std"] = float(conf_.std())

                

                # warn about any unhandled info keys ONLY for the first training step
                if step == 1:
                    _handled_keys = {
                        "W_mean_per_class", "loss_per_class_mean", "pos_per_class",
                        "curriculum_loss", "main_loss", "gating_logits", "gating_weights",
                        "lr", "assigned", "epoch", "step", "lr_groups"
                    }
                    _unhandled = [k for k in info.keys() if k not in _handled_keys]

                    if _unhandled:
                        logger.warning("W&B: unhandled info keys on first batch: {}", _unhandled)

                # finally write to wandb if we have anything
                if logd:
                    try:
                        # logger.warning("W&B: logging per-batch data at step- {}", step)
                        # logger.warning("logging the data dict keys: {}", list(logd.keys()))
                        wandb.log(logd)

                    except Exception:
                        self._log.exception("Failed to wandb.log per-batch data")
        except Exception:
            self._log.exception("Failed to log batch info")

        # --- sample routing logging (counts + gating hist) to TensorBoard ---
        try:
            if self.tb and info is not None:
                assigned = info.get("assigned", None)
                gating_logits = info.get("gating_logits", None)

                if assigned is not None:
                    try:
                        assigned_np = assigned.detach().cpu().numpy() if torch.is_tensor(assigned) else np.array(assigned)
                    except Exception:
                        assigned_np = np.array(assigned)

                    # counts per expert
                    for ex in (0, 1, 2):
                        cnt = int((assigned_np == ex).sum())
                        self.tb.add_scalar(f"expert_usage/batch_count_expert_{ex}", cnt, step)

                    counts = np.array([ (assigned_np == i).sum() for i in (0,1,2) ])
                    self.tb.add_histogram("expert_usage/counts", counts, step)

                if gating_logits is not None and torch.is_tensor(gating_logits):
                    logits_np = gating_logits.detach().cpu().numpy()
                    self.tb.add_histogram("routing/gating_logits", logits_np, step)

                    try:
                        weights_np = torch.nn.functional.softmax(gating_logits, dim=1).detach().cpu().numpy()
                        self.tb.add_histogram("routing/gating_weights", weights_np, step)
                        self.tb.add_scalar("routing/mean_weight_A", weights_np[:, 0].mean(), step)
                        self.tb.add_scalar("routing/mean_weight_B", weights_np[:, 1].mean(), step)
                        self.tb.add_scalar("routing/mean_weight_C", weights_np[:, 2].mean(), step)
                    except Exception:
                        pass
        except Exception:
            self._log.exception("Failed sample routing logging")

    # optional utility to log model graph (call once when you have an example batch)
    def log_model_graph(self, model: torch.nn.Module, example_input: torch.Tensor):
        if self.tb:
            try:
                self.tb.add_graph(model, example_input)
            except Exception:
                self._log.exception("Failed to write model graph to TB")
        if self.use_wandb and wandb is not None:
            try:
                # prefer trainer's logger-aware model unwrapping if needed before watching
                core_model = getattr(model, "module", model)
                # wandb.watch(core_model, log="all", log_freq=100)
            except Exception:
                self._log.exception("Failed to attach W&B watch to model")

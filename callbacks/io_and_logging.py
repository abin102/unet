import os
import csv
import torch  # <--- Added this
import torch.nn as nn
from typing import Optional

from utils.io import ensure_dir, save_state
from .model_logger import ModelLogger

from loguru import logger 

__all__ = ["Checkpoint", "CSVLogger", "ModelLogger"]

class Checkpoint:
    def __init__(self, out_dir, monitor="val/acc", mode="max"):
        self.out_dir = out_dir
        ensure_dir(out_dir)   # ensure dir exists
        self.monitor = monitor
        self.mult = 1 if mode == "max" else -1
        self.best = None
        logger.debug("Checkpoint initialized: out_dir={} monitor={} mode={}", out_dir, monitor, mode)

    def on_train_begin(self, tr):
        logger.debug("Checkpoint.on_train_begin called (no-op)")

    def on_epoch_end(self, tr, epoch, trl, tra, vall, vala):
        try:
            # Determine metric according to monitor.
            stats = getattr(tr, "latest_val_stats", {})
            monitor_key = self.monitor.replace("val/", "") 
            
            if monitor_key in stats:
                metric = stats[monitor_key]
            elif "acc" in self.monitor:
                metric = vala
            else:
                metric = -vall

            improved = self.best is None or self.mult * metric > self.mult * (self.best or -1e9)

            # --- FIX STARTS HERE: Handle DataParallel ---
            # If the model is wrapped in DataParallel, we must save the inner '.module'
            # to keep the keys clean (e.g. 'conv1.weight' instead of 'module.conv1.weight')
            if isinstance(tr.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                raw_model_state = tr.model.module.state_dict()
            else:
                raw_model_state = tr.model.state_dict()
            # --------------------------------------------

            state = {
                "epoch": epoch,
                "model": raw_model_state, # <--- Use the unwrapped state
                "opt": tr.optimizer.state_dict(),
                "sched": tr.scheduler.state_dict() if getattr(tr, "scheduler", None) else None,
            }

            if improved:
                self.best = metric
                best_path = os.path.join(self.out_dir, "best.ckpt")
                save_state(best_path, state)
                logger.info("New best checkpoint saved to {} (metric={:.4f})", best_path, metric)
            else:
                logger.debug("No improvement this epoch (metric={:.4f} best={:.4f})", metric, self.best if self.best else 0)

            last_path = os.path.join(self.out_dir, "last.ckpt")
            save_state(last_path, state)
        except Exception as e:
            logger.error("Error in Checkpoint.on_epoch_end: {}", e, exc_info=True)


class CSVLogger:
    # ... (Rest of the class remains exactly the same) ...
    def __init__(self, out_dir, fmt_acc=True, flush_to_disk=True):
        """
        out_dir: directory where metrics.csv is placed
        fmt_acc: format accuracies as percentage (True -> 97.123) ONLY for standard 'acc' fields.
                 Custom metrics are formatted based on their value.
        flush_to_disk: call os.fsync after flush to force write to disk
        """
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        self.path = os.path.join(out_dir, "metrics.csv")
        self.f = None
        self.w = None
        self.fmt_acc = bool(fmt_acc)
        self.flush_to_disk = bool(flush_to_disk)
        self.extra_metrics = [] # Will store the list from config

        logger.info("CSVLogger initialized. metrics will be written to: {}", self.path)

    def on_train_begin(self, tr):
        try:
            first_time = not os.path.exists(self.path)
            self.f = open(self.path, "a", newline="")
            self.w = csv.writer(self.f)

            # --- DYNAMIC METRIC LOADING ---
            self.extra_metrics = tr.cfg.get("metrics", [])
            
            if not self.extra_metrics:
                pass 

            if first_time:
                header = ["epoch", "train_loss", "train_acc(%)", "val_loss"]
                header += [m for m in self.extra_metrics]
                
                self.w.writerow(header)
                self._flush()
                logger.info("Created new metrics.csv with headers: {}", header)
            else:
                logger.info("Appending to existing metrics.csv at {}", self.path)
        except Exception as e:
            logger.error("Failed to open CSVLogger: {}", e)
            self.f = None

    def _format(self, x):
        if hasattr(x, "item"): x = x.item()
        if isinstance(x, float):
            return f"{x:.4f}"
        return x
        
    def _format_pct(self, x):
        if hasattr(x, "item"): x = x.item()
        if isinstance(x, float):
            if self.fmt_acc: return f"{x * 100:.3f}"
            return f"{x:.3f}"
        return x

    def _flush(self):
        if not self.f: return
        try:
            self.f.flush()
            if self.flush_to_disk: os.fsync(self.f.fileno())
        except Exception: pass

    def on_epoch_end(self, tr, epoch, trl, tra, vall, vala):
        if self.w is None: return

        try:
            row = [
                int(epoch),
                self._format(trl),
                self._format_pct(tra),
                self._format(vall),
            ]

            stats = getattr(tr, "latest_val_stats", {}) or {}
            
            for key in self.extra_metrics:
                val = stats.get(key)
                if val is not None:
                    row.append(self._format(val))
                else:
                    row.append("")

            self.w.writerow(row)
            self._flush()
        except Exception as e:
            logger.error("Failed to write metrics row: {}", e)

    def on_train_end(self, tr):
        try:
            if self.f: self.f.close()
        except Exception:
            pass
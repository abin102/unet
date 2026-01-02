import os
import csv
import torch
import torch.nn as nn
from loguru import logger  # <--- Using Loguru as requested

# Import the centralized saver to handle DDP unwrapping automatically
from utils.checkpoint_utils import save_checkpoint

__all__ = ["Checkpoint", "CSVLogger"]

class Checkpoint:
    def __init__(self, out_dir, monitor="val/acc", mode="max"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        self.monitor = monitor
        self.mult = 1 if mode == "max" else -1
        self.best = None
        logger.debug("Checkpoint initialized: out_dir={} monitor={} mode={}", out_dir, monitor, mode)

    def on_train_begin(self, tr):
        pass

    def on_epoch_end(self, tr, epoch, trl, tra, vall, vala):
        try:
            # 1. Determine Metric Value
            stats = getattr(tr, "latest_val_stats", {})
            monitor_key = self.monitor.replace("val/", "") 
            
            if monitor_key in stats:
                metric = stats[monitor_key]
            elif "acc" in self.monitor:
                metric = vala
            else:
                metric = -vall 

            # 2. Check for Improvement
            improved = (self.best is None) or (self.mult * metric > self.mult * self.best)

            if improved:
                self.best = metric
                # Save Best (Centralized saver handles DDP unwrapping)
                save_checkpoint(
                    path=os.path.join(self.out_dir, "best.ckpt"),
                    trainer=tr,
                    epoch=epoch,
                    is_best=True
                )
                logger.info("New best checkpoint saved! ({}={:.4f})", self.monitor, metric)
            else:
                logger.debug("No improvement. (Best: {:.4f}, Current: {:.4f})", self.best if self.best else 0, metric)

            # 3. Always save 'last.ckpt'
            save_checkpoint(
                path=os.path.join(self.out_dir, "last.ckpt"),
                trainer=tr,
                epoch=epoch,
                is_best=False
            )

        except Exception as e:
            logger.error("Error in Checkpoint.on_epoch_end: {}", e)


class CSVLogger:
    def __init__(self, out_dir, fmt_acc=True, flush_to_disk=True):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
            
        self.path = os.path.join(out_dir, "metrics.csv")
        self.f = None
        self.w = None
        self.fmt_acc = bool(fmt_acc)
        self.flush_to_disk = bool(flush_to_disk)
        self.extra_metrics = [] 

        logger.info("CSVLogger initialized. metrics will be written to: {}", self.path)

    def on_train_begin(self, tr):
        try:
            file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            
            self.f = open(self.path, "a", newline="")
            self.w = csv.writer(self.f)
            self.extra_metrics = tr.cfg.get("metrics", [])

            if not file_exists:
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
                row.append(self._format(val) if val is not None else "")

            self.w.writerow(row)
            self._flush()
        except Exception as e:
            logger.error("Failed to write metrics row: {}", e)

    def on_train_end(self, tr):
        try:
            if self.f: self.f.close()
        except Exception:
            pass
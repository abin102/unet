# callbacks/io_and_logging.py
import os
import csv
from typing import Optional

from utils.io import ensure_dir, save_state
from .model_logger import ModelLogger

from loguru import logger 
# optional: __all__ for clarity
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
            # determine metric according to monitor (simple heuristic)
            metric = vala if "acc" in self.monitor else -vall
            improved = self.best is None or self.mult * metric > self.mult * (self.best or -1e9)

            state = {
                "epoch": epoch,
                "model": tr.model.state_dict(),
                "opt": tr.optimizer.state_dict(),
                "sched": tr.scheduler.state_dict() if getattr(tr, "scheduler", None) else None,
            }

            if improved:
                self.best = metric
                best_path = os.path.join(self.out_dir, "best.ckpt")
                save_state(best_path, state)
                logger.info("New best checkpoint saved to {} (metric={})", best_path, metric)
            else:
                logger.debug("No improvement this epoch (metric={} best={})", metric, self.best)

            last_path = os.path.join(self.out_dir, "last.ckpt")
            save_state(last_path, state)
            logger.debug("Last checkpoint written to {} (epoch={})", last_path, epoch)
        except Exception as e:
            logger.error("Error in Checkpoint.on_epoch_end: {}", e, exc_info=True)



class CSVLogger:
    def __init__(self, out_dir, fmt_acc=True, flush_to_disk=True):
        """
        out_dir: directory where metrics.csv is placed
        fmt_acc: format accuracies as percentage with 3 decimals (True -> 97.123)
        flush_to_disk: call os.fsync after flush to force write to disk
        """
        self.out_dir = out_dir
        # ensure_dir(out_dir) # Ensure this util exists or use os.makedirs
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        self.path = os.path.join(out_dir, "metrics.csv")
        self.f = None
        self.w = None
        self.write_hmt = False
        self.fmt_acc = bool(fmt_acc)
        self.flush_to_disk = bool(flush_to_disk)

        logger.info("CSVLogger initialized. metrics will be written to: {}", self.path)

    def on_train_begin(self, tr):
        try:
            first_time = not os.path.exists(self.path)
            self.f = open(self.path, "a", newline="")
            self.w = csv.writer(self.f)

            self.write_hmt = bool(getattr(tr, "class_splits", None))

            if first_time:
                header = ["epoch", "train_loss", "train_acc(%)", "val_loss", "val_acc(%)"]
                if self.write_hmt:
                    header += ["val_head(%)", "val_medium(%)", "val_tail(%)"]
                
                # --- CHANGE 1: Always add these headers for new files ---
                # Even if the dataset doesn't have them, it's safer to have the columns 
                # ready so the CSV structure is consistent.
                header += ["hard_acc(%)", "medium_acc(%)", "easy_acc(%)"]
                # -------------------------------------------------------

                self.w.writerow(header)
                self._flush()
                logger.info("Created new metrics.csv and wrote header at {}", self.path)
            else:
                logger.info("Appending to existing metrics.csv at {}", self.path)
        except Exception as e:
            logger.error("Failed to open CSVLogger: {}", e)
            self.f = None

    def _format_loss(self, x):
        if isinstance(x, float): return f"{x:.3f}"
        return x

    def _format_acc(self, x):
        if isinstance(x, float):
            if self.fmt_acc: return f"{x * 100:.3f}"
            return f"{x:.3f}"
        return x

    def _flush(self):
        if not self.f: return
        try:
            self.f.flush()
            if self.flush_to_disk:
                os.fsync(self.f.fileno())
        except Exception:
            pass

    def on_epoch_end(self, tr, epoch, trl, tra, vall, vala):
        if self.w is None: return

        try:
            # 1. Standard Metrics
            row = [
                int(epoch),
                self._format_loss(trl),
                self._format_acc(tra),
                self._format_loss(vall),
                self._format_acc(vala),
            ]

            # 2. Existing Head/Medium/Tail Logic
            stats = getattr(tr, "latest_val_stats", {}) or {}
            if self.write_hmt:
                row += [
                    self._format_acc(stats.get("head_acc", "")),
                    self._format_acc(stats.get("medium_acc", "")),
                    self._format_acc(stats.get("tail_acc", "")),
                ]

            # --- CHANGE 2: Safe Bucket Logging ---
            # We use stats.get(). If the key is missing (old dataset), it returns None.
            # If None, we write an empty string "" so the CSV stays clean.
            h = stats.get("hard_acc")
            m = stats.get("medium_acc")
            e = stats.get("easy_acc")

            row.append(self._format_acc(h) if h is not None else "")
            row.append(self._format_acc(m) if m is not None else "")
            row.append(self._format_acc(e) if e is not None else "")
            # -------------------------------------

            self.w.writerow(row)
            self._flush()
        except Exception as e:
            logger.error("Failed to write metrics row: {}", e)

    def on_train_end(self, tr):
        try:
            if self.f: self.f.close()
        except Exception:
            pass
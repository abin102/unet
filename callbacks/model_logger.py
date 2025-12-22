# callbacks/model_logger.py
import os
import traceback
from typing import Optional
from loguru import logger   


class ModelLogger:
    """
    Callback to summarize & save model structure at training start.

    Usage:
        cbs = [ CSVLogger(out_dir), Checkpoint(out_dir), ModelLogger(out_dir) ]
    """

    def __init__(self, out_dir: str, input_size=(1, 3, 64, 64), save_trace: bool = False, enabled: bool = True):
        """
        out_dir: where to write model_structure.txt / model_experts.json
        input_size: dummy input shape used for probing (B, C, H, W)
        save_trace: whether to attempt torch.jit.trace (may fail)
        enabled: if False, callback is a no-op
        """
        self.out_dir = out_dir
        self.input_size = tuple(input_size)
        self.save_trace = bool(save_trace)
        self.enabled = bool(enabled)
        logger.debug("ModelLogger initialized: out_dir={} input_size={} save_trace={} enabled={}",
                     self.out_dir, self.input_size, self.save_trace, self.enabled)

    def _is_main_process(self, tr) -> bool:
        """
        Detect main process / rank 0. Looks for tr.rank or environment variables as fallback.
        """
        # prefer trainer-provided rank if present
        if tr is not None and hasattr(tr, "rank"):
            try:
                is_main = int(getattr(tr, "rank")) == 0
                logger.debug("Detected trainer.rank={} -> is_main={}", getattr(tr, "rank"), is_main)
                return is_main
            except Exception as e:
                logger.debug("Could not parse trainer.rank: {}", e)

        # fallbacks: common env vars in multi-node launchers
        for k in ("RANK", "LOCAL_RANK", "MPI_RANK"):
            v = os.environ.get(k)
            if v is not None:
                try:
                    is_main = int(v) == 0
                    logger.debug("Env var {}={} -> is_main={}", k, v, is_main)
                    return is_main
                except Exception as e:
                    logger.debug("Could not parse env var {}: {}", k, e)

        # default to True when uncertain (single-process)
        logger.debug("Could not determine distributed rank; defaulting to main process (True).")
        return True

    def on_train_begin(self, tr):
        if not self.enabled:
            logger.info("ModelLogger is disabled; skipping model summary.")
            return
        if not self._is_main_process(tr):
            logger.info("Not main process; ModelLogger will not run on this rank.")
            return

        out_dir = getattr(tr, "out_dir", None) or self.out_dir
        logger.info("ModelLogger running. Output directory: {}", out_dir)

        # Lazy import to avoid import-time cycles / missing symbols
        try:
            from utils.model_summary import summarize_and_save_model
        except Exception as e:
            logger.warning("Could not import summarize_and_save_model: {}", e, exc_info=True)
            return

        model_to_inspect = tr.model if hasattr(tr, "model") else getattr(tr, "net", None)
        if model_to_inspect is None:
            logger.warning("Trainer does not expose a model (tr.model or tr.net). Skipping model summary.")
            return

        try:
            logger.debug("Calling summarize_and_save_model with input_size={} save_trace={}",
                         self.input_size, self.save_trace)
            res = summarize_and_save_model(
                model_to_inspect,
                out_dir,
                cfg=getattr(tr, "cfg", None),
                input_size=self.input_size,
                save_trace=self.save_trace
            )
            logger.info("Model summary saved. Paths: {}", res)

            # Optionally attach summary locations to trainer for other callbacks to use
            try:
                tr.model_summary_paths = res
                logger.debug("Attached model_summary_paths to trainer.")
            except Exception as e:
                logger.debug("Failed to attach model_summary_paths to trainer: {}", e)
        except Exception as e:
            # don't crash training if summarization fails
            logger.warning("summarize_and_save_model failed: {}", e, exc_info=True)
            logger.debug("Traceback:\n{}", traceback.format_exc())

    def on_epoch_end(self, tr, epoch, trl, tra, vall, vala):
        # ModelLogger does not perform per-epoch actions, but log at debug level for traceability
        logger.debug("ModelLogger.on_epoch_end called (no-op). epoch={}", epoch)
        return

    def on_train_end(self, tr):
        logger.debug("ModelLogger.on_train_end called (no-op).")
        return

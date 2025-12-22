# utils/logging_utils.py
import os
import sys
import logging
from typing import Optional, Tuple
from pathlib import Path

from loguru import logger as _logger




# --- Robust project-root detection and path helpers ---
def _find_project_root(start_path: Optional[str] = None) -> Path:
    """
    Attempt to locate the project root by searching upward for common markers.
    Falls back to cwd if nothing found.
    """
    start = Path(start_path or Path(__file__).resolve()).resolve()
    # include current path and parents
    candidates = [start] + list(start.parents)
    markers = (".git", "pyproject.toml", "setup.py", "requirements.txt")
    for parent in candidates:
        try:
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        except Exception:
            continue
    # fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def _relpath(path: str) -> str:
    """
    Try to return a path relative to the repository/project root.
    If that fails, return a compact fallback like 'dir/file.py' or filename.
    Never raise.
    """
    try:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(PROJECT_ROOT)
            parts = rel.parts
            if len(parts) <= 3:
                return str(rel)
            # keep only last 3 parts if very deep
            return os.path.join(*parts[-3:])
        except Exception:
            parts = p.parts
            if len(parts) >= 2:
                return os.path.join(parts[-2], parts[-1])
            return p.name
    except Exception:
        try:
            return Path(path).name
        except Exception:
            return "<unknown>"


# Interceptor to route stdlib logging records into loguru
class InterceptHandler(logging.Handler):
    """
    Logging handler that forwards stdlib logging to loguru.
    Subclassing logging.Handler is required for stdlib compatibility.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        level = record.levelname if hasattr(record, "levelname") else "INFO"
        # depth controls caller depth shown by loguru; adjust if needed
        _logger.opt(exception=record.exc_info, depth=6).log(level, message)


def init_logger_and_tb(
    debug: bool,
    debug_dir: str,
    tb_logdir: Optional[str] = None,
    log_level: str = "INFO",
    rotation: str = "50 MB",
    retention: str = "14 days",
    compression: Optional[str] = "zip",
) -> Tuple[object, Optional[object]]:
    """
    Initialize loguru logger and optionally a TensorBoard SummaryWriter.

    Args:
        debug: enable backtrace/diagnose (richer tracebacks)
        debug_dir: per-run directory where run.log will be created
        tb_logdir: if provided and SummaryWriter exists, create TB writer (else None)
        log_level: minimum level (DEBUG/INFO/...)
        rotation/retention/compression: file-sink params for loguru

    Returns:
        (loguru_logger, tb_writer_or_None)
    """
    # ensure output directory exists
    os.makedirs(debug_dir, exist_ok=True)
    log_file = os.path.join(debug_dir, "run.log")

    # --- remove existing sinks to make init idempotent ---
    try:
        _logger.remove()
    except Exception:
        pass

    # --- colored-level customization (nice-to-have) ---
    try:
        _logger.level("TRACE", color="<cyan>")
        _logger.level("DEBUG", color="<blue>")
        _logger.level("INFO",  color="<white>")
        _logger.level("SUCCESS", color="<green>")
        _logger.level("WARNING", color="<yellow>")
        _logger.level("ERROR", color="<red>")
        _logger.level("CRITICAL", color="<red><b>")
    except Exception:
        pass

    # --- Console sink (concise + aligned) ---
    console_fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <7}</level> | "
        "<cyan>{file.name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    _logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        backtrace=debug,
        diagnose=debug,
        format=console_fmt,
    )

    # --- File sink (detailed, relative paths) ---
    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | "
        "{extra[path]}:{function}:{line} - {message}"
    )

    def _inject_relpath(record: dict) -> bool:
        """
        Loguru filter: adds record['extra']['path'] containing a readable path string.
        Always returns True so logging continues. Defensive and never raises.
        """
        try:
            f = record.get("file")
            fpath = None

            # --- START: MODIFIED SECTION ---
            # Check for the 'path' attribute on the file object first
            if hasattr(f, "path"):
                fpath = f.path
            # Fallback for dict-like access (as in original code)
            elif isinstance(f, dict):
                fpath = f.get("path")
            # --- END: MODIFIED SECTION ---
                
            if not fpath:
                # no file info; attempt to use existing extra.path or set unknown
                extra = record.get("extra", {})
                existing = extra.get("path") if isinstance(extra, dict) else None
                record.setdefault("extra", {})["path"] = existing or "<unknown>"
                return True

            record.setdefault("extra", {})["path"] = _relpath(fpath)
        except Exception:
            try:
                record.setdefault("extra", {})["path"] = "<unknown>"
            except Exception:
                pass
        return True
    

    _logger.add(
        log_file,
        level=log_level,
        enqueue=True,
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=debug,
        diagnose=debug,
        format=file_fmt,
        filter=_inject_relpath,
    )

    # --- Intercept stdlib logging (so libraries using logging get routed to loguru) ---
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)
    handler = InterceptHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.handlers = [handler]

    # --- Optional TensorBoard writer (guarded) ---
    tb_writer = None
    if tb_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            SummaryWriter = None

        try:
            os.makedirs(tb_logdir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_logdir)
            _logger.info("TensorBoard enabled at {}", tb_logdir)
        except Exception:
            _logger.exception("Failed to create TensorBoard writer")

    _logger.info("Loguru initialized. Log file: {}", log_file)
    return _logger, tb_writer

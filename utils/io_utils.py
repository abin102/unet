# utils/io_utils.py
import os
import torch
from typing import Optional

def save_crash_debug(debug_dir: Optional[str], tag: str, **data):
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"crash_{tag}.pth")
        serial = {}
        for k,v in data.items():
            try:
                serial[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v
            except Exception:
                serial[k] = str(v)
        torch.save(serial, path)
    except Exception:
        try:
            import logging
            logging.getLogger("Trainer").exception("Failed to save crash debug")
        except Exception:
            pass

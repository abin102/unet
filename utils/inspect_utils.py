# utils/inspect_utils.py
import torch
def tensor_stats(x: torch.Tensor):
    out = {"shape": tuple(x.shape),
           "has_nan": bool(torch.isnan(x).any().item()) if x.is_floating_point() else False,
           "has_inf": bool(torch.isinf(x).any().item()) if x.is_floating_point() else False}
    try:
        out["min"] = float(torch.min(x).cpu())
        out["max"] = float(torch.max(x).cpu())
    except Exception:
        out["min"], out["max"] = None, None
    if x.is_floating_point():
        out["mean"] = float(torch.mean(x).cpu()) if x.numel() else None
    else:
        out["mean"] = None
    if not x.is_floating_point() and x.numel():
        try:
            vals, counts = torch.unique(x, return_counts=True)
            out["unique"] = {int(v.cpu()): int(c.cpu()) for v, c in zip(vals, counts)}
        except Exception:
            out["unique"] = None
    return out

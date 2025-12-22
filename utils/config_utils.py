# utils/config_utils.py
import yaml
from typing import List, Tuple, Any, Dict

def load_cfg(path: str, overrides: List[str] = None) -> Dict:
    """Load YAML and apply CLI overrides like key=value and dotted keys."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for kv in overrides:
            k, v = kv.split("=", 1)
            try:
                v_eval = eval(v)
            except Exception:
                v_eval = v
            cur = cfg
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v_eval
    cfg = _coerce_numbers(cfg)
    return cfg

def _coerce_numbers(obj):
    """Same coercion you had: convert string numbers, tuples, etc."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("(") and s.endswith(")"):
            try:
                a, b = [float(x) for x in s[1:-1].split(",")]
                return (a, b)
            except Exception:
                return obj
        if s.isdigit():
            return int(s)
        try:
            return float(s)
        except ValueError:
            return obj
    return obj

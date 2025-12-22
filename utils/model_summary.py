# utils/model_summary.py
"""
Model summarizer used by callbacks.model_logger.

Exports:
  - summarize_and_save_model(model, out_dir, cfg=None, input_size=(1,3,64,64), save_trace=False)
  - probe_feature_shapes(...)
"""

import json
import pathlib
import traceback
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn
import yaml


def _gather_expert_info(model: nn.Module):
    expert_list = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # heuristic: look for attributes we set on Expert (in_ch/out_dim/etc) or name includes 'expert'
        if any(hasattr(module, a) for a in ("in_ch", "out_dim", "depth", "wide", "large_kernel")) or "expert" in name.lower():
            ex_info = {"name": name, "class": module.__class__.__name__}
            for attr in ("in_ch", "out_dim", "depth", "wide", "large_kernel"):
                if hasattr(module, attr):
                    try:
                        ex_info[attr] = getattr(module, attr)
                    except Exception:
                        ex_info[attr] = None
            ex_info["param_count"] = sum(p.numel() for p in module.parameters())
            ex_info["trainable_param_count"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
            expert_list.append(ex_info)
    return expert_list


def _model_children_list(model: nn.Module):
    return [(name, module.__class__.__name__) for name, module in model.named_children()]


def _param_counts(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def probe_feature_shapes(model: nn.Module, input_size: Tuple[int, int, int, int],
                         probe_fn_name: str = "_extract_backbone_features",
                         device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Attempt to run model._extract_backbone_features(dummy) to get feature shape(s).
    Returns dict: either feature_shape or an error message.
    """
    result = {"success": False}
    try:
        device = torch.device("cpu") if device is None else device
        model_cpu = model.to(device).eval()
        dummy = torch.randn(*input_size, device=device)
        with torch.no_grad():
            if hasattr(model_cpu, probe_fn_name):
                feats = getattr(model_cpu, probe_fn_name)(dummy)
                result["success"] = True
                result["probe_fn"] = probe_fn_name
                result["feature_shape"] = tuple(feats.shape)
            else:
                # fallback: try a forward pass and capture the shape of the returned logits if Tensor
                out = model_cpu(dummy)
                if isinstance(out, torch.Tensor):
                    result["success"] = True
                    result["forward_out_shape"] = tuple(out.shape)
                elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
                    result["success"] = True
                    result["forward_out_shape"] = tuple(out[0].shape)
                else:
                    result["error"] = "probe function not found and forward() did not return tensors to infer shapes"
    except Exception as e:
        result["error"] = f"probe failed: {e}"
        result["traceback"] = traceback.format_exc()
    return result


def summarize_and_save_model(model: nn.Module, out_dir: str, cfg: Optional[dict] = None,
                             input_size: Tuple[int, int, int, int] = (1, 3, 64, 64),
                             save_trace: bool = False) -> Dict[str, Any]:
    """
    Produce a compact text summary printed to stdout and write text + json under out_dir.
    THIS VERSION safely moves the model to CPU for probing/tracing and restores original device/mode.
    Returns dict with paths.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Remember original device and training mode so we can restore them
    try:
        # If model has parameters, get device from first param; else default to cpu
        params = list(model.parameters())
        orig_device = params[0].device if params else torch.device("cpu")
    except Exception:
        orig_device = torch.device("cpu")

    orig_training = model.training

    # We'll perform probing/tracing on CPU to avoid GPU memory and speed issues.
    # Move model to CPU (non-inplace semantics are not available; .to mutates module)
    try:
        model_cpu = model.to(torch.device("cpu"))
    except Exception:
        # fallback: try model.cpu()
        model_cpu = model.cpu()

    # switch to eval mode for deterministic probe/tracing
    model_cpu_eval = model_cpu.eval()

    # Build summary text (this does NOT require model on a specific device)
    lines = []
    lines.append("=== Model summary (compact) ===")
    lines.append(f"Model class: {model.__class__.__name__}")

    # top-level children
    for name, clsname in _model_children_list(model):
        lines.append(f"  {name}: {clsname}")

    total_params, trainable_params = _param_counts(model)
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")

    expert_list = _gather_expert_info(model)
    if expert_list:
        lines.append("--- Experts found ---")
        for ex in expert_list:
            lines.append(f"  {ex['name']}: class={ex['class']} in_ch={ex.get('in_ch')} out_dim={ex.get('out_dim')} depth={ex.get('depth')} params={ex['param_count']:,}")
    else:
        lines.append("No experts detected by heuristic (looking for attributes in_ch/out_dim or name containing 'expert').")

    summary_text = "\n".join(lines)
    # print(summary_text)

    # save text + json
    txt_path = out_path / "model_structure.txt"
    json_path = out_path / "model_experts.json"
    txt_path.write_text(summary_text)
    try:
        json.dump(expert_list, json_path.open("w"), indent=2)
    except Exception:
        json_path.write_text(json.dumps([str(x) for x in expert_list], indent=2))

    # probe feature shapes (run on the CPU-copy, not the original model)
    probe = probe_feature_shapes(model_cpu_eval, input_size=input_size)
    probe_path = out_path / "model_probe.json"
    try:
        json.dump(probe, probe_path.open("w"), indent=2)
    except Exception:
        probe_path.write_text(str(probe))

    # save config copy if provided
    try:
        if cfg is not None:
            cfg_path = out_path / "config_used.yaml"
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"[model_summary] Warning: failed to save config copy: {e}")

    # optional: try to trace (best-effort) -- operate on the CPU eval copy
    trace_path = None
    if save_trace:
        trace_path = out_path / "model_trace.pt"
        try:
            dummy = torch.randn(*input_size)
            with torch.no_grad():
                traced = torch.jit.trace(model_cpu_eval, dummy, strict=False)
            traced.save(str(trace_path))
            print(f"[model_summary] Saved torch.jit trace to {trace_path}")
        except Exception as e:
            print(f"[model_summary] trace failed: {e}")
            trace_path = None

    # Restore original device and training/eval mode on the original model object.
    # Note: model.to(orig_device) mutates the module back to original device
    try:
        model.to(orig_device)
    except Exception:
        # if that fails, attempt .cuda() when orig_device is CUDA
        try:
            if str(orig_device).startswith("cuda"):
                model.cuda()
            else:
                model.cpu()
        except Exception:
            pass

    # restore training/eval mode
    try:
        if orig_training:
            model.train()
        else:
            model.eval()
    except Exception:
        pass

    return {
        "summary_txt": str(txt_path),
        "experts_json": str(json_path),
        "probe_json": str(probe_path),
        "trace_path": str(trace_path) if trace_path is not None else None
    }

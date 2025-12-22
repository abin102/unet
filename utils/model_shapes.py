# utils/model_shapes.py
import torch

def record_shapes(model, sample, device, out_path, amp=True):
    """
    Runs a single real sample through the model and writes input/output
    shapes of every leaf module to `out_path`. Handles tuple outputs (e.g., ResLT).
    """
    lines = []
    hooks = []

    def is_leaf(m):
        return len(list(m.children())) == 0

    def hook(name):
        def _h(module, inp, out):
            in_shape = tuple(inp[0].shape) if isinstance(inp, (list, tuple)) else tuple(inp.shape)
            if isinstance(out, (list, tuple)):
                out_shape = [tuple(o.shape) for o in out]
            else:
                out_shape = tuple(out.shape)
            lines.append(f"{name:<40} in: {in_shape} -> out: {out_shape}")
        return _h

    # register hooks on leaf modules with qualified names
    for name, m in model.named_modules():
        if is_leaf(m):
            hooks.append(m.register_forward_hook(hook(name or m.__class__.__name__)))

    model_was_training = model.training
    model.eval()

    sample = sample.to(device, non_blocking=True)
    ctx = (lambda: torch.amp.autocast("cuda")) if (amp and torch.cuda.is_available()) else torch.no_grad
    with torch.no_grad():
        with (torch.amp.autocast("cuda") if (amp and torch.cuda.is_available()) else torch.no_grad()):
            out = model(sample)

    # also add a summary line for the model output
    if isinstance(out, tuple):
        out_shapes = [tuple(o.shape) for o in out]
    else:
        out_shapes = tuple(out.shape)
    lines.append(f"{'MODEL_OUTPUT':<40} -> {out_shapes}")

    for h in hooks:
        h.remove()
    if model_was_training:
        model.train()

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

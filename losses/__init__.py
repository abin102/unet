# losses/__init__.py  (relevant snippets)

# losses/__init__.py
import torch.nn as nn
from .alr import ALRLoss
from .regularised_alr import RegularisedALRLoss   # <- important: import the module so registration builder can reference it

_LOSSES = {}
REGISTRY = _LOSSES

def register(name):
    def decorator(fn):
        _LOSSES[name] = fn
        return fn
    return decorator

def get_loss(name, **kwargs):
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(_LOSSES.keys())}")
    return _LOSSES[name](**kwargs)

@register("ce")
def build_ce(**kwargs):
    return nn.CrossEntropyLoss()

@register("alr")
def build_alr_loss(class_weights=None, eps=1e-8, reduction="mean", **_):
    if class_weights is not None:
        import torch
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return ALRLoss(class_weights=class_weights, eps=eps, reduction=reduction)

@register("regularised_alr")
def build_regularised_alr_loss(class_weights=None, eps=1e-8, reduction="mean", alr_scale=1.0, lambda_ent=0.0, **_):
    if class_weights is not None:
        import torch
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return RegularisedALRLoss(class_weights=class_weights, eps=eps, reduction=reduction, alr_scale=alr_scale, lambda_ent=lambda_ent)

# ---- import ResLT implementation ----
from .reslt_loss import ResLTLoss

@register("reslt_loss")
def build_reslt_loss(num_classes, head_classes, medium_classes, tail_classes, beta=0.5, **kwargs):
    return ResLTLoss(
        num_classes=num_classes,
        head_classes=head_classes,
        medium_classes=medium_classes,
        tail_classes=tail_classes,
        beta=beta
    )

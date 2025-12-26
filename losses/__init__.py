import torch.nn as nn
from losses.segmentation import CEDiceLoss

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
    return nn.CrossEntropyLoss(**kwargs)

@register("ce_dice")
def build_ce_dice(**kwargs):
    return CEDiceLoss(**kwargs)

# models/__init__.py

from .unet import build_unet
from .reparm_unet import build_reparam_unet

REGISTRY = {
    "unet": build_unet,
    "reparam_unet": build_reparam_unet,
}

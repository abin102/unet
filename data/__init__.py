# ensure project root is on sys.path so "from data import REGISTRY" works
# ensure project root is importable so "from data import REGISTRY" works
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # one level up from scripts/
sys.path.insert(0, str(PROJECT_ROOT))
import random
from loguru import logger

from typing import Iterable, Optional, List
import torchvision as tv
import os
from torch.utils.data import Subset


REGISTRY = {}

def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco
def _make_transforms(image_size=224,
                     to_rgb=False,
                     imagenet_norm=True,
                     mean=None,
                     std=None,
                     train=True):
    """
    Build torchvision transforms with optional ImageNet or custom normalization.
    Prints which normalization is applied for transparency.
    """
    t = []
    if to_rgb:
        t.append(tv.transforms.Grayscale(num_output_channels=3))

    t.append(tv.transforms.Resize(image_size))

    if train:
        t.extend([
            tv.transforms.RandomCrop(image_size, padding=4),
            tv.transforms.RandomHorizontalFlip()
        ])

    t.append(tv.transforms.ToTensor())

    # normalization logic with info print
    if imagenet_norm:
        print(f"[INFO] Applying ImageNet normalization (mean={ [0.485, 0.456, 0.406] }, std={ [0.229, 0.224, 0.225] })")
        t.append(tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]))
    elif mean is not None and std is not None:
        print(f"[INFO] Applying custom normalization (mean={mean}, std={std})")
        t.append(tv.transforms.Normalize(mean=mean, std=std))
    else:
        print(f"[INFO] No normalization applied (imagenet_norm=False and no custom mean/std provided).")

    return tv.transforms.Compose(t)



# --- ADD THIS LINE AT THE BOTTOM ---
from data.ct_dataset import build_ct_dataset 
# This triggers the code in ct_dataset.py that does: REGISTRY["ct_segmentation"] = ...
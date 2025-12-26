import sys
sys.path.append("..")  # add project root

import torchvision as tv
from torch import nn
import torch.nn.functional as F
import torch

REGISTRY = {}

def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

from models.unet import build_unet 
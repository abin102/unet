
# ensures project root (one level up from /scripts) is on sys.path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # one level up from scripts/
sys.path.insert(0, str(PROJECT_ROOT))

import torch, torchvision as tv
src_model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
src_state = src_model.state_dict()
from models import REGISTRY as M
tgt = M['resnet50_torch'](num_classes=10, pretrained=False, cifar_stem=True)
tgt_state = tgt.state_dict()

skipped = []
adapted = []
for k in src_state:
    if k in tgt_state:
        if src_state[k].shape != tgt_state[k].shape:
            adapted.append((k, src_state[k].shape, tgt_state[k].shape))
    else:
        skipped.append(k)
print("adapted examples:", adapted[:10])
print("skipped (src-only) examples:", skipped[:10])
# total params
def numel(state): return sum(v.numel() for v in state.values())
print("src total params:", numel(src_state))
print("tgt total params:", numel(tgt_state))

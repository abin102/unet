# models/mobilenetv2_3bn.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2_3BN(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        base = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1280, num_classes)

        self.BNH = nn.BatchNorm2d(1280)
        self.BNM = nn.BatchNorm2d(1280)
        self.BNT = nn.BatchNorm2d(1280)

    def forward(self, x):
        out = self.features(x)
        h = self.BNH(out)
        m = self.BNM(out)
        t = self.BNT(out)
        fs = torch.cat((h, m, t), dim=0)
        pooled = self.avgpool(fs).flatten(1)
        logits = self.linear(pooled)
        c = logits.size(0) // 3
        return logits[:c], logits[c:2*c], logits[2*c:]



# models/small_cifar.py
import torch
from torch import nn
import torch.nn.functional as F

class SmallCIFARNet(nn.Module):
    """
    Tiny CNN baseline for CIFAR-10/100.
    Use num_classes=100 for CIFAR-100, or 10 for CIFAR-10.
    """
    def __init__(self, num_classes: int = 100, drop_prob: float = 0.25):
        super().__init__()
        # conv stage 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # conv stage 2
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # conv stage 3
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)                       # 32 -> 16
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                       # 16 -> 8
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)                       # 8 -> 4
        x = self.avgpool(x)                    # -> [B, 256, 1, 1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)                    # raw logits -> use with CrossEntropyLoss
        return logits

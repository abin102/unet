import torch.nn as nn
from .resnet_cifar import ResNet_Cifar


class ResNetClassifier(nn.Module):
    """
    Wraps ResNet_Cifar(from the repo of ResLT paper) with a classification head.
    Feature map -> GAP -> (Dropout) -> Linear -> logits
    """
    def __init__(
        self,
        block,                   # e.g., BasicBlock or Bottleneck (must define .expansion)
        num_blocks,              # e.g., [5, 5, 5]
        num_classes=10,
        scale=1,
        groups=1,
        nc=[16, 32, 64],
        drop=0.0                 # optional dropout before the linear head
    ):
        super().__init__()
        # Build the backbone (your existing feature extractor)
        self.backbone = ResNet_Cifar(
            block=block,
            num_blocks=num_blocks,
            num_classes=num_classes,
            scale=scale,
            groups=groups,
            nc=nc
        )
        # Feature dim at the output of the backbone is nc[2] * scale * block.expansion
        feat_dim = (nc[2] * scale) * getattr(block, "expansion", 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop) if drop and drop > 0 else nn.Identity()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feats=False):
        """
        Returns logits of shape (N, num_classes).
        If return_feats=True, also returns the pooled features of shape (N, feat_dim).
        """
        feats_map = self.backbone(x)                    # (N, C, 8, 8) for CIFAR32
        pooled = self.pool(feats_map).flatten(1)        # (N, C)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)                        # (N, num_classes)
        return (logits, pooled) if return_feats else logits

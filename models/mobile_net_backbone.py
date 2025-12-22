# models/mobilenetv2_backbone.py
from typing import Any, Tuple
import torch
from torch import nn
import torchvision as tv
import torch.nn.functional as F

def _load_mobilenetv2(pretrained: bool):
    try:
        weights = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.models.mobilenet_v2(weights=weights)
    except Exception:
        try:
            model = tv.models.mobilenet_v2(pretrained=pretrained)
        except Exception:
            model = tv.models.mobilenet_v2(weights=None)
    return model

class MobileNetV2AsResNetLike(nn.Module):
    """
    Wrap MobileNetV2 and expose a ResNet-like interface that PlugAndPlayExperts expects:
      - conv1, bn1, relu, (optional maxpool)
      - layer1, layer2, layer3
    We implement these by slicing torchvision MobileNetV2.features (a Sequential).
    This keeps downstream code unchanged while allowing MobileNet to be probed like a ResNet.
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 3, cifar_stem: bool = False, **_):
        super().__init__()
        base = _load_mobilenetv2(pretrained=pretrained)

        # optionally adapt first conv channels
        old_conv = base.features[0][0]
        if in_channels != old_conv.in_channels:
            base.features[0][0] = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            )

        # optionally change stride on stem for CIFAR
        if cifar_stem:
            try:
                conv0 = base.features[0][0]
                base.features[0][0] = nn.Conv2d(
                    conv0.in_channels,
                    conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=(1,1),
                    padding=conv0.padding,
                    bias=(conv0.bias is not None)
                )
            except Exception:
                pass

        self._base = base

        # Build ResNet-like attributes by grouping MobileNet features into 4 blocks:
        # conv1 / bn1 / relu  -> around features[0] (conv+bn+relu)
        # then split features into layer1, layer2, layer3 groups.
        # NOTE: the exact split below is a pragmatic mapping that works well for standard torchvision mobilenet_v2.
        feats = list(self._base.features.children())

        # conv1 block (features[0] is usually [ConvBNReLU])
        stem_block = feats[0]
        if isinstance(stem_block, nn.Sequential) and len(stem_block) == 3:
            self.conv1 = stem_block[0]  # The Conv2d
            self.bn1 = stem_block[1]    # The BatchNorm2d
            self.relu = stem_block[2]   # The ReLU6
        else:
            # Fallback if the structure is not what we expect
            self.conv1 = stem_block
            self.bn1 = nn.Identity()
            self.relu = nn.ReLU(inplace=True)
            print("WARNING: MobileNetV2AsResNetLike wrapper could not find Conv/BN/ReLU stem.")        # Now create "layer" groups as contiguous slices of features
       
       
        # Heuristic split: skip the first item (stem), then split the remaining into 3 groups
        remaining = feats[1:]
        L = len(remaining)
        # split roughly into thirds (fine for typical mobilenet depth)
        i1 = max(1, L // 3)
        i2 = max(i1 + 1, 2 * L // 3)

        layer1_seq = nn.Sequential(*remaining[:i1])
        layer2_seq = nn.Sequential(*remaining[i1:i2])
        layer3_seq = nn.Sequential(*remaining[i2:])

        self.layer1 = layer1_seq
        self.layer2 = layer2_seq
        self.layer3 = layer3_seq

        # Note: MobileNet does not have maxpool attribute; define optional identity so plug-and-play checks work
        self.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If someone calls the backbone as a classifier, preserve that behavior:
        # But for plug_and_play, they call conv1->bn1->relu->layerN sequence via wrapper, not this direct forward.
        # We provide a default forward that returns the spatial features after full features stack.
        return self._base.features(x)

# model/backbone.py
"""
DAIIC-ready backbone + attention wrappers.

Provides:
- simple BackboneWrapper for standard use (forward_features + forward)
- daiic_resnet34 / daiic_resnet50 constructors that return modules implementing:
    feature extractor (pretrained convs),
    misclassification-costs attention (W),
    discriminative feature attention (F' via SE-style gating),
    fusion (concat/add) and classifier head.

The returned module's API:
- forward_features(x) -> Tensor [B, C, H, W]
- forward(x) -> dict {
      'logits': Tensor[B, K],
      'probs': Tensor[B, K] (sigmoid),
      'W': Tensor[B, K],
      'F': Tensor[B, C, H, W],
      'Fprime': Tensor[B, C, H, W]
  }
"""
from typing import Callable, Optional
import torch
from torch import nn
import torchvision as tv

# import register from your package (model/__init__.py defines it)
from . import register


# --------------------------
# small utility modules
# --------------------------
class Identity(nn.Module):
    def forward(self, x):
        return x


class SEChannelGate(nn.Module):
    """Squeeze-and-Excitation style channel attention (discriminative feature attention)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, F):
        # F: [B, C, H, W]
        B, C, H, W = F.shape
        s = F.mean(dim=(2, 3))                    # [B, C]
        s = self.act(self.fc1(s))                # [B, hidden]
        s = self.sig(self.fc2(s))                # [B, C]
        return s.view(B, C, 1, 1)                # broadcastable gating


class CostAttention(nn.Module):
    """Misclassification-costs attention network.
       Implementation: 1x1 conv -> GAP -> softmax over classes -> W (per sample).
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        # We will apply GAP then softmax across classes in forward

    def forward(self, F):
        # F: [B, C, H, W]
        x = self.conv1x1(F)            # [B, K, H, W]
        s = x.mean(dim=(2, 3))         # [B, K]
        W = torch.softmax(s/5, dim=1)    # normalize across classes
        return W                       # [B, K]


# --------------------------
# Backbone wrapper
# --------------------------
class BackboneWrapper(nn.Module):
    """
    Simple wrapper that exposes:
      - forward_features(x) -> [B, C, H, W]
      - forward(x) -> logits [B, num_classes]  (GAP -> classifier)
      - .classifier and .classifier_in_features
    """
    def __init__(self, feature_extractor: Callable, classifier: nn.Module, classifier_in_features: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classifier_in_features = classifier_in_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.feature_extractor, nn.Module):
            return self.feature_extractor(x)
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        pooled = nn.functional.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)
        logits = self.classifier(pooled)
        return logits


# --------------------------
# helper to build resnet feature extractor (pre-pool)
# --------------------------
def _make_resnet_feature_extractor(resnet_model: nn.Module, cifar_stem: bool = False, in_channels: int = 3):
    """
    Build a feature extractor up to layer4 for torchvision ResNets.
    Optionally apply CIFAR stem (3x3 conv stride1 + remove maxpool) to preserve spatial res.
    """
    if cifar_stem:
        # replace conv1 and maxpool for CIFAR-style stem
        old = resnet_model.conv1
        resnet_model.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        resnet_model.maxpool = Identity()

    # assemble sequential extractor (same order as torchvision forward up to layer4)
    layers = [
        resnet_model.conv1,
        resnet_model.bn1,
        resnet_model.relu,
        resnet_model.maxpool,
        resnet_model.layer1,
        resnet_model.layer2,
        resnet_model.layer3,
        resnet_model.layer4,
    ]
    return nn.Sequential(*layers)


# --------------------------
# DAIIC composite module
# --------------------------
class DAIICModule(nn.Module):
    """
    Compose: feature_extractor -> cost_attention (W) & discrim_attention (gating) -> fusion -> classifier
    forward(x) returns a dict: logits, probs, W, F, Fprime
    """
    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier_in_features: int,
                 num_classes: int,
                 fusion: str = "concat",      # "concat" or "add"
                 se_reduction: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        assert fusion in ("concat", "add"), "fusion must be 'concat' or 'add'"
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else Identity()

        # channels of feature extractor output (C)
        C = classifier_in_features

        # cost-attention (W) - 1x1 conv producing K maps -> GAP -> softmax
        self.cost_att = CostAttention(in_channels=C, num_classes=num_classes)

        # discriminative feature attention (SE-style)
        self.se = SEChannelGate(channels=C, reduction=se_reduction)

        # classifier: input channels depend on fusion
        if fusion == "concat":
            clf_in = C * 2
        else:  # add
            clf_in = C

        self.classifier = nn.Linear(clf_in, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor):
        """
        Returns dict:
          - logits: [B, K] (raw)
          - probs: sigmoid(logits)
          - W: [B, K] misclassification cost vector
          - F: [B, C, H, W] original feature map
          - Fprime: [B, C, H, W] discriminative gated features
        """
        F = self.forward_features(x)                   # [B, C, H, W]
        # compute W
        W = self.cost_att(F)                           # [B, K]

        # compute SE gating
        g = self.se(F)                                 # [B, C, 1, 1]
        Fprime = F * g                                 # [B, C, H, W]

        # fuse F and Fprime
        if self.fusion == "concat":
            fused = torch.cat([F, Fprime], dim=1)      # [B, 2C, H, W]
        else:  # add
            fused = F + Fprime                         # [B, C, H, W]

        # pool -> classifier
        pooled = nn.functional.adaptive_avg_pool2d(fused, 1).view(fused.size(0), -1)  # [B, clf_in]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)               # [B, K]
        probs = torch.sigmoid(logits)

        return {"logits": logits, "probs": probs, "W": W, "F": F, "Fprime": Fprime}


# --------------------------
# registry constructors
# --------------------------
@register("daiic_resnet34")
def daiic_resnet34(num_classes: int = 100, pretrained: bool = False, in_channels: int = 3,
                   cifar_stem: bool = True, fusion: str = "concat", se_reduction: int = 16,
                   dropout: float = 0.2, high_res: bool = False, **_):
    """
    DAIIC-ready ResNet34. If high_res=True, the last downsampling is replaced with dilation
    so the returned feature maps are higher-resolution (e.g. 8x8 for CIFAR32 instead of 4x4).
    """
    # Tell torchvision to replace the stride in layer4 with dilation when high_res True
    replace = None
    if high_res:
        # replace stride for layer4 only (layer2/layer3 keep their defaults)
        replace = [False, False, True]

    # create the resnet, passing replace_stride_with_dilation if requested
    # torchvision's API accepts replace_stride_with_dilation argument
    kwargs = {}
    if replace is not None:
        kwargs['replace_stride_with_dilation'] = replace

    # pass pretrained weights if requested
    weights = tv.models.ResNet34_Weights.DEFAULT if pretrained else None
    # For newer torchvision, you can use: tv.models.resnet34(weights=weights, replace_stride_with_dilation=replace)
    m = tv.models.resnet34(weights=weights, **kwargs)

    # if using cifar_stem we will replace conv1 and maxpool inside _make_resnet_feature_extractor
    feat_extractor = _make_resnet_feature_extractor(m, cifar_stem=cifar_stem, in_channels=in_channels)
    classifier_in = m.fc.in_features
    daiic = DAIICModule(feature_extractor=feat_extractor,
                        classifier_in_features=classifier_in,
                        num_classes=num_classes,
                        fusion=fusion,
                        se_reduction=se_reduction,
                        dropout=dropout)
    return daiic

@register("resnet34")
def resnet34(num_classes: int = 100, pretrained: bool = False, in_channels: int = 3, cifar_stem: bool = False, **_):
    """
    Standard lightweight wrapper (non-DAIIC): returns BackboneWrapper(feature_extractor, classifier, classifier_in_features)
    Use cifar_stem=True if inputs are small (32x32) to keep spatial resolution.
    """
    weights = tv.models.ResNet34_Weights.DEFAULT if pretrained else None
    m = tv.models.resnet34(weights=weights)
    # adapt first conv if input channels differ (if not using cifar_stem)
    if in_channels != 3 and not cifar_stem:
        old = m.conv1
        m.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=False)

    feat_extractor = _make_resnet_feature_extractor(m, cifar_stem=cifar_stem, in_channels=in_channels)
    classifier_in = m.fc.in_features
    classifier = nn.Linear(classifier_in, num_classes)
    return BackboneWrapper(feat_extractor, classifier, classifier_in)



@register("resnet50")
def resnet50(num_classes: int = 100, pretrained: bool = False, in_channels: int = 3, cifar_stem: bool = False, **_):
    weights = tv.models.ResNet50_Weights.DEFAULT if pretrained else None
    m = tv.models.resnet50(weights=weights)
    if in_channels != 3 and not cifar_stem:
        old = m.conv1
        m.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=False)
    feat_extractor = _make_resnet_feature_extractor(m, cifar_stem=cifar_stem, in_channels=in_channels)
    classifier_in = m.fc.in_features
    classifier = nn.Linear(classifier_in, num_classes)
    return BackboneWrapper(feat_extractor, classifier, classifier_in)


@register("daiic_resnet50")
def daiic_resnet50(num_classes: int = 100, pretrained: bool = True, in_channels: int = 3,
                   cifar_stem: bool = True, fusion: str = "concat", se_reduction: int = 16, dropout: float = 0.2, **_):
    weights = tv.models.ResNet50_Weights.DEFAULT if pretrained else None
    m = tv.models.resnet50(weights=weights)
    feat_extractor = _make_resnet_feature_extractor(m, cifar_stem=cifar_stem, in_channels=in_channels)
    classifier_in = m.fc.in_features
    daiic = DAIICModule(feature_extractor=feat_extractor,
                        classifier_in_features=classifier_in,
                        num_classes=num_classes,
                        fusion=fusion,
                        se_reduction=se_reduction,
                        dropout=dropout)
    return daiic


@register("densenet121")
def densenet121(num_classes: int = 100, pretrained: bool = False, in_channels: int = 3, cifar_stem: bool = False, **_):
    weights = tv.models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = tv.models.densenet121(weights=weights)
    # m.features returns spatial map [B, C, H, W]
    # adapt first conv if needed
    try:
        first = m.features[0]
        if isinstance(first, nn.Conv2d) and in_channels != 3:
            m.features[0] = nn.Conv2d(in_channels, first.out_channels,
                                      kernel_size=first.kernel_size,
                                      stride=first.stride, padding=first.padding, bias=False)
    except Exception:
        pass
    feat_extractor = m.features
    classifier_in = m.classifier.in_features
    classifier = nn.Linear(classifier_in, num_classes)
    return BackboneWrapper(feat_extractor, classifier, classifier_in)

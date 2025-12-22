import sys
sys.path.append("..")  # add project root

import torchvision as tv
from torch import nn
from .resnet_cifar import ResLTResNet32, BasicBlock
from .resnet_cifar_classifier import ResNetClassifier
from .resnet50_cbam import build_resnet50_cbam
from .small_cifar import SmallCIFARNet
import torch.nn.functional as F
import torch

REGISTRY = {}

from .plug_and_play_reparam_expert import PlugAndPlayReparamExperts  
REGISTRY["plug_and_play_reparam_expert"] = PlugAndPlayReparamExperts 

from .plug_and_play_expert import PlugAndPlayExperts
REGISTRY["plug_and_play_expert"] = PlugAndPlayExperts

def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


@register("resnet18")
def resnet18(num_classes=10, pretrained=False, **_):
    weights = tv.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = tv.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

from .resnet32_scratch import ResNet32

@register("resnet32")
def build_resnet32(num_classes=10, **kwargs):
    return ResNet32(num_classes=num_classes)



@register("resnet32_backbone")
def resnet32_backbone(num_classes=None, scale=1, nc=None, **_):
    """
    Build a backbone that returns spatial feature maps using layers from the ResNet32 classifier.
    This reuses the conv1/bn1/layer1/layer2/layer3 modules from ResNet32 and returns their output.
    """
    # instantiate the classifier to obtain the conv layers
    cls_model = ResNet32(num_classes=10)  # num_classes here doesn't matter; we only reuse conv layers

    # lightweight backbone wrapper that reuses layers from the classifier
    class _Backbone(nn.Module):
        def __init__(self, cls_mod):
            super().__init__()
            self.conv1 = cls_mod.conv1
            self.bn1 = cls_mod.bn1
            self.layer1 = cls_mod.layer1
            self.layer2 = cls_mod.layer2
            self.layer3 = cls_mod.layer3

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            return out

    return _Backbone(cls_model)

@register("densenet121")
def densenet121(num_classes=10, pretrained=False, **_):
    weights = tv.models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = tv.models.densenet121(weights=weights)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


# 🔹 Wrapper for ResLTResNet32 → always return a single logit
class ResLTResNet32Wrapper(nn.Module):
    def __init__(self, num_classes=10, scale=1):
        super().__init__()
        self.base = ResLTResNet32(num_classes=num_classes, scale=scale)

    def forward(self, x):
        logitH, logitM, logitT = self.base(x)
        return logitH + logitM + logitT


@register("resltresnet32")
def resltresnet32(num_classes=10, scale=1, **_):
    return ResLTResNet32(num_classes=num_classes, scale=scale)


# 🔹 Second custom model (ResNetClassifier wrapper)
@register("resnet_cifar_classifier")
def resnet_cifar_classifier(num_classes=10, drop=0.0, **_):
    return ResNetClassifier(
        block=BasicBlock,
        num_blocks=[5, 5, 5],   # ResNet32 config
        num_classes=num_classes,
        drop=drop,
    )


# ------------------------
# 🔹 EfficientNetV2-S
# ------------------------
@register("efficientnet_v2_s")
def efficientnet_v2_s(num_classes=10, pretrained=False, in_channels=3, **_):
    weights = tv.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    m = tv.models.efficientnet_v2_s(weights=weights)

    # Replace classifier
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)

    # Replace first conv if input channels != 3
    if in_channels != 3:
        old_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

    return m

@register("resnet50_cbam")
def resnet50_cbam(num_classes=10, pretrained=False, use_cbam=True, in_channels=3, cifar_stem=False, **_):
    return build_resnet50_cbam(num_classes=num_classes, pretrained=pretrained,
                               use_cbam=use_cbam, in_channels=in_channels, cifar_stem=cifar_stem)

@register("resnet50")
def resnet50_cbam(num_classes=10, pretrained=False, use_cbam=False, in_channels=3, cifar_stem=False, **_):
    """  
    for 3 experts use resnet50_torch- which can access features
    """
    return build_resnet50_cbam(num_classes=num_classes, pretrained=pretrained,
                               use_cbam=use_cbam, in_channels=in_channels, cifar_stem=cifar_stem)



# import the factory (put this with the other imports at top of file)

# register it exactly like your other models
@register("small_cifar")
def small_cifar(num_classes=100, drop=0.25, **_):
    return SmallCIFARNet(num_classes=num_classes, drop_prob=drop)



# ---------- add this model registration to models/__init__.py ----------

def _copy_state_dict_adapt(dst_model: torch.nn.Module, src_state: dict):
    """
    Copy weights from src_state into dst_model as fully as possible.
    - Copies exact key+shape matches.
    - Adapts conv1.weight (7x7 -> 3x3 or channel differences).
    - Adapts fc.weight/fc.bias by copying overlapping rows if num_classes differ.
    Returns statistics dict.
    """
    own_state = dst_model.state_dict()
    copied_tensors = 0
    copied_params = 0
    adapted_tensors = 0
    adapted_params = 0
    skipped = []

    # 1) exact matches
    for k, v in src_state.items():
        if k in own_state and own_state[k].shape == v.shape:
            try:
                own_state[k].copy_(v)
                copied_tensors += 1
                copied_params += v.numel()
            except Exception as e:
                skipped.append((k, f"copy_err:{e}"))

    # 2) conv1 adaptation (if present and shapes differ)
    if 'conv1.weight' in src_state and 'conv1.weight' in own_state:
        src_w = src_state['conv1.weight']   # (out_s, in_s, Hs, Ws)
        tgt_w = own_state['conv1.weight']   # (out_t, in_t, Ht, Wt)
        if src_w.shape != tgt_w.shape:
            try:
                sw = src_w.clone()  # keep src intact
                out_s, in_s, Hs, Ws = sw.shape
                out_t, in_t, Ht, Wt = tgt_w.shape

                # 2a) spatial resize if kernel sizes differ
                if (Hs, Ws) != (Ht, Wt):
                    # reshape to (out*in, 1, Hs, Ws) for interpolate
                    sw_reshaped = sw.view(out_s * in_s, 1, Hs, Ws)
                    sw_resized = F.interpolate(sw_reshaped, size=(Ht, Wt), mode='bilinear', align_corners=False)
                    sw = sw_resized.view(out_s, in_s, Ht, Wt)

                # 2b) input channel adaptation
                if in_s != in_t:
                    if in_s == 3 and in_t == 1:
                        # average RGB -> single channel
                        sw = sw.mean(dim=1, keepdim=True)  # (out_s,1,Ht,Wt)
                    else:
                        # general approach: average across source channels to 1, then repeat/trim
                        ch_averaged = sw.mean(dim=1, keepdim=True)  # (out_s,1,Ht,Wt)
                        sw = ch_averaged.repeat(1, in_t, 1, 1)
                        sw = sw[:, :in_t, :, :]

                # 2c) output channel adaptation (rare; try slice or pad zeros)
                if sw.shape[0] != out_t:
                    if sw.shape[0] >= out_t:
                        sw = sw[:out_t, :, :, :]
                    else:
                        pad_amt = out_t - sw.shape[0]
                        pad_tensor = torch.zeros(pad_amt, sw.shape[1], sw.shape[2], sw.shape[3],
                                                 device=sw.device, dtype=sw.dtype)
                        sw = torch.cat([sw, pad_tensor], dim=0)

                # final copy if shape now matches
                if sw.shape == tgt_w.shape:
                    own_state['conv1.weight'].copy_(sw)
                    adapted_tensors += 1
                    adapted_params += sw.numel()
                else:
                    skipped.append(('conv1.weight', (sw.shape, tgt_w.shape)))
            except Exception as e:
                skipped.append(('conv1.weight', f'error:{e}'))

    # 3) fc adaptation (copy overlap of rows/cols)
    if 'fc.weight' in src_state and 'fc.weight' in own_state:
        sw = src_state['fc.weight']  # (out_s, in_s)
        tw = own_state['fc.weight']  # (out_t, in_t)
        try:
            out_s, in_s = sw.shape
            out_t, in_t = tw.shape
            min_out = min(out_s, out_t)
            min_in  = min(in_s, in_t)
            # copy overlapping block
            tw[:min_out, :min_in].copy_(sw[:min_out, :min_in])
            adapted_tensors += 1
            adapted_params += sw[:min_out, :min_in].numel()

            # biases
            if 'fc.bias' in src_state and 'fc.bias' in own_state:
                sb = src_state['fc.bias']
                tb = own_state['fc.bias']
                min_b = min(sb.shape[0], tb.shape[0])
                tb[:min_b].copy_(sb[:min_b])
                adapted_tensors += 1
                adapted_params += min_b
        except Exception as e:
            skipped.append(('fc.weight', f'error:{e}'))

    # commit in-place to model
    dst_model.load_state_dict(own_state)

    stats = {
        'copied_tensors': copied_tensors,
        'copied_params': copied_params,
        'adapted_tensors': adapted_tensors,
        'adapted_params': adapted_params,
        'skipped_examples': skipped
    }
    return stats


@register("resnet50_torch")
def resnet50_torch(num_classes=1000, pretrained=False, in_channels=3, cifar_stem=False, **_):
    """
    Registry entry for vanilla torchvision ResNet-50 with aggressive, safe pretrained loading.
    - num_classes: desired output classes (replaces fc if !=1000)
    - pretrained: try to reuse as many pretrained weights as possible (adapt conv1/fc)
    - in_channels: first conv input channels
    - cifar_stem: if True, replace the 7x7 stride-2 stem with a 3x3 stride-1 stem and remove maxpool
    """
    # 1) construct a torchvision resnet50 (unweighted by default)
    m = None
    try:
        # modern API (preferred) - may raise if torchvision too old/new
        weights = tv.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tv.models.resnet50(weights=weights)
    except Exception:
        try:
            # fallback older API
            m = tv.models.resnet50(pretrained=pretrained)
        except Exception:
            # last resort, plain unpretrained
            m = tv.models.resnet50(weights=None)

    # 2) modify stem if requested (cifar_stem) or input channels differ
    # Note: do this before attempting to copy pretrained weights so adaptation sees final shapes
    if cifar_stem:
        # replace conv1 with 3x3 stride1 conv and remove initial maxpool
        old = m.conv1
        m.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=3, stride=1, padding=1,
                            bias=(old.bias is not None))
        # remove maxpool if exists
        if hasattr(m, "maxpool"):
            m.maxpool = nn.Identity()
    else:
        # only replace input channels if different
        if in_channels != 3:
            old = m.conv1
            m.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                kernel_size=old.kernel_size, stride=old.stride,
                                padding=old.padding, bias=(old.bias is not None))

    # 3) replace final classifier head if requested (keep shape info)
    if num_classes != 1000:
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    # 4) if pretrained requested, load a vanilla torchvision resnet50 state_dict and adaptively copy
    if pretrained:
        try:
            # obtain canonical torchvision resnet50 (pretrained) to use as source of weights
            try:
                src_weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
                src_model = tv.models.resnet50(weights=src_weights)
            except Exception:
                src_model = tv.models.resnet50(pretrained=True)
            src_state = src_model.state_dict()

            stats = _copy_state_dict_adapt(m, src_state)

            # friendly print/log with both tensors and parameter counts
            print(f"[resnet50_torch] pretrained=True: copied {stats['copied_tensors']} tensors "
                  f"({stats['copied_params']:,} params), adapted {stats['adapted_tensors']} tensors "
                  f"({stats['adapted_params']:,} params).")
            if stats['skipped_examples']:
                print(f"[resnet50_torch] skipped/adapt-examples: {stats['skipped_examples'][:5]}")
        except Exception as e:
            print(f"[resnet50_torch] warning: attempted to load pretrained weights but failed: {e}")

    return m


@register("mobilenetv2_backbone")
def mobilenetv2_backbone(num_classes=None, pretrained=False, in_channels=3, cifar_stem=False, **_):
    # return the ResNet-like wrapper whose forward() -> spatial features (N,C,H,W)
    from .mobile_net_backbone import MobileNetV2AsResNetLike
    return MobileNetV2AsResNetLike(pretrained=pretrained, in_channels=in_channels, cifar_stem=cifar_stem)


from .mobilenetv2_3bn import MobileNetV2_3BN

@register("mobilenetv2_3bhn")
def build_mobilenetv2_3bn(num_classes=10, pretrained=False, **kwargs):
    return MobileNetV2_3BN(num_classes=num_classes, pretrained=pretrained)


@register("mobilenetv2")
def build_mobilenetv2(num_classes=10, pretrained=False, in_channels=3, **_):
    # load pretrained mobilenet
    weights = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = tv.models.mobilenet_v2(weights=weights)

    # replace last linear layer
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)

    # adapt first conv for different num input channels
    if in_channels != 3:
        old_conv = m.features[0][0]  # first Conv2d
        m.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

    return m


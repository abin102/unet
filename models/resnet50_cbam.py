# models/resnet50_cbam.py
import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.models.resnet import ResNet, Bottleneck
# NOTE: do NOT import model_urls; newer torchvision removed it.

# ---- CBAM (compact) ----
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True)
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        out = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, use_bn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1) if use_bn else nn.Identity()

    def forward(self, x):
        mx  = torch.max(x, dim=1, keepdim=True)[0]
        avg = torch.mean(x, dim=1, keepdim=True)
        cat = torch.cat([mx, avg], dim=1)   # B,2,H,W
        out = self.conv(cat)
        out = self.bn(out)
        out = torch.sigmoid(out)
        return x * out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ---- Bottleneck that can apply CBAM ----
class BottleneckCBAM(Bottleneck):
    def __init__(self, *args, use_cbam=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cbam = use_cbam
        if use_cbam:
            out_ch = self.conv3.out_channels
            self.cbam = CBAM(out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        if self.use_cbam:
            out = self.cbam(out)

        out = self.relu(out)
        return out

# ---- ResNet50 builder with BottleneckCBAM ----
class ResNet50_CBAM(ResNet):
    def __init__(self, num_classes=10, pretrained=False, use_cbam=True, in_channels=3, cifar_stem=False):
        # ResNet50 config = [3,4,6,3]
        super().__init__(block=BottleneckCBAM, layers=[3, 4, 6, 3], num_classes=num_classes)

        # adjust stem for in_channels first (will be possibly overridden by cifar_stem below)
        if in_channels != 3:
            old = self.conv1
            self.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias
            )

        # CIFAR stem: 3x3, stride=1 and remove initial maxpool
        if cifar_stem:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # remove the initial maxpool so 32x32 doesn't get downsampled too much
            if hasattr(self, "maxpool"):
                self.maxpool = nn.Identity()
            elif hasattr(self, "pool"):
                self.pool = nn.Identity()

        # enable CBAM on all BottleneckCBAM modules if requested
        if use_cbam:
            for m in self.modules():
                if isinstance(m, BottleneckCBAM):
                    m.use_cbam = True
                    m.cbam = CBAM(m.conv3.out_channels)

        # load pretrained weights if requested (best-effort; strict=False)
        if pretrained:
            # try new weights API, fall back safely
            try:
                weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
                model_pretrained = tv.models.resnet50(weights=weights)
                state = model_pretrained.state_dict()
                self.load_state_dict(state, strict=False)
            except Exception:
                try:
                    # older torchvision: weights arg may be different
                    model_pretrained = tv.models.resnet50(pretrained=True)
                    self.load_state_dict(model_pretrained.state_dict(), strict=False)
                except Exception:
                    # last resort: don't fail, just warn
                    print("Warning: pretrained weights requested but could not be loaded for this torchvision version.")

def build_resnet50_cbam(num_classes=10, pretrained=False, use_cbam=True, in_channels=3, cifar_stem=False, **_):
    return ResNet50_CBAM(num_classes=num_classes, pretrained=pretrained, use_cbam=use_cbam, in_channels=in_channels, cifar_stem=cifar_stem)

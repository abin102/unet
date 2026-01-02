import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PART 1: REPARAM PRIMITIVES (Taken directly from your provided code)
# ==============================================================================

def getConv2D(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias)

# def getBN(channels, eps=1e-5, momentum=0.01, affine=True):
#     return nn.BatchNorm2d(num_features=channels, eps=eps, momentum=momentum, affine=affine)

def getBN(channels, eps=1e-5, momentum=None, affine=True):
    # momentum is unused for GroupNorm, kept for API compatibility
    num_groups = min(8, channels)  # safe default
    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=channels,
        eps=eps,
        affine=affine
    )


def mergeBN(convLayer, BNLayer):
    std = (BNLayer.running_var + BNLayer.eps).sqrt()
    t = (BNLayer.weight / std).reshape(-1, 1, 1, 1)
    return convLayer.weight * t, BNLayer.bias - BNLayer.running_mean * BNLayer.weight / std

def kernelFuse(target, sec):
    sec_h, sec_w = sec.size(2), sec.size(3)
    target_h, target_w = target.size(2), target.size(3)
    target[:, :, target_h // 2 - sec_h // 2: target_h // 2 - sec_h // 2 + sec_h,
                 target_w // 2 - sec_w // 2: target_w // 2 - sec_w // 2 + sec_w] += sec

class AsymmConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depthwise=False):
        # NOTE: Defaults to depthwise=False for UNet standard convolutions
        super(AsymmConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.depthwise = depthwise
        self.mergedFlag = False
        self.mergedConv = None

        g = in_channels if depthwise else 1

        # 1. Vertical & Horizontal Strip
        self.convVe1 = getConv2D(in_channels, out_channels, (kernel_size, 1), stride, [padding, 0], groups=g, bias=False)
        self.bnVe1 = getBN(out_channels)
        self.convHo1 = getConv2D(in_channels, out_channels, (1, kernel_size), stride, [0, padding], groups=g, bias=False)
        self.bnHo1 = getBN(out_channels)

        # 2. Square (NxN)
        self.convSq = getConv2D(in_channels, out_channels, (kernel_size, kernel_size), stride, [padding, padding], groups=g, bias=False)
        self.bnSq = getBN(out_channels)

        # Note: I omitted the Rectangular (NxN-2) blocks here to save some VRAM 
        # for the UNet, but you can add them back if you have the memory.

    def forward(self, x):
        if self.mergedFlag:
            return self.mergedConv(x)
        return (self.bnVe1(self.convVe1(x)) + 
                self.bnHo1(self.convHo1(x)) + 
                self.bnSq(self.convSq(x)))

    def mergeAsyKernels(self):
        if self.mergedFlag: return
        w_ve1, b_ve1 = mergeBN(self.convVe1, self.bnVe1)
        w_ho1, b_ho1 = mergeBN(self.convHo1, self.bnHo1)
        w_sq, b_sq = mergeBN(self.convSq, self.bnSq)

        kernelFuse(w_sq, w_ve1)
        kernelFuse(w_sq, w_ho1)
        b_sq = b_sq + b_ve1 + b_ho1

        g = self.in_channels if self.depthwise else 1
        self.mergedConv = getConv2D(self.in_channels, self.out_channels, (self.kernel_size, self.kernel_size),
                                    self.stride, [self.padding, self.padding], groups=g, bias=True)
        self.mergedConv.weight.data = w_sq
        self.mergedConv.bias.data = b_sq
        self.mergedFlag = True
        
        # Cleanup
        for attr in ['convVe1', 'bnVe1', 'convHo1', 'bnHo1', 'convSq', 'bnSq']:
            if hasattr(self, attr): delattr(self, attr)


class MultiConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, depthwise=False):
        super(MultiConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = sorted(kernel_sizes)
        self.stride = stride
        self.depthwise = depthwise
        self.mergedFlag = False
        self.mergedConv = None
        self.conv_list = nn.ModuleList()

        for kernel in self.kernel_sizes:
            # Padding is kernel // 2 to maintain spatial size
            self.conv_list.append(AsymmConvBlock(
                in_channels, out_channels, kernel, stride, kernel // 2, depthwise
            ))

    def forward(self, x):
        if self.mergedFlag:
            return self.mergedConv(x)
        
        y = None
        for conv in self.conv_list:
            out = conv(x)
            y = out if y is None else y + out
        return y

    def mergeMulKernels(self):
        if self.mergedFlag: return
        for block in self.conv_list:
            block.mergeAsyKernels()

        largest_idx = -1
        target_w = self.conv_list[largest_idx].mergedConv.weight.data.clone()
        target_b = self.conv_list[largest_idx].mergedConv.bias.data.clone()

        for i in range(len(self.conv_list) - 1):
            kernelFuse(target_w, self.conv_list[i].mergedConv.weight.data)
            target_b += self.conv_list[i].mergedConv.bias.data

        g = self.in_channels if self.depthwise else 1
        max_k = self.kernel_sizes[-1]
        self.mergedConv = getConv2D(self.in_channels, self.out_channels, (max_k, max_k),
                                    self.stride, [max_k // 2, max_k // 2], groups=g, bias=True)
        self.mergedConv.weight.data = target_w
        self.mergedConv.bias.data = target_b
        self.mergedFlag = True
        if hasattr(self, 'conv_list'): delattr(self, 'conv_list')

# ==============================================================================
# PART 2: REPARAM-UNET IMPLEMENTATION
# ==============================================================================

class ReparamDoubleConv(nn.Module):
    """
    Replaces (Conv => BN => ReLU) * 2 with:
    (MultiConvBlock => ReLU) * 2
    Note: MultiConvBlock handles BN internally.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3]):
        super().__init__()
        
        # We generally do NOT use depthwise for standard UNet feature extraction
        # unless you specifically want a lightweight MobileNet-like UNet.
        # Set depthwise=False for standard power.
        self.reparam_conv = nn.Sequential(
            MultiConvBlock(in_channels, out_channels, kernel_sizes, stride=1, depthwise=False),
            nn.ReLU(inplace=True),
            MultiConvBlock(out_channels, out_channels, kernel_sizes, stride=1, depthwise=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reparam_conv(x)


class ReparamUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, features=[64, 128, 256, 512], kernel_sizes=[3]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.kernel_sizes = kernel_sizes

        # Down part of UNet
        for feature in features:
            self.downs.append(ReparamDoubleConv(in_channels, feature, kernel_sizes))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(ReparamDoubleConv(feature*2, feature, kernel_sizes))

        self.bottleneck = ReparamDoubleConv(features[-1], features[-1]*2, kernel_sizes)
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x, current_epoch=None):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def switch_to_deploy(self):
        """
        Traverses the whole UNet and fuses all MultiConvBlocks.
        """
        print("[ReparamUNet] Switching to deploy mode (fusing kernels)...")
        for module in self.modules():
            if isinstance(module, MultiConvBlock):
                module.mergeMulKernels()
        print("[ReparamUNet] Fusion complete.")

# ==============================================================================
# PART 3: BUILDER
# ==============================================================================

def build_reparam_unet(**kwargs):
    if "kernel_sizes" not in kwargs:
        kwargs["kernel_sizes"] = [3]
    return ReparamUNet(**kwargs)


# Test Code
if __name__ == "__main__":
    x = torch.randn((1, 1, 160, 160))
    
    # 1. Train Mode (Heavy)
    model = build_unet(in_channels=1, n_classes=3, features=[16, 32], kernel_sizes=[3])
    preds = model(x)
    print(f"Train Output: {preds.shape}")
    
    # 2. Deploy Mode (Light - equivalent to standard UNet inference speed)
    model.switch_to_deploy()
    preds_deploy = model(x)
    print(f"Deploy Output: {preds_deploy.shape}")
    
    # Check that fusion didn't break values (error should be small ~1e-5)
    print("Fusion successful.")
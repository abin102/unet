import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional, Tuple, Union, List

# ==============================================================================
# PART 1: REPARAMETERIZATION MATH & BLOCKS (From ERoHPRF)
# ==============================================================================

def getConv2D(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias)

def getBN(channels, eps=1e-5, momentum=0.01, affine=True):
    return nn.BatchNorm2d(num_features=channels, eps=eps, momentum=momentum, affine=affine)

def mergeBN(convLayer, BNLayer):
    std = (BNLayer.running_var + BNLayer.eps).sqrt()
    t = (BNLayer.weight / std).reshape(-1, 1, 1, 1)
    return convLayer.weight * t, BNLayer.bias - BNLayer.running_mean * BNLayer.weight / std

def kernelFuse(target, sec):
    """Fuses a smaller kernel 'sec' into the center of 'target'."""
    sec_h, sec_w = sec.size(2), sec.size(3)
    target_h, target_w = target.size(2), target.size(3)
    target[:, :, target_h // 2 - sec_h // 2: target_h // 2 - sec_h // 2 + sec_h,
                 target_w // 2 - sec_w // 2: target_w // 2 - sec_w // 2 + sec_w] += sec

class AsymmConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depthwise=True):
        super(AsymmConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.depthwise = depthwise
        self.mergedFlag = False
        self.mergedConv = None

        # Logic: If depthwise, we use groups=in_channels
        g = in_channels if depthwise else 1

        # 1. Vertical & Horizontal Strip (Nx1, 1xN)
        self.convVe1 = getConv2D(in_channels, out_channels, (kernel_size, 1), stride, [padding, 0], groups=g, bias=False)
        self.bnVe1 = getBN(out_channels)
        self.convHo1 = getConv2D(in_channels, out_channels, (1, kernel_size), stride, [0, padding], groups=g, bias=False)
        self.bnHo1 = getBN(out_channels)

        # 2. Vertical & Horizontal Rect (NxN-2, N-2xN) - Only if kernel > 3 usually, but kept for consistency
        self.convVek = getConv2D(in_channels, out_channels, (kernel_size, kernel_size - 2), stride, [padding, padding - 1], groups=g, bias=False)
        self.bnVek = getBN(out_channels)
        self.convHok = getConv2D(in_channels, out_channels, (kernel_size - 2, kernel_size), stride, [padding - 1, padding], groups=g, bias=False)
        self.bnHok = getBN(out_channels)

        # 3. Square (NxN)
        self.convSq = getConv2D(in_channels, out_channels, (kernel_size, kernel_size), stride, [padding, padding], groups=g, bias=False)
        self.bnSq = getBN(out_channels)

    def forward(self, x):
        if self.mergedFlag:
            return self.mergedConv(x)
        return (self.bnVe1(self.convVe1(x)) + self.bnVek(self.convVek(x)) +
                self.bnHo1(self.convHo1(x)) + self.bnHok(self.convHok(x)) +
                self.bnSq(self.convSq(x)))

    def mergeAsyKernels(self):
        if self.mergedFlag: return
        # Merge BN into weights
        w_ve1, b_ve1 = mergeBN(self.convVe1, self.bnVe1)
        w_ho1, b_ho1 = mergeBN(self.convHo1, self.bnHo1)
        w_sq, b_sq = mergeBN(self.convSq, self.bnSq)
        w_vek, b_vek = mergeBN(self.convVek, self.bnVek)
        w_hok, b_hok = mergeBN(self.convHok, self.bnHok)

        # Fuse everything into the Square kernel
        kernelFuse(w_sq, w_ve1)
        kernelFuse(w_sq, w_ho1)
        kernelFuse(w_sq, w_vek)
        kernelFuse(w_sq, w_hok)
        b_sq = b_sq + b_ve1 + b_ho1 + b_vek + b_hok

        # Create the final layer
        g = self.in_channels if self.depthwise else 1
        self.mergedConv = getConv2D(self.in_channels, self.out_channels, (self.kernel_size, self.kernel_size),
                                    self.stride, [self.padding, self.padding], groups=g, bias=True)
        self.mergedConv.weight.data = w_sq
        self.mergedConv.bias.data = b_sq
        self.mergedFlag = True
        
        # Cleanup to save memory
        for attr in ['convVe1', 'bnVe1', 'convHo1', 'bnHo1', 'convVek', 'bnVek', 'convHok', 'bnHok', 'convSq', 'bnSq']:
            if hasattr(self, attr): delattr(self, attr)


class MultiConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, depthwise=True):
        super(MultiConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = sorted(kernel_sizes) # Ensure sorted 3, 5, 7
        self.stride = stride
        self.depthwise = depthwise
        self.mergedFlag = False
        self.mergedConv = None
        self.conv_list = nn.ModuleList()

        for kernel in self.kernel_sizes:
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
        # 1. Merge the inner Asymm blocks first
        for block in self.conv_list:
            block.mergeAsyKernels()

        # 2. Fuse smaller kernels into the largest kernel
        largest_idx = -1
        target_w = self.conv_list[largest_idx].mergedConv.weight.data.clone()
        target_b = self.conv_list[largest_idx].mergedConv.bias.data.clone()

        for i in range(len(self.conv_list) - 1):
            kernelFuse(target_w, self.conv_list[i].mergedConv.weight.data)
            target_b += self.conv_list[i].mergedConv.bias.data

        # 3. Create final layer
        g = self.in_channels if self.depthwise else 1
        max_k = self.kernel_sizes[-1]
        self.mergedConv = getConv2D(self.in_channels, self.out_channels, (max_k, max_k),
                                    self.stride, [max_k // 2, max_k // 2], groups=g, bias=True)
        self.mergedConv.weight.data = target_w
        self.mergedConv.bias.data = target_b
        self.mergedFlag = True
        if hasattr(self, 'conv_list'): delattr(self, 'conv_list')


# ==============================================================================
# PART 2: THE EXPERT WRAPPER
# ==============================================================================

class ReparamExpert(nn.Module):
    def __init__(self, in_ch, out_dim=256, depth=1, wide=1, kernel_sizes=[3], dropout=0.0):
        super().__init__()
        # If user passes an integer (e.g. 3), wrap it in list
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
            
        logger.info(f"[ReparamExpert] Init: in={in_ch}, out={out_dim}, depth={depth}, kernels={kernel_sizes}")
        
        layers = []
        cur = in_ch
        
        for i in range(depth):
            nxt = max(1, int(cur * wide))
            
            # Pointwise adjustment if channels change
            if cur != nxt:
                 layers.append(nn.Conv2d(cur, nxt, 1, bias=False))
                 layers.append(nn.BatchNorm2d(nxt))
                 layers.append(nn.ReLU6(inplace=True))
                 cur = nxt

            # The Expert Block using the SPECIFIC kernel list
            layers.append(MultiConvBlock(cur, cur, kernel_sizes=kernel_sizes, stride=1, depthwise=True))
            layers.append(nn.ReLU6(inplace=True))
            
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(cur, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        emb = self.proj(x)
        return emb

    def fuse(self):
        for m in self.modules():
            if isinstance(m, MultiConvBlock):
                m.mergeMulKernels()


# ==============================================================================
# PART 3: UNCERTAINTY HEAD
# ==============================================================================

class UncertaintyHead(nn.Module):
    def __init__(self, in_ch, hidden=128, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_ch, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lin2 = nn.Linear(hidden, 3) # 3 Experts

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        h = self.relu(self.lin1(x))
        h = self.dropout(h)
        logits = self.lin2(h)
        return logits


# ==============================================================================
# PART 4: MAIN PLUG AND PLAY WRAPPER
# ==============================================================================

class PlugAndPlayReparamExperts(nn.Module):
    def __init__(self,
                 backbone: Union[str, nn.Module] = "resnet50",
                 backbone_args: Optional[dict] = None,
                 registry: Optional[dict] = None,
                 feature_layer: str = "layer3",
                 num_classes: int = 1000,
                 adapter_channels: Optional[int] = 256,
                 expert_dims: int = 256,
                 expert_cfgs: Optional[List[dict]] = None,
                 uncertainty_dropout: float = 0.0,
                 probe_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        super().__init__()

        # ---- Resolve backbone (FIXED SECTION) ----
        if isinstance(backbone, str):
            if registry is None:
                logger.critical("[PlugAndPlayReparamExperts] Missing registry for string backbone key!")
                raise ValueError("Backbone string requires registry mapping.")
            ctor = registry.get(backbone)
            if ctor is None:
                logger.critical(f"[PlugAndPlayReparamExperts] Unknown backbone key: {backbone}")
                raise KeyError(f"Backbone '{backbone}' not found.")
            args = backbone_args or {}
            self.backbone = ctor(**args)
            logger.debug(f"[PlugAndPlayReparamExperts] Constructed backbone '{backbone}' with args={args}")
        elif isinstance(backbone, nn.Module):
            self.backbone = backbone
        else:
            raise ValueError("Invalid backbone type.")

        # Probe Channels
        feat_ch = self._probe_backbone(probe_input_size)
        
        # Adapter
        if adapter_channels:
            self.adapter = nn.Sequential(
                nn.Conv2d(feat_ch, adapter_channels, 1, bias=False),
                nn.BatchNorm2d(adapter_channels),
                nn.ReLU(inplace=True),
            )
            feat_ch = adapter_channels
        else:
            self.adapter = None

        # Gating
        self.uncertainty = UncertaintyHead(feat_ch, hidden=max(64, feat_ch // 2), dropout=uncertainty_dropout)

        # Configs
        if expert_cfgs is None:
            # DEFAULT FALLBACK (Matches previous logic if config not provided)
            expert_cfgs = [
                {"depth": 1, "wide": 1, "kernel_sizes": [3]},    # Expert A
                {"depth": 1, "wide": 1, "kernel_sizes": [5]},    # Expert B
                {"depth": 1, "wide": 1, "kernel_sizes": [7]}     # Expert C
            ]

        # Initialize Reparam Experts
        self.expert_a = ReparamExpert(feat_ch, expert_dims, **expert_cfgs[0])
        self.expert_b = ReparamExpert(feat_ch, expert_dims, **expert_cfgs[1])
        self.expert_c = ReparamExpert(feat_ch, expert_dims, **expert_cfgs[2])

        self.classifier = nn.Linear(expert_dims, num_classes)
        logger.success("[PlugAndPlayReparamExperts] Ready. Call .switch_to_deploy() after training.")

    def _probe_backbone(self, input_size):
        # We need to temporarily move to CPU to probe if it's not already
        orig_device = next(self.backbone.parameters(), torch.zeros(1)).device
        
        # Simple probe: Pass dummy input
        # Note: We do NOT move model to CPU here to avoid disturbing training setup
        # unless necessary. We just create dummy input on correct device.
        dummy = torch.randn(*input_size).to(orig_device)
        
        # Ensure eval mode for probe to avoid BN updates
        was_training = self.backbone.training
        self.backbone.eval()
        
        with torch.no_grad():
            try:
                out = self.backbone(dummy)
            except Exception as e:
                # Fallback: try cpu
                logger.warning(f"Probe failed on {orig_device}, trying CPU. Error: {e}")
                self.backbone.cpu()
                dummy = torch.randn(*input_size)
                out = self.backbone(dummy)
                self.backbone.to(orig_device)

        if was_training:
            self.backbone.train()

        return out.shape[1]

    def forward(self, x, current_epoch=-1):
        feats = self.backbone(x)
        feats = self.adapter(feats) if self.adapter else feats

        # Warmup Phase (Default < 50 epochs, adjustable via logic outside)
        if 0 <= current_epoch < 50: 
            e_a = self.expert_a(feats)
            e_b = self.expert_b(feats)
            e_c = self.expert_c(feats)
            emb = (e_a + e_b + e_c) / 3
            logits = self.classifier(emb)
            
            # Return dummy values so train_step.py and WandB don't crash
            dummy_loss = torch.tensor(0.0, device=logits.device)
            dummy_weights = torch.zeros((x.size(0), 3), device=logits.device)
            
            return {
                "logits": logits, 
                "loss_aux": dummy_loss,
                "gating_weights": dummy_weights
            }

        # Gating Phase
        gate_logits = self.uncertainty(feats)
        gate_w = F.softmax(gate_logits, dim=1)

        e_a = self.expert_a(feats)
        e_b = self.expert_b(feats)
        e_c = self.expert_c(feats)

        # Weighted combination
        emb = e_a * gate_w[:, 0:1] + e_b * gate_w[:, 1:2] + e_c * gate_w[:, 2:3]
        logits = self.classifier(emb)

        # Curriculum Loss Calculation
        with torch.no_grad():
            conf = F.softmax(logits, 1).max(1)[0]
            easy = (conf > 0.9).float()
            medium = ((conf >= 0.6) & (conf <= 0.9)).float()
            hard = (conf < 0.6).float()

        loss_easy = (gate_w[:, 1] * easy + gate_w[:, 2] * easy).mean()
        loss_medium = (gate_w[:, 0] * medium + gate_w[:, 2] * medium).mean()
        loss_hard = (gate_w[:, 0] * hard + gate_w[:, 1] * hard).mean()
        
        return {
                "logits": logits, 
                "loss_aux": loss_easy + loss_medium + loss_hard,
                "gating_weights": gate_w.detach() 
            }
    
    def switch_to_deploy(self):
        """
        Collapses all multi-scale experts into single layer convolutions.
        Call this after training and before saving/inference.
        """
        logger.warning("[PlugAndPlayReparamExperts] Switching to DEPLOY mode. This is irreversible!")
        self.expert_a.fuse()
        self.expert_b.fuse()
        self.expert_c.fuse()
        self.eval()

# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    # 1. Create a dummy backbone
    backbone = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.ReLU())
    
    # 2. Init Model
    # Explicitly testing separate branches [3], [5], [7]
    cfgs = [
        {"depth": 1, "wide": 1, "kernel_sizes": [3]},
        {"depth": 1, "wide": 1, "kernel_sizes": [5]},
        {"depth": 1, "wide": 1, "kernel_sizes": [7]}
    ]
    
    model = PlugAndPlayReparamExperts(backbone=backbone, probe_input_size=(1, 3, 224, 224), expert_cfgs=cfgs)
    
    # 3. Test Forward
    inp = torch.randn(2, 3, 224, 224)
    out = model(inp, current_epoch=55) # > 50 to test gating
    print("Training Output Shape:", out['logits'].shape)
    print("Gating Weights Shape:", out['gating_weights'].shape)
    
    # 4. Test Fusion
    model.switch_to_deploy()
    out_fused = model(inp, current_epoch=55)
    print("Fused Output Shape:", out_fused['logits'].shape)
    
    # Verify outputs are close (error due to float precision is expected)
    diff = (out['logits'] - out_fused['logits']).abs().mean()
    print(f"Difference after fusion: {diff.item():.8f}")
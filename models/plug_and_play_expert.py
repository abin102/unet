# models/plug_and_play_expert.py
"""
Plug-and-play wrapper: attach 3 experts + uncertainty head to any backbone.

Includes structured Loguru logging for every significant event:
 - DEBUG: detailed tensor shapes, control flow, and fine-grained checkpoints
 - INFO: initialization, high-level execution progress
 - WARNING: recoverable or unexpected conditions
 - ERROR/CRITICAL: serious issues, bad configurations, or exceptions
"""

from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False
    wandb = None


# ============================ Basic Block ============================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        logger.debug(f"[ConvBlock] Initialized: in={in_ch}, out={out_ch}, kernel={kernel_size}, pad={padding}")

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        logger.trace(f"[ConvBlock] Forward: input={x.shape}, output={out.shape}")
        return out


# ============================ Expert ============================
class Expert(nn.Module):
    def __init__(self, in_ch, out_dim=256, depth=1, wide=1, large_kernel=False, dropout=0.0):
        super().__init__()
        logger.info(f"[Expert] Initializing: in_ch={in_ch}, out_dim={out_dim}, depth={depth}, wide={wide}, "
                    f"large_kernel={large_kernel}, dropout={dropout}")

        self.in_ch = in_ch
        self.out_dim = out_dim
        self.depth = depth
        self.wide = wide
        self.large_kernel = large_kernel

        layers = []
        cur = in_ch
        for i in range(depth):
            k = 5 if (large_kernel and i == depth - 1) else 3
            pad = k // 2
            nxt = max(1, int(cur * wide))
            layers.append(ConvBlock(cur, nxt, kernel_size=k, padding=pad))
            logger.debug(f"[Expert] Layer {i}: {cur}→{nxt}, kernel={k}")
            cur = nxt

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            logger.debug(f"[Expert] Added Dropout2d({dropout})")

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(cur, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        emb = self.proj(x)
        logger.trace(f"[Expert] Forward: input={x.shape}, emb={emb.shape}")
        return emb


# ============================ Uncertainty Head ============================
class UncertaintyHead(nn.Module):
    def __init__(self, in_ch, hidden=128, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_ch, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lin2 = nn.Linear(hidden, 3)
        logger.info(f"[UncertaintyHead] Initialized: in_ch={in_ch}, hidden={hidden}, dropout={dropout}")

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        h = self.relu(self.lin1(x))
        h = self.dropout(h)
        logits = self.lin2(h)
        logger.trace(f"[UncertaintyHead] Forward: pooled={x.shape}, logits={logits.shape}")
        return logits


# ============================ Main Wrapper ============================
class PlugAndPlayExperts(nn.Module):
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
                 probe_input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                 curriculum_loss_multiplier: float = 0.01,):
        super().__init__()

        logger.info(f"[PlugAndPlayExperts] Init: backbone={backbone}, layer={feature_layer}, "
                    f"num_classes={num_classes}, adapter={adapter_channels}, expert_dims={expert_dims}")

        # ---- Resolve backbone ----
        if isinstance(backbone, str):
            if registry is None:
                logger.critical("[PlugAndPlayExperts] Missing registry for string backbone key!")
                raise ValueError("Backbone string requires registry mapping.")
            ctor = registry.get(backbone)
            if ctor is None:
                logger.critical(f"[PlugAndPlayExperts] Unknown backbone key: {backbone}")
                raise KeyError(f"Backbone '{backbone}' not found.")
            args = backbone_args or {}
            self.backbone = ctor(**args)
            logger.debug(f"[PlugAndPlayExperts] Constructed backbone '{backbone}' with args={args}")
        elif isinstance(backbone, nn.Module):
            self.backbone = backbone
            logger.debug("[PlugAndPlayExperts] Backbone passed as pre-built nn.Module")
        else:
            logger.error("[PlugAndPlayExperts] Invalid backbone type.")
            raise ValueError("Invalid backbone type.")

        # ---- Probe channels ----
        feat_ch, feat_shape = self._probe_backbone(probe_input_size)
        logger.info(f"[PlugAndPlayExperts] Backbone probe complete: feat_ch={feat_ch}, feat_shape={feat_shape}")

        # ---- Adapter ----
        if adapter_channels is not None:
            self.adapter = nn.Sequential(
                nn.Conv2d(feat_ch, adapter_channels, 1, bias=False),
                nn.BatchNorm2d(adapter_channels),
                nn.ReLU(inplace=True),
            )
            feat_ch_after_adapter = adapter_channels
            logger.info(f"[Adapter] Built 1x1 conv adapter: {feat_ch}→{adapter_channels}")
        else:
            self.adapter = None
            feat_ch_after_adapter = feat_ch
            logger.warning("[Adapter] None configured — using raw backbone features.")

        # ---- Gating Head ----
        self.uncertainty = UncertaintyHead(feat_ch_after_adapter,
                                           hidden=max(64, feat_ch_after_adapter // 2),
                                           dropout=uncertainty_dropout)

        # ---- Experts ----
        if expert_cfgs is None:
            expert_cfgs = [
                {"depth": 1, "wide": 1},
                {"depth": 2, "wide": 1},
                {"depth": 3, "wide": 2, "large_kernel": True}
            ]
            logger.debug("[PlugAndPlayExperts] Using default expert configs.")
        if len(expert_cfgs) != 3:
            logger.critical("[PlugAndPlayExperts] Must specify exactly 3 experts.")
            raise AssertionError("expert_cfgs must have length 3")

        self.expert_a = Expert(feat_ch_after_adapter, expert_dims, **expert_cfgs[0])
        self.expert_b = Expert(feat_ch_after_adapter, expert_dims, **expert_cfgs[1])
        self.expert_c = Expert(feat_ch_after_adapter, expert_dims, **expert_cfgs[2])
        logger.info("[PlugAndPlayExperts] All experts initialized.")

        # ---- Classifier ----
        self.classifier = nn.Linear(expert_dims, num_classes)
        logger.info(f"[Classifier] Linear: in={expert_dims}, out={num_classes}")

        self._feat_ch, self._feat_shape = feat_ch, feat_shape
        self._print_summary()

    # ============================================================
    def _probe_backbone(self, input_size):
        logger.debug(f"[Probe] Running backbone probe with dummy input {input_size}")
        orig_training = self.backbone.training
        orig_device = next(self.backbone.parameters(), torch.zeros(1)).device

        try:
            dummy = torch.randn(*input_size, device="cpu")
            self.backbone.to("cpu").eval()
            with torch.no_grad():
                x = self.backbone(dummy)
                if isinstance(x, torch.Tensor) and x.ndim == 4:
                    return x.shape[1], tuple(x.shape)
                logger.warning("[Probe] Non-4D output; trying alternative modules.")
            raise RuntimeError("Failed to infer feature shape")
        except Exception as e:
            logger.exception(f"[Probe] Failed: {e}")
            raise
        finally:
            self.backbone.to(orig_device)
            if orig_training: self.backbone.train()
            logger.debug("[Probe] Restored backbone to original device/mode.")

    # ============================================================
    def forward(self, x, return_gate_info=False, current_epoch=-1):
        logger.debug(f"[Forward] Start: shape={tuple(x.shape)}, epoch={current_epoch}")

        # ---- Feature extraction ----
        feats = self.backbone(x)
        if not (isinstance(feats, torch.Tensor) and feats.ndim == 4):
            logger.critical("[Forward] Backbone failed to return spatial features.")
            raise RuntimeError("Invalid feature map output from backbone.")
        logger.trace(f"[Forward] Extracted feats={feats.shape}")

        feats_adapt = self.adapter(feats) if self.adapter else feats
        logger.trace(f"[Forward] Adapter output={feats_adapt.shape}")

        # ---- Stage 1: Warm-up ----
        if 0 <= current_epoch < 50:
            logger.info(f"[Forward] Warm-up phase (epoch={current_epoch})")
            e_a, e_b, e_c = [ex(feats_adapt) for ex in (self.expert_a, self.expert_b, self.expert_c)]
            emb = (e_a + e_b + e_c) / 3
            logits = self.classifier(emb)
            dummy_loss = torch.tensor(0.0, device=logits.device)
            gating_weights = torch.zeros((x.size(0), 3), device=logits.device)
            logger.debug("the input given to the model is {}", x.shape)
            #  logits, dummy_loss
            # if current_epoch ==1:

            #     logger.debug("returning dummy loss for logging purpose {}", dummy_loss)
            #     logger.debug("returning gating weights {}", gating_weights)
            dummy_confidence = torch.zeros(x.size(0), device=logits.device)
            return {"logits": logits,
                    "model_type": "MoE",
                    "name": "PlugAndPlayExperts",
                    "expert_count": 3,
                    "curriculum_loss": dummy_loss,
                    "to_plot":{"gating_weights": gating_weights, "curriculum_loss":dummy_loss,
                               "confidence": dummy_confidence}}

        # ---- Stage 2: Gating ----
        logger.info(f"[Forward] Gating phase (epoch={current_epoch})")
        gating_logits = self.uncertainty(feats_adapt)
        gating_weights = F.softmax(gating_logits, dim=1)
        logger.debug(f"[Forward] Gating weights mean={gating_weights.mean(0).detach().cpu().numpy()}")

        e_a, e_b, e_c = [ex(feats_adapt) for ex in (self.expert_a, self.expert_b, self.expert_c)]
        emb = e_a * gating_weights[:, 0:1] + e_b * gating_weights[:, 1:2] + e_c * gating_weights[:, 2:3]
        logits = self.classifier(emb)

        # ---- Curriculum loss ----
        with torch.no_grad():
            conf = F.softmax(logits, 1).max(1)[0]
            easy = (conf > 0.9).float()
            medium = ((conf >= 0.6) & (conf <= 0.9)).float()
            hard = (conf < 0.6).float()

        loss_easy = (gating_weights[:, 1] * easy + gating_weights[:, 2] * easy).mean()
        loss_medium = (gating_weights[:, 0] * medium + gating_weights[:, 2] * medium).mean()
        loss_hard = (gating_weights[:, 0] * hard + gating_weights[:, 1] * hard).mean()
        total_loss = loss_easy + loss_medium + loss_hard

        logger.debug(f"[Loss] easy={loss_easy:.6f}, medium={loss_medium:.6f}, hard={loss_hard:.6f}, total={total_loss:.6f}")


        return {"logits": logits,
                "model_type": "MoE",
                "name": "PlugAndPlayExperts",
                "expert_count": 3,
                "curriculum_loss": total_loss,
                "to_plot":{"gating_weights": gating_weights, "curriculum_loss":total_loss,
                           "confidence": conf}}

    # ============================================================
    def _print_summary(self):
        logger.info(f"[Summary] Backbone feat_ch={self._feat_ch}, shape={self._feat_shape}")
        if self.adapter:
            logger.info("[Summary] Adapter present.")
        for name, ex in zip(["A", "B", "C"], [self.expert_a, self.expert_b, self.expert_c]):
            logger.info(f"[Expert{name}] in_ch={ex.in_ch}, out_dim={ex.out_dim}, depth={ex.depth}, wide={ex.wide}, "
                        f"large_kernel={ex.large_kernel}")

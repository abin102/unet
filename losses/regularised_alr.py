# losses/regularised_alr.py
import torch
import torch.nn as nn
from typing import Optional
from .alr import alr_loss_fn as _alr_loss_fn

class RegularisedALRLoss(nn.Module):
    """
    Blended Cross-Entropy + ALR loss with optional entropy regularization on W.

    loss = (1 - alpha_alr) * CE + alpha_alr * ALR_scaled - lambda_ent * Ent(W)

    Forward returns (loss, info_dict) where info_dict contains:
      - loss_ce
      - loss_alr
      - loss_per_sample (ALR per-sample weighted loss entries)
      - loss_per_class_mean
      - pos_per_class
      - W_mean_per_class
      - W_top1_mean
      - W_entropy_mean
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
        reduction: str = "mean",
        alr_scale: float = 1.0,
        lambda_ent: float = 0.0,
        alpha_alr: float = 0.1,   # fraction of loss coming from ALR (0..1)
    ):
        super().__init__()
        self.class_weights = class_weights
        self.eps = eps
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction
        self.alr_scale = float(alr_scale)
        self.lambda_ent = float(lambda_ent)
        self.alpha_alr = float(alpha_alr)

    def _prepare_W(self, W: Optional[torch.Tensor], B: int, K: int, device, dtype):
        """
        Ensure W is [B,K], non-negative and row-normalized.
        If W is None, return uniform W.
        """
        if W is None:
            return torch.full((B, K), 1.0 / float(K), device=device, dtype=dtype)
        Wt = W.to(device).float().clamp(min=0.0)
        rs = Wt.sum(dim=1, keepdim=True)
        zero_mask = rs == 0
        if zero_mask.any():
            Wt = Wt + (zero_mask.float() / float(K))
            rs = Wt.sum(dim=1, keepdim=True)
        Wt = Wt / (rs + 1e-12)
        return Wt

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, W: Optional[torch.Tensor] = None):
        """
        Args:
          logits: [B, K]
          targets: [B] (ints) or [B,K] multi-hot
          W: [B, K] attention weights (may be None)
        Returns:
          (loss, info)
        """
        device = logits.device
        B, K = logits.shape

        # --- CE loss (standard single-label cross-entropy) ---
        # Trainer sends integer labels for CE path, ensure shape ()
        if targets.dim() > 1 and targets.size(1) == K:
            # multi-hot -> convert to class indices by argmax for CE
            targets_ce = targets.argmax(dim=1).view(-1)
        else:
            targets_ce = targets.view(-1)
        ce_loss = nn.functional.cross_entropy(logits, targets_ce, reduction=self.reduction)

        # --- ALR (functional) ---
        # call functional ALR with only the args it supports
        base_alr = _alr_loss_fn(
            logits=logits,
            targets=targets,
            W=W,
            class_weights=self.class_weights,
            eps=self.eps,
            reduction=self.reduction,
        )

        # apply scaling for ALR component
        if self.alr_scale != 1.0:
            base_alr = base_alr * float(self.alr_scale)

        # --- entropy regularizer on W (optional) ---
        ent = None
        Wt = self._prepare_W(W, B, K, device, logits.dtype)
        if self.lambda_ent and self.lambda_ent != 0.0:
            ent_per_sample = - (Wt * (Wt.clamp(min=1e-12).log())).sum(dim=1)
            ent = ent_per_sample.mean()  # scalar
            # subtract lambda * entropy (to encourage higher entropy)
            base_alr = base_alr - float(self.lambda_ent) * ent

        # --- Blend CE + ALR ---
        alpha = float(self.alpha_alr)
        loss = (1.0 - alpha) * ce_loss + alpha * base_alr

        # --- Diagnostics (mirror previous ALR info) ---
        with torch.no_grad():
            # y -> multi-hot
            if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
                labels = targets.view(-1).long()
                y = torch.zeros((labels.size(0), K), device=device, dtype=torch.float32)
                y.scatter_(1, labels.unsqueeze(1), 1.0)
            else:
                y = targets.float().to(device)

            probs = torch.sigmoid(logits)
            loss_per_entry = -(y * torch.log(probs + self.eps) + (1.0 - y) * torch.log(1.0 - probs + self.eps))  # [B,K]

            # use normalized Wt for diagnostics
            weighted = (Wt * (self.class_weights.view(1, K) if self.class_weights is not None else 1.0) * loss_per_entry)+1

            loss_per_class_mean = weighted.mean(dim=0)           # [K]
            pos_per_class = y.sum(dim=0)                         # [K]
            W_mean_per_class = Wt.mean(dim=0)                    # [K]
            W_top1_mean = float(Wt.max(dim=1).values.mean())
            W_entropy_mean = float((-(Wt * (Wt.clamp(1e-12).log())).sum(dim=1)).mean())

        info = {
            "loss_ce": ce_loss.detach() if torch.is_tensor(ce_loss) else ce_loss,
            "loss_alr": base_alr.detach() if torch.is_tensor(base_alr) else base_alr,
            "loss_per_sample": weighted.sum(dim=1).detach(),
            "loss_per_class_mean": loss_per_class_mean.detach(),
            "pos_per_class": pos_per_class.detach(),
            "W_mean_per_class": W_mean_per_class.detach(),
            "W_top1_mean": W_top1_mean,
            "W_entropy_mean": W_entropy_mean,
        }

        return loss, info

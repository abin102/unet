# loss/alr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def _ensure_multi_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer class labels (B,) or (B,1) into multi-hot (one-hot) (B, K).
    If targets is already float/binary with shape (B,K) it is returned as float tensor.
    """
    if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
        # single-label integer form -> one-hot
        labels = targets.view(-1).long()
        out = torch.zeros((labels.size(0), num_classes), device=targets.device, dtype=torch.float32)
        out.scatter_(1, labels.unsqueeze(1), 1.0)
        return out
    else:
        # assume already multi-hot (could be float or int)
        return targets.float()


def alr_loss_fn(logits: torch.Tensor,
                targets: torch.Tensor,
                W: torch.Tensor,
                class_weights: Optional[torch.Tensor] = None,
                eps: float = 1e-8,
                reduction: str = "mean") -> torch.Tensor:
    """
    Attention-Augmented Logistic Regression (ALR) loss (functional).

    Args:
      logits: [B, K] raw model outputs
      targets: [B, K] multi-hot (0/1) or [B] integer labels
      W: [B, K] per-sample per-class attention weights (softmax across K recommended)
      class_weights: optional [K] static multiplicative class weights (applied after W)
      eps: small constant for numeric stability
      reduction: "mean" | "sum" | "none"  ("none" returns per-sample loss vector [B])

    Returns:
      scalar loss (if reduction != "none") or tensor [B] of per-sample losses
    """
    B, K = logits.shape
    probs = torch.sigmoid(logits)

    y = _ensure_multi_hot(targets, num_classes=K)  # [B, K]
    # BCE per entry
    loss_per_entry = -(y * torch.log(probs + eps) + (1.0 - y) * torch.log(1.0 - probs + eps))  # [B,K]

    # optional static class weights
    if class_weights is not None:
        # ensure shape [K]
        cw = class_weights.view(1, K).to(logits.device).float()
    else:
        cw = 1.0

    # combine learned W and static class weights
    weighted = W * cw * loss_per_entry  # [B,K]

    loss_per_sample = weighted.sum(dim=1)  # [B]
    if reduction == "mean":
        return loss_per_sample.mean()
    elif reduction == "sum":
        return loss_per_sample.sum()
    elif reduction == "none":
        return loss_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class ALRLoss(nn.Module):
    """
    nn.Module wrapper around alr_loss_fn that returns diagnostics.

    Usage:
        criterion = ALRLoss(class_weights=None)
        loss, info = criterion(logits, targets, W)
        info is a dict containing per-sample loss and per-class diagnostics.

    Returns:
      loss: scalar tensor
      info: dict {
          'loss_per_sample': Tensor[B],
          'loss_per_class_mean': Tensor[K],
          'pos_per_class': Tensor[K],    # number of positive (target=1) per class in the batch
          'W_mean_per_class': Tensor[K], # mean W per class across batch
      }
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.class_weights = class_weights
        self.eps = eps
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
          logits: [B, K]
          targets: [B] (ints) or [B, K] multi-hot
          W: [B, K] learned attention weights

        Returns:
          loss (scalar or per-sample per reduction), info dict
        """
        B, K = logits.shape
        y = _ensure_multi_hot(targets, num_classes=K)

        # compute base losses
        probs = torch.sigmoid(logits)
        loss_per_entry = -(y * torch.log(probs + self.eps) + (1.0 - y) * torch.log(1.0 - probs + self.eps))  # [B,K]

        # class weights
        if self.class_weights is not None:
            cw = self.class_weights.view(1, K).to(logits.device).float()
        else:
            cw = 1.0

        weighted = W * cw * loss_per_entry  # [B,K]
        loss_per_sample = weighted.sum(dim=1)  # [B]

        # reduction
        if self.reduction == "mean":
            loss = loss_per_sample.mean()
        elif self.reduction == "sum":
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        # diagnostics
        with torch.no_grad():
            loss_per_class_mean = weighted.mean(dim=0)           # [K]
            pos_per_class = y.sum(dim=0)                         # [K]
            W_mean_per_class = W.mean(dim=0)                     # [K]

        info = {
            "loss_per_sample": loss_per_sample.detach(),
            "loss_per_class_mean": loss_per_class_mean.detach(),
            "pos_per_class": pos_per_class.detach(),
            "W_mean_per_class": W_mean_per_class.detach(),
        }
        return loss, info

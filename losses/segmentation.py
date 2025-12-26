import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(
        target, num_classes=probs.shape[1]
    ).permute(0, 3, 1, 2).float()

    intersection = (probs * target_oh).sum((2, 3))
    union = probs.sum((2, 3)) + target_oh.sum((2, 3))
    return 1.0 - ((2 * intersection + eps) / (union + eps)).mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, ignore_index=-100):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        return (
            self.ce_weight * self.ce(logits, target)
            + self.dice_weight * dice_loss(logits, target)
        )

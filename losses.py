import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(1),
            targets,
            reduction='none'
        )

        probs = torch.sigmoid(logits.squeeze(1))

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce

        return loss.mean()


class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha=0.35,
        beta=0.65,
        smooth=1.0
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits.squeeze(1))

        targets = targets.float()

        tp = (probs * targets).sum(dim=(1, 2))

        fp = ((1 - targets) * probs).sum(dim=(1, 2))

        fn = (targets * (1 - probs)).sum(dim=(1, 2))

        tversky = (
            tp + self.smooth
        ) / (
            tp
            + self.alpha * fp
            + self.beta * fn
            + self.smooth
        )

        return 1 - tversky.mean()


class CombinedLoss(nn.Module):

    def __init__(
        self,
        focal_weight=0.5,
        tversky_weight=0.5,
        alpha=0.7,
        beta=0.3
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

        self.focal = smp.losses.FocalLoss(
            mode="binary"
        )

        self.tversky = smp.losses.TverskyLoss(
            mode="binary",
            alpha=alpha,
            beta=beta
        )

    def forward(self, logits, targets):

        focal_loss = self.focal(logits, targets)

        tversky_loss = self.tversky(logits, targets)

        return (
            self.focal_weight * focal_loss
            + self.tversky_weight * tversky_loss
        )
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_pos_weight = pos_weight is not None
        if self.use_pos_weight:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        else:
            self.register_buffer("pos_weight", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    def forward(self, inputs, targets, smooth=1):
        # Flatten the inputs and targets
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Calculate Binary Cross Entropy Loss
        bce_kwargs = {}
        if self.use_pos_weight:
            bce_kwargs["pos_weight"] = self.pos_weight.to(inputs.device)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, **bce_kwargs)

        # Calculate Dice Loss
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        # Combine BCE and Dice Loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        probs = torch.sigmoid(inputs)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1 - tversky_index) ** self.gamma


class FocalTverskyLoss(TverskyLoss):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)


def build_loss(
    loss_name: str = "bce_dice",
    pos_weight: float | None = None,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_gamma: float = 1.33,
):
    normalized = loss_name.strip().lower().replace("-", "_")

    if normalized == "bce_dice":
        return BCEDiceLoss(pos_weight=pos_weight)
    if normalized == "tversky":
        return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=1.0)
    if normalized == "focal_tversky":
        return FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
        )

    raise ValueError(
        f"Unsupported loss_name: {loss_name}. "
        "Choose from 'bce_dice', 'tversky', or 'focal_tversky'."
    )

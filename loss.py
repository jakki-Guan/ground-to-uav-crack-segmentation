import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # Calculate Dice Loss
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        # Combine BCE and Dice Loss
        total_loss = bce_loss + dice_loss

        return total_loss
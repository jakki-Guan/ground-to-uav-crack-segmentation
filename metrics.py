import torch

def iou_score(preds, targets, threshold=0.5,eps=1e-6):
    """
    Calculate the Intersection over Union (IoU) score.

    Args:
        preds (torch.Tensor): Predicted probabilities (logits) of shape (N, C, H, W).
        targets (torch.Tensor): Ground truth binary masks of shape (N, C, H, W).
        threshold (float): Threshold to convert predicted probabilities to binary masks.

    Returns:
        float: IoU score.
    """
    # Apply sigmoid to get probabilities and then threshold to get binary masks
    probs = torch.sigmoid(preds)
    preds_binary = (probs > threshold).float()

    # Flatten the tensors
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate intersection and union
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection

    # Calculate IoU
    iou = (intersection + eps) / (union + eps)

    return iou.item()

def f1_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    Calculate the F1 score.

    Args:
        preds (torch.Tensor): Predicted probabilities (logits) of shape (N, C, H, W).
        targets (torch.Tensor): Ground truth binary masks of shape (N, C, H, W).
        threshold (float): Threshold to convert predicted probabilities to binary masks.

    Returns:
        float: F1 score.
    """
    # Apply sigmoid to get probabilities and then threshold to get binary masks
    probs = torch.sigmoid(preds)
    preds_binary = (probs > threshold).float()

    # Flatten the tensors
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate true positives, false positives, and false negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    # Calculate F1 score
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)


    return f1.item()
import torch

from postprocess import ConnectedComponentPostprocessConfig, logits_to_binary_mask


def confusion_stats(
    preds,
    targets,
    threshold: float = 0.5,
    eps: float = 1e-6,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
):
    preds_binary = logits_to_binary_mask(
        logits=preds,
        threshold=threshold,
        postprocess_config=postprocess_config,
    )

    preds_flat = preds_binary.reshape(-1)
    targets_flat = targets.reshape(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }


def precision_score(
    preds,
    targets,
    threshold: float = 0.5,
    eps: float = 1e-6,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
):
    return confusion_stats(
        preds,
        targets,
        threshold=threshold,
        eps=eps,
        postprocess_config=postprocess_config,
    )["precision"]


def recall_score(
    preds,
    targets,
    threshold: float = 0.5,
    eps: float = 1e-6,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
):
    return confusion_stats(
        preds,
        targets,
        threshold=threshold,
        eps=eps,
        postprocess_config=postprocess_config,
    )["recall"]


def iou_score(
    preds,
    targets,
    threshold: float = 0.5,
    eps: float = 1e-6,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
):
    return confusion_stats(
        preds,
        targets,
        threshold=threshold,
        eps=eps,
        postprocess_config=postprocess_config,
    )["iou"]


def f1_score(
    preds,
    targets,
    threshold: float = 0.5,
    eps: float = 1e-6,
    postprocess_config: ConnectedComponentPostprocessConfig | None = None,
):
    return confusion_stats(
        preds,
        targets,
        threshold=threshold,
        eps=eps,
        postprocess_config=postprocess_config,
    )["f1"]

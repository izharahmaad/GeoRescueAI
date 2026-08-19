"""Segmentation metrics and confusion-matrix utilities."""
from __future__ import annotations

import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    indices = target[valid] * num_classes + pred[valid]
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics_from_confusion(cm: torch.Tensor) -> dict[str, float | list[float]]:
    cm = cm.to(torch.float64)
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = torch.where(denom_iou > 0, tp / denom_iou, torch.zeros_like(tp))
    dice = torch.where(denom_dice > 0, 2 * tp / denom_dice, torch.zeros_like(tp))
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    valid_iou = denom_iou > 0
    return {
        "per_class_iou": iou.tolist(),
        "miou": float(iou[valid_iou].mean()) if valid_iou.any() else 0.0,
        "dice": float(dice.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
    }

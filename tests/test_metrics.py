import torch

from georescue.metrics import confusion_matrix, metrics_from_confusion


def test_confusion_matrix_and_metrics() -> None:
    target = torch.tensor([[0, 1], [1, 0]])
    pred = torch.tensor([[0, 1], [0, 0]])
    cm = confusion_matrix(pred, target, 2)
    assert cm.tolist() == [[2, 0], [1, 1]]
    result = metrics_from_confusion(cm)
    assert 0 <= result["miou"] <= 1
    assert 0 <= result["dice"] <= 1


def test_perfect_segmentation() -> None:
    target = torch.tensor([[0, 1], [1, 0]])
    cm = confusion_matrix(target, target, 2)
    result = metrics_from_confusion(cm)
    assert result["miou"] == 1.0
    assert result["dice"] == 1.0

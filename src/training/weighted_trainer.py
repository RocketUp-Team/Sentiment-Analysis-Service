"""Custom Trainer with class-weighted CrossEntropyLoss for imbalanced datasets.

The standard HuggingFace Trainer uses unweighted CrossEntropyLoss, which
causes the model to under-predict minority classes when training data is
imbalanced.  ``WeightedTrainer`` overrides ``compute_loss`` to apply
per-class weights inversely proportional to class frequency.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import Trainer


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from an array of integer labels.

    Weights are computed as ``n_samples / (n_classes * count_per_class)``
    (the *balanced* formula used by scikit-learn).  Classes absent from
    *labels* receive a weight of ``1.0`` so that they neither dominate
    nor vanish.

    Returns a 1-D float32 tensor of shape ``(num_classes,)``.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    n_samples = len(labels)

    weights = np.ones(num_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = n_samples / (num_classes * counts[nonzero])

    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """HuggingFace Trainer with class-weighted CrossEntropyLoss.

    Parameters
    ----------
    class_weights : torch.Tensor
        1-D tensor of per-class weights (length == ``num_labels``).
        Created by :func:`compute_class_weights`.
    *args, **kwargs
        Forwarded to the base ``Trainer``.
    """

    def __init__(self, *, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self._class_weights.to(logits.device),
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

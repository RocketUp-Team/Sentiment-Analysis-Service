"""Class weight computation for imbalanced datasets.

Tính class weights dựa trên phân phối nhãn thực tế trong training set,
sử dụng chiến lược 'balanced' của sklearn — weight tỷ lệ nghịch với tần suất.

    w_i = n_total / (n_classes * n_samples_in_class_i)

Weights được trả về dưới dạng ``torch.Tensor`` sẵn sàng truyền vào
``torch.nn.CrossEntropyLoss(weight=...)``
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def compute_class_weights(
    labels: Sequence[int],
    *,
    num_labels: int,
) -> torch.Tensor:
    """Tính class weights từ danh sách integer labels trong training set.

    Args:
        labels:     Danh sách integer label (0-indexed) của toàn bộ training set.
        num_labels: Tổng số class (xác định kích thước tensor output).

    Returns:
        FloatTensor shape ``(num_labels,)`` — phần tử thứ i là weight của class i.
        Nếu một class không xuất hiện trong ``labels``, weight của nó là 1.0
        (neutral — không penalize, không reward).

    Example::

        >>> weights = compute_class_weights([0, 0, 0, 1, 1, 2], num_labels=3)
        # class 0: nhiều nhất → weight thấp nhất
        # class 2: ít nhất    → weight cao nhất
    """
    labels_array = np.array(labels, dtype=int)
    present_classes = np.unique(labels_array)

    weights_np = np.ones(num_labels, dtype=np.float32)

    if len(present_classes) < 2:
        # Degenerate case: chỉ có 1 class, giữ nguyên weight = 1
        return torch.from_numpy(weights_np)

    n_total = len(labels_array)
    for cls_idx in present_classes:
        n_cls = int((labels_array == cls_idx).sum())
        weights_np[cls_idx] = n_total / (len(present_classes) * n_cls)

    return torch.from_numpy(weights_np)

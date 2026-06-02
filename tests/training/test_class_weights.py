"""Unit tests for src/training/class_weights.py"""

from __future__ import annotations

import pytest
import torch

from src.training.class_weights import compute_class_weights


class TestComputeClassWeights:
    """compute_class_weights(labels, num_labels)"""

    def test_balanced_two_class_returns_equal_weights(self):
        # 50% class 0, 50% class 1 → weights bằng nhau
        labels = [0, 0, 0, 1, 1, 1]
        weights = compute_class_weights(labels, num_labels=2)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)
        # Khi balanced: weight = n/(n_classes * n_cls) = 6/(2*3) = 1.0
        assert weights[0] == pytest.approx(1.0, rel=1e-4)
        assert weights[1] == pytest.approx(1.0, rel=1e-4)

    def test_imbalanced_minority_gets_higher_weight(self):
        # 80% class 0, 20% class 1 → class 1 phải có weight cao hơn
        labels = [0] * 80 + [1] * 20
        weights = compute_class_weights(labels, num_labels=2)

        assert weights[1] > weights[0]

    def test_severely_imbalanced_three_class(self):
        # Mô phỏng UIT-VSFC: positive 50%, negative 46%, neutral 4%
        pos_count = 500
        neg_count = 460
        neu_count = 40
        labels = [2] * pos_count + [0] * neg_count + [1] * neu_count  # 0=neg, 1=neu, 2=pos
        weights = compute_class_weights(labels, num_labels=3)

        # neutral (class 1) phải có weight cao nhất
        assert weights[1] > weights[2]
        assert weights[1] > weights[0]
        # positive và negative gần nhau (chênh lệch nhỏ)
        assert abs(weights[2].item() - weights[0].item()) < 0.5

    def test_weight_formula_is_inverse_frequency(self):
        # w_i = n_total / (n_classes * n_cls_i)
        labels = [0, 0, 0, 0, 1, 1]  # 4 vs 2 → IR=2x
        weights = compute_class_weights(labels, num_labels=2)

        n_total = 6
        n_classes = 2
        expected_w0 = n_total / (n_classes * 4)  # = 0.75
        expected_w1 = n_total / (n_classes * 2)  # = 1.50

        assert weights[0] == pytest.approx(expected_w0, rel=1e-4)
        assert weights[1] == pytest.approx(expected_w1, rel=1e-4)

    def test_absent_class_gets_neutral_weight_of_one(self):
        # Class 2 không có sample → weight = 1.0 (không penalize, không reward)
        labels = [0, 0, 1, 1]
        weights = compute_class_weights(labels, num_labels=3)

        assert weights[2] == pytest.approx(1.0, rel=1e-4)

    def test_num_labels_determines_output_size(self):
        labels = [0, 1]
        weights = compute_class_weights(labels, num_labels=5)

        assert weights.shape == (5,)

    def test_single_class_returns_all_ones(self):
        labels = [0, 0, 0]
        weights = compute_class_weights(labels, num_labels=2)

        # Degenerate case: chỉ có 1 class → giữ nguyên weight=1
        assert (weights == 1.0).all()

    def test_dtype_is_float32(self):
        labels = [0, 1, 2]
        weights = compute_class_weights(labels, num_labels=3)

        assert weights.dtype == torch.float32

    def test_large_imbalance_ratio_sentiment_vi(self):
        # Mô phỏng thực tế UIT-VSFC (IR ~11.5x)
        labels = [2] * 5643 + [0] * 5325 + [1] * 458  # VIT train split sizes
        weights = compute_class_weights(labels, num_labels=3)

        # neutral (1) phải có weight ít nhất 8x so với positive (2)
        ratio = weights[1].item() / weights[2].item()
        assert ratio > 8.0

"""Tests for WeightedTrainer and compute_class_weights."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.weighted_trainer import WeightedTrainer, compute_class_weights


class TestComputeClassWeights:
    """Unit tests for compute_class_weights."""

    def test_balanced_dataset_returns_uniform_weights(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        weights = compute_class_weights(labels, num_classes=3)

        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        np.testing.assert_allclose(weights.numpy(), [1.0, 1.0, 1.0], atol=1e-6)

    def test_imbalanced_dataset_upweights_minority(self):
        # 2 negative, 3 neutral, 10 positive → positive is majority
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        weights = compute_class_weights(labels, num_classes=3)

        assert weights[0] > weights[1] > weights[2]
        # negative (count=2): 15 / (3*2)  = 2.5
        # neutral  (count=3): 15 / (3*3)  ≈ 1.667
        # positive (count=10): 15 / (3*10) = 0.5
        np.testing.assert_allclose(
            weights.numpy(), [2.5, 15 / 9, 0.5], atol=1e-5
        )

    def test_missing_class_gets_default_weight(self):
        labels = np.array([0, 0, 2, 2])  # class 1 is absent
        weights = compute_class_weights(labels, num_classes=3)

        assert weights[1] == 1.0  # default for absent class
        # present classes: 4 / (3*2) = 0.667
        np.testing.assert_allclose(weights[0].item(), 4 / (3 * 2), atol=1e-6)
        np.testing.assert_allclose(weights[2].item(), 4 / (3 * 2), atol=1e-6)

    def test_single_class_produces_single_weight(self):
        labels = np.array([0, 0, 0])
        weights = compute_class_weights(labels, num_classes=1)

        assert weights.shape == (1,)
        np.testing.assert_allclose(weights.numpy(), [1.0], atol=1e-6)

    def test_returns_tensor_type(self):
        labels = np.array([0, 1])
        weights = compute_class_weights(labels, num_classes=2)

        assert isinstance(weights, torch.Tensor)


class TestWeightedTrainer:
    """Integration-style tests for WeightedTrainer.compute_loss."""

    @pytest.fixture()
    def class_weights(self):
        return torch.tensor([2.5, 1.0, 0.5], dtype=torch.float32)

    def test_compute_loss_applies_class_weights(self, class_weights):
        class FakeModel:
            def __call__(self, **kwargs):
                class Output:
                    logits = torch.tensor(
                        [[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]],
                        dtype=torch.float32,
                    )

                return Output()

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer._class_weights = class_weights

        inputs = {
            "labels": torch.tensor([1, 0]),
            "input_ids": torch.tensor([[1], [2]]),
        }
        loss = trainer.compute_loss(FakeModel(), inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

        # Verify loss differs from unweighted version
        unweighted_fn = torch.nn.CrossEntropyLoss()
        logits = torch.tensor(
            [[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]], dtype=torch.float32
        )
        unweighted_loss = unweighted_fn(logits, torch.tensor([1, 0]))
        assert not torch.allclose(loss, unweighted_loss)

    def test_compute_loss_return_outputs(self, class_weights):
        class FakeOutput:
            logits = torch.tensor(
                [[0.1, 0.8, 0.1]], dtype=torch.float32
            )

        class FakeModel:
            def __call__(self, **kwargs):
                return FakeOutput()

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer._class_weights = class_weights

        inputs = {
            "labels": torch.tensor([1]),
            "input_ids": torch.tensor([[1]]),
        }
        result = trainer.compute_loss(FakeModel(), inputs, return_outputs=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, outputs = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(outputs, FakeOutput)

    def test_labels_removed_from_inputs(self, class_weights):
        """Verify labels are popped so they don't reach the model forward pass."""

        class FakeModel:
            def __call__(self, **kwargs):
                assert "labels" not in kwargs, "labels should be popped from inputs"

                class Output:
                    logits = torch.tensor([[0.1, 0.8, 0.1]], dtype=torch.float32)

                return Output()

        trainer = WeightedTrainer.__new__(WeightedTrainer)
        trainer._class_weights = class_weights

        inputs = {
            "labels": torch.tensor([0]),
            "input_ids": torch.tensor([[1]]),
        }
        trainer.compute_loss(FakeModel(), inputs)

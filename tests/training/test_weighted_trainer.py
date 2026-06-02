"""Unit tests for src/training/weighted_trainer.py"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.weighted_trainer import WeightedLossTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Fakes / Stubs
# ─────────────────────────────────────────────────────────────────────────────

class FakeOutputs:
    """Minimal stand-in for HuggingFace model outputs."""
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    """Forward pass returns FakeOutputs with fixed logits."""
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def __call__(self, **kwargs) -> FakeOutputs:
        return FakeOutputs(self._logits)


class FakeTrainingArguments:
    """Minimal TrainingArguments stub (WeightedLossTrainer only reads .device)."""
    # We don't actually use args in compute_loss, so an empty object is fine.
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_trainer(
    num_labels: int = 3,
    weights: list[float] | None = None,
) -> WeightedLossTrainer:
    """Build a WeightedLossTrainer bypassing heavy __init__ via __new__."""
    if weights is None:
        weights = [1.0] * num_labels
    trainer = object.__new__(WeightedLossTrainer)
    trainer._class_weights = torch.tensor(weights, dtype=torch.float32)
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedLossTrainerComputeLoss:
    """WeightedLossTrainer.compute_loss()"""

    def test_returns_scalar_loss_tensor(self):
        batch_size, num_labels = 4, 3
        logits = torch.randn(batch_size, num_labels)
        labels = torch.tensor([0, 1, 2, 0])
        inputs = {"input_ids": torch.ones(batch_size, 10, dtype=torch.long), "labels": labels}

        trainer = _build_trainer(num_labels=num_labels)
        loss = trainer.compute_loss(FakeModel(logits), inputs, return_outputs=False)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_return_outputs_true_returns_tuple(self):
        batch_size, num_labels = 2, 3
        logits = torch.randn(batch_size, num_labels)
        labels = torch.tensor([0, 1])
        inputs = {"labels": labels}

        trainer = _build_trainer(num_labels=num_labels)
        result = trainer.compute_loss(FakeModel(logits), inputs, return_outputs=True)

        assert isinstance(result, tuple)
        loss, outputs = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(outputs, FakeOutputs)

    def test_weighted_loss_differs_from_unweighted_for_imbalanced_data(self):
        """Minority class gets higher weight \u2192 total weighted loss differs from uniform.

        NOTE: A single-sample batch is insufficient here because CrossEntropyLoss
        with reduction='mean' normalises by weight[label], cancelling the weight
        for a lone sample.  A two-sample batch with mixed labels avoids that.
        """
        # sample 0: strongly predicts class 0 (correct, label=0)
        # sample 1: strongly predicts class 0 (WRONG, true label=1 = minority)
        logits = torch.tensor([
            [5.0, -5.0, -5.0],
            [5.0, -5.0, -5.0],
        ])
        labels = torch.tensor([0, 1])
        inputs = {"labels": labels}

        # Uniform weights: both classes treated identically
        unweighted_trainer = _build_trainer(num_labels=3, weights=[1.0, 1.0, 1.0])
        loss_unweighted = unweighted_trainer.compute_loss(
            FakeModel(logits), inputs.copy(), return_outputs=False
        )

        # Minority boost: the wrong sample (class 1) penalised 5\u00d7 harder
        weighted_trainer = _build_trainer(num_labels=3, weights=[1.0, 5.0, 1.0])
        loss_weighted = weighted_trainer.compute_loss(
            FakeModel(logits), inputs.copy(), return_outputs=False
        )

        # Weighted loss is strictly greater: the high-weight wrong prediction
        # dominates the normalised mean more than with uniform weights.
        assert loss_weighted.item() > loss_unweighted.item()

    def test_matching_logits_and_labels_gives_low_loss(self):
        """Model dự đoán đúng hoàn toàn → loss gần 0."""
        # Class 1: logit rất cao
        logits = torch.tensor([[0.0, 100.0, 0.0]])
        labels = torch.tensor([1])
        inputs = {"labels": labels}

        trainer = _build_trainer(num_labels=3, weights=[1.0, 1.5, 1.0])
        loss = trainer.compute_loss(FakeModel(logits), inputs)

        assert loss.item() < 0.01

    def test_weights_moved_to_logit_device(self):
        """Class weights phải tự động di chuyển sang cùng device với logits."""
        logits = torch.randn(2, 3)  # CPU tensor
        labels = torch.tensor([0, 2])
        inputs = {"labels": labels}

        # Weights khởi tạo trên CPU (như thông thường)
        trainer = _build_trainer(num_labels=3, weights=[1.0, 2.0, 1.0])
        # Không nên raise RuntimeError về device mismatch
        loss = trainer.compute_loss(FakeModel(logits), inputs)

        assert loss.device == logits.device

    def test_labels_are_popped_from_inputs(self):
        """inputs dict phải không còn 'labels' sau khi compute_loss chạy."""
        logits = torch.randn(2, 3)
        labels = torch.tensor([0, 1])
        inputs = {"labels": labels, "input_ids": torch.ones(2, 5, dtype=torch.long)}

        trainer = _build_trainer(num_labels=3)
        trainer.compute_loss(FakeModel(logits), inputs)

        assert "labels" not in inputs

    def test_agrees_with_manual_cross_entropy(self):
        """Loss phải khớp với torch.nn.CrossEntropyLoss(weight=...) tính thủ công."""
        weights_list = [0.5, 2.0, 1.0]
        logits = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.1, 2.5]])
        labels = torch.tensor([1, 2])
        inputs = {"labels": labels}

        trainer = _build_trainer(num_labels=3, weights=weights_list)
        our_loss = trainer.compute_loss(FakeModel(logits), inputs)

        ref_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(weights_list, dtype=torch.float32)
        )(logits, labels)

        assert our_loss.item() == pytest.approx(ref_loss.item(), rel=1e-5)


class TestWeightedLossTrainerIsSubclass:
    def test_is_subclass_of_trainer(self):
        from transformers import Trainer
        assert issubclass(WeightedLossTrainer, Trainer)

"""Tests for evaluation metrics computation — model predictions are mocked."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from contracts.model_interface import PredictionResult


# ── Test Data ──────────────────────────────────────────────────

def _make_test_df() -> pd.DataFrame:
    """Minimal sentences DataFrame with known labels."""
    return pd.DataFrame(
        {
            "text": [
                "The food was great",
                "Terrible service",
                "It was okay",
                "Amazing pasta",
                "Bad experience",
                "Normal place",
            ],
            "sentiment": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
            ],
            "split": ["test"] * 6,
        }
    )


def _make_mock_model(predictions: list[str]) -> MagicMock:
    """Create a mock model that returns fixed predictions."""
    mock = MagicMock()
    results = [
        PredictionResult(
            sentiment=s, confidence=0.9, aspects=[], sarcasm_flag=False
        )
        for s in predictions
    ]
    mock.predict_batch.return_value = results
    return mock


# ── evaluate_on_dataset ───────────────────────────────────────

class TestEvaluateOnDataset:
    def test_returns_early_on_empty_split(self):
        from src.model.evaluate import evaluate_on_dataset
        df = _make_test_df()
        mock_model = _make_mock_model([])
        # Try evaluating on a split that does not exist
        metrics = evaluate_on_dataset(mock_model, df, split="val", batch_size=32)
        assert metrics["n_samples"] == 0
        assert "error" in metrics

    def test_returns_required_metric_keys(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        # Perfect predictions
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)

        required_keys = {
            "split",
            "n_samples",
            "accuracy",
            "f1_macro",
            "f1_per_class",
            "precision_macro",
            "recall_macro",
            "mean_confidence",
            "classification_report",
            "confusion_matrix",
        }
        assert required_keys.issubset(metrics.keys())

    def test_perfect_predictions_accuracy_is_one(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_n_samples_matches_split(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        assert metrics["n_samples"] == 6

    def test_f1_per_class_has_three_values(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        assert len(metrics["f1_per_class"]) == 3

    def test_confusion_matrix_is_3x3(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_batching_works_with_small_batch_size(self):
        """Batch size smaller than dataset should still work."""
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds[:2])  # batch of 2
        # Need mock_model.predict_batch to handle multiple calls
        mock_model.predict_batch.side_effect = [
            [
                PredictionResult(
                    sentiment=s, confidence=0.9, aspects=[], sarcasm_flag=False
                )
                for s in preds[i : i + 2]
            ]
            for i in range(0, len(preds), 2)
        ]

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=2)
        assert metrics["n_samples"] == 6
        assert metrics["accuracy"] == pytest.approx(1.0)


# ── log_to_mlflow ─────────────────────────────────────────────

class TestLogToMlflow:
    @patch("src.model.evaluate.mlflow")
    def test_calls_mlflow_log_params_and_metrics(self, mock_mlflow):
        from src.model.evaluate import log_to_mlflow
        from src.model.config import ModelConfig

        config = ModelConfig()
        metrics = {
            "accuracy": 0.8,
            "f1_macro": 0.75,
            "f1_per_class": [0.7, 0.8, 0.75],
            "precision_macro": 0.76,
            "recall_macro": 0.74,
            "mean_confidence": 0.85,
            "n_samples": 100,
            "device": "cpu",
            "confusion_matrix": [[30, 5, 5], [3, 25, 2], [2, 3, 25]],
            "classification_report": "dummy report",
        }
        params_yaml = {
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "model_experiment_name": "test_exp",
            }
        }

        log_to_mlflow(config, metrics, params_yaml)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()

    @patch("src.model.evaluate.mlflow")
    def test_uses_default_experiment_name(self, mock_mlflow):
        from src.model.evaluate import log_to_mlflow
        from src.model.config import ModelConfig

        config = ModelConfig()
        metrics = {
            "accuracy": 0.8,
            "f1_macro": 0.75,
            "f1_per_class": [0.7, 0.8, 0.75],
            "precision_macro": 0.76,
            "recall_macro": 0.74,
            "mean_confidence": 0.85,
            "n_samples": 100,
            "device": "cpu",
            "confusion_matrix": [[30, 5, 5], [3, 25, 2], [2, 3, 25]],
            "classification_report": "dummy report",
        }
        params_yaml = {}  # No mlflow config

        log_to_mlflow(config, metrics, params_yaml)

        mock_mlflow.set_experiment.assert_called_once_with("sentiment_baseline")

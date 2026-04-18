"""Tests for evaluation metrics computation — model predictions are mocked."""
import json
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
        metrics = evaluate_on_dataset(mock_model, df, split="val")
        assert metrics["n_samples"] == 0
        assert "error" in metrics

    def test_returns_required_metric_keys(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        # Perfect predictions
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test")

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

        metrics = evaluate_on_dataset(mock_model, df, split="test")
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_n_samples_matches_split(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test")
        assert metrics["n_samples"] == 6

    def test_f1_per_class_has_three_values(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test")
        assert len(metrics["f1_per_class"]) == 3

    def test_confusion_matrix_is_3x3(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test")
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_evaluate_on_dataset_delegates_chunking_to_predict_batch(self):
        """evaluate_on_dataset should call predict_batch once with all texts."""
        from unittest.mock import MagicMock
        from src.model.evaluate import evaluate_on_dataset

        texts = [f"text {i}" for i in range(12)]
        df = pd.DataFrame({
            "text": texts,
            "sentiment": ["positive"] * 4 + ["negative"] * 4 + ["neutral"] * 4,
            "split": ["test"] * 12,
        })

        mock_model = MagicMock()
        mock_model.predict_batch.return_value = [
            PredictionResult(sentiment="positive", confidence=0.9) for _ in range(4)
        ] + [
            PredictionResult(sentiment="negative", confidence=0.9) for _ in range(4)
        ] + [
            PredictionResult(sentiment="neutral", confidence=0.9) for _ in range(4)
        ]

        evaluate_on_dataset(mock_model, df, split="test")

        mock_model.predict_batch.assert_called_once()
        call_args = mock_model.predict_batch.call_args
        assert call_args[0][0] == texts


    def test_raises_on_prediction_count_mismatch(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        mock_model = MagicMock()
        mock_model.predict_batch.return_value = [
            PredictionResult(
                sentiment="positive", confidence=0.9, aspects=[], sarcasm_flag=False
            )
        ]

        with pytest.raises(ValueError, match="Prediction count does not match"):
            evaluate_on_dataset(mock_model, df, split="test")


# ── _log_reporting_artifacts ───────────────────────────────────

class TestLogReportingArtifacts:
    def test_logs_three_artifacts(self):
        from src.model.evaluate import _log_reporting_artifacts

        mock_mlflow = MagicMock()
        metrics = {
            "confusion_matrix": [[30, 5, 5], [3, 25, 2], [2, 3, 25]],
            "classification_report": "dummy report",
            "accuracy": 0.8,
            "f1_macro": 0.75,
        }

        _log_reporting_artifacts(mock_mlflow, metrics)

        assert mock_mlflow.log_artifact.call_count == 3

    def test_artifact_filenames(self):
        from src.model.evaluate import _log_reporting_artifacts

        mock_mlflow = MagicMock()
        metrics = {
            "confusion_matrix": [[30, 5, 5], [3, 25, 2], [2, 3, 25]],
            "classification_report": "dummy report",
            "accuracy": 0.8,
            "f1_macro": 0.75,
        }

        _log_reporting_artifacts(mock_mlflow, metrics)

        logged_paths = [
            call.args[0] for call in mock_mlflow.log_artifact.call_args_list
        ]
        filenames = [p.rsplit("/", 1)[-1] for p in logged_paths]
        assert "confusion_matrix.png" in filenames
        assert "classification_report.txt" in filenames
        assert "metrics_summary.json" in filenames


# ── log_to_mlflow ─────────────────────────────────────────────

class TestLogToMlflow:
    def test_calls_mlflow_log_params_and_metrics(self):
        import src.model.evaluate as evaluate_module
        from src.model.config import ModelConfig

        config = ModelConfig()
        mock_mlflow = MagicMock()
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

        with patch.object(
            evaluate_module.importlib, "import_module", return_value=mock_mlflow
        ) as mock_import_module, patch.object(
            evaluate_module, "_log_reporting_artifacts"
        ) as mock_log_reporting_artifacts:
            evaluate_module.log_to_mlflow(config, metrics, params_yaml)

        mock_import_module.assert_called_once_with("mlflow")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_log_reporting_artifacts.assert_called_once_with(mock_mlflow, metrics)

    def test_uses_default_experiment_name(self):
        import src.model.evaluate as evaluate_module
        from src.model.config import ModelConfig

        config = ModelConfig()
        mock_mlflow = MagicMock()
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

        with patch.object(
            evaluate_module.importlib, "import_module", return_value=mock_mlflow
        ), patch.object(evaluate_module, "_log_reporting_artifacts") as mock_log_reporting_artifacts:
            evaluate_module.log_to_mlflow(config, metrics, params_yaml)

        mock_mlflow.set_experiment.assert_called_once_with("sentiment_baseline")
        mock_log_reporting_artifacts.assert_called_once_with(mock_mlflow, metrics)


class TestMainCli:
    @patch("src.model.evaluate.pd.read_csv")
    @patch("src.model.baseline.BaselineModelInference")
    @patch("src.model.evaluate.ModelConfig")
    @patch("src.model.evaluate.load_params", return_value={})
    @patch("src.model.evaluate._project_root")
    def test_returns_error_when_processed_sentences_missing(
        self,
        mock_project_root,
        mock_load_params,
        mock_model_config,
        mock_baseline_model,
        mock_read_csv,
        tmp_path,
    ):
        from src.model.evaluate import main

        mock_project_root.return_value = tmp_path

        result = main()

        assert result == 1
        mock_load_params.assert_called_once()
        mock_model_config.assert_not_called()
        mock_baseline_model.assert_not_called()
        mock_read_csv.assert_not_called()

    @patch("src.model.evaluate.log_to_mlflow")
    @patch("src.model.evaluate.evaluate_on_dataset")
    @patch("src.model.baseline.BaselineModelInference")
    @patch("src.model.evaluate.ModelConfig")
    @patch("src.model.evaluate.load_params", return_value={})
    @patch("src.model.evaluate._project_root")
    def test_writes_baseline_metrics_report_on_success(
        self,
        mock_project_root,
        mock_load_params,
        mock_model_config,
        mock_baseline_model,
        mock_evaluate_on_dataset,
        mock_log_to_mlflow,
        tmp_path,
    ):
        from src.model.evaluate import main

        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        pd.DataFrame(
            {"text": ["ok"], "sentiment": ["positive"], "split": ["test"]}
        ).to_csv(processed_dir / "sentences.csv", index=False)

        mock_project_root.return_value = tmp_path
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_baseline_model.return_value = mock_model
        metrics = {
            "split": "test",
            "n_samples": 1,
            "accuracy": 1.0,
            "f1_macro": 1.0,
            "f1_per_class": [1.0, 1.0, 1.0],
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "mean_confidence": 0.9,
            "classification_report": "dummy report",
            "confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        mock_evaluate_on_dataset.return_value = metrics

        result = main()

        report_path = tmp_path / "data" / "reports" / "baseline_metrics.json"
        assert result == 0
        assert report_path.exists()
        assert json.loads(report_path.read_text(encoding="utf-8"))["accuracy"] == 1.0
        mock_log_to_mlflow.assert_called_once()

    @patch("src.model.evaluate.log_to_mlflow", side_effect=RuntimeError("mlflow down"))
    @patch("src.model.evaluate.evaluate_on_dataset")
    @patch("src.model.baseline.BaselineModelInference")
    @patch("src.model.evaluate.ModelConfig")
    @patch("src.model.evaluate.load_params", return_value={})
    @patch("src.model.evaluate._project_root")
    def test_returns_success_when_mlflow_logging_fails(
        self,
        mock_project_root,
        mock_load_params,
        mock_model_config,
        mock_baseline_model,
        mock_evaluate_on_dataset,
        mock_log_to_mlflow,
        tmp_path,
        caplog,
    ):
        from src.model.evaluate import main

        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        pd.DataFrame(
            {"text": ["ok"], "sentiment": ["positive"], "split": ["test"]}
        ).to_csv(processed_dir / "sentences.csv", index=False)

        mock_project_root.return_value = tmp_path
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_baseline_model.return_value = mock_model
        mock_evaluate_on_dataset.return_value = {
            "split": "test",
            "n_samples": 1,
            "accuracy": 1.0,
            "f1_macro": 1.0,
            "f1_per_class": [1.0, 1.0, 1.0],
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "mean_confidence": 0.9,
            "classification_report": "dummy report",
            "confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        }

        result = main()

        report_path = tmp_path / "data" / "reports" / "baseline_metrics.json"
        assert result == 0
        assert report_path.exists()
        assert any("MLflow logging failed" in record.message for record in caplog.records)
        mock_log_to_mlflow.assert_called_once()

"""CLI script: evaluate baseline model on processed data and log results to MLflow.

Usage: python -m src.model.evaluate
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from contracts.model_interface import ModelInference
from src.data.utils import load_params
from src.model.config import ModelConfig
from src.training.mlflow_callback import resolve_pipeline_tracking_uri

logger = logging.getLogger(__name__)

LABELS = ["positive", "negative", "neutral"]


def save_metrics_report(metrics: dict, output_path: Path) -> None:
    """Persist evaluation metrics to a JSON report file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def evaluate_on_dataset(
    model: ModelInference,
    sentences_df: pd.DataFrame,
    split: str = "test",
) -> dict:
    """Evaluate a model on a named dataset split."""
    split_df = sentences_df[sentences_df["split"] == split].copy()
    if split_df.empty:
        return {
            "split": split,
            "n_samples": 0,
            "error": f"No samples found for split '{split}'",
        }

    texts = split_df["text"].astype(str).tolist()
    true_labels = split_df["sentiment"].astype(str).tolist()

    batch_results = model.predict_batch(texts, skip_absa=True)
    predictions = [result.sentiment for result in batch_results]
    confidences = [float(result.confidence) for result in batch_results]

    if len(predictions) != len(true_labels):
        raise ValueError(
            "Prediction count does not match ground truth size: "
            f"{len(predictions)} != {len(true_labels)}"
        )

    return {
        "split": split,
        "n_samples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "f1_macro": float(
            f1_score(true_labels, predictions, labels=LABELS, average="macro")
        ),
        "f1_per_class": f1_score(
            true_labels,
            predictions,
            labels=LABELS,
            average=None,
        ).astype(float).tolist(),
        "precision_macro": float(
            precision_score(
                true_labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true_labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "mean_confidence": float(np.mean(confidences)),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=LABELS,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predictions,
            labels=LABELS,
        ).tolist(),
    }


def _log_reporting_artifacts(mlflow_client, metrics: dict) -> None:
    """Render and upload evaluation artifacts for an MLflow run."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        figure, axis = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(metrics["confusion_matrix"]),
            display_labels=LABELS,
        ).plot(ax=axis)
        axis.set_title("Baseline Model Confusion Matrix")
        cm_path = temp_path / "confusion_matrix.png"
        figure.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        mlflow_client.log_artifact(str(cm_path))

        report_path = temp_path / "classification_report.txt"
        report_path.write_text(metrics["classification_report"], encoding="utf-8")
        mlflow_client.log_artifact(str(report_path))

        summary_path = temp_path / "metrics_summary.json"
        summary_path.write_text(
            json.dumps(
                {"metrics": metrics},
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        mlflow_client.log_artifact(str(summary_path))


def log_to_mlflow(config: ModelConfig, metrics: dict, params_yaml: dict) -> None:
    """Log baseline-model evaluation params, metrics, and artifacts to MLflow.

    URI priority:
      1. MLFLOW_TRACKING_URI env var  (docker-compose sets http://mlflow:5000,
                                       local .env sets DagsHub URL)
      2. params.yaml  mlflow.tracking_uri
      3. hardcoded fallback http://localhost:5000
    """
    import os
    mlflow_client = importlib.import_module("mlflow")
    mlflow_config = params_yaml.get("mlflow", {})
    tracking_uri = resolve_pipeline_tracking_uri(mlflow_config)
    experiment_name = mlflow_config.get(
        "model_experiment_name",
        "sentiment_baseline",
    )

    mlflow_client.set_tracking_uri(tracking_uri)
    mlflow_client.set_experiment(experiment_name)

    run_params = {
        "model_name": config.model_name,
        "model_type": "baseline_pretrained",
        "max_length": config.max_length,
        "device": str(metrics.get("device", "cpu")),
        "fine_tuned": False,
        "absa_enabled": False,
    }
    run_metrics = {
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "n_samples": int(metrics["n_samples"]),
    }

    with mlflow_client.start_run(run_name="baseline_roberta"):
        mlflow_client.log_params(run_params)
        mlflow_client.log_metrics(run_metrics)

        for index, label in enumerate(LABELS):
            mlflow_client.log_metric(
                f"f1_{label}",
                float(metrics["f1_per_class"][index]),
            )

        _log_reporting_artifacts(mlflow_client, metrics)

    logger.info(
        "Logged MLflow evaluation run with accuracy=%.4f f1_macro=%.4f",
        run_metrics["accuracy"],
        run_metrics["f1_macro"],
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    """Evaluate the baseline model on processed test data and log to MLflow."""
    from src.model.baseline import BaselineModelInference

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    root = _project_root()
    params = load_params(str(root / "params.yaml"))

    sentences_path = root / "data" / "processed" / "sentences.csv"
    if not sentences_path.exists():
        logger.error("Processed data not found at %s", sentences_path)
        return 1

    config = ModelConfig()
    model = BaselineModelInference(config)
    sentences_df = pd.read_csv(sentences_path)
    metrics = evaluate_on_dataset(model, sentences_df, split="test")
    if metrics.get("n_samples", 0) == 0:
        logger.error(metrics.get("error", "Evaluation produced no samples."))
        return 1

    metrics["device"] = str(model.device)
    save_metrics_report(metrics, root / "data" / "reports" / "baseline_metrics.json")
    try:
        log_to_mlflow(config, metrics, params)
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)

    print(f"\n{'=' * 50}")
    print("Baseline Evaluation Results (test split)")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"\n{metrics['classification_report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

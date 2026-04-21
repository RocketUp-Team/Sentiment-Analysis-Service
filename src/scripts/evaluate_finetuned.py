"""CLI entrypoint for evaluating finetuned Phase 2 adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sklearn.metrics import classification_report

import pandas as pd

from src.model.baseline import BaselineModelInference
from src.model.config import ModelConfig
from src.training.metrics import build_metrics_payload
from src.training.task_configs import get_task_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned Phase 2 adapters.")
    parser.add_argument(
        "--task",
        required=True,
        choices=("sarcasm", "sentiment"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics_finetuned.json"),
    )
    return parser.parse_args(argv)


def _load_evaluation_frame(task_name: str, root: Path) -> pd.DataFrame:
    if task_name == "sarcasm":
        return pd.read_csv(root / "data" / "raw" / "sarcasm.csv")

    df_en = pd.read_csv(root / "data" / "raw" / "sentiment_en.csv")
    df_vi = pd.read_csv(root / "data" / "raw" / "sentiment_vi.csv")
    return pd.concat([df_en, df_vi], ignore_index=True)


def _select_evaluation_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df:
        return df
    return df[df["split"].astype(str) == "test"].copy()


def _resolve_languages(df: pd.DataFrame) -> list[str]:
    if "lang" in df:
        return df["lang"].fillna("en").astype(str).tolist()
    return ["en"] * len(df)


def _resolve_true_label(label_value, label_names: tuple[str, ...]) -> str:
    if isinstance(label_value, str) and label_value in label_names:
        return label_value
    return label_names[int(label_value)]


def evaluate(
    task_name: str,
    *,
    root: Path | None = None,
    max_samples: int | None = None,
) -> dict:
    """Run evaluation and return full metrics payload.

    MLflow context is managed by the CALLER if logging is desired.

    Returns dict with keys from build_metrics_payload() plus:
    - y_true: list[str] — ground truth labels
    - y_pred: list[str] — predicted labels
    """
    root = root or Path(__file__).resolve().parents[2]
    task = get_task_config(task_name)

    config = ModelConfig(
        mode="finetuned",
        sentiment_adapter_path=str(root / "models" / "adapters" / "sentiment"),
        sarcasm_adapter_path=str(root / "models" / "adapters" / "sarcasm"),
    )
    inference = BaselineModelInference(config=config)
    df = _select_evaluation_rows(_load_evaluation_frame(task_name, root))

    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    texts = df["text"].tolist()
    languages = _resolve_languages(df)

    results = inference.predict_batch(texts, lang="en", skip_absa=True)

    y_pred: list[str] = []
    y_true: list[str] = []
    for row, result in zip(df.itertuples(index=False), results, strict=True):
        if task_name == "sarcasm":
            y_pred.append(task.label_names[1] if result.sarcasm_flag else task.label_names[0])
        else:
            y_pred.append(result.sentiment)
        y_true.append(_resolve_true_label(row.label, task.label_names))

    metrics_payload = build_metrics_payload(
        y_true=y_true,
        y_pred=y_pred,
        languages=languages,
        label_names=task.label_names,
    )
    metrics_payload["y_true"] = y_true
    metrics_payload["y_pred"] = y_pred
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper — writes JSON reports. Backwards-compatible with DVC."""
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = evaluate(args.task, root=root, max_samples=100)

    args.output.write_text(
        json.dumps(
            {
                "task": args.task,
                "overall_f1": metrics_payload["overall_f1"],
                "n_samples": len(metrics_payload["y_true"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output.parent.joinpath("per_language_f1.json").write_text(
        json.dumps({"per_lang_f1": metrics_payload["per_lang_f1"]}, indent=2),
        encoding="utf-8",
    )
    label_names = ["positive", "negative", "neutral"]
    clf_report_dict = classification_report(
        metrics_payload["y_true"],
        metrics_payload["y_pred"],
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )
    clf_report_str = classification_report(
        metrics_payload["y_true"],
        metrics_payload["y_pred"],
        labels=label_names,
        zero_division=0,
    )
    args.output.parent.joinpath("fairness_report.json").write_text(
        json.dumps(
            {
                "overall_f1": metrics_payload["overall_f1"],
                "per_lang_f1": metrics_payload["per_lang_f1"],
                "per_lang_gap": metrics_payload["per_lang_gap"],
                "sample_counts": metrics_payload["sample_counts"],
                "confusion_matrices": metrics_payload["per_lang_confusion_matrices"],
                "classification_report": clf_report_dict,
                "classification_report_text": clf_report_str,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation for {args.task} completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

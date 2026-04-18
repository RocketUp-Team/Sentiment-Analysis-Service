"""CLI entrypoint for evaluating finetuned Phase 2 adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def main(argv: list[str] | None = None) -> int:
    """Run finetuned adapter evaluation for a supported task."""
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    task = get_task_config(args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = ModelConfig(
        mode="finetuned",
        sentiment_adapter_path=str(root / "models" / "adapters" / "sentiment"),
        sarcasm_adapter_path=str(root / "models" / "adapters" / "sarcasm"),
    )
    inference = BaselineModelInference(config=config)
    df = _select_evaluation_rows(_load_evaluation_frame(args.task, root))
    df = df.sample(n=min(100, len(df)), random_state=42)
    texts = df["text"].tolist()
    languages = _resolve_languages(df)

    # predict_batch accepts a single `lang` for the entire batch (used only for
    # allowlist validation today).  We pass "en" because en is always in the
    # supported-languages allowlist and the model sees raw text regardless of this
    # parameter.  Per-row language information is captured in `languages` below and
    # is used for per-language metric computation, which is what matters here.
    results = inference.predict_batch(texts, lang="en", skip_absa=True)

    y_pred: list[str] = []
    y_true: list[str] = []
    for row, result in zip(df.itertuples(index=False), results, strict=True):
        if args.task == "sarcasm":
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

    args.output.write_text(
        json.dumps(
            {
                "task": args.task,
                "overall_f1": metrics_payload["overall_f1"],
                "n_samples": len(texts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output.parent.joinpath("per_language_f1.json").write_text(
        json.dumps({"per_lang_f1": metrics_payload["per_lang_f1"]}, indent=2),
        encoding="utf-8",
    )
    args.output.parent.joinpath("fairness_report.json").write_text(
        json.dumps(
            {
                "overall_f1": metrics_payload["overall_f1"],
                "per_lang_f1": metrics_payload["per_lang_f1"],
                "per_lang_gap": metrics_payload["per_lang_gap"],
                "sample_counts": metrics_payload["sample_counts"],
                "confusion_matrices": metrics_payload["per_lang_confusion_matrices"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation for {args.task} completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

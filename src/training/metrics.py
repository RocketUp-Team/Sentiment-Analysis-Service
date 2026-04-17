"""Metrics helpers for Phase 2 finetuning and evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sklearn.metrics import confusion_matrix, f1_score


def compute_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> float:
    """Return macro-F1 across the supplied label set."""
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(label_names),
            average="macro",
            zero_division=0,
        )
    )


def compute_per_language_f1(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    languages: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, float]:
    """Return per-language macro-F1 scores."""
    grouped_true: dict[str, list[str]] = {}
    grouped_pred: dict[str, list[str]] = {}

    for truth, pred, lang in zip(y_true, y_pred, languages, strict=True):
        grouped_true.setdefault(lang, []).append(truth)
        grouped_pred.setdefault(lang, []).append(pred)

    per_language: dict[str, float] = {}
    for lang in sorted(grouped_true):
        active_labels = [
            label
            for label in label_names
            if label in grouped_true[lang] or label in grouped_pred[lang]
        ]
        per_language[lang] = compute_macro_f1(
            grouped_true[lang],
            grouped_pred[lang],
            active_labels,
        )

    return per_language


def build_confusion_matrix(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> list[list[int]]:
    """Return a JSON-safe confusion matrix."""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(label_names),
    ).tolist()


def build_metrics_payload(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    languages: Sequence[str],
    label_names: Sequence[str],
) -> dict:
    """Build the fairness-aware metrics payload used by Phase 2 evaluation."""
    per_lang_f1 = compute_per_language_f1(
        y_true=y_true,
        y_pred=y_pred,
        languages=languages,
        label_names=label_names,
    )

    per_lang_confusion_matrices = {}
    for lang in sorted(set(languages)):
        lang_true = [truth for truth, row_lang in zip(y_true, languages, strict=True) if row_lang == lang]
        lang_pred = [pred for pred, row_lang in zip(y_pred, languages, strict=True) if row_lang == lang]
        per_lang_confusion_matrices[lang] = build_confusion_matrix(
            y_true=lang_true,
            y_pred=lang_pred,
            label_names=label_names,
        )

    f1_values = list(per_lang_f1.values())
    per_lang_gap = max(f1_values) - min(f1_values) if f1_values else 0.0

    return {
        "overall_f1": compute_macro_f1(y_true, y_pred, label_names),
        "per_lang_f1": per_lang_f1,
        "per_lang_gap": float(per_lang_gap),
        "sample_counts": dict(Counter(languages)),
        "confusion_matrix": build_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            label_names=label_names,
        ),
        "per_lang_confusion_matrices": per_lang_confusion_matrices,
    }

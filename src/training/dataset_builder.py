"""Dataset normalization helpers for Phase 2 finetuning."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def normalize_text_for_hash(text: str) -> str:
    """Normalize text for duplicate detection."""
    return " ".join(str(text).strip().lower().split())


def dedup_rows(rows: Iterable[dict]) -> list[dict]:
    """Deduplicate row dicts by normalized text, keeping first occurrence."""
    deduped: list[dict] = []
    seen: set[str] = set()

    for row in rows:
        key = normalize_text_for_hash(row["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))

    return deduped


def build_stratify_labels(frame: pd.DataFrame) -> pd.Series:
    """Build the `lang x label` stratification key used for sentiment splits."""
    return frame["lang"].astype(str) + "__" + frame["label"].astype(str)

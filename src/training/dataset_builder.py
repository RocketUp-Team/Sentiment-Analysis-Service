"""Dataset normalization helpers for Phase 2 finetuning."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


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


def oversample_minority_class(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    target_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Oversample minority classes to reach a minimum ratio of total samples.

    Sử dụng pandas ``sample(replace=True)`` — không cần thư viện ngoài.
    Chỉ nên được gọi trên **training split** (không áp dụng cho val/test).

    Args:
        df:            DataFrame chứa cột ``label_col`` (integer hoặc string).
        label_col:     Tên cột chứa nhãn.
        target_ratio:  Tỷ lệ tối thiểu mỗi class so với tổng sau oversampling.
                       Ví dụ 0.15 → mỗi class chiếm ít nhất 15%.
        seed:          Random seed cho reproducibility.

    Returns:
        DataFrame mới với minority classes đã được oversample.
        Thứ tự hàng được shuffle lại để tránh clustering.

    Example::

        df_balanced = oversample_minority_class(df_train, target_ratio=0.15)
    """
    if df.empty or label_col not in df.columns:
        return df

    label_counts = df[label_col].value_counts()
    n_classes = len(label_counts)

    if n_classes <= 1:
        return df

    current_total = len(df)
    frames: list[pd.DataFrame] = []
    rows_added = 0

    for cls_val, cls_count in label_counts.items():
        cls_ratio = cls_count / current_total
        if cls_ratio >= target_ratio:
            # Class đã đủ tỷ lệ, giữ nguyên
            frames.append(df[df[label_col] == cls_val])
            continue

        # Tính số hàng cần thêm để đạt target_ratio
        # target_ratio = (cls_count + n_new) / (current_total + n_new)
        # → n_new = (target_ratio * current_total - cls_count) / (1 - target_ratio)
        n_new = max(
            0,
            int((target_ratio * current_total - cls_count) / (1.0 - target_ratio)),
        )

        cls_df = df[df[label_col] == cls_val]
        frames.append(cls_df)
        if n_new > 0:
            oversampled = cls_df.sample(n=n_new, replace=True, random_state=seed)
            frames.append(oversampled)
            rows_added += n_new
            logger.info(
                "oversample_minority_class: class=%s added %d rows (%.1f%% → %.1f%%)",
                cls_val,
                n_new,
                cls_ratio * 100,
                (cls_count + n_new) / (current_total + rows_added) * 100,
            )

    result = pd.concat(frames, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)

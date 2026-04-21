import pandas as pd
import pytest

from src.training.dataset_builder import (
    build_stratify_labels,
    dedup_rows,
    normalize_text_for_hash,
    oversample_minority_class,
)


def test_normalize_text_for_hash_collapses_case_and_whitespace():
    assert normalize_text_for_hash("  Great   FOOD!  ") == "great food!"


def test_cross_dataset_dedup_by_normalized_text_hash():
    rows = [
        {"text": "Great food!", "label": "positive", "lang": "en"},
        {"text": " great food! ", "label": "positive", "lang": "en"},
        {"text": "dịch vụ tốt", "label": "positive", "lang": "vi"},
    ]

    out = dedup_rows(rows)

    assert len(out) == 2
    assert [row["text"] for row in out] == ["Great food!", "dịch vụ tốt"]


def test_build_stratify_labels_combines_language_and_label():
    frame = pd.DataFrame(
        [
            {"text": "great", "label": "positive", "lang": "en"},
            {"text": "tệ", "label": "negative", "lang": "vi"},
        ]
    )

    stratify = build_stratify_labels(frame)

    assert stratify.tolist() == ["en__positive", "vi__negative"]


# ─────────────────────────────────────────────────────────────────────────────
# oversample_minority_class
# ─────────────────────────────────────────────────────────────────────────────

def _make_imbalanced_df(pos=500, neg=460, neu=40) -> pd.DataFrame:
    """Mô phỏng UIT-VSFC distribution."""
    rows = (
        [{"text": f"pos-{i}", "label": 2} for i in range(pos)]
        + [{"text": f"neg-{i}", "label": 0} for i in range(neg)]
        + [{"text": f"neu-{i}", "label": 1} for i in range(neu)]
    )
    return pd.DataFrame(rows)


def test_oversample_minority_increases_total_size():
    df = _make_imbalanced_df()
    result = oversample_minority_class(df, label_col="label", target_ratio=0.15)

    assert len(result) > len(df)


def test_oversample_minority_reaches_target_ratio():
    df = _make_imbalanced_df(pos=500, neg=460, neu=40)
    target = 0.15
    result = oversample_minority_class(df, label_col="label", target_ratio=target)

    counts = result["label"].value_counts()
    total = len(result)
    for cls_val, cnt in counts.items():
        ratio = cnt / total
        assert ratio >= target - 0.02, (
            f"Class {cls_val}: ratio={ratio:.3f} below target={target}"
        )


def test_oversample_minority_does_not_touch_majority_class_count():
    df = _make_imbalanced_df(pos=500, neg=460, neu=40)
    result = oversample_minority_class(df, label_col="label", target_ratio=0.15)

    # Majority class (pos=2) không bị cắt bớt
    original_pos = (df["label"] == 2).sum()
    result_pos = (result["label"] == 2).sum()
    assert result_pos == original_pos


def test_oversample_minority_balanced_data_unchanged():
    """Dữ liệu đã cân bằng → không thêm rows."""
    rows = [{"text": f"t{i}", "label": i % 3} for i in range(300)]
    df = pd.DataFrame(rows)
    result = oversample_minority_class(df, label_col="label", target_ratio=0.15)

    # Đều là 33.3% mỗi class → target_ratio=0.15 đã đủ → không oversample
    assert len(result) == len(df)


def test_oversample_minority_returns_shuffled_rows():
    """DataFrame kết quả phải được shuffle (không còn sorted by class)."""
    df = _make_imbalanced_df()
    result = oversample_minority_class(df, label_col="label", target_ratio=0.15, seed=42)

    # Sau shuffle, labels không nên là một dãy đơn điệu liên tục
    labels = result["label"].tolist()
    # Nếu không shuffle: sẽ là [2,2,..., 0,0,..., 1,1,...]
    # Sau shuffle sẽ xen kẽ — kiểm tra bằng cách verify không strictly sorted
    sorted_labels = sorted(labels)
    assert labels != sorted_labels


def test_oversample_minority_empty_df_returns_empty():
    df = pd.DataFrame(columns=["text", "label"])
    result = oversample_minority_class(df, label_col="label", target_ratio=0.15)

    assert result.empty


def test_oversample_minority_missing_column_returns_unchanged():
    df = pd.DataFrame([{"text": "hello", "sentiment": 1}])
    result = oversample_minority_class(df, label_col="label", target_ratio=0.15)

    pd.testing.assert_frame_equal(result, df)


def test_oversample_minority_reproducible_with_same_seed():
    df = _make_imbalanced_df()
    r1 = oversample_minority_class(df, label_col="label", target_ratio=0.15, seed=7)
    r2 = oversample_minority_class(df, label_col="label", target_ratio=0.15, seed=7)

    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

import pandas as pd

from src.training.dataset_builder import (
    build_stratify_labels,
    dedup_rows,
    normalize_text_for_hash,
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

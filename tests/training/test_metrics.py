import pytest

from src.training.metrics import build_metrics_payload, compute_per_language_f1


LABEL_NAMES = ("negative", "neutral", "positive")


def test_compute_per_language_f1_is_one_for_perfect_predictions():
    y_true = ["negative", "positive", "neutral", "positive"]
    y_pred = ["negative", "positive", "neutral", "positive"]
    languages = ["en", "en", "vi", "vi"]

    per_language = compute_per_language_f1(
        y_true=y_true,
        y_pred=y_pred,
        languages=languages,
        label_names=LABEL_NAMES,
    )

    assert per_language == {"en": 1.0, "vi": 1.0}


def test_build_metrics_payload_contains_required_phase2_keys():
    payload = build_metrics_payload(
        y_true=["negative", "positive", "neutral", "positive"],
        y_pred=["negative", "negative", "neutral", "positive"],
        languages=["en", "en", "vi", "vi"],
        label_names=LABEL_NAMES,
    )

    assert set(payload) == {
        "overall_f1",
        "per_lang_f1",
        "per_lang_gap",
        "sample_counts",
        "confusion_matrix",
        "per_lang_confusion_matrices",
    }
    assert payload["sample_counts"] == {"en": 2, "vi": 2}
    assert payload["per_lang_gap"] == pytest.approx(0.6666666666666667)
    assert len(payload["confusion_matrix"]) == 3
    assert len(payload["per_lang_confusion_matrices"]["en"]) == 3

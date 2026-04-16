import json
import os
import types

import numpy as np
import pandas as pd

import src.data.validators as validators_module
from src.data.validators import (
    MISSING_SPLIT_KEY,
    DataQualityValidator,
    log_quality_report_to_mlflow,
    save_report,
)


def _base_params(**overrides):
    p = {
        "min_samples": 100,
        "max_null_ratio": 0.01,
        "expected_labels": {
            "sentiment": ["positive", "negative", "neutral"],
            "aspect": ["food", "service", "ambiance", "price", "location", "general"],
        },
    }
    p.update(overrides)
    return p


def _assert_rich_report_shape(report: dict) -> None:
    assert "total_samples" in report
    assert "splits" in report
    assert "aspect_distribution" in report
    assert "text_length_stats" in report
    assert "checks_passed" in report
    assert "errors" in report
    assert "warnings" in report
    for split_name, info in report["splits"].items():
        assert isinstance(split_name, str)
        assert set(info.keys()) >= {"samples", "null_ratio", "label_distribution"}
        assert isinstance(info["samples"], int)
        assert isinstance(info["null_ratio"], float)
        assert isinstance(info["label_distribution"], dict)


def test_data_quality_validator_happy_path(tmp_path):
    os.makedirs(tmp_path / "processed")
    sentences_df = pd.DataFrame(
        [{"sentence_id": str(i), "text": "foo", "sentiment": "positive", "split": "train"} for i in range(200)]
    )
    aspects_df = pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "positive"} for i in range(200)]
    )
    sentences_df.to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    aspects_df.to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params()).validate(str(tmp_path / "processed"))
    _assert_rich_report_shape(report)
    assert report["checks_passed"] is True
    assert report["total_samples"] == 200
    assert report["splits"]["train"]["samples"] == 200
    assert report["splits"]["train"]["label_distribution"]["positive"] == 200
    assert report["aspect_distribution"]["food"] == 200
    assert "mean" in report["text_length_stats"]
    assert "min" in report["text_length_stats"]
    assert "max" in report["text_length_stats"]
    assert "median" in report["text_length_stats"]
    assert "p95" in report["text_length_stats"]


def test_validate_fails_on_missing_required_sentence_columns(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame([{"sentence_id": "1", "text": "x", "sentiment": "positive"}]).to_csv(
        tmp_path / "processed" / "sentences.csv", index=False
    )
    pd.DataFrame(
        [{"sentence_id": "1", "aspect_category": "food", "sentiment": "positive"}]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=1)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("sentence" in e.lower() and "split" in e.lower() for e in report["errors"])


def test_validate_fails_on_missing_required_aspect_columns(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame(
        [{"sentence_id": "1", "text": "foo", "sentiment": "positive", "split": "train"}] * 5
    ).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame([{"sentence_id": "1", "aspect_category": "food"}]).to_csv(
        tmp_path / "processed" / "aspects.csv", index=False
    )

    report = DataQualityValidator(_base_params(min_samples=1)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("aspect" in e.lower() and "sentiment" in e.lower() for e in report["errors"])


def test_validate_fails_when_sentence_null_ratio_exceeds_threshold(tmp_path):
    os.makedirs(tmp_path / "processed")
    rows = [{"sentence_id": str(i), "text": "ok", "sentiment": "positive", "split": "train"} for i in range(100)]
    rows[0]["text"] = np.nan
    pd.DataFrame(rows).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "positive"} for i in range(100)]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=50, max_null_ratio=0.005)).validate(
        str(tmp_path / "processed")
    )
    assert report["checks_passed"] is False
    assert any("null" in e.lower() for e in report["errors"])


def test_validate_fails_on_invalid_sentence_sentiment(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame(
        [{"sentence_id": str(i), "text": "ok", "sentiment": "bad_label", "split": "train"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "positive"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params()).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("sentiment" in e.lower() for e in report["errors"])


def test_validate_fails_on_invalid_aspect_category(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame(
        [{"sentence_id": str(i), "text": "ok", "sentiment": "positive", "split": "train"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [
            {
                "sentence_id": str(i),
                "aspect_category": "not_a_real_aspect_label",
                "sentiment": "positive",
            }
            for i in range(120)
        ]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params()).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("aspect_category" in e.lower() for e in report["errors"])
    assert any("not_a_real_aspect_label" in e for e in report["errors"])


def test_validate_fails_on_invalid_aspect_sentiment(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame(
        [{"sentence_id": str(i), "text": "ok", "sentiment": "positive", "split": "train"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "weird"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params()).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("aspect" in e.lower() and "sentiment" in e.lower() for e in report["errors"])


def test_validate_fails_when_total_samples_below_min_even_if_split_groups_large(tmp_path):
    """Explicit total row count must satisfy min_samples (same param as per-split)."""
    os.makedirs(tmp_path / "processed")
    rows = [{"sentence_id": str(i), "text": "ok", "sentiment": "positive", "split": "train"} for i in range(80)]
    pd.DataFrame(rows).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "positive"} for i in range(80)]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=100)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("total_samples" in e.lower() and "min_samples" in e.lower() for e in report["errors"])


def test_validate_fails_when_split_below_min_samples(tmp_path):
    os.makedirs(tmp_path / "processed")
    train = [{"sentence_id": f"t{i}", "text": "a", "sentiment": "positive", "split": "train"} for i in range(40)]
    test = [{"sentence_id": f"s{i}", "text": "b", "sentiment": "positive", "split": "test"} for i in range(120)]
    pd.DataFrame(train + test).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    aspects = (
        [{"sentence_id": f"t{i}", "aspect_category": "food", "sentiment": "positive"} for i in range(40)]
        + [{"sentence_id": f"s{i}", "aspect_category": "food", "sentiment": "positive"} for i in range(120)]
    )
    pd.DataFrame(aspects).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=100)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("train" in e.lower() and "min_samples" in e.lower() for e in report["errors"])


def test_validate_emits_warnings_for_missing_expected_sentiment_classes(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame(
        [{"sentence_id": str(i), "text": "ok", "sentiment": "positive", "split": "train"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "positive"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=50)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is True
    assert any("negative" in w.lower() or "neutral" in w.lower() for w in report["warnings"])


def test_split_stats_include_null_split_bucket_and_sum_to_total(tmp_path):
    os.makedirs(tmp_path / "processed")
    rows = (
        [{"sentence_id": f"t{i}", "text": "a", "sentiment": "positive", "split": "train"} for i in range(60)]
        + [{"sentence_id": f"n{i}", "text": "b", "sentiment": "negative", "split": None} for i in range(40)]
    )
    pd.DataFrame(rows).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    aspects = (
        [{"sentence_id": f"t{i}", "aspect_category": "food", "sentiment": "positive"} for i in range(60)]
        + [{"sentence_id": f"n{i}", "aspect_category": "food", "sentiment": "negative"} for i in range(40)]
    )
    pd.DataFrame(aspects).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=40)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is True
    assert MISSING_SPLIT_KEY in report["splits"]
    assert report["splits"]["train"]["samples"] == 60
    assert report["splits"][MISSING_SPLIT_KEY]["samples"] == 40
    assert (
        report["splits"]["train"]["samples"] + report["splits"][MISSING_SPLIT_KEY]["samples"]
        == report["total_samples"]
    )


def test_validate_per_split_null_ratio_and_label_distribution(tmp_path):
    os.makedirs(tmp_path / "processed")
    rows = []
    for i in range(50):
        rows.append(
            {"sentence_id": f"a{i}", "text": "x", "sentiment": "positive", "split": "train"}
        )
    for i in range(50):
        rows.append(
            {"sentence_id": f"b{i}", "text": "y", "sentiment": "negative", "split": "test"}
        )
    pd.DataFrame(rows).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    aspects = (
        [{"sentence_id": f"a{i}", "aspect_category": "food", "sentiment": "positive"} for i in range(50)]
        + [{"sentence_id": f"b{i}", "aspect_category": "service", "sentiment": "negative"} for i in range(50)]
    )
    pd.DataFrame(aspects).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=40)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is True
    assert report["splits"]["train"]["samples"] == 50
    assert report["splits"]["test"]["samples"] == 50
    assert report["splits"]["train"]["null_ratio"] == 0.0
    assert report["splits"]["train"]["label_distribution"]["positive"] == 50
    assert report["splits"]["test"]["label_distribution"]["negative"] == 50
    dist = report["aspect_distribution"]
    assert dist["food"] == 50 and dist["service"] == 50


def test_validate_fails_on_empty_aspects_file(tmp_path):
    os.makedirs(tmp_path / "processed")
    pd.DataFrame(
        [{"sentence_id": str(i), "text": "ok", "sentiment": "positive", "split": "train"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(columns=["sentence_id", "aspect_category", "sentiment"]).to_csv(
        tmp_path / "processed" / "aspects.csv", index=False
    )

    report = DataQualityValidator(_base_params(min_samples=50)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("aspect" in e.lower() and "row" in e.lower() for e in report["errors"])


def test_invalid_sentence_sentiment_all_null_lists_placeholder_example(tmp_path):
    os.makedirs(tmp_path / "processed")
    rows = [{"sentence_id": str(i), "text": "ok", "sentiment": None, "split": "train"} for i in range(120)]
    pd.DataFrame(rows).to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    pd.DataFrame(
        [{"sentence_id": str(i), "aspect_category": "food", "sentiment": "positive"} for i in range(120)]
    ).to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=50)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    err = next(e for e in report["errors"] if "sentiment" in e.lower())
    assert "<null_or_na>" in err


def test_validate_fails_on_orphan_aspects(tmp_path):
    os.makedirs(tmp_path / "processed")
    sentences_df = pd.DataFrame(
        [{"sentence_id": "1", "text": "foo", "sentiment": "positive", "split": "train"}] * 100
    )
    aspects_df = pd.DataFrame(
        [
            {"sentence_id": "1", "aspect_category": "food", "sentiment": "positive"},
            {"sentence_id": "missing", "aspect_category": "food", "sentiment": "positive"},
        ]
    )
    sentences_df.to_csv(tmp_path / "processed" / "sentences.csv", index=False)
    aspects_df.to_csv(tmp_path / "processed" / "aspects.csv", index=False)

    report = DataQualityValidator(_base_params(min_samples=50)).validate(str(tmp_path / "processed"))
    assert report["checks_passed"] is False
    assert any("orphan" in e.lower() for e in report["errors"])


def test_save_report_writes_json_and_creates_parent_dirs(tmp_path):
    report = {
        "total_samples": 1,
        "splits": {"train": {"samples": 1, "null_ratio": 0.0, "label_distribution": {"positive": 1}}},
        "aspect_distribution": {"food": 1},
        "text_length_stats": {"mean": 3.0, "min": 3, "max": 3, "median": 3.0, "p95": 3.0},
        "checks_passed": True,
        "errors": [],
        "warnings": [],
    }
    out = tmp_path / "nested" / "dir" / "quality_report.json"
    save_report(report, str(out))
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["checks_passed"] is True
    assert loaded["total_samples"] == 1
    assert loaded["splits"]["train"]["samples"] == 1


def test_log_quality_report_to_mlflow_returns_false_when_dependency_missing(
    tmp_path, monkeypatch, caplog
):
    report = {
        "total_samples": 1,
        "splits": {"train": {"samples": 1, "null_ratio": 0.0, "label_distribution": {"positive": 1}}},
        "aspect_distribution": {"food": 1},
        "text_length_stats": {"mean": 3.0, "min": 3, "max": 3, "median": 3.0, "p95": 3.0},
        "checks_passed": True,
        "errors": [],
        "warnings": [],
    }
    report_path = tmp_path / "quality_report.json"
    save_report(report, str(report_path))

    def _missing_module(_: str):
        raise ModuleNotFoundError("No module named 'mlflow'")

    monkeypatch.setattr(validators_module.importlib, "import_module", _missing_module)

    with caplog.at_level("WARNING"):
        logged = log_quality_report_to_mlflow(
            report,
            report_path,
            {"tracking_uri": "http://localhost:5000", "experiment_name": "data_preprocessing"},
        )

    assert logged is False
    assert any("mlflow" in record.message.lower() for record in caplog.records)


def test_log_quality_report_to_mlflow_logs_expected_metrics(tmp_path, monkeypatch):
    report = {
        "total_samples": 3,
        "splits": {
            "train": {"samples": 2, "null_ratio": 0.0, "label_distribution": {"positive": 2}},
            "val": {"samples": 1, "null_ratio": 0.0, "label_distribution": {"negative": 1}},
        },
        "aspect_distribution": {"food": 2, "service": 1},
        "text_length_stats": {"mean": 3.0, "min": 3, "max": 3, "median": 3.0, "p95": 3.0},
        "checks_passed": True,
        "errors": [],
        "warnings": [],
    }
    report_path = tmp_path / "quality_report.json"
    save_report(report, str(report_path))
    calls: list[tuple[str, object]] = []

    class _FakeRun:
        def __init__(self, run_name: str):
            self._run_name = run_name

        def __enter__(self):
            calls.append(("start_run", self._run_name))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("end_run", self._run_name))
            return False

    fake_mlflow = types.SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("tracking_uri", uri)),
        set_experiment=lambda name: calls.append(("experiment", name)),
        start_run=lambda run_name: _FakeRun(run_name),
        log_artifact=lambda path: calls.append(("artifact", path)),
        log_metric=lambda name, value: calls.append(("metric", (name, value))),
        log_params=lambda params: calls.append(("params", params)),
    )

    monkeypatch.setattr(
        validators_module.importlib,
        "import_module",
        lambda module_name: fake_mlflow if module_name == "mlflow" else None,
    )
    monkeypatch.setattr(
        validators_module.subprocess,
        "check_output",
        lambda *args, **kwargs: b"abc123\n",
    )

    logged = log_quality_report_to_mlflow(
        report,
        report_path,
        {"tracking_uri": "http://localhost:5000", "experiment_name": "data_preprocessing"},
    )

    assert logged is True
    assert ("tracking_uri", "http://localhost:5000") in calls
    assert ("experiment", "data_preprocessing") in calls
    assert ("start_run", "data_vabc123") in calls
    assert ("artifact", str(report_path)) in calls
    assert ("metric", ("total_samples", 3)) in calls
    assert ("metric", ("passed_quality_checks", 1)) in calls
    assert ("metric", ("train_samples", 2)) in calls
    assert ("metric", ("val_null_ratio", 0.0)) in calls
    params_call = next(value for name, value in calls if name == "params")
    assert params_call["dataset"] == "semeval2014_restaurants"
    assert params_call["git_version"] == "abc123"

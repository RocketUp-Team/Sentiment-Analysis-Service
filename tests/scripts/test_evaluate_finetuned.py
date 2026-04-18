import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.scripts import evaluate_finetuned
from src.scripts.evaluate_finetuned import main, parse_args


def test_evaluate_finetuned_accepts_task_and_output_path(tmp_path):
    output_path = tmp_path / "metrics.json"

    args = parse_args(
        ["--task", "sentiment", "--output", str(output_path)]
    )

    assert args.task == "sentiment"
    assert args.output == output_path


def test_evaluate_finetuned_defaults_to_phase2_reports():
    args = parse_args(["--task", "sarcasm"])

    assert isinstance(args.output, Path)
    assert args.output == Path("reports/metrics_finetuned.json")


def test_evaluate_finetuned_main_writes_sarcasm_reports_from_inference(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "reports" / "metrics.json"
    read_csv_calls: list[str] = []
    inference_calls: list[dict] = []
    created_configs = []

    def fake_read_csv(path):
        read_csv_calls.append(Path(path).name)
        return pd.DataFrame(
            [
                {"text": "dry joke", "label": 0},
                {"text": "wow totally sincere", "label": 1},
            ]
        )

    class FakeInference:
        def __init__(self, config) -> None:
            created_configs.append(config)

        def predict_batch(self, texts, lang: str, skip_absa: bool):
            inference_calls.append(
                {"texts": list(texts), "lang": lang, "skip_absa": skip_absa}
            )
            return [
                SimpleNamespace(sentiment="neutral", sarcasm_flag=False),
                SimpleNamespace(sentiment="neutral", sarcasm_flag=True),
            ]

    def fake_build_metrics_payload(*, y_true, y_pred, languages, label_names):
        assert y_true == ["irony", "non_irony"]
        assert y_pred == ["non_irony", "irony"]
        assert languages == ["en", "en"]
        assert label_names == ("non_irony", "irony")
        return {
            "overall_f1": 1.0,
            "per_lang_f1": {"en": 1.0},
            "per_lang_gap": 0.0,
            "sample_counts": {"en": 2},
            "per_lang_confusion_matrices": {"en": [[1, 0], [0, 1]]},
        }

    monkeypatch.setattr(evaluate_finetuned.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        evaluate_finetuned, "BaselineModelInference", FakeInference
    )
    monkeypatch.setattr(
        evaluate_finetuned, "build_metrics_payload", fake_build_metrics_payload
    )

    exit_code = main(["--task", "sarcasm", "--output", str(output_path)])

    assert exit_code == 0
    assert read_csv_calls == ["sarcasm.csv"]
    assert inference_calls == [
        {
            "texts": ["wow totally sincere", "dry joke"],
            "lang": "en",
            "skip_absa": True,
        }
    ]
    assert created_configs[0].mode == "finetuned"
    assert created_configs[0].sentiment_adapter_path.endswith(
        "models/adapters/sentiment"
    )
    assert created_configs[0].sarcasm_adapter_path.endswith("models/adapters/sarcasm")
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "task": "sarcasm",
        "overall_f1": 1.0,
        "n_samples": 2,
    }
    assert json.loads(
        output_path.parent.joinpath("per_language_f1.json").read_text(encoding="utf-8")
    ) == {"per_lang_f1": {"en": 1.0}}
    assert json.loads(
        output_path.parent.joinpath("fairness_report.json").read_text(encoding="utf-8")
    ) == {
        "overall_f1": 1.0,
        "per_lang_f1": {"en": 1.0},
        "per_lang_gap": 0.0,
        "sample_counts": {"en": 2},
        "confusion_matrices": {"en": [[1, 0], [0, 1]]},
    }


def test_evaluate_finetuned_main_uses_test_split_when_present(tmp_path, monkeypatch):
    output_path = tmp_path / "reports" / "metrics.json"
    inference_calls: list[dict] = []

    def fake_read_csv(path):
        assert Path(path).name == "sarcasm.csv"
        return pd.DataFrame(
            [
                {"text": "train example", "label": 0, "split": "train"},
                {"text": "heldout first", "label": 1, "split": "test"},
                {"text": "heldout second", "label": 0, "split": "test"},
            ]
        )

    class FakeInference:
        def __init__(self, config) -> None:
            self.config = config

        def predict_batch(self, texts, lang: str, skip_absa: bool):
            inference_calls.append(
                {"texts": list(texts), "lang": lang, "skip_absa": skip_absa}
            )
            return [
                SimpleNamespace(sentiment="neutral", sarcasm_flag=True),
                SimpleNamespace(sentiment="neutral", sarcasm_flag=False),
            ]

    def fake_build_metrics_payload(*, y_true, y_pred, languages, label_names):
        assert y_true == ["non_irony", "irony"]
        assert y_pred == ["irony", "non_irony"]
        assert languages == ["en", "en"]
        assert label_names == ("non_irony", "irony")
        return {
            "overall_f1": 0.5,
            "per_lang_f1": {"en": 0.5},
            "per_lang_gap": 0.0,
            "sample_counts": {"en": 2},
            "per_lang_confusion_matrices": {"en": [[1, 0], [1, 0]]},
        }

    monkeypatch.setattr(evaluate_finetuned.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        evaluate_finetuned, "BaselineModelInference", FakeInference
    )
    monkeypatch.setattr(
        evaluate_finetuned, "build_metrics_payload", fake_build_metrics_payload
    )

    exit_code = main(["--task", "sarcasm", "--output", str(output_path)])

    assert exit_code == 0
    assert inference_calls == [
        {
            "texts": ["heldout second", "heldout first"],
            "lang": "en",
            "skip_absa": True,
        }
    ]


def test_evaluate_finetuned_main_uses_multilingual_sentiment_dataset(tmp_path, monkeypatch):
    output_path = tmp_path / "sentiment" / "metrics.json"
    read_csv_calls: list[str] = []

    def fake_read_csv(path):
        filename = Path(path).name
        read_csv_calls.append(filename)
        if filename == "sentiment_en.csv":
            return pd.DataFrame(
                [
                    {"text": "great", "label": 2, "lang": "en"},
                    {"text": "bad", "label": 0, "lang": "en"},
                ]
            )
        if filename == "sentiment_vi.csv":
            return pd.DataFrame(
                [
                    {"text": "tot", "label": 2, "lang": "vi"},
                ]
            )
        raise AssertionError(f"Unexpected csv path: {path}")

    class FakeInference:
        def __init__(self, config) -> None:
            self.config = config

        def predict_batch(self, texts, lang: str, skip_absa: bool):
            assert texts == ["great", "bad", "tot"]
            assert lang == "en"
            assert skip_absa is True
            return [
                SimpleNamespace(sentiment="positive", sarcasm_flag=False),
                SimpleNamespace(sentiment="negative", sarcasm_flag=False),
                SimpleNamespace(sentiment="neutral", sarcasm_flag=False),
            ]

    def fake_build_metrics_payload(*, y_true, y_pred, languages, label_names):
        assert y_true == ["positive", "negative", "positive"]
        assert y_pred == ["positive", "negative", "neutral"]
        assert languages == ["en", "en", "vi"]
        assert label_names == ("negative", "neutral", "positive")
        return {
            "overall_f1": 0.8,
            "per_lang_f1": {"en": 1.0, "vi": 0.0},
            "per_lang_gap": 1.0,
            "sample_counts": {"en": 2, "vi": 1},
            "per_lang_confusion_matrices": {
                "en": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                "vi": [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            },
        }

    monkeypatch.setattr(evaluate_finetuned.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        evaluate_finetuned, "BaselineModelInference", FakeInference
    )
    monkeypatch.setattr(
        evaluate_finetuned, "build_metrics_payload", fake_build_metrics_payload
    )

    exit_code = main(["--task", "sentiment", "--output", str(output_path)])

    assert exit_code == 0
    assert read_csv_calls == ["sentiment_en.csv", "sentiment_vi.csv"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "task": "sentiment",
        "overall_f1": 0.8,
        "n_samples": 3,
    }
    assert json.loads(
        output_path.parent.joinpath("per_language_f1.json").read_text(encoding="utf-8")
    ) == {"per_lang_f1": {"en": 1.0, "vi": 0.0}}
    assert json.loads(
        output_path.parent.joinpath("fairness_report.json").read_text(encoding="utf-8")
    ) == {
        "overall_f1": 0.8,
        "per_lang_f1": {"en": 1.0, "vi": 0.0},
        "per_lang_gap": 1.0,
        "sample_counts": {"en": 2, "vi": 1},
        "confusion_matrices": {
            "en": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
            "vi": [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
        },
    }


def test_evaluate_returns_metrics_payload_dict(tmp_path, monkeypatch):
    """evaluate() returns the full metrics payload without writing files."""
    inference_calls: list[dict] = []

    def fake_read_csv(path):
        filename = Path(path).name
        if filename == "sentiment_en.csv":
            return pd.DataFrame(
                [
                    {"text": "great", "label": 2, "lang": "en"},
                    {"text": "bad", "label": 0, "lang": "en"},
                ]
            )
        return pd.DataFrame([{"text": "tot", "label": 2, "lang": "vi"}])

    class FakeInference:
        def __init__(self, config) -> None:
            self.config = config

        def predict_batch(self, texts, lang: str, skip_absa: bool):
            inference_calls.append({"texts": list(texts), "lang": lang})
            return [
                SimpleNamespace(sentiment="positive", sarcasm_flag=False),
                SimpleNamespace(sentiment="negative", sarcasm_flag=False),
                SimpleNamespace(sentiment="neutral", sarcasm_flag=False),
            ]

    expected_payload = {
        "overall_f1": 0.8,
        "per_lang_f1": {"en": 1.0, "vi": 0.0},
        "per_lang_gap": 1.0,
        "sample_counts": {"en": 2, "vi": 1},
        "confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        "per_lang_confusion_matrices": {
            "en": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
            "vi": [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
        },
    }

    def fake_build_metrics_payload(*, y_true, y_pred, languages, label_names):
        return expected_payload

    monkeypatch.setattr(evaluate_finetuned.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(evaluate_finetuned, "BaselineModelInference", FakeInference)
    monkeypatch.setattr(evaluate_finetuned, "build_metrics_payload", fake_build_metrics_payload)

    from src.scripts.evaluate_finetuned import evaluate

    result = evaluate("sentiment", root=tmp_path, max_samples=None)

    assert result["overall_f1"] == 0.8
    assert result["per_lang_f1"] == {"en": 1.0, "vi": 0.0}
    assert result["per_lang_gap"] == 1.0
    assert "y_true" in result
    assert "y_pred" in result

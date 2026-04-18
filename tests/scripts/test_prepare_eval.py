"""Tests for src/scripts/prepare_eval.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.scripts import prepare_eval
from src.scripts.prepare_eval import main, parse_args


def test_parse_args_returns_default_output_dir():
    args = parse_args([])
    assert args.output_dir == Path("data/eval")


def test_parse_args_accepts_custom_output_dir(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path / "eval")])
    assert args.output_dir == tmp_path / "eval"


def _make_sentiment_csv(path: Path, lang: str, n_per_label: int = 5) -> None:
    """Write a minimal sentiment CSV with train/test splits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in ("negative", "neutral", "positive"):
        for i in range(n_per_label):
            rows.append(
                {
                    "text": f"{lang}-{label}-{i}",
                    "label": label,
                    "lang": lang,
                    "source": f"test_source_{lang}",
                    "split": "test" if i < 2 else "train",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_main_writes_mixed_lang_eval_csv(tmp_path, monkeypatch):
    """main() writes both output files to the given output directory."""
    out_dir = tmp_path / "data" / "eval"

    sample_df = pd.DataFrame([
        {"text": "great", "label": "positive", "lang": "en", "source": "src"},
        {"text": "tốt", "label": "positive", "lang": "vi", "source": "src"},
    ])

    # Patch _load_sentiment_test_rows so no real raw files are needed on disk
    monkeypatch.setattr(prepare_eval, "_load_sentiment_test_rows", lambda root: sample_df)

    result = main(["--output-dir", str(out_dir)])

    assert result == 0
    assert (out_dir / "mixed_lang_eval.csv").exists()
    assert (out_dir / "vi_sarcasm_eval.csv").exists()
    written = pd.read_csv(out_dir / "mixed_lang_eval.csv")
    assert len(written) == 2


def test_load_sentiment_test_rows_filters_to_test_split(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    _make_sentiment_csv(raw_dir / "sentiment_en.csv", lang="en", n_per_label=5)
    _make_sentiment_csv(raw_dir / "sentiment_vi.csv", lang="vi", n_per_label=5)

    result = prepare_eval._load_sentiment_test_rows(tmp_path)

    # n_per_label=5, split=test for i<2 → 2 test rows per label × 3 labels × 2 langs = 12
    assert len(result) == 12
    assert set(result["lang"].unique()) == {"en", "vi"}
    assert set(result.columns) == {"text", "label", "lang", "source"}


def test_load_sentiment_test_rows_raises_when_no_data(tmp_path):
    with pytest.raises(RuntimeError, match="No sentiment raw data found"):
        prepare_eval._load_sentiment_test_rows(tmp_path)


def test_vi_sarcasm_probe_rows_have_required_columns():
    df = pd.DataFrame(prepare_eval._VI_SARCASM_PROBE_ROWS)
    assert {"text", "label", "lang", "source"}.issubset(df.columns)
    assert set(df["lang"]) == {"vi"}
    assert set(df["label"]).issubset({"irony", "non_irony"})
    assert len(df) >= 4  # must have at least some probes


def test_load_sentiment_test_rows_works_without_split_column(tmp_path):
    """When 'split' column is absent, all rows are included."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    df = pd.DataFrame(
        [
            {"text": "good", "label": "positive", "lang": "en", "source": "x"},
            {"text": "bad", "label": "negative", "lang": "en", "source": "x"},
        ]
    )
    df.to_csv(raw_dir / "sentiment_en.csv", index=False)
    # no sentiment_vi.csv

    result = prepare_eval._load_sentiment_test_rows(tmp_path)
    assert len(result) == 2
    assert list(result.columns) == ["text", "label", "lang", "source"]

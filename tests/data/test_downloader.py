from pathlib import Path

import pandas as pd
import pytest

import src.data.downloader as downloader
from src.data.downloader import (
    EXPECTED_RAW_ASPECT_COLUMNS,
    EXPECTED_RAW_SENTENCE_COLUMNS,
    SchemaError,
    build_sarcasm_frame,
    build_uit_vsfc_frame,
    download_sarcasm_dataset,
    download_sentiment_datasets,
    main,
    validate_raw_schema,
    write_placeholder_raw_csvs,
)


def test_validate_raw_schema():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo", "split": "train"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1", "aspect_category": "food", "sentiment": "positive"}])

    validate_raw_schema(sentences_df, aspects_df)  # Should not raise

    with pytest.raises(SchemaError):
        validate_raw_schema(pd.DataFrame([{"text": "missing id"}]), aspects_df)


def test_validate_raw_schema_rejects_empty_sentences():
    sentences_df = pd.DataFrame(columns=sorted(EXPECTED_RAW_SENTENCE_COLUMNS))
    aspects_df = pd.DataFrame(
        [{"sentence_id": "1", "aspect_category": "food", "sentiment": "positive"}]
    )
    with pytest.raises(SchemaError):
        validate_raw_schema(sentences_df, aspects_df)


def test_validate_raw_schema_rejects_empty_aspects():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo", "split": "train"}])
    aspects_df = pd.DataFrame(columns=sorted(EXPECTED_RAW_ASPECT_COLUMNS))
    with pytest.raises(SchemaError):
        validate_raw_schema(sentences_df, aspects_df)


def test_validate_raw_schema_rejects_missing_aspect_columns():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo", "split": "train"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1", "aspect_category": "food"}])
    with pytest.raises(SchemaError) as excinfo:
        validate_raw_schema(sentences_df, aspects_df)
    assert "sentiment" in str(excinfo.value).lower()


def test_write_placeholder_raw_csvs_writes_schema_valid_stub(tmp_path):
    raw_dir = tmp_path / "raw"
    write_placeholder_raw_csvs(
        raw_dir,
        dataset_name="stub_ds",
        splits=["train", "dev"],
    )

    sentences_df = pd.read_csv(raw_dir / "sentences.csv")
    aspects_df = pd.read_csv(raw_dir / "aspects.csv")
    validate_raw_schema(sentences_df, aspects_df)

    assert set(sentences_df["split"]) == {"train", "dev"}
    assert sentences_df["text"].str.contains("stub_ds").all()
    assert sentences_df["sentence_id"].tolist() == aspects_df["sentence_id"].tolist()
    assert len(sentences_df) == 4000
    assert len(aspects_df) == 4000


def test_build_sarcasm_frame_adds_split_lang_and_source_columns():
    split_frames = {
        "train": pd.DataFrame(
            [
                {"text": "love it", "label": 1},
                {"text": "sure, amazing", "label": 0},
            ]
        ),
        "test": pd.DataFrame([{"text": "wow", "label": 1}]),
    }

    result = build_sarcasm_frame(split_frames)

    assert result.columns.tolist() == ["text", "label", "lang", "split", "source"]
    assert result["lang"].tolist() == ["en", "en", "en"]
    assert result["split"].tolist() == ["train", "train", "test"]
    assert result["source"].tolist() == ["tweet_eval_irony"] * 3


def test_build_uit_vsfc_frame_maps_numeric_labels_to_project_strings():
    result = build_uit_vsfc_frame(
        sentences=["rất tốt", "bình thường", "quá tệ"],
        labels=[2, 1, 0],
        split="validation",
    )

    assert result.columns.tolist() == ["text", "label", "lang", "split", "source"]
    assert result["label"].tolist() == ["positive", "neutral", "negative"]
    assert result["lang"].tolist() == ["vi", "vi", "vi"]
    assert result["split"].tolist() == ["validation", "validation", "validation"]


class _FakeSplit:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


def test_download_sarcasm_dataset_writes_csv_with_metadata(tmp_path, monkeypatch):
    def fake_load_dataset(name: str, config: str):
        assert (name, config) == ("tweet_eval", "irony")
        return {"train": _FakeSplit(pd.DataFrame([{"text": "sure", "label": 1}]))}

    monkeypatch.setattr(downloader, "load_dataset", fake_load_dataset)

    out_path = tmp_path / "raw" / "sarcasm.csv"
    download_sarcasm_dataset(out_path)

    result = pd.read_csv(out_path)
    assert result.columns.tolist() == ["text", "label", "lang", "source"]
    assert result.iloc[0].to_dict() == {
        "text": "sure",
        "label": 1,
        "lang": "en",
        "source": "tweet_eval_irony",
    }


def test_download_sentiment_datasets_writes_english_and_vietnamese_csvs(tmp_path, monkeypatch):
    english_loader_calls: list[tuple[str, dict[str, str]]] = []
    vietnamese_loader_calls: list[tuple[str, dict[str, str]]] = []
    english_data_files = {
        "train": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/train/*.parquet",
        "validation": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/validation/*.parquet",
        "test": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/test/*.parquet",
    }
    vietnamese_data_files = {
        "train": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/train/*.parquet",
        "validation": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/validation/*.parquet",
        "test": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/test/*.parquet",
    }

    def fake_load_dataset(name: str, config: str | None = None, **kwargs):
        if name == "parquet":
            assert config is None
            data_files = kwargs["data_files"]
            if data_files == english_data_files:
                english_loader_calls.append((name, data_files))
                return {
                    "train": _FakeSplit(
                        pd.DataFrame([{"text": "great", "source": "sem_eval_2017", "label": "positive"}])
                    )
                }
            if data_files == vietnamese_data_files:
                vietnamese_loader_calls.append((name, data_files))
                return {
                    "train": _FakeSplit(pd.DataFrame([{"sentence": "rat hay", "sentiment": 2, "topic": 0}])),
                    "validation": _FakeSplit(
                        pd.DataFrame([{"sentence": "binh thuong", "sentiment": 1, "topic": 1}])
                    ),
                    "test": _FakeSplit(pd.DataFrame([{"sentence": "te", "sentiment": 0, "topic": 2}])),
                }
        raise AssertionError(f"unexpected dataset request: {(name, config)}")

    monkeypatch.setattr(downloader, "load_dataset", fake_load_dataset)

    en_path = tmp_path / "raw" / "sentiment_en.csv"
    vi_path = tmp_path / "raw" / "sentiment_vi.csv"

    download_sentiment_datasets(en_path, vi_path)

    en_result = pd.read_csv(en_path)
    vi_result = pd.read_csv(vi_path)

    assert en_result.columns.tolist() == ["text", "label", "lang", "source"]
    assert en_result.iloc[0].to_dict() == {
        "text": "great",
        "label": "positive",
        "lang": "en",
        "source": "multilingual_sentiments",
    }
    assert english_loader_calls == [
        ("parquet", english_data_files)
    ]
    assert vietnamese_loader_calls == [
        ("parquet", vietnamese_data_files)
    ]

    assert vi_result.columns.tolist() == ["text", "label", "lang", "source", "split"]
    assert vi_result.to_dict("records") == [
        {
            "text": "rat hay",
            "label": "positive",
            "lang": "vi",
            "source": "uit_vsfc",
            "split": "train",
        },
        {
            "text": "binh thuong",
            "label": "neutral",
            "lang": "vi",
            "source": "uit_vsfc",
            "split": "validation",
        },
        {
            "text": "te",
            "label": "negative",
            "lang": "vi",
            "source": "uit_vsfc",
            "split": "test",
        },
    ]


def test_main_dispatches_semeval_task(monkeypatch):
    calls: dict[str, object] = {}

    def fake_load_params(path: str):
        calls["params_path"] = path
        return {"data": {"dataset_name": "restaurants", "splits": ["train", "test"]}}

    def fake_extract(raw_dir: Path, *, dataset_name: str, splits: list[str]):
        calls["raw_dir"] = raw_dir
        calls["dataset_name"] = dataset_name
        calls["splits"] = splits
        return raw_dir / "sentences.csv", raw_dir / "aspects.csv"

    monkeypatch.setattr(downloader, "load_params", fake_load_params)
    monkeypatch.setattr(downloader, "extract_semeval_xmls", fake_extract)

    main(["--task", "semeval"])

    root = Path(downloader.__file__).resolve().parents[2]
    assert calls == {
        "params_path": str(root / "params.yaml"),
        "raw_dir": root / "data" / "raw",
        "dataset_name": "restaurants",
        "splits": ["train", "test"],
    }


@pytest.mark.parametrize(
    ("task", "expected_name"),
    [("sarcasm", "sarcasm.csv"), ("sentiment", "sentiment_en.csv")],
)
def test_main_dispatches_hf_tasks(task, expected_name, monkeypatch):
    calls: dict[str, tuple[Path, ...]] = {}

    def fake_sarcasm(out_path: Path):
        calls["sarcasm"] = (out_path,)

    def fake_sentiment(en_out_path: Path, vi_out_path: Path):
        calls["sentiment"] = (en_out_path, vi_out_path)

    monkeypatch.setattr(downloader, "download_sarcasm_dataset", fake_sarcasm)
    monkeypatch.setattr(downloader, "download_sentiment_datasets", fake_sentiment)

    main(["--task", task])

    root = Path(downloader.__file__).resolve().parents[2]
    if task == "sarcasm":
        assert calls == {"sarcasm": (root / "data" / "raw" / "sarcasm.csv",)}
    else:
        assert calls == {
            "sentiment": (
                root / "data" / "raw" / "sentiment_en.csv",
                root / "data" / "raw" / "sentiment_vi.csv",
            )
        }
    assert expected_name in str(next(iter(calls.values()))[0])

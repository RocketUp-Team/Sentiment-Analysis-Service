import pandas as pd
import pytest

from src.data.downloader import (
    EXPECTED_RAW_ASPECT_COLUMNS,
    EXPECTED_RAW_SENTENCE_COLUMNS,
    SchemaError,
    build_sarcasm_frame,
    build_uit_vsfc_frame,
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

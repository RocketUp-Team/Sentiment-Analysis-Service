import pandas as pd
import pytest

from src.data.downloader import (
    EXPECTED_RAW_ASPECT_COLUMNS,
    EXPECTED_RAW_SENTENCE_COLUMNS,
    SchemaError,
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

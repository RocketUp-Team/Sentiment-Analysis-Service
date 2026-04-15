import pandas as pd
import pytest

from src.data.pipeline import PreprocessingPipeline, _build_transforms_from_params
from src.data.transforms.base import BaseTransform
from src.data.transforms.duplicate_remover import DuplicateRemover
from src.data.transforms.label_mapper import LabelMapper
from src.data.transforms.length_filter import LengthFilter
from src.data.transforms.sentiment_deriver import SentimentDeriver
from src.data.transforms.splitter import Splitter
from src.data.transforms.text_cleaner import TextCleaner


class BadTransform(BaseTransform):
    """Drops a required sentence column so validate_output fails."""

    required_sentence_columns = ["sentence_id", "text"]

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return sentences_df.drop(columns=["text"]).copy(), aspects_df.copy()


class StripSentenceIdTransform(BaseTransform):
    required_sentence_columns: list[str] = []
    required_aspect_columns: list[str] = []

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return sentences_df.drop(columns=["sentence_id"]).copy(), aspects_df.copy()


def _minimal_params(*, remove_duplicates: bool) -> dict:
    return {
        "label_mapping": {"aspect_categories": {}},
        "sentiment_derivation": {"mixed_strategy": "negative_priority"},
        "preprocessing": {
            "lowercase": True,
            "strip_whitespace": True,
            "remove_duplicates": remove_duplicates,
            "min_text_length": 3,
        },
        "data": {
            "max_text_length": 2000,
            "validation_ratio": 0.1,
            "split_seed": 42,
        },
    }


def test_preprocessing_pipeline_cascade():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "text": "same"},
            {"sentence_id": "2", "text": "same"},
        ]
    )
    aspects_df = pd.DataFrame(
        [
            {"sentence_id": "1", "val": "a"},
            {"sentence_id": "2", "val": "b"},
        ]
    )

    pipeline = PreprocessingPipeline([DuplicateRemover()])
    res_s, res_a = pipeline.run(sentences_df, aspects_df)

    assert len(res_s) == 1
    assert res_s.iloc[0]["sentence_id"] == "1"

    # Cascade should remove aspect for sentence_id "2"
    assert len(res_a) == 1
    assert res_a.iloc[0]["sentence_id"] == "1"


def test_pipeline_run_raises_from_validate_output_on_invalid_transform_output():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "ok"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1", "aspect_category": "food", "sentiment": "positive"}])

    pipeline = PreprocessingPipeline([BadTransform()])
    with pytest.raises(ValueError, match="dropped required sentence columns"):
        pipeline.run(sentences_df, aspects_df)


def test_build_transforms_from_params_order_with_duplicate_remover():
    transforms = _build_transforms_from_params(_minimal_params(remove_duplicates=True))
    assert [type(t) for t in transforms] == [
        LabelMapper,
        SentimentDeriver,
        TextCleaner,
        DuplicateRemover,
        LengthFilter,
        Splitter,
    ]


def test_build_transforms_from_params_order_without_duplicate_remover():
    transforms = _build_transforms_from_params(_minimal_params(remove_duplicates=False))
    assert [type(t) for t in transforms] == [
        LabelMapper,
        SentimentDeriver,
        TextCleaner,
        LengthFilter,
        Splitter,
    ]


def test_pipeline_run_raises_when_aspects_missing_sentence_id_at_input():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "a"}])
    aspects_df = pd.DataFrame([{"foo": 1}])
    with pytest.raises(ValueError, match="aspects_df is missing required column 'sentence_id'"):
        PreprocessingPipeline([DuplicateRemover()]).run(sentences_df, aspects_df)


def test_pipeline_run_raises_when_sentences_missing_sentence_id_at_input():
    sentences_df = pd.DataFrame([{"text": "a"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1"}])
    with pytest.raises(ValueError, match="sentences_df is missing required column 'sentence_id'"):
        PreprocessingPipeline([DuplicateRemover()]).run(sentences_df, aspects_df)


def test_pipeline_run_raises_when_sentence_id_missing_after_transform():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "a"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1"}])
    with pytest.raises(
        ValueError,
        match=r"after StripSentenceIdTransform.*sentences_df is missing required column 'sentence_id'",
    ):
        PreprocessingPipeline([StripSentenceIdTransform()]).run(sentences_df, aspects_df)

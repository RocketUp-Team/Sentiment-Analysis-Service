import pandas as pd
import pytest

from src.data.transforms.base import BaseTransform
from src.data.transforms.label_mapper import LabelMapper
from src.data.transforms.sentiment_deriver import SentimentDeriver


class DummyTransform(BaseTransform):
    required_sentence_columns = ["id"]
    required_aspect_columns = ["sentiment"]

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return sentences_df, aspects_df


def test_base_transform_validation():
    t = DummyTransform()
    sentences_df = pd.DataFrame([{"id": 1}])
    aspects_df = pd.DataFrame([{"sentiment": "positive"}])

    # Should not raise
    t.validate_output(sentences_df, aspects_df)

    # Should raise missing sentence columns
    bad_sentences = pd.DataFrame([{"wrong_col": 1}])
    with pytest.raises(ValueError, match="dropped required sentence columns"):
        t.validate_output(bad_sentences, aspects_df)

    # Should raise missing aspect columns
    bad_aspects = pd.DataFrame([{"wrong_col": "positive"}])
    with pytest.raises(ValueError, match="dropped required aspect columns"):
        t.validate_output(sentences_df, bad_aspects)

    # Should raise empty
    with pytest.raises(ValueError, match="produced empty sentences"):
        t.validate_output(pd.DataFrame(), aspects_df)


def test_label_mapper():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo"}])
    aspects_df = pd.DataFrame(
        [
            {
                "sentence_id": "1",
                "aspect_category": "anecdotes/miscellaneous",
                "sentiment": "positive",
            },
            {"sentence_id": "1", "aspect_category": "food", "sentiment": "conflict"},
        ]
    )

    mapper = LabelMapper(aspect_categories={"anecdotes/miscellaneous": "general"})
    _, new_aspects = mapper.transform(sentences_df, aspects_df)

    assert len(new_aspects) == 1
    assert new_aspects.iloc[0]["aspect_category"] == "general"
    assert new_aspects.iloc[0]["sentiment"] == "positive"


def test_sentiment_deriver():
    sentences_df = pd.DataFrame(
        [{"sentence_id": "1", "text": "foo"}, {"sentence_id": "2", "text": "bar"}]
    )
    aspects_df = pd.DataFrame(
        [
            {"sentence_id": "1", "sentiment": "positive"},
            {"sentence_id": "1", "sentiment": "negative"},
            {"sentence_id": "2", "sentiment": "positive"},
        ]
    )

    deriver = SentimentDeriver(mixed_strategy="negative_priority")
    new_sentences, _ = deriver.transform(sentences_df, aspects_df)

    assert (
        new_sentences.loc[new_sentences["sentence_id"] == "1", "sentiment"].iloc[0]
        == "negative"
    )
    assert (
        new_sentences.loc[new_sentences["sentence_id"] == "2", "sentiment"].iloc[0]
        == "positive"
    )

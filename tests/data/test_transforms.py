import pandas as pd
import pytest

from src.data.transforms.base import BaseTransform


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

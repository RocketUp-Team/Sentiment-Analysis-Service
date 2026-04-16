import pandas as pd

from .base import BaseTransform


class LengthFilter(BaseTransform):
    required_sentence_columns = ["text"]
    required_aspect_columns = []

    def __init__(self, min_length: int = 3, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = sentences_df.copy()
        lengths = df["text"].str.len()
        df = df[(lengths >= self.min_length) & (lengths <= self.max_length)].copy()
        return df, aspects_df.copy()

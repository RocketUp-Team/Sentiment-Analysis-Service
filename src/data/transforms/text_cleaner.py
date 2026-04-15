import pandas as pd

from .base import BaseTransform


class TextCleaner(BaseTransform):
    required_sentence_columns = ["text"]
    required_aspect_columns = []

    def __init__(self, lowercase: bool = True, strip_whitespace: bool = True):
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = sentences_df.copy()
        text = df["text"].fillna("").astype(str)

        if self.lowercase:
            text = text.str.lower()

        if self.strip_whitespace:
            text = text.str.replace(r"\s+", " ", regex=True).str.strip()

        df["text"] = text
        df = df[df["text"].str.len() > 0].copy()

        return df, aspects_df.copy()

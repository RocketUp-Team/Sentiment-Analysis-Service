import pandas as pd

from .base import BaseTransform


class DuplicateRemover(BaseTransform):
    required_sentence_columns = ["sentence_id", "text"]
    required_aspect_columns = []

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        deduped = sentences_df.drop_duplicates(subset=["text"], keep="first").copy()
        return deduped, aspects_df.copy()

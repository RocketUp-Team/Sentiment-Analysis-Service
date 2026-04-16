from typing import Dict

import pandas as pd

from .base import BaseTransform


class LabelMapper(BaseTransform):
    required_sentence_columns = []
    required_aspect_columns = ["aspect_category", "sentiment"]

    def __init__(self, aspect_categories: Dict[str, str]):
        self.aspect_categories = aspect_categories

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = aspects_df.copy()
        df = df[df["sentiment"] != "conflict"].copy()

        if self.aspect_categories:
            df["aspect_category"] = df["aspect_category"].replace(
                self.aspect_categories
            )

        return sentences_df.copy(), df

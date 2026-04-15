import math

import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseTransform


class Splitter(BaseTransform):
    required_sentence_columns = ["split", "sentiment"]
    required_aspect_columns = []

    def __init__(self, validation_ratio: float, seed: int):
        self.validation_ratio = validation_ratio
        self.seed = seed

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_mask = sentences_df["split"] == "train"
        train_df = sentences_df[train_mask].copy()
        test_df = sentences_df[~train_mask].copy()

        if len(train_df) < 2:
            return sentences_df, aspects_df

        val_size = math.ceil(len(train_df) * self.validation_ratio)
        sentiment_counts = train_df["sentiment"].value_counts(dropna=False)
        if val_size < sentiment_counts.size or sentiment_counts.min() < 2:
            return sentences_df, aspects_df

        train_final, val_df = train_test_split(
            train_df,
            test_size=self.validation_ratio,
            random_state=self.seed,
            stratify=train_df["sentiment"],
        )
        train_final = train_final.copy()
        val_df = val_df.copy()
        val_df["split"] = "val"

        result = pd.concat([train_final, val_df, test_df], ignore_index=True)
        return result, aspects_df

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseTransform


class Splitter(BaseTransform):
    required_sentence_columns = ["split", "sentiment"]
    required_aspect_columns = []

    def __init__(self, validation_ratio: float, seed: int):
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must satisfy 0 < validation_ratio < 1")
        self.validation_ratio = validation_ratio
        self.seed = seed

    def _validate_stratification_feasibility(self, train_df: pd.DataFrame) -> None:
        class_counts = train_df["sentiment"].value_counts(dropna=False)
        n_classes = len(class_counts)
        val_size = math.ceil(len(train_df) * self.validation_ratio)
        train_size = len(train_df) - val_size

        if (class_counts < 2).any():
            raise ValueError(
                "Splitter cannot stratify train data: each sentiment class must "
                "have at least 2 train rows"
            )

        if val_size < n_classes:
            raise ValueError(
                "Splitter cannot stratify train data: validation split would be "
                f"too small for {n_classes} classes"
            )

        if train_size < n_classes:
            raise ValueError(
                "Splitter cannot stratify train data: train remainder would be "
                f"too small for {n_classes} classes"
            )

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_mask = sentences_df["split"] == "train"
        train_df = sentences_df[train_mask].copy()
        test_df = sentences_df[~train_mask].copy()

        if len(train_df) < 2:
            return sentences_df.copy(), aspects_df.copy()

        self._validate_stratification_feasibility(train_df)

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

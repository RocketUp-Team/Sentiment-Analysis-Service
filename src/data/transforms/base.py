from abc import ABC, abstractmethod

import pandas as pd


class BaseTransform(ABC):
    required_sentence_columns: list[str] = []
    required_aspect_columns: list[str] = []

    @abstractmethod
    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def validate_output(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> None:
        if sentences_df.empty:
            raise ValueError(f"{self.name} produced empty sentences DataFrame")

        if self.required_sentence_columns:
            missing = set(self.required_sentence_columns) - set(sentences_df.columns)
            if missing:
                raise ValueError(
                    f"{self.name} dropped required sentence columns: {missing}"
                )

        if self.required_aspect_columns:
            missing = set(self.required_aspect_columns) - set(aspects_df.columns)
            if missing:
                raise ValueError(
                    f"{self.name} dropped required aspect columns: {missing}"
                )

    @property
    def name(self) -> str:
        return self.__class__.__name__

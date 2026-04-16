import pandas as pd

from .base import BaseTransform


class SentimentDeriver(BaseTransform):
    required_sentence_columns = ["sentence_id", "sentiment"]
    required_aspect_columns = []
    supported_mixed_strategies = {"negative_priority"}

    def __init__(self, mixed_strategy: str = "negative_priority"):
        if mixed_strategy not in self.supported_mixed_strategies:
            raise ValueError(
                f"Unsupported mixed_strategy: {mixed_strategy}. "
                f"Supported values: {sorted(self.supported_mixed_strategies)}"
            )
        self.mixed_strategy = mixed_strategy

    def _derive_overall_sentiment(self, aspect_sentiments: list[str]) -> str:
        sentiments = {
            sentiment for sentiment in aspect_sentiments if pd.notna(sentiment)
        }
        if not sentiments:
            return "neutral"
        if len(sentiments) == 1:
            return sentiments.pop()
        if self.mixed_strategy == "negative_priority" and "negative" in sentiments:
            return "negative"
        return "positive"

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        res_sentences = sentences_df.copy()

        if aspects_df.empty:
            res_sentences["sentiment"] = "neutral"
            return res_sentences, aspects_df

        grouped = (
            aspects_df.groupby("sentence_id")["sentiment"].apply(list).reset_index()
        )
        grouped["sentiment_derived"] = grouped["sentiment"].apply(
            self._derive_overall_sentiment
        )

        res_sentences = pd.merge(
            res_sentences,
            grouped[["sentence_id", "sentiment_derived"]],
            on="sentence_id",
            how="left",
        )
        res_sentences["sentiment"] = res_sentences["sentiment_derived"].fillna(
            "neutral"
        )
        res_sentences.drop(columns=["sentiment_derived"], inplace=True)

        return res_sentences, aspects_df

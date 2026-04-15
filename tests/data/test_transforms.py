import pandas as pd
import pytest

from src.data.transforms.base import BaseTransform
from src.data.transforms.duplicate_remover import DuplicateRemover
from src.data.transforms.label_mapper import LabelMapper
from src.data.transforms.length_filter import LengthFilter
from src.data.transforms.sentiment_deriver import SentimentDeriver
from src.data.transforms.splitter import Splitter
from src.data.transforms.text_cleaner import TextCleaner


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


def test_sentiment_deriver_neutral_for_sentence_without_matching_aspects():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo"}])
    aspects_df = pd.DataFrame([{"sentence_id": "2", "sentiment": "positive"}])

    deriver = SentimentDeriver()
    new_sentences, _ = deriver.transform(sentences_df, aspects_df)

    assert new_sentences.iloc[0]["sentiment"] == "neutral"


def test_sentiment_deriver_neutral_for_empty_aspects_df():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo"}])
    aspects_df = pd.DataFrame(columns=["sentence_id", "sentiment"])

    deriver = SentimentDeriver()
    new_sentences, _ = deriver.transform(sentences_df, aspects_df)

    assert list(new_sentences["sentiment"]) == ["neutral"]


def test_sentiment_deriver_all_negative_and_all_neutral_groups():
    sentences_df = pd.DataFrame(
        [{"sentence_id": "1", "text": "foo"}, {"sentence_id": "2", "text": "bar"}]
    )
    aspects_df = pd.DataFrame(
        [
            {"sentence_id": "1", "sentiment": "negative"},
            {"sentence_id": "1", "sentiment": "negative"},
            {"sentence_id": "2", "sentiment": "neutral"},
        ]
    )

    deriver = SentimentDeriver()
    new_sentences, _ = deriver.transform(sentences_df, aspects_df)

    assert (
        new_sentences.loc[new_sentences["sentence_id"] == "1", "sentiment"].iloc[0]
        == "negative"
    )
    assert (
        new_sentences.loc[new_sentences["sentence_id"] == "2", "sentiment"].iloc[0]
        == "neutral"
    )


def test_sentiment_deriver_mixed_positive_and_neutral_is_positive():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo"}])
    aspects_df = pd.DataFrame(
        [
            {"sentence_id": "1", "sentiment": "positive"},
            {"sentence_id": "1", "sentiment": "neutral"},
        ]
    )

    deriver = SentimentDeriver()
    new_sentences, _ = deriver.transform(sentences_df, aspects_df)

    assert new_sentences.iloc[0]["sentiment"] == "positive"


def test_sentiment_deriver_invalid_strategy_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported mixed_strategy"):
        SentimentDeriver(mixed_strategy="typo")


def test_sentiment_deriver_returns_aspects_unchanged_by_value():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1", "sentiment": "positive"}])

    deriver = SentimentDeriver()
    _, new_aspects = deriver.transform(sentences_df, aspects_df)

    assert new_aspects.equals(aspects_df)


def test_text_cleaner_required_columns():
    assert TextCleaner.required_sentence_columns == ["text"]
    assert TextCleaner.required_aspect_columns == []


def test_text_cleaner_lowercase_false_preserves_casing_and_strips_whitespace():
    sentences_df = pd.DataFrame(
        [{"sentence_id": "1", "text": "  THE Food   was great  \n"}]
    )
    aspects_df = pd.DataFrame()

    cleaner = TextCleaner(lowercase=False, strip_whitespace=True)
    new_sentences, _ = cleaner.transform(sentences_df, aspects_df)

    assert new_sentences.iloc[0]["text"] == "THE Food was great"


def test_text_cleaner_strip_whitespace_false_preserves_spacing_while_lowercasing():
    sentences_df = pd.DataFrame(
        [{"sentence_id": "1", "text": "  THE Food   was great  \n"}]
    )
    aspects_df = pd.DataFrame()

    cleaner = TextCleaner(lowercase=True, strip_whitespace=False)
    new_sentences, _ = cleaner.transform(sentences_df, aspects_df)

    assert new_sentences.iloc[0]["text"] == "  the food   was great  \n"


def test_text_cleaner_drops_empty_and_whitespace_only_rows_when_stripping():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "text": "   "},
            {"sentence_id": "2", "text": "  THE Food   was great  \n"},
        ]
    )
    aspects_df = pd.DataFrame()

    cleaner = TextCleaner(lowercase=True, strip_whitespace=True)
    new_sentences, _ = cleaner.transform(sentences_df, aspects_df)

    assert list(new_sentences["sentence_id"]) == ["2"]
    assert list(new_sentences["text"]) == ["the food was great"]


def test_text_cleaner_does_not_mutate_original_sentences_df():
    sentences_df = pd.DataFrame(
        [{"sentence_id": "1", "text": "  THE Food   was great  \n"}]
    )
    original = sentences_df.copy(deep=True)
    aspects_df = pd.DataFrame()

    cleaner = TextCleaner(lowercase=True, strip_whitespace=True)
    cleaner.transform(sentences_df, aspects_df)

    assert sentences_df.equals(original)


def test_text_cleaner_returns_aspects_unchanged_by_value():
    sentences_df = pd.DataFrame([{"sentence_id": "1", "text": "foo"}])
    aspects_df = pd.DataFrame([{"sentence_id": "1", "sentiment": "positive"}])

    cleaner = TextCleaner(lowercase=True, strip_whitespace=True)
    _, new_aspects = cleaner.transform(sentences_df, aspects_df)

    assert new_aspects.equals(aspects_df)


def test_duplicate_remover():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "text": "same text"},
            {"sentence_id": "2", "text": "same text"},
        ]
    )
    remover = DuplicateRemover()
    new_sentences, _ = remover.transform(sentences_df, pd.DataFrame())

    assert len(new_sentences) == 1
    assert new_sentences.iloc[0]["sentence_id"] == "1"


def test_length_filter():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "text": "abc"},
            {"sentence_id": "2", "text": "ab"},
            {"sentence_id": "3", "text": "abcd"},
        ]
    )
    filterer = LengthFilter(min_length=3, max_length=3)
    new_sentences, _ = filterer.transform(sentences_df, pd.DataFrame())

    assert len(new_sentences) == 1
    assert new_sentences.iloc[0]["sentence_id"] == "1"


def test_splitter():
    sentences_df = pd.DataFrame(
        [{"sentence_id": str(i), "split": "train", "sentiment": "positive"} for i in range(10)]
        + [
            {"sentence_id": str(i + 10), "split": "test", "sentiment": "negative"}
            for i in range(5)
        ]
    )

    splitter = Splitter(validation_ratio=0.2, seed=42)
    new_sentences, _ = splitter.transform(sentences_df, pd.DataFrame())

    val_df = new_sentences[new_sentences["split"] == "val"]
    train_df = new_sentences[new_sentences["split"] == "train"]
    test_df = new_sentences[new_sentences["split"] == "test"]

    assert len(val_df) == 2
    assert len(train_df) == 8
    assert len(test_df) == 5


def test_splitter_multiclass_stratified_split_preserves_each_class_in_val():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "split": "train", "sentiment": "positive"},
            {"sentence_id": "2", "split": "train", "sentiment": "positive"},
            {"sentence_id": "3", "split": "train", "sentiment": "positive"},
            {"sentence_id": "4", "split": "train", "sentiment": "positive"},
            {"sentence_id": "5", "split": "train", "sentiment": "positive"},
            {"sentence_id": "6", "split": "train", "sentiment": "negative"},
            {"sentence_id": "7", "split": "train", "sentiment": "negative"},
            {"sentence_id": "8", "split": "train", "sentiment": "negative"},
            {"sentence_id": "9", "split": "train", "sentiment": "neutral"},
            {"sentence_id": "10", "split": "train", "sentiment": "neutral"},
            {"sentence_id": "11", "split": "test", "sentiment": "negative"},
            {"sentence_id": "12", "split": "test", "sentiment": "neutral"},
        ]
    )

    splitter = Splitter(validation_ratio=0.25, seed=42)
    new_sentences, _ = splitter.transform(sentences_df, pd.DataFrame())

    val_sentiments = set(new_sentences.loc[new_sentences["split"] == "val", "sentiment"])

    assert val_sentiments == {"positive", "negative", "neutral"}


def test_splitter_preserves_non_train_rows_without_relabeling():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "split": "train", "sentiment": "positive"},
            {"sentence_id": "2", "split": "train", "sentiment": "negative"},
            {"sentence_id": "3", "split": "train", "sentiment": "positive"},
            {"sentence_id": "4", "split": "train", "sentiment": "negative"},
            {"sentence_id": "5", "split": "test", "sentiment": "neutral"},
        ]
    )

    splitter = Splitter(validation_ratio=0.5, seed=42)
    new_sentences, _ = splitter.transform(sentences_df, pd.DataFrame())

    test_rows = new_sentences[new_sentences["sentence_id"] == "5"]

    assert list(test_rows["split"]) == ["test"]
    assert "val" not in set(test_rows["split"])


def test_splitter_returns_original_data_when_train_rows_less_than_two():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "split": "train", "sentiment": "positive"},
            {"sentence_id": "2", "split": "test", "sentiment": "negative"},
        ]
    )
    aspects_df = pd.DataFrame([{"sentence_id": "1", "sentiment": "positive"}])

    splitter = Splitter(validation_ratio=0.2, seed=42)
    new_sentences, new_aspects = splitter.transform(sentences_df, aspects_df)

    assert new_sentences.equals(sentences_df)
    assert new_aspects.equals(aspects_df)


def test_splitter_imbalanced_train_labels_raise_value_error():
    sentences_df = pd.DataFrame(
        [
            {"sentence_id": "1", "split": "train", "sentiment": "positive"},
            {"sentence_id": "2", "split": "train", "sentiment": "positive"},
            {"sentence_id": "3", "split": "train", "sentiment": "positive"},
            {"sentence_id": "4", "split": "train", "sentiment": "positive"},
            {"sentence_id": "5", "split": "train", "sentiment": "negative"},
        ]
    )

    splitter = Splitter(validation_ratio=0.5, seed=42)

    with pytest.raises(ValueError, match="cannot stratify"):
        splitter.transform(sentences_df, pd.DataFrame())


def test_splitter_invalid_validation_ratio_raises_value_error():
    with pytest.raises(ValueError, match="validation_ratio"):
        Splitter(validation_ratio=1.0, seed=42)

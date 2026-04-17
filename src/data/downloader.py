"""Raw dataset acquisition helpers and schema checks.

This module parses the SemEval-2014 XML datasets directly into structured Pandas DataFrames.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.data.utils import load_params

EXPECTED_RAW_SENTENCE_COLUMNS = {"sentence_id", "text", "split"}
EXPECTED_RAW_ASPECT_COLUMNS = {"sentence_id", "aspect_category", "sentiment"}
_UIT_SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
_MULTILINGUAL_SENTIMENTS_EN_FILES = {
    "train": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/train/*.parquet",
    "validation": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/validation/*.parquet",
    "test": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/test/*.parquet",
}
_UIT_VSFC_FILES = {
    "train": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/train/*.parquet",
    "validation": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/validation/*.parquet",
    "test": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/test/*.parquet",
}


class SchemaError(Exception):
    """Raised when raw sentence/aspect frames do not match the expected schema."""


def _load_hf_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def build_sarcasm_frame(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Normalize loaded tweet-eval irony splits into a single training frame."""
    rows: list[pd.DataFrame] = []
    for split, frame in split_frames.items():
        normalized = frame.loc[:, ["text", "label"]].copy()
        normalized["lang"] = "en"
        normalized["split"] = split
        normalized["source"] = "tweet_eval_irony"
        rows.append(normalized)

    if not rows:
        return pd.DataFrame(columns=["text", "label", "lang", "split", "source"])

    return pd.concat(rows, ignore_index=True)


def build_uit_vsfc_frame(
    *,
    sentences: list[str],
    labels: list[int],
    split: str,
) -> pd.DataFrame:
    """Normalize UIT-VSFC split files into the shared sentiment frame."""
    if len(sentences) != len(labels):
        raise ValueError("UIT-VSFC sentence and label counts must match")

    return pd.DataFrame(
        {
            "text": sentences,
            "label": [_UIT_SENTIMENT_MAP[label] for label in labels],
            "lang": ["vi"] * len(labels),
            "split": [split] * len(labels),
            "source": ["uit_vsfc"] * len(labels),
        }
    )


def download_sarcasm_dataset(out_path: Path) -> None:
    """Download the tweet_eval irony training split and persist it as CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = _load_hf_dataset("tweet_eval", "irony")
    df = build_sarcasm_frame(
        {split: split_dataset.to_pandas() for split, split_dataset in dataset.items()}
    )
    df[["text", "label", "lang", "split", "source"]].to_csv(out_path, index=False)
    print(f"Saved sarcasm dataset to {out_path}")


def download_sentiment_datasets(en_out_path: Path, vi_out_path: Path) -> None:
    """Download English and Vietnamese sentiment datasets and persist them as CSV."""
    en_out_path.parent.mkdir(parents=True, exist_ok=True)
    vi_out_path.parent.mkdir(parents=True, exist_ok=True)

    en_ds = _load_hf_dataset("parquet", data_files=_MULTILINGUAL_SENTIMENTS_EN_FILES)
    en_frames: list[pd.DataFrame] = []
    for split, split_ds in en_ds.items():
        split_df = split_ds.to_pandas().loc[:, ["text", "label"]].copy()
        split_df["lang"] = "en"
        split_df["source"] = "multilingual_sentiments"
        split_df["split"] = split
        en_frames.append(split_df)

    en_df = pd.concat(en_frames, ignore_index=True)
    en_df[["text", "label", "lang", "source", "split"]].to_csv(en_out_path, index=False)

    vi_ds = _load_hf_dataset("parquet", data_files=_UIT_VSFC_FILES)
    vi_frames: list[pd.DataFrame] = []
    for split, split_ds in vi_ds.items():
        split_df = split_ds.to_pandas()
        vi_frames.append(
            build_uit_vsfc_frame(
                sentences=split_df["sentence"].tolist(),
                labels=split_df["sentiment"].tolist(),
                split=split,
            )
        )

    vi_df = pd.concat(vi_frames, ignore_index=True)
    vi_df[["text", "label", "lang", "source", "split"]].to_csv(vi_out_path, index=False)
    print(f"Saved sentiment datasets to {en_out_path} and {vi_out_path}")


def validate_raw_schema(sentences_df: pd.DataFrame, aspects_df: pd.DataFrame) -> None:
    sentence_cols = set(sentences_df.columns)
    aspect_cols = set(aspects_df.columns)

    missing_sentences = EXPECTED_RAW_SENTENCE_COLUMNS - sentence_cols
    if missing_sentences:
        raise SchemaError(
            f"sentences dataframe missing required columns: {sorted(missing_sentences)}"
        )

    missing_aspects = EXPECTED_RAW_ASPECT_COLUMNS - aspect_cols
    if missing_aspects:
        raise SchemaError(
            f"aspects dataframe missing required columns: {sorted(missing_aspects)}"
        )

    if sentences_df.shape[0] == 0 or aspects_df.shape[0] == 0:
        raise SchemaError("raw sentence and aspect frames must each contain at least one row")


def extract_semeval_xmls(
    raw_dir: str | Path,
    *,
    dataset_name: str,
    splits: list[str],
) -> tuple[Path, Path]:
    """Extract real SemEval-2014 XML files into raw CSV frames aligned with ``data.*`` params."""
    if not splits:
        raise ValueError("splits must be a non-empty list to generate placeholder raw rows")

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    external_dir = raw_path.parent / "external" / "semeval2014"
    split_file_map = {
        "train": external_dir / "Restaurants_Train_v2.xml",
        "test": external_dir / "Restaurants_Test_Gold.xml",
    }

    sentence_rows: list[dict[str, str]] = []
    aspect_rows: list[dict[str, str]] = []
    
    for split in splits:
        xml_path = split_file_map.get(split)
        if not xml_path or not xml_path.exists():
            print(f"Warning: Missing real dataset file for split '{split}': {xml_path}")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for sentence in root.findall("sentence"):
            s_id = sentence.get("id")
            text_node = sentence.find("text")
            text = text_node.text if text_node is not None else ""
            
            sentence_rows.append({
                "sentence_id": str(s_id),
                "text": text,
                "split": split,
            })
            
            aspect_categories = sentence.find("aspectCategories")
            if aspect_categories is not None:
                for aspect in aspect_categories.findall("aspectCategory"):
                    category = aspect.get("category")
                    sentiment = aspect.get("polarity")
                    if sentiment and str(sentiment).strip().lower() not in {"none", "null", "na"}:
                        aspect_rows.append({
                            "sentence_id": str(s_id),
                            "aspect_category": category,
                            "sentiment": sentiment,
                        })

    if not sentence_rows or not aspect_rows:
        raise RuntimeError("No matching sentences or aspects extracted from XML.")

    sentences_df = pd.DataFrame(sentence_rows)
    aspects_df = pd.DataFrame(aspect_rows)

    # ensure columns exist
    for col in EXPECTED_RAW_SENTENCE_COLUMNS:
        if col not in sentences_df:
            sentences_df[col] = pd.Series(dtype=str)
    for col in EXPECTED_RAW_ASPECT_COLUMNS:
        if col not in aspects_df:
            aspects_df[col] = pd.Series(dtype=str)

    sentences_df = sentences_df[sorted(EXPECTED_RAW_SENTENCE_COLUMNS)]
    aspects_df = aspects_df[sorted(EXPECTED_RAW_ASPECT_COLUMNS)]

    sentences_path = raw_path / "sentences.csv"
    aspects_path = raw_path / "aspects.csv"
    sentences_df.to_csv(sentences_path, index=False)
    aspects_df.to_csv(aspects_path, index=False)
    return sentences_path, aspects_path


def write_placeholder_raw_csvs(
    raw_dir: str | Path,
    *,
    dataset_name: str,
    splits: list[str],
) -> tuple[Path, Path]:
    """Generate dummy SemEval-2014 style row stubs in raw CSV format."""
    if not splits:
        raise ValueError("splits must be a non-empty list to generate placeholder raw rows")

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    sentence_rows: list[dict[str, str]] = []
    aspect_rows: list[dict[str, str]] = []

    # create 2000 rows per split to match existing tests
    for i, split in enumerate(splits):
        for j in range(2000):
            s_id = f"{i}_{j}"
            sentence_rows.append({
                "sentence_id": s_id,
                "text": f"Placeholder text {j} for dataset {dataset_name} in split {split}",
                "split": split,
            })
            aspect_rows.append({
                "sentence_id": s_id,
                "aspect_category": "food",
                "sentiment": "neutral",
            })

    sentences_df = pd.DataFrame(sentence_rows)
    aspects_df = pd.DataFrame(aspect_rows)

    sentences_path = raw_path / "sentences.csv"
    aspects_path = raw_path / "aspects.csv"
    sentences_df.to_csv(sentences_path, index=False)
    aspects_df.to_csv(aspects_path, index=False)
    return sentences_path, aspects_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="semeval",
        choices=["semeval", "sarcasm", "sentiment"],
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]

    if args.task == "semeval":
        params = load_params(str(root / "params.yaml"))
        data = params["data"]
        extract_semeval_xmls(
            root / "data" / "raw",
            dataset_name=str(data["dataset_name"]),
            splits=list(data["splits"]),
        )
    elif args.task == "sarcasm":
        download_sarcasm_dataset(root / "data" / "raw" / "sarcasm.csv")
    elif args.task == "sentiment":
        download_sentiment_datasets(
            root / "data" / "raw" / "sentiment_en.csv",
            root / "data" / "raw" / "sentiment_vi.csv",
        )


if __name__ == "__main__":
    main()

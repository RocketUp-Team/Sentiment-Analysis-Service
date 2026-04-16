"""Raw dataset acquisition helpers and schema checks.

This module parses the SemEval-2014 XML datasets directly into structured Pandas DataFrames.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.data.utils import load_params

EXPECTED_RAW_SENTENCE_COLUMNS = {"sentence_id", "text", "split"}
EXPECTED_RAW_ASPECT_COLUMNS = {"sentence_id", "aspect_category", "sentiment"}


class SchemaError(Exception):
    """Raised when raw sentence/aspect frames do not match the expected schema."""


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


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    params = load_params(str(root / "params.yaml"))
    data = params["data"]
    extract_semeval_xmls(
        root / "data" / "raw",
        dataset_name=str(data["dataset_name"]),
        splits=list(data["splits"]),
    )

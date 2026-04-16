"""Raw dataset acquisition helpers and schema checks.

This module currently writes placeholder CSVs so the DVC stage shape and schema
validation can be exercised before the real SemEval XML ingestion is implemented.
"""

from __future__ import annotations

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


def write_placeholder_raw_csvs(
    raw_dir: str | Path,
    *,
    dataset_name: str,
    splits: list[str],
) -> tuple[Path, Path]:
    """Write minimal non-empty stub raw CSVs aligned with ``data.*`` params.

    This is intentionally a temporary stand-in for the real SemEval XML parser.
    """
    if not splits:
        raise ValueError("splits must be a non-empty list to generate placeholder raw rows")

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    sentence_rows: list[dict[str, str]] = []
    aspect_rows: list[dict[str, str]] = []
    for split in splits:
        for i in range(2000):
            sentence_id = f"placeholder-{split}-{i}"
            sentence_rows.append(
                {
                    "sentence_id": sentence_id,
                    "text": f"placeholder raw row for {dataset_name} ({split}) - sample {i}",
                    "split": split,
                }
            )
            aspect_rows.append(
                {
                    "sentence_id": sentence_id,
                    "aspect_category": "general",
                    "sentiment": "neutral",
                }
            )

    sentences_df = pd.DataFrame(sentence_rows)[sorted(EXPECTED_RAW_SENTENCE_COLUMNS)]
    aspects_df = pd.DataFrame(aspect_rows)[sorted(EXPECTED_RAW_ASPECT_COLUMNS)]

    sentences_path = raw_path / "sentences.csv"
    aspects_path = raw_path / "aspects.csv"
    sentences_df.to_csv(sentences_path, index=False)
    aspects_df.to_csv(aspects_path, index=False)
    return sentences_path, aspects_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    params = load_params(str(root / "params.yaml"))
    data = params["data"]
    write_placeholder_raw_csvs(
        root / "data" / "raw",
        dataset_name=str(data["dataset_name"]),
        splits=list(data["splits"]),
    )

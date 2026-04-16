"""Preprocessing pipeline: ordered transforms with aspect cascade and validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.transforms.base import BaseTransform
from src.data.transforms.duplicate_remover import DuplicateRemover
from src.data.transforms.label_mapper import LabelMapper
from src.data.transforms.length_filter import LengthFilter
from src.data.transforms.sentiment_deriver import SentimentDeriver
from src.data.transforms.splitter import Splitter
from src.data.transforms.text_cleaner import TextCleaner
from src.data.utils import load_params

logger = logging.getLogger("data_pipeline")


def _require_sentence_id_columns(
    sentences_df: pd.DataFrame, aspects_df: pd.DataFrame, *, after: str
) -> None:
    """Cascade requires ``sentence_id`` on both frames; fail fast if missing."""
    if "sentence_id" not in sentences_df.columns:
        raise ValueError(
            f"Cannot cascade orphan aspects after {after}: "
            "sentences_df is missing required column 'sentence_id'"
        )
    if "sentence_id" not in aspects_df.columns:
        raise ValueError(
            f"Cannot cascade orphan aspects after {after}: "
            "aspects_df is missing required column 'sentence_id'"
        )


class PreprocessingPipeline:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def run(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run transforms in order; cascade aspects by ``sentence_id``; validate after each step."""
        s = sentences_df.copy()
        a = aspects_df.copy()
        _require_sentence_id_columns(s, a, after="pipeline input")
        for t in self.transforms:
            s, a = t.transform(s, a)
            _require_sentence_id_columns(s, a, after=t.name)
            a = a[a["sentence_id"].isin(s["sentence_id"])].copy()
            t.validate_output(s, a)
            logger.info(
                "%s: sentences=%d aspects=%d",
                t.name,
                len(s),
                len(a),
            )
        return s, a


def _build_transforms_from_params(params: dict) -> list[BaseTransform]:
    pre = params["preprocessing"]
    data_cfg = params["data"]
    transforms: list[BaseTransform] = [
        LabelMapper(aspect_categories=params["label_mapping"]["aspect_categories"]),
        SentimentDeriver(
            mixed_strategy=params["sentiment_derivation"]["mixed_strategy"]
        ),
        TextCleaner(
            lowercase=pre["lowercase"],
            strip_whitespace=pre["strip_whitespace"],
        ),
    ]
    if pre["remove_duplicates"]:
        transforms.append(DuplicateRemover())
    transforms.append(
        LengthFilter(
            min_length=pre["min_text_length"],
            max_length=data_cfg["max_text_length"],
        )
    )
    transforms.append(
        Splitter(
            validation_ratio=data_cfg["validation_ratio"],
            seed=data_cfg["split_seed"],
        )
    )
    return transforms


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    root = Path(__file__).resolve().parents[2]
    params = load_params(str(root / "params.yaml"))
    pipeline = PreprocessingPipeline(_build_transforms_from_params(params))
    sentences = pd.read_csv(root / "data" / "raw" / "sentences.csv", dtype=str)
    aspects = pd.read_csv(root / "data" / "raw" / "aspects.csv", dtype=str)
    out_s, out_a = pipeline.run(sentences, aspects)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_s.to_csv(out_dir / "sentences.csv", index=False)
    out_a.to_csv(out_dir / "aspects.csv", index=False)

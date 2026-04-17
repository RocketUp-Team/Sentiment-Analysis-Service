"""CLI script to prepare held-out evaluation sets for Phase 2 finetuned evaluation.

This script extracts evaluation slices from the raw downloaded datasets and writes
them to ``data/eval/`` so that downstream DVC stages have a stable, tracked input.

Outputs
-------
data/eval/mixed_lang_eval.csv
    Balanced EN + VI sample drawn from the test splits of the sentiment datasets.
    Columns: text, label, lang, source.

data/eval/vi_sarcasm_eval.csv
    Vietnamese sarcasm probe set.  Because no labelled Vietnamese sarcasm data exists
    in Phase 2, this file contains a small set of manually-curated probe texts with
    informational labels (confidence is expected to be low; results are gated behind
    the null-safe fallback documented in the Phase 2 design).
    Columns: text, label, lang, source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Probe rows for the vi-sarcasm eval file.
# These are manually curated examples used to track Vietnamese sarcasm signal.
# Labels are informational; the target F1 is >= 0.45 (Phase 2 design).
# ---------------------------------------------------------------------------
_VI_SARCASM_PROBE_ROWS: list[dict] = [
    {"text": "Ừ thì đúng là tuyệt vời lắm", "label": "irony", "lang": "vi", "source": "manual_probe"},
    {"text": "Dịch vụ siêu tốt, chờ 2 tiếng mới được phục vụ", "label": "irony", "lang": "vi", "source": "manual_probe"},
    {"text": "Chất lượng hoàn hảo, chỉ bị hỏng ngay ngày đầu tiên thôi", "label": "irony", "lang": "vi", "source": "manual_probe"},
    {"text": "Cảm ơn vì đã phục vụ rất chu đáo", "label": "non_irony", "lang": "vi", "source": "manual_probe"},
    {"text": "Sản phẩm tốt, đáng mua", "label": "non_irony", "lang": "vi", "source": "manual_probe"},
    {"text": "Thức ăn ngon, phục vụ nhiệt tình", "label": "non_irony", "lang": "vi", "source": "manual_probe"},
]


def _load_sentiment_test_rows(root: Path) -> pd.DataFrame:
    """Load test-split rows from raw sentiment CSVs."""
    en_path = root / "data" / "raw" / "sentiment_en.csv"
    vi_path = root / "data" / "raw" / "sentiment_vi.csv"

    frames: list[pd.DataFrame] = []
    for path, lang_fallback in [(en_path, "en"), (vi_path, "vi")]:
        if not path.exists():
            print(f"Warning: {path} not found, skipping.", flush=True)
            continue
        df = pd.read_csv(path)
        if "lang" not in df.columns:
            df["lang"] = lang_fallback
        # Keep only the test split when split information is available.
        if "split" in df.columns:
            df = df[df["split"].astype(str) == "test"].copy()
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No sentiment raw data found under data/raw/.  "
            "Run the download_sentiment DVC stage first."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Sample at most 200 rows per language for a fast, balanced mixed-lang eval.
    sampled_parts: list[pd.DataFrame] = []
    for lang, group in combined.groupby("lang"):
        sampled_parts.append(group.sample(n=min(200, len(group)), random_state=42))

    return pd.concat(sampled_parts, ignore_index=True)[["text", "label", "lang", "source"]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2 held-out eval sets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval"),
        help="Directory to write eval CSV files into.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    out_dir: Path = root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- mixed-language sentiment eval ----------------------------------------
    mixed_df = _load_sentiment_test_rows(root)
    mixed_out = out_dir / "mixed_lang_eval.csv"
    mixed_df.to_csv(mixed_out, index=False)
    print(f"Wrote {len(mixed_df)} rows to {mixed_out}", flush=True)

    # --- Vietnamese sarcasm probe set -----------------------------------------
    vi_sarcasm_df = pd.DataFrame(_VI_SARCASM_PROBE_ROWS)
    vi_out = out_dir / "vi_sarcasm_eval.csv"
    vi_sarcasm_df.to_csv(vi_out, index=False)
    print(f"Wrote {len(vi_sarcasm_df)} rows to {vi_out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

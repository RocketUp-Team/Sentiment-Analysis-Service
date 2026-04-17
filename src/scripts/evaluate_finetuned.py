"""CLI entrypoint for evaluating finetuned Phase 2 adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate finetuned Phase 2 adapters.")
    parser.add_argument(
        "--task",
        required=True,
        choices=("sarcasm", "sentiment"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics_finetuned.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Placeholder evaluation entrypoint with validated CLI args."""
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "task": args.task,
        "overall_f1": 0.0,
        "n_samples": 0,
    }
    per_language_payload = {
        "per_lang_f1": {"en": 0.0, "vi": 0.0, "other": 0.0},
    }
    fairness_payload = {
        "overall_f1": 0.0,
        "per_lang_f1": {"en": 0.0, "vi": 0.0, "other": 0.0},
        "per_lang_gap": 0.0,
        "sample_counts": {"en": 0, "vi": 0, "other": 0},
        "confusion_matrices": {},
    }

    args.output.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    args.output.parent.joinpath("per_language_f1.json").write_text(
        json.dumps(per_language_payload, indent=2),
        encoding="utf-8",
    )
    args.output.parent.joinpath("fairness_report.json").write_text(
        json.dumps(fairness_payload, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

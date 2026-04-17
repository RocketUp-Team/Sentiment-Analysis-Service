"""CLI entrypoint for Phase 2 finetuning."""

from __future__ import annotations

import argparse
import getpass
import logging
import subprocess
import sys

from src.training.mlflow_callback import build_run_tags, resolve_tracking_uri
from src.training.task_configs import get_task_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse finetuning CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Phase 2 finetuning.")
    parser.add_argument(
        "--task",
        required=True,
        choices=("sarcasm", "sentiment"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a lightweight smoke configuration instead of full training.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLflow tracking URI.",
    )
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    """Resolve the requested task and print a run summary."""
    args = parse_args(argv)
    task = get_task_config(args.task)
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    tags = build_run_tags(
        task=task.name,
        git_sha=_git_sha(),
        device="cpu",
        environment="local",
        dataset_version=task.dataset_version,
        seed=task.seed,
        user=getpass.getuser(),
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(
        "Prepared %s finetuning run (smoke=%s, tracking_uri=%s, tags=%s)",
        task.name,
        args.smoke,
        tracking_uri,
        tags,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

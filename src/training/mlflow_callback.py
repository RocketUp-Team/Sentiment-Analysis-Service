"""MLflow helpers for Phase 2 finetuning runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


REQUIRED_TAGS = [
    "task",
    "git_sha",
    "device",
    "environment",
    "dataset_version",
    "seed",
    "user",
]


def resolve_tracking_uri(explicit_uri: str | None = None) -> str:
    """Resolve the MLflow tracking URI from CLI or environment."""
    return explicit_uri or os.getenv("MLFLOW_TRACKING_URI") or "file:./mlruns"


def build_run_tags(
    *,
    task: str,
    git_sha: str,
    device: str,
    environment: str,
    dataset_version: str,
    seed: int,
    user: str,
) -> dict[str, Any]:
    """Build the required Phase 2 MLflow tag payload."""
    return {
        "task": task,
        "git_sha": git_sha,
        "device": device,
        "environment": environment,
        "dataset_version": dataset_version,
        "seed": str(seed),
        "user": user,
    }

"""Task-specific training configuration for Phase 2 finetuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Training settings shared by CLI, trainer, and evaluation flows."""

    name: str
    adapter_name: str
    num_labels: int
    label_names: tuple[str, ...]
    languages: tuple[str, ...]
    epochs: int
    learning_rate: float = 2e-4
    batch_size: int = 16        # Aligned with params.yaml training.*.batch_size
    gradient_accumulation_steps: int = 2  # Aligned with params.yaml; effective batch = 32
    max_length: int = 128
    dataset_version: str = "v1"
    seed: int = 42


_TASKS: dict[str, TaskConfig] = {
    "sarcasm": TaskConfig(
        name="sarcasm",
        adapter_name="sarcasm",
        num_labels=2,
        label_names=("non_irony", "irony"),
        languages=("en",),
        epochs=3,
        dataset_version="tweet_eval_irony_v1",
    ),
    "sentiment": TaskConfig(
        name="sentiment",
        adapter_name="sentiment",
        num_labels=3,
        label_names=("negative", "neutral", "positive"),
        languages=("en", "vi"),
        epochs=5,
        dataset_version="multilingual_sentiment_v1",
    ),
}


def get_task_config(task: str) -> TaskConfig:
    """Return the immutable config for a supported training task."""
    try:
        return _TASKS[task]
    except KeyError as exc:
        raise ValueError(f"Unknown training task: {task}") from exc


def list_task_configs() -> tuple[TaskConfig, ...]:
    """Return all supported task configs in registry order."""
    return tuple(_TASKS.values())

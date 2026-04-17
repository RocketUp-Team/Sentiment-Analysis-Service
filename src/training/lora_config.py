"""LoRA configuration factory for Phase 2 finetuning."""

from __future__ import annotations

from peft import LoraConfig, TaskType

from src.training.task_configs import TaskConfig, get_task_config


def build_lora_config(task: str | TaskConfig) -> LoraConfig:
    """Build a shared LoRA config for a supported training task."""
    task_config = get_task_config(task) if isinstance(task, str) else task

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )

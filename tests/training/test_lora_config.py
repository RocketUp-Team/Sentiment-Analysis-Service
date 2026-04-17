from peft import LoraConfig, TaskType

from src.training.lora_config import build_lora_config


def test_default_lora_targets_query_value():
    config = build_lora_config(task="sarcasm")

    assert isinstance(config, LoraConfig)
    assert set(config.target_modules) == {"query", "value"}


def test_lora_hyperparameters_match_phase2_defaults():
    config = build_lora_config(task="sentiment")

    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.task_type == TaskType.SEQ_CLS
    assert config.bias == "none"

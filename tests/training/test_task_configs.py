import pytest

from src.training.task_configs import TaskConfig, get_task_config


def test_sentiment_task_is_multilingual():
    task = get_task_config("sentiment")

    assert isinstance(task, TaskConfig)
    assert set(task.languages) == {"en", "vi"}
    assert task.num_labels == 3
    assert task.epochs == 5


def test_sarcasm_task_is_english_binary():
    task = get_task_config("sarcasm")

    assert task.languages == ("en",)
    assert task.num_labels == 2
    assert task.label_names == ("non_irony", "irony")
    assert task.epochs == 3


def test_unknown_task_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown training task"):
        get_task_config("absa")

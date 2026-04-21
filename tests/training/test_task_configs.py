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


def test_sentiment_task_enables_both_balancing_techniques():
    task = get_task_config("sentiment")

    assert task.use_class_weights is True
    assert task.oversample_minority is True
    assert 0.0 < task.oversample_target_ratio <= 1.0


def test_sarcasm_task_uses_class_weights_only():
    task = get_task_config("sarcasm")

    assert task.use_class_weights is True
    # Sarcasm IR=1.52x → oversample không cần thiết
    assert task.oversample_minority is False


def test_task_config_default_balance_fields():
    # TaskConfig mặc định: class weights bật, oversample tắt
    task = TaskConfig(
        name="test",
        adapter_name="test",
        num_labels=2,
        label_names=("a", "b"),
        languages=("en",),
        epochs=1,
    )

    assert task.use_class_weights is True
    assert task.oversample_minority is False
    assert task.oversample_target_ratio == 0.15

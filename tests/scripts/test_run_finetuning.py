from src.scripts.run_finetuning import parse_args
from src.training.mlflow_callback import REQUIRED_TAGS, build_run_tags, resolve_tracking_uri


def test_run_finetuning_uses_local_mlflow_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_tracking_uri() == "file:./mlruns"


def test_parse_args_accepts_supported_tasks():
    args = parse_args(["--task", "sarcasm", "--smoke"])

    assert args.task == "sarcasm"
    assert args.smoke is True


def test_build_run_tags_contains_required_schema():
    tags = build_run_tags(
        task="sentiment",
        git_sha="abc1234",
        device="cpu",
        environment="local",
        dataset_version="v1",
        seed=42,
        user="tester",
    )

    assert REQUIRED_TAGS == [
        "task",
        "git_sha",
        "device",
        "environment",
        "dataset_version",
        "seed",
        "user",
    ]
    assert set(REQUIRED_TAGS).issubset(tags)

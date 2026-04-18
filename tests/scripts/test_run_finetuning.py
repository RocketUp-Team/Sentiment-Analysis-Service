from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest
from transformers import TrainingArguments

from src.scripts import run_finetuning
from src.scripts.run_finetuning import parse_args
from src.training.task_configs import get_task_config
from src.training.mlflow_callback import (
    REQUIRED_TAGS,
    build_run_tags,
    resolve_pipeline_tracking_uri,
    resolve_tracking_uri,
)


def test_run_finetuning_uses_local_mlflow_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_tracking_uri() == "file:./mlruns"


def test_resolve_pipeline_tracking_uri_prefers_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/r.mlflow")

    assert (
        resolve_pipeline_tracking_uri({"tracking_uri": "http://localhost:5000"})
        == "https://dagshub.com/u/r.mlflow"
    )


def test_resolve_pipeline_tracking_uri_uses_yaml_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert (
        resolve_pipeline_tracking_uri({"tracking_uri": "http://mlflow:5000"})
        == "http://mlflow:5000"
    )


def test_resolve_pipeline_tracking_uri_fallback_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_pipeline_tracking_uri({}) == "http://localhost:5000"


def test_resolve_pipeline_tracking_uri_no_fallback_when_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert resolve_pipeline_tracking_uri({}, fallback=None) == ""


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


class FakeSplitDataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.map_calls: list[dict] = []
        self.train = [{"split": "train"}]
        self.test = [{"split": "test"}]

    def map(self, func, batched: bool):
        self.map_calls.append({"batched": batched})
        func({"text": [row["text"] for row in self.rows]})
        return self


class FakeDatasetFactory:
    def __init__(self) -> None:
        self.rows_by_split: dict[str, list[dict]] = {}
        self.datasets_by_split: dict[str, FakeSplitDataset] = {}

    def from_list(self, rows: list[dict]) -> FakeSplitDataset:
        split_name = "train" if "train" not in self.rows_by_split else "test"
        self.rows_by_split[split_name] = rows
        dataset = FakeSplitDataset(rows)
        self.datasets_by_split[split_name] = dataset
        return dataset


class FakeDatasetDict(dict):
    def map(self, func, batched: bool):
        for dataset in self.values():
            dataset.map(func, batched=batched)
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, texts, truncation: bool, max_length: int):
        self.calls.append(
            {
                "texts": list(texts),
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        return {
            "input_ids": [[101] for _ in texts],
            "attention_mask": [[1] for _ in texts],
        }


class FakeAutoTokenizer:
    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.model_name: str | None = None

    def from_pretrained(self, model_name: str) -> FakeTokenizer:
        self.model_name = model_name
        return self.tokenizer


class FakeAutoModelForSequenceClassification:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def from_pretrained(self, model_name: str, num_labels: int):
        call = {"model_name": model_name, "num_labels": num_labels}
        self.calls.append(call)
        return call


class FakePeftModel:
    def __init__(self, base_model) -> None:
        self.base_model = base_model
        self.print_trainable_parameters_called = False
        self.saved_paths: list[str] = []

    def print_trainable_parameters(self) -> None:
        self.print_trainable_parameters_called = True

    def save_pretrained(self, path: str) -> None:
        self.saved_paths.append(path)


class FakeTrainer:
    instances: list["FakeTrainer"] = []

    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class=None,
        data_collator=None,
    ) -> None:
        self.kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_class,
            "data_collator": data_collator,
        }
        self.train_calls = 0
        type(self).instances.append(self)

    def train(self) -> None:
        self.train_calls += 1


class FakeTrainingArguments:
    instances: list["FakeTrainingArguments"] = []

    def __init__(self, *, eval_strategy=None, evaluation_strategy=None, **kwargs) -> None:
        assert eval_strategy == "epoch"
        assert evaluation_strategy is None
        self.kwargs = {"eval_strategy": eval_strategy, **kwargs}
        type(self).instances.append(self)


class FakeLegacyTrainingArguments:
    def __init__(self, *, evaluation_strategy=None, **kwargs) -> None:
        assert evaluation_strategy == "epoch"
        self.kwargs = {"evaluation_strategy": evaluation_strategy, **kwargs}


class FakeModernTrainingArguments:
    def __init__(self, *, eval_strategy=None, **kwargs) -> None:
        assert eval_strategy == "epoch"
        self.kwargs = {"eval_strategy": eval_strategy, **kwargs}


class FakeLegacyTrainer:
    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer=None,
        data_collator=None,
    ) -> None:
        self.kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "data_collator": data_collator,
        }


class FakeModernTrainer:
    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class=None,
        data_collator=None,
    ) -> None:
        self.kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_class,
            "data_collator": data_collator,
        }


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.tags: list[dict] = []
        self.start_run_calls = 0

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def set_experiment(self, experiment_name: str) -> None:
        self.experiments.append(experiment_name)

    def set_tags(self, tags: dict) -> None:
        self.tags.append(tags)

    def start_run(self):
        self.start_run_calls += 1
        return nullcontext()


def _build_rows(count: int, label: str | int, lang: str = "en") -> list[dict]:
    return [
        {"text": f"text-{idx}", "label": label, "lang": lang}
        for idx in range(count)
    ]


def _install_training_fakes(monkeypatch, read_csv_tables: dict[str, pd.DataFrame]):
    dataset_factory = FakeDatasetFactory()
    tokenizer = FakeTokenizer()
    auto_tokenizer = FakeAutoTokenizer(tokenizer)
    auto_model = FakeAutoModelForSequenceClassification()
    fake_peft_model: dict[str, FakePeftModel] = {}
    fake_mlflow = FakeMlflow()
    dedup_inputs: list[list[dict]] = []
    collator_calls: list[FakeTokenizer] = []
    lora_calls: list[str] = []

    def fake_read_csv(path):
        return read_csv_tables[Path(path).name].copy()

    def fake_dedup_rows(rows: list[dict]) -> list[dict]:
        dedup_inputs.append(rows)
        return rows

    def fake_data_collator_with_padding(*, tokenizer):
        collator_calls.append(tokenizer)
        return {"tokenizer": tokenizer}

    def fake_build_lora_config(task):
        lora_calls.append(task.name)
        return {"task": task.name}

    def fake_get_peft_model(model, lora_config):
        peft_model = FakePeftModel(model)
        fake_peft_model["model"] = peft_model
        fake_peft_model["lora_config"] = lora_config
        return peft_model

    monkeypatch.setattr(run_finetuning.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(run_finetuning, "dedup_rows", fake_dedup_rows)
    monkeypatch.setattr(run_finetuning, "Dataset", dataset_factory)
    monkeypatch.setattr(run_finetuning, "DatasetDict", FakeDatasetDict)
    monkeypatch.setattr(run_finetuning, "AutoTokenizer", auto_tokenizer)
    monkeypatch.setattr(run_finetuning, "AutoModelForSequenceClassification", auto_model)
    monkeypatch.setattr(run_finetuning, "DataCollatorWithPadding", fake_data_collator_with_padding)
    monkeypatch.setattr(run_finetuning, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(run_finetuning, "Trainer", FakeTrainer)
    monkeypatch.setattr(run_finetuning, "build_lora_config", fake_build_lora_config)
    monkeypatch.setattr(run_finetuning, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(run_finetuning, "mlflow", fake_mlflow)
    monkeypatch.setattr(run_finetuning, "_git_sha", lambda: "abc1234")
    monkeypatch.setattr(run_finetuning.getpass, "getuser", lambda: "tester")

    FakeTrainer.instances = []
    FakeTrainingArguments.instances = []

    return {
        "dataset_factory": dataset_factory,
        "tokenizer": tokenizer,
        "auto_tokenizer": auto_tokenizer,
        "auto_model": auto_model,
        "peft_model": fake_peft_model,
        "mlflow": fake_mlflow,
        "dedup_inputs": dedup_inputs,
        "collator_calls": collator_calls,
        "lora_calls": lora_calls,
    }


def test_main_smoke_runs_short_trainer_loop_without_mlflow_run(monkeypatch):
    tables = {
        "sarcasm.csv": pd.DataFrame(_build_rows(40, "1")),
    }
    fakes = _install_training_fakes(monkeypatch, tables)

    result = run_finetuning.main(["--task", "sarcasm", "--smoke"])

    assert result == 0
    assert len(fakes["dedup_inputs"]) == 1
    assert len(fakes["dedup_inputs"][0]) == 32
    assert {type(row["label"]) for row in fakes["dedup_inputs"][0]} == {int}
    assert len(fakes["dataset_factory"].rows_by_split["train"]) == 28
    assert len(fakes["dataset_factory"].rows_by_split["test"]) == 4
    assert fakes["auto_tokenizer"].model_name == "xlm-roberta-base"
    assert fakes["tokenizer"].calls == [
        {
            "texts": [row["text"] for row in fakes["dataset_factory"].rows_by_split["train"]],
            "truncation": True,
            "max_length": 128,
        },
        {
            "texts": [row["text"] for row in fakes["dataset_factory"].rows_by_split["test"]],
            "truncation": True,
            "max_length": 128,
        }
    ]
    assert fakes["auto_model"].calls == [
        {"model_name": "xlm-roberta-base", "num_labels": 3}
    ]
    assert fakes["lora_calls"] == ["sarcasm"]
    assert fakes["peft_model"]["lora_config"] == {"task": "sarcasm"}
    assert fakes["peft_model"]["model"].print_trainable_parameters_called is True
    assert FakeTrainingArguments.instances[0].kwargs["num_train_epochs"] == 1
    assert FakeTrainingArguments.instances[0].kwargs["report_to"] == "none"
    assert FakeTrainingArguments.instances[0].kwargs["eval_strategy"] == "epoch"
    assert FakeTrainingArguments.instances[0].kwargs["save_strategy"] == "epoch"
    assert FakeTrainer.instances[0].train_calls == 1
    assert (
        FakeTrainer.instances[0].kwargs["train_dataset"]
        is fakes["dataset_factory"].datasets_by_split["train"]
    )
    assert (
        FakeTrainer.instances[0].kwargs["eval_dataset"]
        is fakes["dataset_factory"].datasets_by_split["test"]
    )
    assert FakeTrainer.instances[0].kwargs["processing_class"] is fakes["tokenizer"]
    assert FakeTrainingArguments.instances[0].kwargs["output_dir"].endswith(
        "models/adapters_smoke/sarcasm"
    )
    assert fakes["peft_model"]["model"].saved_paths == [
        str(
            Path(run_finetuning.__file__).resolve().parents[2]
            / "models"
            / "adapters_smoke"
            / "sarcasm"
        )
    ]
    assert fakes["mlflow"].tracking_uris == ["file:./mlruns"]
    assert fakes["mlflow"].experiments == ["phase2_finetuning_sarcasm"]
    assert fakes["mlflow"].start_run_calls == 0
    assert fakes["mlflow"].tags == []


def test_main_full_sentiment_training_uses_mlflow_and_label_mapping(monkeypatch):
    def build_rows(lang: str) -> list[dict]:
        rows = []
        for label_name in ("negative", "neutral", "positive"):
            for idx in range(10):
                rows.append(
                    {
                        "text": f"{lang}-{label_name}-{idx}",
                        "label": label_name,
                        "lang": lang,
                    }
                )
        return rows

    tables = {
        "sentiment_en.csv": pd.DataFrame(build_rows("en")),
        "sentiment_vi.csv": pd.DataFrame(build_rows("vi")),
    }
    fakes = _install_training_fakes(monkeypatch, tables)
    monkeypatch.setattr(run_finetuning.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_finetuning.torch.backends.mps, "is_available", lambda: False)

    result = run_finetuning.main(
        ["--task", "sentiment", "--tracking-uri", "http://mlflow.local"]
    )

    assert result == 0
    assert len(fakes["dedup_inputs"]) == 1
    assert set(row["label"] for row in fakes["dedup_inputs"][0]) == {0, 1, 2}
    assert len(fakes["dataset_factory"].rows_by_split["train"]) == 54
    assert len(fakes["dataset_factory"].rows_by_split["test"]) == 6
    assert fakes["auto_model"].calls == [
        {"model_name": "xlm-roberta-base", "num_labels": 3}
    ]
    assert FakeTrainingArguments.instances[0].kwargs["output_dir"].endswith(
        "models/adapters/sentiment"
    )
    assert FakeTrainingArguments.instances[0].kwargs["num_train_epochs"] == 5
    assert FakeTrainingArguments.instances[0].kwargs["report_to"] == ["mlflow"]
    assert FakeTrainer.instances[0].train_calls == 1
    assert fakes["mlflow"].tracking_uris == ["http://mlflow.local"]
    assert fakes["mlflow"].experiments == ["phase2_finetuning_sentiment"]
    assert fakes["mlflow"].start_run_calls == 1
    assert fakes["mlflow"].tags == [
        {
            "task": "sentiment",
            "git_sha": "abc1234",
            "device": "cuda",
            "environment": "local",
            "dataset_version": "multilingual_sentiment_v1",
            "seed": "42",
            "user": "tester",
        }
    ]


def test_main_sentiment_raises_for_unmapped_labels(monkeypatch):
    tables = {
        "sentiment_en.csv": pd.DataFrame([{"text": "bad", "label": "mixed", "lang": "en"}]),
        "sentiment_vi.csv": pd.DataFrame([{"text": "tot", "label": "positive", "lang": "vi"}]),
    }
    _install_training_fakes(monkeypatch, tables)

    with pytest.raises(ValueError, match="Unmapped sentiment labels: mixed"):
        run_finetuning.main(["--task", "sentiment"])


def test_split_rows_stratifies_multilingual_sentiment_eval_set():
    rows: list[dict] = []
    for lang in ("en", "vi"):
        for label in (0, 1, 2):
            for idx in range(10):
                rows.append(
                    {
                        "text": f"{lang}-{label}-{idx}",
                        "label": label,
                        "lang": lang,
                    }
                )

    split_rows = run_finetuning._split_rows_for_training(get_task_config("sentiment"), rows)

    def group_counts(items):
        counts: dict[tuple[str, int], int] = {}
        for row in items:
            key = (row["lang"], row["label"])
            counts[key] = counts.get(key, 0) + 1
        return counts

    assert len(split_rows["train"]) == 54
    assert len(split_rows["test"]) == 6
    assert set(group_counts(split_rows["test"]).values()) == {1}
    assert set(group_counts(split_rows["train"]).values()) == {9}


def test_build_training_args_returns_real_transformers_object_for_smoke(tmp_path):
    args = run_finetuning._build_training_args(
        get_task_config("sarcasm"),
        tmp_path / "models" / "adapters_smoke" / "sarcasm",
        epochs=1,
        smoke=True,
    )

    assert isinstance(args, TrainingArguments)
    assert args.output_dir.endswith("models/adapters_smoke/sarcasm")
    assert args.eval_strategy.value == "epoch"
    assert args.report_to == []


def test_build_training_args_uses_legacy_evaluation_strategy_name(tmp_path):
    args = run_finetuning._build_training_args(
        get_task_config("sarcasm"),
        tmp_path / "models" / "adapters_smoke" / "sarcasm",
        epochs=1,
        smoke=True,
        training_arguments_cls=FakeLegacyTrainingArguments,
    )

    assert args.kwargs["evaluation_strategy"] == "epoch"


def test_build_trainer_uses_legacy_tokenizer_parameter():
    trainer = run_finetuning._build_trainer(
        model="model",
        training_args="args",
        train_dataset="train",
        eval_dataset="eval",
        tokenizer="tokenizer",
        data_collator="collator",
        trainer_cls=FakeLegacyTrainer,
    )

    assert trainer.kwargs["tokenizer"] == "tokenizer"
    assert "processing_class" not in trainer.kwargs


def test_build_trainer_uses_modern_processing_class_parameter():
    trainer = run_finetuning._build_trainer(
        model="model",
        training_args="args",
        train_dataset="train",
        eval_dataset="eval",
        tokenizer="tokenizer",
        data_collator="collator",
        trainer_cls=FakeModernTrainer,
    )

    assert trainer.kwargs["processing_class"] == "tokenizer"
    assert "tokenizer" not in trainer.kwargs


def test_main_sentiment_smoke_skips_brittle_stratified_split(monkeypatch):
    def build_en_rows() -> list[dict]:
        rows = []
        for label_name in ("negative", "neutral", "positive"):
            for idx in range(10):
                rows.append(
                    {
                        "text": f"en-{label_name}-{idx}",
                        "label": label_name,
                        "lang": "en",
                    }
                )
        return rows

    tables = {
        "sentiment_en.csv": pd.DataFrame(build_en_rows()),
        "sentiment_vi.csv": pd.DataFrame(
            [
                {"text": "vi-negative-0", "label": "negative", "lang": "vi"},
                {"text": "vi-neutral-0", "label": "neutral", "lang": "vi"},
                {"text": "vi-positive-0", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-1", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-2", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-3", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-4", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-5", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-6", "label": "positive", "lang": "vi"},
                {"text": "vi-positive-7", "label": "positive", "lang": "vi"},
            ]
        ),
    }
    fakes = _install_training_fakes(monkeypatch, tables)

    result = run_finetuning.main(["--task", "sentiment", "--smoke"])

    assert result == 0
    assert len(fakes["dataset_factory"].rows_by_split["train"]) == 28
    assert len(fakes["dataset_factory"].rows_by_split["test"]) == 4

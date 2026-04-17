"""CLI entrypoint for Phase 2 finetuning."""

from __future__ import annotations

import argparse
import getpass
import inspect
import logging
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.training.dataset_builder import build_stratify_labels, dedup_rows
from src.training.lora_config import build_lora_config
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


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_output_dir(root: Path, task_name: str, smoke: bool) -> Path:
    parent = "adapters_smoke" if smoke else "adapters"
    return root / "models" / parent / task_name


def _build_training_args(
    task,
    output_dir: Path,
    *,
    epochs: int,
    smoke: bool,
    training_arguments_cls=None,
):
    training_arguments_cls = training_arguments_cls or TrainingArguments
    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=task.learning_rate,
        per_device_train_batch_size=task.batch_size,
        per_device_eval_batch_size=task.batch_size,
        gradient_accumulation_steps=task.gradient_accumulation_steps,
        num_train_epochs=epochs,
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=task.seed,
        report_to=["mlflow"] if not smoke else "none",
    )
    parameter_names = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameter_names:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return training_arguments_cls(**kwargs)

def _build_trainer(
    *,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    trainer_cls=None,
):
    trainer_cls = trainer_cls or Trainer
    kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    parameter_names = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameter_names:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return trainer_cls(**kwargs)


def _split_rows_for_training(task, rows: list[dict], *, smoke: bool = False) -> dict[str, list[dict]]:
    stratify = None
    if task.name == "sentiment" and not smoke:
        stratify = build_stratify_labels(pd.DataFrame(rows)).tolist()

    train_rows, test_rows = train_test_split(
        rows,
        test_size=0.1,
        random_state=task.seed,
        stratify=stratify,
    )
    return {"train": list(train_rows), "test": list(test_rows)}


def _load_training_frame(task, root: Path) -> pd.DataFrame:
    if task.name == "sarcasm":
        df = pd.read_csv(root / "data" / "raw" / "sarcasm.csv")
        df["label"] = df["label"].astype(int)
        return df

    df_en = pd.read_csv(root / "data" / "raw" / "sentiment_en.csv")
    df_vi = pd.read_csv(root / "data" / "raw" / "sentiment_vi.csv")
    df = pd.concat([df_en, df_vi], ignore_index=True)
    label2id = {name: idx for idx, name in enumerate(task.label_names)}
    source_labels = df["label"].astype(str)
    df["label"] = source_labels.map(label2id)
    if df["label"].isna().any():
        unmapped = sorted(source_labels[df["label"].isna()].unique().tolist())
        raise ValueError(f"Unmapped sentiment labels: {', '.join(unmapped)}")
    df["label"] = df["label"].astype(int)
    return df


def _model_num_labels(task) -> int:
    # Keep both finetuned adapters compatible with the shared 3-logit backbone.
    return 3 if task.name == "sarcasm" else task.num_labels


def main(argv: list[str] | None = None) -> int:
    """Run the requested finetuning workflow."""
    args = parse_args(argv)
    task = get_task_config(args.task)
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    tags = build_run_tags(
        task=task.name,
        git_sha=_git_sha(),
        device=_detect_device(),
        environment="local",
        dataset_version=task.dataset_version,
        seed=task.seed,
        user=getpass.getuser(),
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting %s finetuning run", task.name)

    root = Path(__file__).resolve().parents[2]
    df = _load_training_frame(task, root)

    epochs = task.epochs
    if args.smoke:
        df = df.head(32)
        epochs = 1

    rows = df.to_dict("records")
    clean_rows = dedup_rows(rows)
    split_rows = _split_rows_for_training(task, clean_rows, smoke=args.smoke)
    hf_dataset = DatasetDict(
        {
            split_name: Dataset.from_list(split_rows[split_name])
            for split_name in ("train", "test")
        }
    )

    base_model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=task.max_length)

    tokenized_ds = hf_dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=_model_num_labels(task),
    )
    lora_config = build_lora_config(task)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    output_dir = _resolve_output_dir(root, task.name, args.smoke)
    training_args = _build_training_args(task, output_dir, epochs=epochs, smoke=args.smoke)

    trainer = _build_trainer(
        model=peft_model,
        training_args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"phase2_finetuning_{task.name}")

    if not args.smoke:
        with mlflow.start_run():
            mlflow.set_tags(tags)
            trainer.train()
            peft_model.save_pretrained(str(output_dir))
            logging.info("Training complete. Adapter saved to %s", output_dir)
    else:
        trainer.train()
        peft_model.save_pretrained(str(output_dir))
        logging.info("Smoke test training complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

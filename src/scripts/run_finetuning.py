"""CLI entrypoint for Phase 2 finetuning."""

from __future__ import annotations

import argparse
import getpass
import logging
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
from datasets import Dataset
from peft import get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.training.dataset_builder import dedup_rows
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


def main(argv: list[str] | None = None) -> int:
    """Run the requested finetuning workflow."""
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
    logging.info("Starting %s finetuning run", task.name)

    root = Path(__file__).resolve().parents[2]
    if task.name == "sarcasm":
        df = pd.read_csv(root / "data" / "raw" / "sarcasm.csv")
        df["label"] = df["label"].astype(int)
    else:
        df_en = pd.read_csv(root / "data" / "raw" / "sentiment_en.csv")
        df_vi = pd.read_csv(root / "data" / "raw" / "sentiment_vi.csv")
        df = pd.concat([df_en, df_vi], ignore_index=True)
        label2id = {name: idx for idx, name in enumerate(task.label_names)}
        df["label"] = df["label"].map(label2id)

    epochs = task.epochs
    if args.smoke:
        df = df.head(32)
        epochs = 1

    rows = df.to_dict("records")
    clean_rows = dedup_rows(rows)
    hf_dataset = Dataset.from_list(clean_rows)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=task.seed)

    base_model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=task.max_length)

    tokenized_ds = hf_dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=task.num_labels,
    )
    lora_config = build_lora_config(task)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    output_dir = root / "models" / "adapters" / task.name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=task.learning_rate,
        per_device_train_batch_size=task.batch_size,
        per_device_eval_batch_size=task.batch_size,
        gradient_accumulation_steps=task.gradient_accumulation_steps,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=task.seed,
        report_to=["mlflow"] if not args.smoke else "none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,
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

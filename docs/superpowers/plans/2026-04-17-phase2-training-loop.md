# Phase 2 Finetuning Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the actual HuggingFace Trainer loop, metrics evaluation, and DVC data download steps to make the Phase 2 finetuning ready for execution on Colab.

**Architecture:** 
1. Expand `downloader.py` to handle downloading HuggingFace datasets (`tweet_eval`, `multilingual-sentiments`, `vietnamese_students_feedback`) to CSVs for DVC tracking. Update `dvc.yaml` to pass `--task` to the downloader.
2. Complete `run_finetuning.py` to load data, wrap the base model with LoRA (`peft`), and run `Trainer`.
3. Complete `evaluate_finetuned.py` to load the trained adapters, run inference, and compute real metrics.
4. Provide a lightweight Colab notebook (`.ipynb`) as an execution wrapper.

**Tech Stack:** Python, HuggingFace `transformers`, `peft`, `datasets`, `mlflow`, `pandas`, `pytest`

---

### Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add missing ML dependencies**

Modify `requirements.txt` to include `peft` and `datasets`. Keep existing versions but add:
```txt
peft>=0.7.0,<0.20.0
datasets>=2.0.0,<5.0.0
```
(Append them to the end of the file).

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add peft and datasets to requirements"
```

### Task 2: Implement HuggingFace Dataset Downloader

**Files:**
- Modify: `src/data/downloader.py`
- Modify: `dvc.yaml`

- [ ] **Step 1: Refactor `downloader.py` `__main__` and add HF download logic**

Add `datasets` import and argparse to `src/data/downloader.py`. Create functions to download the required HF datasets and save them to CSV.

```python
# Add to imports in src/data/downloader.py:
import argparse
from datasets import load_dataset

# Add functions:
def download_sarcasm_dataset(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("tweet_eval", "irony")
    df = dataset["train"].to_pandas()
    df["lang"] = "en"
    df["source"] = "tweet_eval_irony"
    df.to_csv(out_path, index=False)
    print(f"Saved sarcasm dataset to {out_path}")

def download_sentiment_datasets(en_out_path: Path, vi_out_path: Path):
    en_out_path.parent.mkdir(parents=True, exist_ok=True)
    vi_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Multilingual sentiments (English)
    en_ds = load_dataset("tyqiangz/multilingual-sentiments", "english")
    en_df = en_ds["train"].to_pandas()
    en_df["lang"] = "en"
    en_df["source"] = "multilingual_sentiments"
    en_df.to_csv(en_out_path, index=False)
    
    # Vietnamese students feedback
    vi_ds = load_dataset("uitnlp/vietnamese_students_feedback")
    vi_df = vi_ds["train"].to_pandas()
    vi_df["label"] = vi_df["sentiment"].map(_UIT_SENTIMENT_MAP)
    vi_df["text"] = vi_df["sentence"]
    vi_df["lang"] = "vi"
    vi_df["source"] = "uit_vsfc"
    vi_df[["text", "label", "lang", "source", "split"]].to_csv(vi_out_path, index=False)
    print(f"Saved sentiment datasets to {en_out_path} and {vi_out_path}")

# Replace the existing `if __name__ == "__main__":` block with:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="semeval", choices=["semeval", "sarcasm", "sentiment"])
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[2]
    
    if args.task == "semeval":
        params = load_params(str(root / "params.yaml"))
        data = params["data"]
        extract_semeval_xmls(
            root / "data" / "raw",
            dataset_name=str(data["dataset_name"]),
            splits=list(data["splits"]),
        )
    elif args.task == "sarcasm":
        download_sarcasm_dataset(root / "data" / "raw" / "sarcasm.csv")
    elif args.task == "sentiment":
        download_sentiment_datasets(
            root / "data" / "raw" / "sentiment_en.csv",
            root / "data" / "raw" / "sentiment_vi.csv"
        )
```

- [ ] **Step 2: Update `dvc.yaml`**

In `dvc.yaml`, update the `download`, `download_sarcasm` and `download_sentiment` commands to pass the `--task` flag.

```yaml
# Update lines 2-3:
  download:
    cmd: python3 -m src.data.downloader --task semeval

# Update lines 51-52:
  download_sarcasm:
    cmd: python3 -m src.data.downloader --task sarcasm

# Update lines 60-61:
  download_sentiment:
    cmd: python3 -m src.data.downloader --task sentiment
```

- [ ] **Step 3: Test the downloader**

Run: `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m src.data.downloader --task sarcasm`
Expected: Successfully downloads `tweet_eval` and saves `data/raw/sarcasm.csv`.

- [ ] **Step 4: Commit**

```bash
git add src/data/downloader.py dvc.yaml
git commit -m "feat: add HuggingFace dataset downloader support for Phase 2"
```

### Task 3: Implement Training Loop in `run_finetuning.py`

**Files:**
- Modify: `src/scripts/run_finetuning.py`

- [ ] **Step 1: Add training logic**

Replace the placeholder `main()` logic in `src/scripts/run_finetuning.py` with actual `Trainer` code.

```python
# Add to imports:
import pandas as pd
import torch
import mlflow
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import get_peft_model
from src.training.lora_config import build_lora_config
from src.training.dataset_builder import dedup_rows

# Replace the main() function body (after tags are built) with:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting %s finetuning run", task.name)
    
    # 1. Load Data
    root = Path(__file__).resolve().parents[2]
    if task.name == "sarcasm":
        df = pd.read_csv(root / "data" / "raw" / "sarcasm.csv")
        df["label"] = df["label"].astype(int)
    else:
        df_en = pd.read_csv(root / "data" / "raw" / "sentiment_en.csv")
        df_vi = pd.read_csv(root / "data" / "raw" / "sentiment_vi.csv")
        df = pd.concat([df_en, df_vi], ignore_index=True)
        label2id = {name: i for i, name in enumerate(task.label_names)}
        df["label"] = df["label"].map(label2id)
    
    epochs = task.epochs
    if args.smoke:
        df = df.head(32)
        epochs = 1

    # Deduplicate and convert to HF Dataset
    rows = df.to_dict("records")
    clean_rows = dedup_rows(rows)
    hf_dataset = Dataset.from_list(clean_rows)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=task.seed)

    # 2. Tokenize
    base_model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=task.max_length)
    
    tokenized_ds = hf_dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Setup
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        num_labels=task.num_labels
    )
    lora_config = build_lora_config(task)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 4. Train
    output_dir = root / "models" / "adapters" / task.name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=task.learning_rate,
        per_device_train_batch_size=task.batch_size,
        per_device_eval_batch_size=task.batch_size,
        gradient_accumulation_steps=task.gradient_accumulation_steps,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
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
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"phase2_finetuning_{task.name}"
    mlflow.set_experiment(experiment_name)

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
```

- [ ] **Step 2: Test the script**

Run: `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m src.scripts.run_finetuning --task sarcasm --smoke`
Expected: Prints trainable parameters, runs a short 1-epoch loop on 32 samples, and saves adapter.

- [ ] **Step 3: Commit**

```bash
git add src/scripts/run_finetuning.py
git commit -m "feat: implement Trainer loop and LoRA setup in run_finetuning.py"
```

### Task 4: Complete Evaluation Script

**Files:**
- Modify: `src/scripts/evaluate_finetuned.py`

- [ ] **Step 1: Add inference logic**

Replace the stub JSON writing with actual model loading and inference logic.

```python
# Add to imports:
import pandas as pd
from src.model.baseline import BaselineModelInference
from src.model.config import ModelConfig
from src.training.task_configs import get_task_config
from src.training.metrics import build_metrics_payload

# Update main() to perform inference:
    # Inside main(), replace the stub payload generation:
    root = Path(__file__).resolve().parents[2]
    task = get_task_config(args.task)
    
    # Configure model for finetuned mode
    config = ModelConfig(
        mode="finetuned",
        sentiment_adapter_path=str(root / "models" / "adapters" / "sentiment"),
        sarcasm_adapter_path=str(root / "models" / "adapters" / "sarcasm")
    )
    inference = BaselineModelInference(config=config)
    
    # Load dataset based on task
    if args.task == "sarcasm":
        df = pd.read_csv(root / "data" / "raw" / "sarcasm.csv")
    else:
        df_en = pd.read_csv(root / "data" / "raw" / "sentiment_en.csv")
        df_vi = pd.read_csv(root / "data" / "raw" / "sentiment_vi.csv")
        df = pd.concat([df_en, df_vi], ignore_index=True)

    # Use a small sample to avoid taking forever locally
    df = df.sample(n=min(100, len(df)), random_state=42)
    texts = df["text"].tolist()
    languages = df.get("lang", ["en"] * len(df)).tolist()
    
    # Run batch inference
    results = inference.predict_batch(texts, lang="en", skip_absa=True)
    
    y_pred = []
    y_true = []
    for i, res in enumerate(results):
        if args.task == "sarcasm":
            y_pred.append(task.label_names[1] if res.sarcasm_flag else task.label_names[0])
            y_true.append(task.label_names[int(df.iloc[i]["label"])])
        else:
            y_pred.append(res.sentiment)
            y_true.append(task.label_names[int(df.iloc[i]["label"])])

    metrics_payload = build_metrics_payload(
        y_true=y_true,
        y_pred=y_pred,
        languages=languages,
        label_names=task.label_names
    )
    
    args.output.write_text(json.dumps({
        "task": args.task,
        "overall_f1": metrics_payload["overall_f1"],
        "n_samples": len(texts)
    }, indent=2), encoding="utf-8")
    
    args.output.parent.joinpath("per_language_f1.json").write_text(
        json.dumps({"per_lang_f1": metrics_payload["per_lang_f1"]}, indent=2),
        encoding="utf-8",
    )
    
    args.output.parent.joinpath("fairness_report.json").write_text(
        json.dumps({
            "overall_f1": metrics_payload["overall_f1"],
            "per_lang_f1": metrics_payload["per_lang_f1"],
            "per_lang_gap": metrics_payload["per_lang_gap"],
            "sample_counts": metrics_payload["sample_counts"],
            "confusion_matrices": metrics_payload["per_lang_confusion_matrices"]
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Evaluation for {args.task} completed.")
```

- [ ] **Step 2: Test evaluation script**

Run: `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m src.scripts.evaluate_finetuned --task sarcasm`

- [ ] **Step 3: Commit**

```bash
git add src/scripts/evaluate_finetuned.py
git commit -m "feat: implement inference and metrics computation in evaluate_finetuned.py"
```

### Task 5: Create Colab Notebook wrapper

**Files:**
- Create: `notebooks/colab_finetuning.ipynb`

- [ ] **Step 1: Create the Notebook JSON**

Create `notebooks/colab_finetuning.ipynb` containing a simple JSON structure with cells to:
1. Clone repo
2. Install requirements
3. Run python training scripts.

Here is the exact content to write:

```json
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "header"
      },
      "source": [
        "# Sentiment Analysis Service - Phase 2 Finetuning\n",
        "\n",
        "This notebook acts as an execution wrapper for the DVC training pipeline. It runs the scripts defined in the repository."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "setup"
      },
      "outputs": [],
      "source": [
        "!git clone https://github.com/RocketUp-Team/Sentiment-Analysis-Service.git\n",
        "%cd Sentiment-Analysis-Service\n",
        "!git checkout feature/absa-sarcasm-phase2\n",
        "!pip install -r requirements.txt\n",
        "!pip install datasets peft accelerate"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "data"
      },
      "outputs": [],
      "source": [
        "print(\"Downloading datasets...\")\n",
        "!python -m src.data.downloader --task sarcasm\n",
        "!python -m src.data.downloader --task sentiment"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "train_sarcasm"
      },
      "outputs": [],
      "source": [
        "print(\"Training Sarcasm Adapter...\")\n",
        "!python -m src.scripts.run_finetuning --task sarcasm"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "train_sentiment"
      },
      "outputs": [],
      "source": [
        "print(\"Training Sentiment Adapter...\")\n",
        "!python -m src.scripts.run_finetuning --task sentiment"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "export"
      },
      "outputs": [],
      "source": [
        "print(\"Compressing models for download...\")\n",
        "!tar -czvf adapters.tar.gz models/adapters/"
      ]
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
mkdir -p notebooks
git add notebooks/colab_finetuning.ipynb
git commit -m "feat: add Colab execution notebook for Phase 2 finetuning"
```

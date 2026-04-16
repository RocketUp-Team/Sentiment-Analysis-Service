# ABSA & Sarcasm Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the `config.py` for Multilingual support, implement a hybrid data loader (Yelp + Sarcasm), and create a LoRA fine-tuning training loop tracked by MLflow.

**Architecture:** We will construct a `data_mixer.py` to pull and normalize Sarcasm data from HuggingFace and existing processed datasets. Then, using `peft` and HuggingFace `Trainer`, we will implement `trainer.py` to fine-tune the core baseline models via LoRA with MLflow metric tracking.

**Tech Stack:** `transformers`, `peft`, `mlflow`, `datasets`, `pandas`

---

### Task 1: Update Configuration for Multilingual Support

**Files:**
- Modify: `src/model/config.py`
- Test: `tests/test_model_config.py` (assuming existence or create if needed)

- [ ] **Step 1: Write/Update the test**

```python
# test_model_config.py
from src.model.config import ModelConfig

def test_multilingual_models_are_configured():
    config = ModelConfig()
    assert config.model_name == "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    assert config.absa_model_name == "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_config.py -v`
Expected: FAIL due to old English models inside `config.py`.

- [ ] **Step 3: Write minimal implementation**

Modify `src/model/config.py`:
```python
@dataclass(frozen=True)
class ModelConfig:
    # ...
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    # ...
    absa_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    # ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/config.py tests/test_model_config.py
git commit -m "feat: migrate to multilingual baseline models"
```

---

### Task 2: Implement Data Mixer

**Files:**
- Create: `src/training/data_mixer.py`
- Create: `tests/test_data_mixer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_mixer.py
import pandas as pd
from src.training.data_mixer import load_hybrid_dataset

def test_load_hybrid_dataset():
    # Mocking or calling the real loader function
    dataset = load_hybrid_dataset(yelp_path="data/processed/sentences.csv", sample_size=10)
    
    assert "text" in dataset.column_names
    assert "sentiment" in dataset.column_names
    assert "is_sarcasm" in dataset.column_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_mixer.py -v`
Expected: FAIL with "module src.training.data_mixer not found"

- [ ] **Step 3: Write minimal implementation**

```python
# src/training/data_mixer.py
import pandas as pd
from datasets import Dataset, load_dataset

def load_hybrid_dataset(yelp_path: str, sample_size: int = None) -> Dataset:
    # Load processed yelp data
    df_yelp = pd.read_csv(yelp_path)
    if sample_size:
        df_yelp = df_yelp.head(sample_size)
        
    df_yelp["is_sarcasm"] = False
    
    # Load tweet_eval irony for sarcasm
    sarcasm_ds = load_dataset("tweet_eval", "irony", split="train")
    df_sarcasm = sarcasm_ds.to_pandas()
    if sample_size:
        df_sarcasm = df_sarcasm.head(sample_size)
    
    # Map tweet_eval irony labels (0: non_irony, 1: irony)
    df_sarcasm["is_sarcasm"] = df_sarcasm["label"] == 1
    # Assign neutral as placeholder sentiment for pure sarcasm subset if no sentiment is provided
    df_sarcasm["sentiment"] = "neutral" 
    
    # Keep only aligned columns
    df_yelp_sub = df_yelp[["text", "sentiment", "is_sarcasm"]]
    df_sarcasm_sub = df_sarcasm[["text", "sentiment", "is_sarcasm"]]
    
    merged_df = pd.concat([df_yelp_sub, df_sarcasm_sub], ignore_index=True)
    return Dataset.from_pandas(merged_df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_mixer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/data_mixer.py tests/test_data_mixer.py
git commit -m "feat: implement hybrid data loader"
```

---

### Task 3: Implement Trainer with MLflow and LoRA

**Files:**
- Create: `src/training/trainer.py`

- [ ] **Step 1: Write placeholder implementation (Skipping full heavy test for HF Trainer config)**

We will implement the wrapper for the MLflow & LoRA setup directly, as unit-testing the HuggingFace `Trainer` internally requires downloading large models which isn't viable in CI without mocks.

```python
# src/training/trainer.py
import os
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from src.model.config import ModelConfig

def setup_and_train(dataset, output_dir="model_output"):
    config = ModelConfig()
    
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Sentiment_ABSA_Finetuning")
    
    # Load base model & Tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Tokenize Dataset
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config.max_length)
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    
    # LoRA Config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"]  # Typically Q and V in attention
    )
    peft_model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_ds,
    )
    
    with mlflow.start_run():
        mlflow.log_params({"lora_rank": 8, "base_model": config.model_name})
        trainer.train()
        
        # Save model internally
        peft_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
        mlflow.pytorch.log_model(peft_model, "peft_model")
```

- [ ] **Step 2: Commit**

```bash
git add src/training/trainer.py
git commit -m "feat: setup LoRA training loop and mlflow tracking"
```

---

### Task 4: Execution Script

**Files:**
- Create: `src/scripts/run_finetuning.py`

- [ ] **Step 1: Write execution script**

```python
# src/scripts/run_finetuning.py
from src.training.data_mixer import load_hybrid_dataset
from src.training.trainer import setup_and_train
import os

if __name__ == "__main__":
    # Ensure processed data exists
    yelp_path = "data/processed/sentences.csv"
    if not os.path.exists(yelp_path):
        raise FileNotFoundError(f"Missing {yelp_path}")
        
    print("Loading Hybrid Dataset...")
    dataset = load_hybrid_dataset(yelp_path)
    
    print("Starting Fine-tuning with MLFlow Tracking...")
    setup_and_train(dataset)
    print("Finetuning completed successfully.")
```

- [ ] **Step 2: Commit**

```bash
git add src/scripts/run_finetuning.py
git commit -m "feat: create run finetuning execution script"
```

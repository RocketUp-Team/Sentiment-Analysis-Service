> **SUPERSEDED:** Replaced by `docs/superpowers/plans/2026-04-17-absa-sarcasm-finetuning-phase2.md`. Do not execute this Phase 1 plan.

# ABSA & Sarcasm Fine-Tuning (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the `tweet_eval` (irony) dataset, set up a multi-task fine-tuning pipeline with XLM-RoBERTa managed via PEFT/LoRA, and integrate MLflow for tracking.

**Architecture:** We adapt the downloader to ingest English sarcasm data (`tweet_eval`). We introduce a PEFT configuration module, setting up LoRA for Sequence Classification. Finally, we establish the training script entrypoint integrated with MLflow, logging metrics and hyper-parameters, and mapping out the dependency updates in `requirements.txt`. (Note: Phase 2 for mDeBERTa ABSA testing will be covered in subsequent plans).

**Tech Stack:** PyTorch, Transformers, PEFT, MLflow, Datasets, DVC.

---

### Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing test**
Run: `python -c "import peft, datasets, accelerate"`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 2: Write minimal implementation**
Append to `requirements.txt`:
```text
peft>=0.5.0
datasets>=2.14.0
sentencepiece>=0.1.99
accelerate>=0.21.0
```

- [ ] **Step 3: Run test to verify it passes**
Run: `pip install -r requirements.txt && python -c "import peft, datasets, accelerate"`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add requirements.txt
git commit -m "build: add fine-tuning dependencies"
```

### Task 2: Update Downloader for Sarcasm Dataset

**Files:**
- Modify: `src/data/downloader.py`
- Create: `tests/data/test_downloader_sarcasm.py`

- [ ] **Step 1: Write the failing test**
Create `tests/data/test_downloader_sarcasm.py`:
```python
from pathlib import Path
import pandas as pd
from src.data.downloader import download_sarcasm_data

def test_download_sarcasm_data(tmp_path: Path):
    dest = tmp_path / "sarcasm.csv"
    download_sarcasm_data(dest)
    assert dest.exists()
    df = pd.DataFrame(pd.read_csv(dest))
    assert list(df.columns) == ["text", "label"]
    assert len(df) > 0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/data/test_downloader_sarcasm.py -v`
Expected: FAIL (ImportError: cannot import name 'download_sarcasm_data')

- [ ] **Step 3: Write minimal implementation**
Modify `src/data/downloader.py` to add the download method:
```python
def download_sarcasm_data(dest_path: str | Path) -> None:
    from datasets import load_dataset
    dataset = load_dataset("tweet_eval", "irony", split="train")
    df = dataset.to_pandas()
    df = df.rename(columns={"tweet": "text", "label": "label"})
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
```
Also in `src/data/downloader.py`, update the `if __name__ == "__main__":` block to invoke it:
```python
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    params = load_params(str(root / "params.yaml"))
    data = params["data"]
    extract_semeval_xmls(
        root / "data" / "raw",
        dataset_name=str(data["dataset_name"]),
        splits=list(data["splits"]),
    )
    # ADDED: Invocation for sarcasm data download
    download_sarcasm_data(root / "data" / "raw" / "sarcasm.csv")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/data/test_downloader_sarcasm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/data/downloader.py tests/data/test_downloader_sarcasm.py
git commit -m "feat(data): add sarcasm dataset downloader"
```

### Task 3: Establish PEFT/LoRA Configuration

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/lora_config.py`
- Create: `tests/model/test_lora_config.py`

- [ ] **Step 1: Write the failing test**
Create `tests/model/test_lora_config.py`:
```python
from peft import LoraConfig
from src.training.lora_config import get_lora_config

def test_get_lora_config():
    config = get_lora_config()
    assert isinstance(config, LoraConfig)
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.target_modules == {"query", "value"} # Using set for target modules representation
    assert config.modules_to_save == ["classifier"]
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/model/test_lora_config.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src.training')

- [ ] **Step 3: Write minimal implementation**
Create `src/training/__init__.py` (empty file)
Create `src/training/lora_config.py`:
```python
from peft import LoraConfig, TaskType

def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        modules_to_save=["classifier"]
    )
```

> **Note on TaskType.SEQ_CLS:** Using `TaskType.SEQ_CLS` handles standard sequence classification. For genuine multi-task handling (ABSA + Sarcasm) without losing task-specific projections, this standard architecture proxy functions for Phase 1 testing but might necessitate custom parallel classifiers on top of your LoRA wrappers in Phase 2.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/model/test_lora_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/training/ tests/model/test_lora_config.py
git commit -m "feat(training): establish PEFT/LoRA configuration module"
```

### Task 4: Develop MLflow Trainer Script Entrypoint

**Files:**
- Create: `src/scripts/__init__.py`
- Create: `src/scripts/run_finetuning.py`
- Create: `tests/model/test_run_finetuning.py`

- [ ] **Step 1: Write the failing test**
Create `tests/model/test_run_finetuning.py`:
```python
import subprocess
from pathlib import Path

def test_finetuning_script_help():
    # Make sure script responds successfully to --help
    result = subprocess.run(
        ["python", "-m", "src.scripts.run_finetuning", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/model/test_run_finetuning.py -v`
Expected: FAIL (No module named 'src.scripts')

- [ ] **Step 3: Write minimal implementation**
Create `src/scripts/__init__.py` (empty file).
Modify `params.yaml` to add tracking variables:
```yaml
finetune:
  epochs: 3
  batch_size: 16
```

Create `src/scripts/run_finetuning.py`:
```python
"""Entrypoint for model fine-tuning tracking with MLflow."""
import argparse
import mlflow
from src.training.lora_config import get_lora_config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune multi-task model.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    # Provide a basic integration checkpoint that PEFT can import 
    config = get_lora_config()
    print(f"Loaded LoRA Config: {config.r} alpha {config.lora_alpha}")
    
    with mlflow.start_run(run_name="sarcasm-finetuning"):
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        print("MLflow run started successfully (dry run).")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/model/test_run_finetuning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/scripts/ tests/model/test_run_finetuning.py
git commit -m "feat(training): create finetuning mlflow script entrypoint"
```

### Task 5: Integrate Fine-Tuning Stage into DVC

**Files:**
- Modify: `dvc.yaml`

- [ ] **Step 1: Write the failing test (DVC dry run)**
Create or modify a test to check if the dvc stage exists (optional, or just do it via CLI):
Run: `dvc stage list`
Expected: FAIL (or missing `finetune` stage in output)

- [ ] **Step 2: Write minimal implementation**
Append the new stage to `dvc.yaml`. Update the commands to leverage DVC interpolation and add the necessary dataset dependencies and model configuration output paths. 
```yaml
  finetune:
    cmd: python -m src.scripts.run_finetuning --epochs ${finetune.epochs} --batch_size ${finetune.batch_size}
    deps:
      - src/scripts/run_finetuning.py
      - src/training/
      - data/raw/sarcasm.csv
    params:
      - finetune
      - mlflow
    outs:
      - models/sarcasm_lora/
```

- [ ] **Step 3: Run test to verify it passes**
Run: `dvc status`
Expected: PASS (It will show the pipeline structure without breaking)

- [ ] **Step 4: Commit**
```bash
git add dvc.yaml
git commit -m "build: integrate finetuning stage into DVC"
```

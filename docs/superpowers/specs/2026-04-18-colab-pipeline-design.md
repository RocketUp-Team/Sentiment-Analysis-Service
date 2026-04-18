# Colab Full-Pipeline Notebook — Design Spec

**Date:** 2026-04-18
**Status:** Approved
**Scope:** Single Colab notebook that runs the entire ML pipeline (train → evaluate → ONNX export → visualize → DVC push) with real-time DagsHub/MLflow monitoring.

## 1. Problem & Motivation

The current workflow requires too many manual steps after training on Colab:

1. Train on Colab → download `adapters.tar.gz`
2. Locally: untar → ONNX export → DVC push → Docker build

**Goal:** One-click "Run All" on Colab. After completion, locally just `git pull && docker build`.

All training, evaluation, visualization, and model pushing happen on Colab. Results are viewable on DagsHub/MLflow in real-time.

## 2. Approach

**Python-Orchestrated Notebook** — import Python functions directly from repo scripts, wrapped in a parent MLflow run with nested child runs. Combined with DVC push + git push for model versioning.

Key decisions:
- **DVC push** (not MLflow Model Registry) for model storage — Dockerfile already uses `dvc pull`
- **Git tags** for model versioning — `model-v1.0`, `model-v2.0`
- **Parent MLflow run** for pipeline monitoring with nested training runs
- **Refactor `run_finetuning.py`** to separate training logic from CLI wrapper

## 3. Notebook Structure (8 Sections)

### Section 1: Setup

```python
!git clone https://github.com/RocketUp-Team/Sentiment-Analysis-Service.git
%cd Sentiment-Analysis-Service
!git checkout feature/ai-core
!pip install -r requirements.txt
!pip install datasets peft accelerate mlflow dvc
```

### Section 2: DagsHub & Credentials Config

Read from Colab Secrets:

| Secret | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/trungdq.ts/Sentiment-Analysis-Service.mlflow` |
| `DAGSHUB_USER` | DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub access token |
| `GITHUB_TOKEN` | For `git push` updated `dvc.lock` + tags |
| `MODEL_VERSION` | e.g. `v2.0` — explicit version for this training run |

**Pre-flight checks:**
- Verify all secrets are present
- Test MLflow connection (`mlflow.search_experiments()`)
- Configure DVC remote auth
- Configure git remote with token for push

### Section 3: Data Download

```python
!python -m src.data.downloader --task sarcasm
!python -m src.data.downloader --task sentiment
```

Log tag `stage_data_download=complete` on parent run.

### Section 4: Training (Nested MLflow Runs)

```python
from src.scripts.run_finetuning import train  # refactored function

with mlflow.start_run(run_name=f"pipeline_{model_version}") as parent_run:
    # Train sarcasm
    with mlflow.start_run(run_name="train_sarcasm", nested=True):
        sarcasm_result = train("sarcasm")

    # Train sentiment
    with mlflow.start_run(run_name="train_sentiment", nested=True):
        sentiment_result = train("sentiment")
```

Each nested run logs: training loss/eval curves per epoch, hyperparameters, LoRA config, trainable params.

### Section 5: Evaluation

```python
from src.scripts.evaluate_finetuned import evaluate  # refactored function

sarcasm_metrics = evaluate("sarcasm")
sentiment_metrics = evaluate("sentiment")

# Log summary metrics to parent run
mlflow.log_metrics({
    "sarcasm_f1": sarcasm_metrics["overall_f1"],
    "sentiment_f1": sentiment_metrics["overall_f1"],
    "fairness_gap": sentiment_metrics["per_lang_gap"],
})
mlflow.set_tag("stage_evaluation", "complete")
```

Also run baseline evaluation for comparison:
```python
from src.model.evaluate import evaluate_on_dataset
baseline_metrics = evaluate_on_dataset(baseline_model, sentences_df, split="test")
```

### Section 6: ONNX Export

```python
!python -m src.scripts.export_onnx --adapter-name sentiment
!python -m src.scripts.export_onnx --adapter-name sarcasm
mlflow.set_tag("stage_onnx_export", "complete")
```

Run ONNX benchmark:
```python
!python -m src.scripts.benchmark_onnx --samples 1000 --batch-size 32 --output reports/onnx_benchmark.json
```

### Section 7: Visualization (11 Outputs)

All plots saved as PNG and uploaded to DagsHub MLflow artifacts.

#### Output 1: Training Curves
- Source: `trainer.state.log_history` (Python object from Section 4)
- Chart: Line plot — training loss & eval loss per epoch, for both tasks
- Format: 1 PNG with 2 subplots (sarcasm + sentiment)

#### Output 2: Evaluation Metrics Table
- Source: `evaluate()` return dicts
- Chart: Pandas DataFrame displayed inline + logged as JSON artifact
- Columns: Task, F1, Accuracy, Precision, Recall, N_Samples

#### Output 3: Confusion Matrices
- Source: `metrics_payload["confusion_matrix"]`
- Chart: `ConfusionMatrixDisplay` heatmap
- Format: 2 PNGs (sarcasm + sentiment)

#### Output 4: Per-Language Fairness Chart
- Source: `metrics_payload["per_lang_f1"]`
- Chart: Grouped bar chart — F1 per language (EN vs VI)
- Format: 1 PNG

#### Output 5: SHAP Waterfall Plots
- Source: Call `generate_shap_plots.py` logic
- Chart: SHAP waterfall for 2-3 representative samples
- Format: PNGs

#### Output 6: ONNX Benchmark Results
- Source: `reports/onnx_benchmark.json`
- Chart: Bar chart comparing latency/throughput — PyTorch vs FP32 vs INT8
- Format: 1 PNG + table

#### Output 7: MLflow Experiment Screenshots
- Source: DagsHub MLflow UI (user captures manually after notebook completes)
- Format: Screenshots for LaTeX report

#### Output 8: Baseline vs Finetuned Comparison
- Source: `baseline_metrics` vs `sentiment_metrics` from Sections 4-5
- Chart: Grouped bar chart — metrics comparison
- Format: 1 PNG

#### Output 9: Dataset Label Distribution
- Source: `df["label"].value_counts()` for each task
- Chart: Bar chart showing label distribution per task
- Format: 2 PNGs (sarcasm + sentiment)

#### Output 10: LoRA Architecture Summary
- Source: `peft_model.print_trainable_parameters()` captured output
- Chart: Text table — total params, trainable params, % trainable
- Format: Inline display + text artifact

#### Output 11: Training Time & GPU Resource Log
- Source: `torch.cuda.get_device_name()`, `trainer.state` timestamps
- Chart: Text summary — GPU type, training time per task, peak memory
- Format: Inline display + logged as MLflow params

**Artifact upload:**
```python
plot_dir = Path("reports/plots")
for png in plot_dir.glob("*.png"):
    mlflow.log_artifact(str(png), artifact_path="report_plots")
```

### Section 8: DVC Push + Git Push + Versioning

```python
version = get_secret("MODEL_VERSION")  # e.g. "v2.0"

# 1. Push ONNX models to DagsHub storage
!dvc push models/onnx/sentiment_fp32 models/onnx/sarcasm_fp32

# 2. Configure git with token
token = get_secret("GITHUB_TOKEN")
!git remote set-url origin https://{token}@github.com/RocketUp-Team/Sentiment-Analysis-Service.git
!git config user.email "colab@pipeline"
!git config user.name "Colab Pipeline"

# 3. Commit updated dvc.lock
!git add dvc.lock
!git commit -m "chore: update models to {version}"
!git push origin feature/ai-core

# 4. Create and push version tag
!git tag model-{version}
!git push origin model-{version}

# 5. Log version to MLflow
mlflow.log_param("model_version", version)
mlflow.set_tag("git_tag", f"model-{version}")
mlflow.set_tag("stage_push", "complete")
```

## 4. Code Refactoring Required

### 4a. `src/scripts/run_finetuning.py`

**Change:** Extract `train()` function from `main()`.

```python
def train(task_name: str, *, smoke: bool = False) -> dict:
    """Run training workflow. MLflow run context managed by caller.

    Returns dict with keys:
    - adapter_path: Path to saved adapter
    - eval_metrics: dict from trainer evaluation
    - trainable_params: str summary from peft
    - log_history: list of training step logs
    """
    task = get_task_config(task_name)
    # ... existing logic (load data, tokenize, build model, train)
    # ... but NO mlflow.start_run() — caller manages this
    trainer.train()
    peft_model.save_pretrained(str(output_dir))
    return {
        "adapter_path": str(output_dir),
        "eval_metrics": trainer.evaluate(),
        "trainable_params": peft_model.get_nb_trainable_parameters(),
        "log_history": trainer.state.log_history,
    }

def main(argv=None) -> int:
    """CLI wrapper — preserves existing DVC pipeline compatibility."""
    args = parse_args(argv)
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"phase2_finetuning_{args.task}")
    tags = build_run_tags(...)
    with mlflow.start_run():
        mlflow.set_tags(tags)
        train(args.task, smoke=args.smoke)
    return 0
```

**Backward compatibility:** `main()` still works for `python -m src.scripts.run_finetuning --task sarcasm` and DVC pipeline.

### 4b. `src/scripts/evaluate_finetuned.py`

**Change:** Extract `evaluate()` function from `main()`.

```python
def evaluate(task_name: str) -> dict:
    """Run evaluation and return full metrics payload.

    Returns dict with keys from build_metrics_payload():
    - overall_f1, per_lang_f1, per_lang_gap, sample_counts,
    - confusion_matrix, per_lang_confusion_matrices
    """
    # ... existing logic (load model, predict, compute metrics)
    return metrics_payload

def main(argv=None) -> int:
    """CLI wrapper — writes JSON reports."""
    args = parse_args(argv)
    metrics = evaluate(args.task)
    # ... write JSON files (existing code)
    return 0
```

### 4c. No changes to other files

- `export_onnx.py` — called via CLI from notebook
- `mlflow_callback.py` — used as-is
- `metrics.py` — used as-is
- `generate_shap_plots.py` — called via CLI or import
- `benchmark_onnx.py` — called via CLI

## 5. MLflow Run Hierarchy on DagsHub

```
Experiment: "phase2_full_pipeline"
└── Run: pipeline_v2.0 (PARENT)
    │
    ├── Nested Run: train_sarcasm
    │   ├── Params: lr=0.0002, epochs=3, batch_size=16, lora_r=16...
    │   ├── Metrics: train_loss (per step), eval_loss (per epoch), eval_f1...
    │   └── Tags: task=sarcasm, device=cuda, git_sha=abc1234
    │
    ├── Nested Run: train_sentiment
    │   ├── Params: lr=0.0002, epochs=5, batch_size=16, lora_r=16...
    │   ├── Metrics: train_loss (per step), eval_loss (per epoch), eval_f1...
    │   └── Tags: task=sentiment, device=cuda, git_sha=abc1234
    │
    ├── Params (parent):
    │   ├── model_version: "v2.0"
    │   ├── gpu_type: "Tesla T4"
    │   ├── base_model: "xlm-roberta-base"
    │   └── branch: "feature/ai-core"
    │
    ├── Metrics (parent):
    │   ├── sarcasm_f1: 0.85
    │   ├── sentiment_f1: 0.78
    │   ├── fairness_gap: 0.05
    │   ├── onnx_speedup_fp32: 2.3x
    │   └── onnx_speedup_int8: 4.1x
    │
    ├── Tags (parent):
    │   ├── stage_data_download: complete
    │   ├── stage_training: complete
    │   ├── stage_evaluation: complete
    │   ├── stage_onnx_export: complete
    │   ├── stage_push: complete
    │   └── git_tag: model-v2.0
    │
    └── Artifacts (parent):
        └── report_plots/
            ├── training_curves.png
            ├── confusion_matrix_sarcasm.png
            ├── confusion_matrix_sentiment.png
            ├── fairness_chart.png
            ├── baseline_vs_finetuned.png
            ├── dataset_distribution_sarcasm.png
            ├── dataset_distribution_sentiment.png
            ├── shap_sample_1.png
            ├── shap_sample_2.png
            ├── onnx_benchmark.png
            └── metrics_summary.json
```

## 6. Error Handling

| Scenario | Detection | Recovery |
|---|---|---|
| Colab disconnect during training | Parent run stays `RUNNING` on DagsHub | Re-run notebook from beginning |
| 1 task succeeds, 1 fails | Successful nested run shows `FINISHED` | Re-run failed cell only |
| DVC push auth error | Pre-flight check catches in Section 2 | Fix secret, re-run Section 2 |
| Git push fail | Notebook prints `dvc.lock` content as fallback | Copy to local manually |
| ONNX export fail | Adapters already saved on Colab disk | Re-run export cell only |
| MLflow connection fail | Pre-flight check catches in Section 2 | Fix URI/token, re-run Section 2 |

**Pre-flight checks (Section 2):**
```python
# Fail fast before any training
assert mlflow_uri, "❌ MLFLOW_TRACKING_URI not set"
assert dagshub_token, "❌ DAGSHUB_TOKEN not set"
assert github_token, "❌ GITHUB_TOKEN not set"
assert model_version, "❌ MODEL_VERSION not set"

# Test connections
mlflow.set_tracking_uri(mlflow_uri)
experiments = mlflow.search_experiments()
print(f"✅ MLflow connected — {len(experiments)} experiments found")

# Test DVC remote
# (configure + dvc status)
```

## 7. Local Workflow After Colab

After notebook completes successfully:

```bash
# 1. Pull latest code + dvc.lock
git pull origin feature/ai-core

# 2. Build Docker (pulls ONNX models from DagsHub via DVC)
docker build \
  --build-arg DAGSHUB_USERNAME=trungdq.ts \
  --build-arg DAGSHUB_TOKEN=<token> \
  -t sentiment-api .

# 3. Run
docker-compose up

# --- Rollback to previous version ---
git checkout model-v1.0 -- dvc.lock
docker build --build-arg ... -t sentiment-api .
```

## 8. Colab Secrets Required (Summary)

| Secret Name | Value | Used By |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/trungdq.ts/Sentiment-Analysis-Service.mlflow` | MLflow tracking |
| `DAGSHUB_USER` | `trungdq.ts` | MLflow auth + DVC auth |
| `DAGSHUB_TOKEN` | DagsHub access token | MLflow auth + DVC auth |
| `GITHUB_TOKEN` | GitHub Personal Access Token | `git push` dvc.lock + tags |
| `MODEL_VERSION` | e.g. `v2.0` | Git tag + MLflow param |

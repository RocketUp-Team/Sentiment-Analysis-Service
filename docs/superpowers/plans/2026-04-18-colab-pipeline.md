# Colab Full-Pipeline Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-click "Run All" Colab notebook that runs train → evaluate → ONNX export → visualize → DVC push, with real-time DagsHub/MLflow monitoring.

**Architecture:** Refactor `run_finetuning.py` and `evaluate_finetuned.py` to extract importable `train()` and `evaluate()` functions (MLflow-context-free). Then build an 8-section Colab notebook that orchestrates these functions inside a parent MLflow run with nested child runs, ending with DVC push + git tag versioning.

**Tech Stack:** Python, MLflow, DVC, PEFT/LoRA, HuggingFace Transformers, ONNX, matplotlib, SHAP, Google Colab

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| MODIFY | `src/scripts/run_finetuning.py` | Extract `train()` function; `main()` becomes thin CLI wrapper |
| MODIFY | `src/scripts/evaluate_finetuned.py` | Extract `evaluate()` function; `main()` becomes thin CLI wrapper |
| MODIFY | `tests/scripts/test_run_finetuning.py` | Add `test_train_returns_result_dict`, update FakeTrainer |
| MODIFY | `tests/scripts/test_evaluate_finetuned.py` | Add `test_evaluate_returns_metrics_payload` |
| CREATE | `notebooks/colab_full_pipeline.ipynb` | 8-section pipeline notebook |

---

### Task 1: Extract `train()` from `run_finetuning.py`

**Files:**
- Modify: `src/scripts/run_finetuning.py`
- Modify: `tests/scripts/test_run_finetuning.py`

- [ ] **Step 1: Add FakeTrainerState and update FakeTrainer to support return values**

In `tests/scripts/test_run_finetuning.py`, update `FakeTrainer` to expose `.state.log_history` and add a `FakePeftModel.get_nb_trainable_parameters` method:

```python
# Add inside FakePeftModel class (after save_pretrained method, around line 168):
    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        return (1000, 100000)

# Replace FakeTrainer class entirely (around line 171):
class FakeTrainerState:
    def __init__(self) -> None:
        self.log_history: list[dict] = [
            {"loss": 0.5, "epoch": 1},
            {"eval_loss": 0.4, "epoch": 1},
        ]

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
        self.state = FakeTrainerState()
        type(self).instances.append(self)

    def train(self) -> None:
        self.train_calls += 1

    def evaluate(self) -> dict:
        return {"eval_loss": 0.4, "eval_f1": 0.85}
```

- [ ] **Step 2: Write failing test for `train()` return value**

Add at the end of `tests/scripts/test_run_finetuning.py`:

```python
def test_train_returns_result_dict_with_expected_keys(monkeypatch):
    tables = {
        "sarcasm.csv": pd.DataFrame(_build_rows(40, "1")),
    }
    fakes = _install_training_fakes(monkeypatch, tables)

    from src.scripts.run_finetuning import train

    result = train("sarcasm", smoke=True)

    assert isinstance(result, dict)
    assert result["adapter_path"].endswith("models/adapters_smoke/sarcasm")
    assert result["eval_metrics"] == {"eval_loss": 0.4, "eval_f1": 0.85}
    assert result["trainable_params"] == (1000, 100000)
    assert isinstance(result["log_history"], list)
    assert result["log_history"][0]["loss"] == 0.5
    assert result["peft_model"] is fakes["peft_model"]["model"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/scripts/test_run_finetuning.py::test_train_returns_result_dict_with_expected_keys -v`
Expected: FAIL — `train` not importable or doesn't exist yet.

- [ ] **Step 4: Implement `train()` and refactor `main()`**

In `src/scripts/run_finetuning.py`, add `train()` before `main()` (after `_model_num_labels`), then simplify `main()`:

```python
def train(
    task_name: str,
    *,
    smoke: bool = False,
    root: Path | None = None,
) -> dict:
    """Run the full training pipeline for one task.

    MLflow run context is managed by the CALLER — this function does NOT call
    mlflow.start_run(), mlflow.set_experiment(), or mlflow.set_tracking_uri().
    The HuggingFace Trainer will log metrics to whichever MLflow run is active
    in the caller's context via report_to=["mlflow"].

    Returns dict with keys:
    - adapter_path: str — path to saved LoRA adapter
    - eval_metrics: dict — from trainer.evaluate()
    - trainable_params: tuple — (trainable, total) from peft
    - log_history: list[dict] — trainer.state.log_history for plotting
    - peft_model: PeftModel — reference for downstream SHAP/inspection
    """
    root = root or Path(__file__).resolve().parents[2]
    task = get_task_config(task_name)
    df = _load_training_frame(task, root)

    epochs = task.epochs
    if smoke:
        df = df.head(32)
        epochs = 1

    rows = df.to_dict("records")
    clean_rows = dedup_rows(rows)
    split_rows = _split_rows_for_training(task, clean_rows, smoke=smoke)
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

    output_dir = _resolve_output_dir(root, task.name, smoke)
    training_args = _build_training_args(task, output_dir, epochs=epochs, smoke=smoke)

    trainer = _build_trainer(
        model=peft_model,
        training_args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    peft_model.save_pretrained(str(output_dir))
    logging.info("Training complete. Adapter saved to %s", output_dir)

    return {
        "adapter_path": str(output_dir),
        "eval_metrics": eval_metrics,
        "trainable_params": peft_model.get_nb_trainable_parameters(),
        "log_history": trainer.state.log_history,
        "peft_model": peft_model,
    }
```

Then replace `main()` with:

```python
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

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"phase2_finetuning_{task.name}")

    if not args.smoke:
        with mlflow.start_run():
            mlflow.set_tags(tags)
            train(args.task, smoke=False)
    else:
        train(args.task, smoke=True)

    return 0
```

> **⚠️ MLflow Contract:** `train()` does NOT call `mlflow.start_run()`, `mlflow.set_experiment()`, or `mlflow.set_tracking_uri()`. It uses `report_to=["mlflow"]` in TrainingArguments so HF Trainer logs to whichever run the caller has active.

- [ ] **Step 5: Run all training tests**

Run: `pytest tests/scripts/test_run_finetuning.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/scripts/run_finetuning.py tests/scripts/test_run_finetuning.py
git commit -m "refactor: extract train() from run_finetuning.py for notebook import"
```

---

### Task 2: Extract `evaluate()` from `evaluate_finetuned.py`

**Files:**
- Modify: `src/scripts/evaluate_finetuned.py`
- Modify: `tests/scripts/test_evaluate_finetuned.py`

- [ ] **Step 1: Write failing test for `evaluate()` return value**

Add at the end of `tests/scripts/test_evaluate_finetuned.py`:

```python
def test_evaluate_returns_metrics_payload_dict(tmp_path, monkeypatch):
    """evaluate() returns the full metrics payload without writing files."""
    inference_calls: list[dict] = []

    def fake_read_csv(path):
        return pd.DataFrame(
            [
                {"text": "great", "label": 2, "lang": "en"},
                {"text": "bad", "label": 0, "lang": "en"},
                {"text": "tot", "label": 2, "lang": "vi"},
            ]
        )

    class FakeInference:
        def __init__(self, config) -> None:
            self.config = config

        def predict_batch(self, texts, lang: str, skip_absa: bool):
            inference_calls.append({"texts": list(texts), "lang": lang})
            return [
                SimpleNamespace(sentiment="positive", sarcasm_flag=False),
                SimpleNamespace(sentiment="negative", sarcasm_flag=False),
                SimpleNamespace(sentiment="neutral", sarcasm_flag=False),
            ]

    expected_payload = {
        "overall_f1": 0.8,
        "per_lang_f1": {"en": 1.0, "vi": 0.0},
        "per_lang_gap": 1.0,
        "sample_counts": {"en": 2, "vi": 1},
        "confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        "per_lang_confusion_matrices": {
            "en": [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
            "vi": [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
        },
    }

    def fake_build_metrics_payload(*, y_true, y_pred, languages, label_names):
        return expected_payload

    monkeypatch.setattr(evaluate_finetuned.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(evaluate_finetuned, "BaselineModelInference", FakeInference)
    monkeypatch.setattr(evaluate_finetuned, "build_metrics_payload", fake_build_metrics_payload)

    from src.scripts.evaluate_finetuned import evaluate

    result = evaluate("sentiment", root=tmp_path, max_samples=None)

    assert result["overall_f1"] == 0.8
    assert result["per_lang_f1"] == {"en": 1.0, "vi": 0.0}
    assert result["per_lang_gap"] == 1.0
    assert "y_true" in result
    assert "y_pred" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_evaluate_finetuned.py::test_evaluate_returns_metrics_payload_dict -v`
Expected: FAIL — `evaluate` not importable.

- [ ] **Step 3: Implement `evaluate()` and refactor `main()`**

In `src/scripts/evaluate_finetuned.py`, add `evaluate()` before `main()`:

```python
def evaluate(
    task_name: str,
    *,
    root: Path | None = None,
    max_samples: int | None = None,
) -> dict:
    """Run evaluation and return full metrics payload.

    MLflow context is managed by the CALLER if logging is desired.

    Returns dict with keys from build_metrics_payload() plus:
    - y_true: list[str] — ground truth labels
    - y_pred: list[str] — predicted labels
    """
    root = root or Path(__file__).resolve().parents[2]
    task = get_task_config(task_name)

    config = ModelConfig(
        mode="finetuned",
        sentiment_adapter_path=str(root / "models" / "adapters" / "sentiment"),
        sarcasm_adapter_path=str(root / "models" / "adapters" / "sarcasm"),
    )
    inference = BaselineModelInference(config=config)
    df = _select_evaluation_rows(_load_evaluation_frame(task_name, root))

    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    texts = df["text"].tolist()
    languages = _resolve_languages(df)

    results = inference.predict_batch(texts, lang="en", skip_absa=True)

    y_pred: list[str] = []
    y_true: list[str] = []
    for row, result in zip(df.itertuples(index=False), results, strict=True):
        if task_name == "sarcasm":
            y_pred.append(task.label_names[1] if result.sarcasm_flag else task.label_names[0])
        else:
            y_pred.append(result.sentiment)
        y_true.append(_resolve_true_label(row.label, task.label_names))

    metrics_payload = build_metrics_payload(
        y_true=y_true,
        y_pred=y_pred,
        languages=languages,
        label_names=task.label_names,
    )
    metrics_payload["y_true"] = y_true
    metrics_payload["y_pred"] = y_pred
    return metrics_payload
```

Then simplify `main()` to call `evaluate()`:

```python
def main(argv: list[str] | None = None) -> int:
    """CLI wrapper — writes JSON reports. Backwards-compatible with DVC."""
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metrics_payload = evaluate(args.task, root=root, max_samples=100)

    args.output.write_text(
        json.dumps(
            {
                "task": args.task,
                "overall_f1": metrics_payload["overall_f1"],
                "n_samples": len(metrics_payload["y_true"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output.parent.joinpath("per_language_f1.json").write_text(
        json.dumps({"per_lang_f1": metrics_payload["per_lang_f1"]}, indent=2),
        encoding="utf-8",
    )
    args.output.parent.joinpath("fairness_report.json").write_text(
        json.dumps(
            {
                "overall_f1": metrics_payload["overall_f1"],
                "per_lang_f1": metrics_payload["per_lang_f1"],
                "per_lang_gap": metrics_payload["per_lang_gap"],
                "sample_counts": metrics_payload["sample_counts"],
                "confusion_matrices": metrics_payload["per_lang_confusion_matrices"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation for {args.task} completed.")
    return 0
```

- [ ] **Step 4: Run all evaluation tests**

Run: `pytest tests/scripts/test_evaluate_finetuned.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/scripts/evaluate_finetuned.py tests/scripts/test_evaluate_finetuned.py
git commit -m "refactor: extract evaluate() from evaluate_finetuned.py for notebook import"
```

---

### Task 3: Create `notebooks/colab_full_pipeline.ipynb`

**Files:**
- Create: `notebooks/colab_full_pipeline.ipynb`

> **Note:** This is a Colab notebook (JSON). Since it runs on Colab with GPU, there are no local unit tests. Verification is structural (valid JSON, correct imports, correct function calls).

- [ ] **Step 1: Create the notebook file**

Create `notebooks/colab_full_pipeline.ipynb` with 8 sections matching the spec:

**Section 1 — Setup:** Clone repo, install deps, GPU validation.
**Section 2 — Credentials:** Read Colab Secrets, pre-flight checks (MLflow connection, DVC remote, git remote).
**Section 3 — Data Download:** `!python -m src.data.downloader --task sarcasm/sentiment`
**Section 4 — Training:** Inside parent MLflow run, nested child runs calling `train()` with `_adapter_exists()` skip logic.
**Section 5 — Evaluation:** Call `evaluate()` for both tasks, log summary metrics to parent run.
**Section 6 — ONNX Export:** CLI calls to `export_onnx.py` and `benchmark_onnx.py`.
**Section 7 — Visualization:** 10 automated outputs (training curves, confusion matrices, fairness chart, SHAP, ONNX benchmark, baseline comparison, dataset distribution, LoRA summary, GPU resource log). All PNGs uploaded as MLflow artifacts.
**Section 8 — DVC Push + Git Push:** Push ONNX models via DVC, commit `dvc.lock`, create git tag `model-{version}`.

The notebook cells must use these exact imports from refactored code:
```python
from src.scripts.run_finetuning import train
from src.scripts.evaluate_finetuned import evaluate
```

Key design points:
- `train()` is called inside `mlflow.start_run(run_name="train_sarcasm", nested=True)` — the notebook manages MLflow context
- `evaluate()` returns metrics_payload including `y_true`, `y_pred` for confusion matrix plotting
- `_adapter_exists()` enables idempotent re-runs (Colab disconnect recovery)
- All 5 Colab Secrets: `MLFLOW_TRACKING_URI`, `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `GITHUB_TOKEN`, `MODEL_VERSION`

- [ ] **Step 2: Verify notebook is valid JSON**

Run: `python -m json.tool notebooks/colab_full_pipeline.ipynb > /dev/null`
Expected: Exit 0 (valid JSON)

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_full_pipeline.ipynb
git commit -m "feat: add full-pipeline Colab notebook with MLflow monitoring"
```

---

## Verification Plan

### Automated Tests

```bash
# Run all script tests to verify refactoring didn't break anything
pytest tests/scripts/ -v

# Validate notebook JSON
python -m json.tool notebooks/colab_full_pipeline.ipynb > /dev/null
```

### Manual Verification

- Open `notebooks/colab_full_pipeline.ipynb` on Google Colab
- Verify all sections render correctly as markdown/code cells
- Verify `train()` and `evaluate()` imports work from notebook context
- Full pipeline execution requires Colab GPU runtime + Colab Secrets configured

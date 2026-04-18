# Hardware Profiles & DVC Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-detect hardware (Colab L4 vs local Mac) and use optimal training params while keeping effective batch size fixed for reproducibility, then sync `dvc.yaml` and `params.yaml` accordingly.

**Architecture:** Create a `HardwareProfile` dataclass in `src/training/hardware_profiles.py` with two fixed profiles. The training pipeline auto-detects the environment via `torch.cuda.is_available()` and selects the matching profile. `params.yaml` stores both profiles and `effective_batch_size` for DVC tracking. `TaskConfig` no longer holds hardware-specific fields.

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, DVC, pytest

**Spec:** [2026-04-18-hardware-profiles-dvc-sync-design.md](file:///Users/trungshin/learning/Sentiment-Analysis-Service/docs/superpowers/specs/2026-04-18-hardware-profiles-dvc-sync-design.md)

---

### Task 1: Create `hardware_profiles.py` with tests (TDD)

**Files:**
- Create: `src/training/hardware_profiles.py`
- Create: `tests/training/test_hardware_profiles.py`

- [ ] **Step 1: Write failing tests for HardwareProfile and detect_profile**

```python
# tests/training/test_hardware_profiles.py
from unittest.mock import patch

import pytest

from src.training.hardware_profiles import (
    COLAB_L4,
    LOCAL_MAC,
    HardwareProfile,
    detect_profile,
    get_profile,
)


class TestHardwareProfileValues:
    """Verify that profiles maintain effective_batch_size = 32."""

    def test_colab_l4_effective_batch_is_32(self):
        effective = COLAB_L4.batch_size * COLAB_L4.gradient_accumulation_steps
        assert effective == 32

    def test_local_mac_effective_batch_is_32(self):
        effective = LOCAL_MAC.batch_size * LOCAL_MAC.gradient_accumulation_steps
        assert effective == 32

    def test_colab_l4_has_bf16_enabled(self):
        assert COLAB_L4.bf16 is True

    def test_local_mac_has_bf16_disabled(self):
        assert LOCAL_MAC.bf16 is False

    def test_profiles_are_frozen(self):
        with pytest.raises(AttributeError):
            COLAB_L4.batch_size = 999


class TestDetectProfile:
    @patch("src.training.hardware_profiles.torch.cuda.is_available", return_value=True)
    @patch(
        "src.training.hardware_profiles.torch.cuda.get_device_name",
        return_value="NVIDIA L4",
    )
    def test_detect_cuda_returns_colab_l4(self, _mock_name, _mock_cuda):
        profile = detect_profile()
        assert profile is COLAB_L4

    @patch("src.training.hardware_profiles.torch.cuda.is_available", return_value=False)
    def test_detect_no_cuda_returns_local_mac(self, _mock_cuda):
        profile = detect_profile()
        assert profile is LOCAL_MAC


class TestGetProfile:
    def test_get_colab_l4(self):
        assert get_profile("colab_l4") is COLAB_L4

    def test_get_local_mac(self):
        assert get_profile("local_mac") is LOCAL_MAC

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown hardware profile"):
            get_profile("aws_p4d")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_hardware_profiles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.hardware_profiles'`

- [ ] **Step 3: Implement hardware_profiles.py**

```python
# src/training/hardware_profiles.py
"""Auto-detect hardware and return optimal training profile."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware-specific training config.

    Chỉ ảnh hưởng tốc độ training, KHÔNG ảnh hưởng kết quả model.
    Kết quả model được đảm bảo bởi effective_batch_size
    (batch_size × gradient_accumulation_steps) cố định ở tất cả profiles.
    """

    name: str
    batch_size: int
    gradient_accumulation_steps: int
    bf16: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    dataloader_persistent_workers: bool
    dataloader_prefetch_factor: int | None
    optim: str


COLAB_L4 = HardwareProfile(
    name="colab_l4",
    batch_size=32,
    gradient_accumulation_steps=1,
    bf16=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=2,
    optim="adamw_torch_fused",
)

LOCAL_MAC = HardwareProfile(
    name="local_mac",
    batch_size=8,
    gradient_accumulation_steps=4,
    bf16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=False,
    dataloader_prefetch_factor=None,
    optim="adamw_torch",
)

_PROFILES = {"colab_l4": COLAB_L4, "local_mac": LOCAL_MAC}


def detect_profile() -> HardwareProfile:
    """Auto-detect hardware and return the matching profile.

    Detection logic:
    - torch.cuda.is_available() == True  → colab_l4
    - otherwise                          → local_mac
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Detected CUDA GPU: %s → using colab_l4 profile", gpu_name)
        return COLAB_L4

    logger.info("No CUDA GPU detected → using local_mac profile")
    return LOCAL_MAC


def get_profile(name: str) -> HardwareProfile:
    """Return a named profile. Raises ValueError if not found."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(_PROFILES)
        raise ValueError(
            f"Unknown hardware profile: {name}. Valid: {valid}"
        ) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_hardware_profiles.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/hardware_profiles.py tests/training/test_hardware_profiles.py
git commit -m "feat: add hardware profiles with auto-detect for Colab L4 vs local Mac"
```

---

### Task 2: Update `task_configs.py` — remove hardware params

**Files:**
- Modify: `src/training/task_configs.py:8-23`
- Modify: `tests/training/test_task_configs.py`

- [ ] **Step 1: Update existing tests to remove batch_size/grad_accum assertions**

Verify that existing tests in `tests/training/test_task_configs.py` do NOT assert on `batch_size` or `gradient_accumulation_steps`. Current tests only check `languages`, `num_labels`, `epochs`, `label_names` — no changes needed to test file.

Run: `python -m pytest tests/training/test_task_configs.py -v`
Expected: All 3 tests PASS (baseline before changes)

- [ ] **Step 2: Remove batch_size and gradient_accumulation_steps from TaskConfig**

Edit `src/training/task_configs.py` lines 8-23. Replace:

```python
@dataclass(frozen=True)
class TaskConfig:
    """Training settings shared by CLI, trainer, and evaluation flows."""

    name: str
    adapter_name: str
    num_labels: int
    label_names: tuple[str, ...]
    languages: tuple[str, ...]
    epochs: int
    learning_rate: float = 2e-4
    batch_size: int = 128  # Tăng từ 64 → 128: bf16 giảm VRAM ~50%, L4 dư sức chạy.
    gradient_accumulation_steps: int = 1
    max_length: int = 128
    dataset_version: str = "v1"
    seed: int = 42
```

With:

```python
@dataclass(frozen=True)
class TaskConfig:
    """Task-specific training settings.

    Hardware-specific params (batch_size, gradient_accumulation_steps, bf16, etc.)
    are managed by HardwareProfile in src/training/hardware_profiles.py.
    """

    name: str
    adapter_name: str
    num_labels: int
    label_names: tuple[str, ...]
    languages: tuple[str, ...]
    epochs: int
    learning_rate: float = 2e-4
    max_length: int = 128
    dataset_version: str = "v1"
    seed: int = 42
```

- [ ] **Step 3: Run task_configs tests to verify no regression**

Run: `python -m pytest tests/training/test_task_configs.py -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/training/task_configs.py
git commit -m "refactor: remove hardware params from TaskConfig (moved to HardwareProfile)"
```

---

### Task 3: Update `run_finetuning.py` — use HardwareProfile

**Files:**
- Modify: `src/scripts/run_finetuning.py:1-31` (imports)
- Modify: `src/scripts/run_finetuning.py:77-113` (`_build_training_args`)
- Modify: `src/scripts/run_finetuning.py:184-241` (`train`)

- [ ] **Step 1: Add import for hardware_profiles**

Edit `src/scripts/run_finetuning.py` line 30. Replace:

```python
from src.training.task_configs import get_task_config
```

With:

```python
from src.training.hardware_profiles import HardwareProfile, detect_profile
from src.training.task_configs import get_task_config
```

- [ ] **Step 2: Update `_build_training_args` to use HardwareProfile**

Edit `src/scripts/run_finetuning.py` lines 77-113. Replace the entire function:

```python
def _build_training_args(
    task,
    hw_profile: HardwareProfile,
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
        per_device_train_batch_size=hw_profile.batch_size,
        per_device_eval_batch_size=hw_profile.batch_size,
        gradient_accumulation_steps=hw_profile.gradient_accumulation_steps,
        num_train_epochs=epochs,
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=task.seed,
        report_to=["mlflow"] if not smoke else "none",
        # === Hardware-profile-driven optimizations ===
        bf16=hw_profile.bf16,
        dataloader_num_workers=0 if smoke else hw_profile.dataloader_num_workers,
        dataloader_pin_memory=hw_profile.dataloader_pin_memory,
        dataloader_persistent_workers=(
            hw_profile.dataloader_persistent_workers
            if not smoke and hw_profile.dataloader_num_workers > 0
            else False
        ),
        optim=hw_profile.optim,
    )
    if (
        hw_profile.dataloader_prefetch_factor is not None
        and not smoke
        and hw_profile.dataloader_num_workers > 0
    ):
        kwargs["dataloader_prefetch_factor"] = hw_profile.dataloader_prefetch_factor

    parameter_names = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in parameter_names:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return training_arguments_cls(**kwargs)
```

- [ ] **Step 3: Update `train()` to call detect_profile()**

Edit `src/scripts/run_finetuning.py` lines 204-241. Replace these lines inside `train()`:

```python
    root = root or Path(__file__).resolve().parents[2]
    task = get_task_config(task_name)
    df = _load_training_frame(task, root)
```

With:

```python
    root = root or Path(__file__).resolve().parents[2]
    task = get_task_config(task_name)
    hw_profile = detect_profile()
    logging.info(
        "Hardware profile: %s (batch=%d, grad_accum=%d, bf16=%s)",
        hw_profile.name,
        hw_profile.batch_size,
        hw_profile.gradient_accumulation_steps,
        hw_profile.bf16,
    )
    df = _load_training_frame(task, root)
```

And update the `_build_training_args` call on line 241. Replace:

```python
    training_args = _build_training_args(task, output_dir, epochs=epochs, smoke=smoke)
```

With:

```python
    training_args = _build_training_args(task, hw_profile, output_dir, epochs=epochs, smoke=smoke)
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v --ignore=tests/test_api.py`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/scripts/run_finetuning.py
git commit -m "refactor: use HardwareProfile in training pipeline for auto hardware detection"
```

---

### Task 4: Update `params.yaml` — new profile structure

**Files:**
- Modify: `params.yaml:35-55`

- [ ] **Step 1: Replace training section with profile structure**

Edit `params.yaml` lines 35-55. Replace:

```yaml
training:
  sarcasm:
    dataset_version: "tweet_eval_irony_v1"
    model_name: "xlm-roberta-base"
    adapter_output_dir: "models/adapters/sarcasm"
    learning_rate: 0.0002
    epochs: 3
    batch_size: 16
    gradient_accumulation_steps: 2
    max_length: 128
    seed: 42
  sentiment:
    dataset_version: "multilingual_sentiment_v1"
    model_name: "xlm-roberta-base"
    adapter_output_dir: "models/adapters/sentiment"
    learning_rate: 0.0002
    epochs: 5
    batch_size: 16
    gradient_accumulation_steps: 2
    max_length: 128
    seed: 42
```

With:

```yaml
training:
  effective_batch_size: 32

  sarcasm:
    dataset_version: "tweet_eval_irony_v1"
    model_name: "xlm-roberta-base"
    adapter_output_dir: "models/adapters/sarcasm"
    learning_rate: 0.0002
    epochs: 3
    max_length: 128
    seed: 42

  sentiment:
    dataset_version: "multilingual_sentiment_v1"
    model_name: "xlm-roberta-base"
    adapter_output_dir: "models/adapters/sentiment"
    learning_rate: 0.0002
    epochs: 5
    max_length: 128
    seed: 42

  hardware_profiles:
    colab_l4:
      batch_size: 32
      gradient_accumulation_steps: 1
      bf16: true
      dataloader_num_workers: 2
      dataloader_pin_memory: true
      dataloader_persistent_workers: true
      dataloader_prefetch_factor: 2
      optim: "adamw_torch_fused"

    local_mac:
      batch_size: 8
      gradient_accumulation_steps: 4
      bf16: false
      dataloader_num_workers: 0
      dataloader_pin_memory: false
      dataloader_persistent_workers: false
      dataloader_prefetch_factor: null
      optim: "adamw_torch"
```

- [ ] **Step 2: Commit**

```bash
git add params.yaml
git commit -m "config: restructure training params with hardware profiles and effective_batch_size"
```

---

### Task 5: Update `dvc.yaml` — sync params tracking + add SHAP stage

**Files:**
- Modify: `dvc.yaml:88-113` (finetune stages params)
- Modify: `dvc.yaml` (add generate_shap_plots stage after benchmark_onnx)

- [ ] **Step 1: Update finetune_sarcasm params**

Edit `dvc.yaml` lines 94-97. Replace:

```yaml
    params:
      - training.sarcasm
      - mlflow
      - adapters
```

With:

```yaml
    params:
      - training.effective_batch_size
      - training.sarcasm
      - training.hardware_profiles
      - mlflow
      - adapters
```

- [ ] **Step 2: Update finetune_sentiment params**

Edit `dvc.yaml` lines 108-111. Replace:

```yaml
    params:
      - training.sentiment
      - mlflow
      - adapters
```

With:

```yaml
    params:
      - training.effective_batch_size
      - training.sentiment
      - training.hardware_profiles
      - mlflow
      - adapters
```

- [ ] **Step 3: Add generate_shap_plots stage**

Append after the `benchmark_onnx` stage (after line 164):

```yaml
  generate_shap_plots:
    cmd: python3 -m src.scripts.generate_shap_plots --output-dir reports/shap_plots
    deps:
      - src/scripts/generate_shap_plots.py
      - models/adapters/sentiment
      - models/adapters/sarcasm
    outs:
      - reports/shap_plots/
```

- [ ] **Step 4: Commit**

```bash
git add dvc.yaml
git commit -m "config: sync dvc.yaml with hardware profiles params and add SHAP stage"
```

---

### Task 6: Fix Colab notebook `export_onnx` call

**Files:**
- Modify: `notebooks/colab_full_pipeline.ipynb` (Section 6, line 186-187 in JSON)

- [ ] **Step 1: Update Section 6 ONNX export cell**

Edit `notebooks/colab_full_pipeline.ipynb`. In the Section 6 code cell, replace the source:

```json
"source": [
    "!python -m src.scripts.export_onnx\n",
    "!python -m src.scripts.benchmark_onnx\n"
]
```

With:

```json
"source": [
    "!python -m src.scripts.export_onnx --adapter-name sentiment\n",
    "!python -m src.scripts.export_onnx --adapter-name sarcasm\n",
    "!python -m src.scripts.benchmark_onnx\n"
]
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/colab_full_pipeline.ipynb
git commit -m "fix: export both sentiment and sarcasm ONNX models in Colab notebook"
```

---

### Task 7: Run full test suite and verify DVC

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v --ignore=tests/test_api.py`
Expected: All tests PASS including new `test_hardware_profiles.py`

- [ ] **Step 2: Verify DVC can parse the updated config**

Run: `dvc params diff`
Expected: Shows the params changes (new `effective_batch_size`, `hardware_profiles`, removed per-task `batch_size`/`gradient_accumulation_steps`). No parsing errors.

- [ ] **Step 3: Verify DVC stage graph is valid**

Run: `dvc dag`
Expected: Shows the pipeline DAG including the new `generate_shap_plots` stage connected after `finetune_*` stages. No errors.

- [ ] **Step 4: Final commit if any formatting fixes needed**

```bash
git status
# If clean, skip. If formatting changes needed:
# git add -A && git commit -m "chore: formatting cleanup"
```

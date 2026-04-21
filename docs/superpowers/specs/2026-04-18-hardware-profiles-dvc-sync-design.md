# Hardware Profiles & DVC Pipeline Sync

## Problem

Training trên Colab L4 GPU chỉ sử dụng ~11% VRAM (2.4/22.5 GB) và ~6.6% system RAM (3.5/53 GB), dẫn đến training chậm. Nguyên nhân gốc:

1. **Mất đồng bộ config**: `params.yaml` (DVC track) có `batch_size=16, grad_accum=2` nhưng code thực tế (`task_configs.py`) dùng `batch_size=128, grad_accum=1`. DVC không biết training params đã thay đổi.
2. **Không có cơ chế environment-aware**: Cùng một bộ params cho cả local Mac và Colab L4, trong khi hai môi trường có hardware capacity rất khác nhau.
3. **DataLoader chưa tối ưu**: Thiếu `persistent_workers`, `prefetch_factor` khiến GPU phải chờ CPU load data.

## Giải pháp: Environment Profiles

Tách training config thành 2 phần:

- **Training dynamics** (cố định): `effective_batch_size`, `learning_rate`, `epochs`, `seed` — quyết định chất lượng model, giống nhau mọi môi trường.
- **Hardware config** (thay đổi theo môi trường): `batch_size`, `gradient_accumulation_steps`, `bf16`, `dataloader_*`, `optim` — chỉ ảnh hưởng tốc độ.

Hai profiles cố định:
- `colab_l4`: Colab GPU L4 (22.5 GB VRAM)
- `local_mac`: macOS local development

Server CPU chỉ dùng cho inference (load model + predict), không cần training profile.

---

## Proposed Changes

### 1. params.yaml — Restructure training config

#### [MODIFY] params.yaml

Thay thế `batch_size` và `gradient_accumulation_steps` trong mỗi task bằng cấu trúc profile:

```yaml
training:
  effective_batch_size: 32    # batch_size × grad_accum, cố định cho reproducibility

  sarcasm:
    dataset_version: "tweet_eval_irony_v1"
    model_name: "xlm-roberta-base"
    adapter_output_dir: "models/adapters/sarcasm"
    learning_rate: 0.0002
    epochs: 3
    max_length: 128
    seed: 42
    # Bỏ: batch_size, gradient_accumulation_steps

  sentiment:
    dataset_version: "multilingual_sentiment_v1"
    model_name: "xlm-roberta-base"
    adapter_output_dir: "models/adapters/sentiment"
    learning_rate: 0.0002
    epochs: 5
    max_length: 128
    seed: 42
    # Bỏ: batch_size, gradient_accumulation_steps

  hardware_profiles:
    colab_l4:
      batch_size: 32
      gradient_accumulation_steps: 1    # 32×1 = 32
      bf16: true
      dataloader_num_workers: 2
      dataloader_pin_memory: true
      dataloader_persistent_workers: true
      dataloader_prefetch_factor: 2
      optim: "adamw_torch_fused"

    local_mac:
      batch_size: 8
      gradient_accumulation_steps: 4    # 8×4 = 32
      bf16: false
      dataloader_num_workers: 0
      dataloader_pin_memory: false
      dataloader_persistent_workers: false
      dataloader_prefetch_factor: null
      optim: "adamw_torch"
```

Lý do chọn `effective_batch_size: 32`:
- Giá trị ban đầu hoạt động tốt trên local (`16×2=32`)
- Nằm trong khoảng khuyến nghị cho LoRA fine-tuning (16–64)
- Đủ lớn để training ổn định, đủ nhỏ để generalize tốt

---

### 2. src/training/hardware_profiles.py — Module auto-detect mới

#### [NEW] hardware_profiles.py

```python
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
    Kết quả model được đảm bảo bởi effective_batch_size (batch_size × gradient_accumulation_steps)
    cố định ở tất cả profiles.
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
        raise ValueError(f"Unknown hardware profile: {name}. Valid: {valid}") from exc
```

---

### 3. src/training/task_configs.py — Bỏ hardware params

#### [MODIFY] task_configs.py

Xoá `batch_size` và `gradient_accumulation_steps` khỏi `TaskConfig`:

```diff
 @dataclass(frozen=True)
 class TaskConfig:
     name: str
     adapter_name: str
     num_labels: int
     label_names: tuple[str, ...]
     languages: tuple[str, ...]
     epochs: int
     learning_rate: float = 2e-4
-    batch_size: int = 128
-    gradient_accumulation_steps: int = 1
     max_length: int = 128
     dataset_version: str = "v1"
     seed: int = 42
```

---

### 4. src/scripts/run_finetuning.py — Dùng HardwareProfile

#### [MODIFY] run_finetuning.py

**4a. Import mới:**
```python
from src.training.hardware_profiles import HardwareProfile, detect_profile
```

**4b. `_build_training_args` nhận `HardwareProfile`:**
```diff
 def _build_training_args(
     task,
+    hw_profile: HardwareProfile,
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
-        per_device_train_batch_size=task.batch_size,
-        per_device_eval_batch_size=task.batch_size,
-        gradient_accumulation_steps=task.gradient_accumulation_steps,
+        per_device_train_batch_size=hw_profile.batch_size,
+        per_device_eval_batch_size=hw_profile.batch_size,
+        gradient_accumulation_steps=hw_profile.gradient_accumulation_steps,
         num_train_epochs=epochs,
         save_strategy="epoch",
         load_best_model_at_end=True,
         seed=task.seed,
         report_to=["mlflow"] if not smoke else "none",
-        bf16=_on_cuda,
-        dataloader_num_workers=0 if smoke else 4,
-        dataloader_pin_memory=_on_cuda,
-        optim="adamw_torch_fused" if _on_cuda else "adamw_torch",
+        bf16=hw_profile.bf16,
+        dataloader_num_workers=0 if smoke else hw_profile.dataloader_num_workers,
+        dataloader_pin_memory=hw_profile.dataloader_pin_memory,
+        dataloader_persistent_workers=(
+            hw_profile.dataloader_persistent_workers
+            if not smoke and hw_profile.dataloader_num_workers > 0
+            else False
+        ),
+        optim=hw_profile.optim,
     )
+    if hw_profile.dataloader_prefetch_factor is not None and not smoke:
+        kwargs["dataloader_prefetch_factor"] = hw_profile.dataloader_prefetch_factor
```

**4c. `train()` gọi `detect_profile()`:**
```diff
 def train(task_name, *, smoke=False, root=None):
     root = root or Path(__file__).resolve().parents[2]
     task = get_task_config(task_name)
+    hw_profile = detect_profile()
     # ... existing code ...
-    training_args = _build_training_args(task, output_dir, epochs=epochs, smoke=smoke)
+    training_args = _build_training_args(task, hw_profile, output_dir, epochs=epochs, smoke=smoke)
```

---

### 5. dvc.yaml — Sync params & thêm SHAP stage

#### [MODIFY] dvc.yaml

**5a. Finetune stages — track params mới:**

```diff
   finetune_sarcasm:
     cmd: python3 -m src.scripts.run_finetuning --task sarcasm
     deps:
       - src/scripts/run_finetuning.py
       - src/training/
       - data/raw/sarcasm.csv
     params:
-      - training.sarcasm
+      - training.effective_batch_size
+      - training.sarcasm
+      - training.hardware_profiles
       - mlflow
       - adapters

   finetune_sentiment:
     cmd: python3 -m src.scripts.run_finetuning --task sentiment
     deps:
       - src/scripts/run_finetuning.py
       - src/training/
       - data/raw/sentiment_en.csv
       - data/raw/sentiment_vi.csv
     params:
-      - training.sentiment
+      - training.effective_batch_size
+      - training.sentiment
+      - training.hardware_profiles
       - mlflow
       - adapters
```

**5b. Thêm SHAP stage:**

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

---

### 6. Colab notebook — Fix export_onnx call

#### [MODIFY] colab_full_pipeline.ipynb (Section 6)

Notebook hiện chỉ chạy `export_onnx` không truyền `--adapter-name` (mặc định chỉ sentiment). Cần export cả sarcasm để khớp với `dvc.yaml`:

```diff
-!python -m src.scripts.export_onnx
+!python -m src.scripts.export_onnx --adapter-name sentiment
+!python -m src.scripts.export_onnx --adapter-name sarcasm
 !python -m src.scripts.benchmark_onnx
```

---

## Scope ngoài (không thay đổi)

- **Server CPU inference**: Không cần profile, chỉ load model và predict. Code inference (`BaselineModelInference`) không bị ảnh hưởng bởi thay đổi này.
- **`evaluate_finetuned` DVC stage**: Hiện chỉ `--task sentiment`. Colab evaluate cả hai nhưng đây là design choice riêng của notebook, không cần sync ngược vào DVC.

---

## Verification Plan

### Automated Tests

1. **Unit test `hardware_profiles.py`**: Test `detect_profile()` trả đúng profile khi mock `torch.cuda.is_available()`
2. **Unit test `_build_training_args`**: Verify training args sử dụng values từ `HardwareProfile` thay vì `TaskConfig`
3. **Existing tests**: Chạy toàn bộ test suite đảm bảo không regression

```bash
python -m pytest tests/ -v
```

### Manual Verification

1. **Local Mac**: Chạy `python -m src.scripts.run_finetuning --task sarcasm --smoke` → confirm log hiện "using local_mac profile" và batch_size=8
2. **Colab L4**: Chạy notebook Section 4 → confirm log hiện "using colab_l4 profile" và batch_size=32, bf16=True
3. **DVC repro**: Chạy `dvc repro finetune_sarcasm` trên local → verify DVC track đúng `training.hardware_profiles` params

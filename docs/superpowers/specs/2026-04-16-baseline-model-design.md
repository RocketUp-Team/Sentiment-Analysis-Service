# Baseline Model Design — Sentiment Analysis Service

> **Mục tiêu:** Xây dựng `BaselineModelInference` implement `ModelInference` interface bằng pre-trained RoBERTa, tích hợp MLflow evaluation tracking.

## Bối cảnh

- **Dự án:** Sentiment Analysis Service (Topic 8, DDM501)
- **Phạm vi:** Sub-project B.1 trong AI Core & Modeling (Trung phụ trách)
- **Phụ thuộc:** Handoff package (`contracts/`) ✅, Data preprocessing (`src/data/`) ✅
- **Output sẽ được dùng bởi:** Fine-tuning (B.2), ABSA implementation (B.3), Backend integration (Quân)

## Quyết định đã chốt

| Quyết định             | Giá trị                                                  | Lý do                                                                           |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Scope                  | Baseline Model Only                                      | MVP first — có model thật chạy xuyên suốt trước khi fine-tune                   |
| Pre-trained model      | `cardiffnlp/twitter-roberta-base-sentiment-latest`       | 3-class output khớp schema, RoBERTa đúng gợi ý rubric, domain review/short text |
| Environment            | Local M1 Pro, auto-detect device                         | CPU đủ nhanh cho inference (~80-150ms), MPS available khi cần                   |
| Device detection       | Auto-detect: cuda → mps → cpu                            | Portable cho Colab GPU sau này                                                  |
| ABSA trong baseline    | `aspects=[]` (empty list)                                | Baseline chỉ overall sentiment, ABSA là task riêng                              |
| Sarcasm trong baseline | `sarcasm_flag=False`                                     | Không có sarcasm model trong baseline                                           |
| MLflow                 | Evaluation logging — baseline run là experiment đầu tiên | Rubric "multiple experiments" = baseline vs fine-tuned comparison               |
| SHAP                   | Real implementation dùng `shap.Explainer`                | Rubric Responsible AI yêu cầu explainability                                    |

---

## 1. Kiến trúc tổng thể

### File Structure

```
src/
└── model/
    ├── __init__.py           # Export BaselineModelInference, get_device
    ├── device.py             # Device auto-detection utility
    ├── config.py             # Model configuration dataclass
    ├── baseline.py           # BaselineModelInference(ModelInference)
    └── evaluate.py           # CLI: evaluate model on processed data + MLflow log

tests/
└── model/
    ├── __init__.py
    ├── test_device.py        # Test device detection logic
    ├── test_baseline.py      # Test BaselineModelInference (mock model weights)
    └── test_evaluate.py      # Test evaluation functions
```

### Nguyên tắc thiết kế

- **Modular:** Mỗi file 1 trách nhiệm, consistent với `src/data/` pattern
- **Interface compliance:** `BaselineModelInference` implement đúng `ModelInference` abstract class từ `contracts/model_interface.py`
- **Portable:** Device detection tự động, không hardcode
- **Extensible:** Fine-tune sprint sau chỉ thêm `finetuned.py` vào cùng package

### Môi trường thực thi

| Thành phần             | Giá trị                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| **Môi trường**         | Local (MacBook Pro M1 Pro, 16GB RAM)                                     |
| **Python interpreter** | `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python` |
| **Device**             | Auto-detect (cuda → mps → cpu)                                           |
| **Model size**         | ~500MB (download 1 lần, HuggingFace cache)                               |
| **Memory usage**       | ~1GB khi load model                                                      |

### Boundary: Baseline → Fine-tuning (Sub-project B.2)

- **Baseline outputs:** `BaselineModelInference` class implement đầy đủ `ModelInference` interface
- **Fine-tuning inputs:** Kế thừa architecture, thay model weights bằng fine-tuned version
- **MLflow baseline run:** Là experiment đầu tiên để so sánh với fine-tuned runs
- **Contract:** Backend (Quân) có thể swap `MockModelInference` → `BaselineModelInference` ngay lập tức

---

## 2. Device Detection

### `device.py`

```python
import torch

def get_device() -> torch.device:
    """Auto-detect best available device.

    Priority: CUDA (Colab/NVIDIA) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

---

## 3. Model Configuration

### `config.py`

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 512
    default_lang: str = "en"
    supported_languages: tuple[str, ...] = ("en",)
    label_map: dict[int, str] = field(default_factory=lambda: {
        0: "negative",
        1: "neutral",
        2: "positive",
    })
```

> **`label_map`:** cardiffnlp model output: index 0=negative, 1=neutral, 2=positive. Khớp hoàn toàn với project schema `["positive", "negative", "neutral"]`.

---

## 4. BaselineModelInference — Core Class

### `baseline.py`

```python
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline as hf_pipeline

from contracts.model_interface import ModelInference, PredictionResult, SHAPResult
from contracts.errors import UnsupportedLanguageError
from src.model.config import ModelConfig
from src.model.device import get_device


class BaselineModelInference(ModelInference):
    """Implement ModelInference interface bằng pre-trained RoBERTa.

    - Overall sentiment: predict từ model
    - ABSA aspects: trả [] (baseline không có ABSA)
    - SHAP: dùng shap.Explainer với HuggingFace pipeline
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ):
        self._config = config or ModelConfig()
        self._device = device or get_device()
        self._model = None
        self._tokenizer = None
        self._hf_pipeline = None  # Lazy-init cho SHAP
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer + model, move to device."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._config.model_name
        ).to(self._device)
        self._model.eval()

    def _check_language(self, lang: str) -> None:
        if lang not in self._config.supported_languages:
            raise UnsupportedLanguageError(
                f"Language '{lang}' not supported. Supported: {self._config.supported_languages}"
            )

    # ── Inference ──────────────────────────────────────────────

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        """Predict overall sentiment. aspects=[] cho baseline."""
        self._check_language(lang)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()

        return PredictionResult(
            sentiment=self._config.label_map[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            aspects=[],
            sarcasm_flag=False,
        )

    def predict_batch(self, texts: list[str], lang: str = "en") -> list[PredictionResult]:
        """Batch predict — tokenize batch cho efficiency."""
        self._check_language(lang)
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        results = []
        for i in range(len(texts)):
            pred_idx = probs[i].argmax().item()
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(probs[i][pred_idx].item(), 4),
                    aspects=[],
                    sarcasm_flag=False,
                )
            )
        return results

    # ── SHAP Explainability ────────────────────────────────────

    def _get_classification_pipeline(self):
        """Lazy-init HuggingFace pipeline cho SHAP explainer."""
        if self._hf_pipeline is None:
            self._hf_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                top_k=None,
            )
        return self._hf_pipeline

    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        """SHAP values cho mỗi token — phục vụ explainability UI."""
        self._check_language(lang)
        pipe = self._get_classification_pipeline()
        explainer = shap.Explainer(pipe)
        shap_values = explainer([text])

        # Lấy SHAP values cho predicted class
        pred = self.predict_single(text, lang)
        class_idx = {v: k for k, v in self._config.label_map.items()}[pred.sentiment]

        tokens = shap_values.data[0].tolist()
        values = shap_values.values[0][:, class_idx].tolist()
        base = float(shap_values.base_values[0][class_idx])

        return SHAPResult(
            tokens=tokens,
            shap_values=values,
            base_value=base,
        )

    # ── Properties ─────────────────────────────────────────────

    @property
    def supported_languages(self) -> list[str]:
        return list(self._config.supported_languages)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
```

### Key Design Decisions

| Decision                 | Rationale                                                           |
| ------------------------ | ------------------------------------------------------------------- |
| `torch.no_grad()`        | Tắt gradient computation → tiết kiệm memory + tăng tốc inference    |
| `model.eval()`           | Tắt dropout → kết quả deterministic                                 |
| `round(confidence, 4)`   | Tránh floating point noise trong JSON response                      |
| Lazy-init `_hf_pipeline` | Không tốn memory nếu không gọi SHAP                                 |
| Batch tokenization       | `padding=True` + batch input → hiệu quả hơn vòng lặp predict_single |

### Performance Estimates (M1 Pro, 16GB)

| Tác vụ                             | CPU             | MPS             |
| ---------------------------------- | --------------- | --------------- |
| Load model (first time)            | ~3-5s           | ~3-5s           |
| Single inference                   | ~80-150ms       | ~20-50ms        |
| Batch inference (32 samples)       | ~30-50ms/sample | ~10-20ms/sample |
| SHAP explanation                   | ~500ms-2s       | ~500ms-2s       |
| Full dataset eval (~3,800 samples) | ~6-8 phút       | ~2-3 phút       |

---

## 5. Evaluation & MLflow Logging

### `evaluate.py`

```python
"""CLI script: evaluate baseline model trên processed data, log kết quả lên MLflow.

Usage: python -m src.model.evaluate
"""
import json
import logging
import pandas as pd
import mlflow
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.model.baseline import BaselineModelInference
from src.model.config import ModelConfig
from src.data.utils import load_params

logger = logging.getLogger("model_evaluate")
LABELS = ["positive", "negative", "neutral"]


def evaluate_on_dataset(
    model: BaselineModelInference,
    sentences_df: pd.DataFrame,
    split: str = "test",
    batch_size: int = 32,
) -> dict:
    """Evaluate model trên 1 split, trả dict metrics."""
    df = sentences_df[sentences_df["split"] == split].copy()
    texts = df["text"].tolist()
    true_labels = df["sentiment"].tolist()

    # Batch predict
    pred_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        pred_results.extend(model.predict_batch(batch))

    pred_labels = [r.sentiment for r in pred_results]
    pred_confidences = [r.confidence for r in pred_results]

    return {
        "split": split,
        "n_samples": len(texts),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_macro": f1_score(true_labels, pred_labels, labels=LABELS, average="macro"),
        "f1_per_class": f1_score(
            true_labels, pred_labels, labels=LABELS, average=None
        ).tolist(),
        "precision_macro": precision_score(
            true_labels, pred_labels, labels=LABELS, average="macro"
        ),
        "recall_macro": recall_score(
            true_labels, pred_labels, labels=LABELS, average="macro"
        ),
        "mean_confidence": sum(pred_confidences) / len(pred_confidences),
        "classification_report": classification_report(
            true_labels, pred_labels, labels=LABELS
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, pred_labels, labels=LABELS
        ).tolist(),
    }


def log_to_mlflow(config: ModelConfig, metrics: dict, params_yaml: dict) -> None:
    """Log evaluation results lên MLflow."""
    tracking_uri = params_yaml.get("mlflow", {}).get(
        "tracking_uri", "http://localhost:5000"
    )
    experiment_name = params_yaml.get("mlflow", {}).get(
        "experiment_name", "sentiment_analysis"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="baseline_roberta"):
        # Params
        mlflow.log_params({
            "model_name": config.model_name,
            "model_type": "baseline_pretrained",
            "max_length": config.max_length,
            "device": str(metrics.get("device", "cpu")),
            "fine_tuned": False,
            "absa_enabled": False,
        })

        # Metrics
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "mean_confidence": metrics["mean_confidence"],
            "n_samples": metrics["n_samples"],
        })

        # Per-class F1
        for i, label in enumerate(LABELS):
            mlflow.log_metric(f"f1_{label}", metrics["f1_per_class"][i])

        # Artifact: confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=metrics["confusion_matrix"],
            display_labels=LABELS,
        ).plot(ax=ax)
        plt.title("Baseline Model — Confusion Matrix")
        cm_path = "/tmp/confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        # Artifact: classification report
        report_path = "/tmp/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(metrics["classification_report"])
        mlflow.log_artifact(report_path)

        logger.info(
            f"MLflow run logged: F1={metrics['f1_macro']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = load_params("params.yaml")
    config = ModelConfig()
    model = BaselineModelInference(config)

    sentences_df = pd.read_csv("data/processed/sentences.csv")
    metrics = evaluate_on_dataset(model, sentences_df, split="test")
    metrics["device"] = str(model._device)

    log_to_mlflow(config, metrics, params)

    print(f"\n{'='*50}")
    print(f"Baseline Evaluation Results (test split)")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"\n{metrics['classification_report']}")
```

### MLflow Tracking Summary

| Loại          | Nội dung                                                                         |
| ------------- | -------------------------------------------------------------------------------- |
| **Params**    | model_name, model_type, device, max_length, fine_tuned, absa_enabled             |
| **Metrics**   | accuracy, f1_macro, f1_per_class, precision_macro, recall_macro, mean_confidence |
| **Artifacts** | confusion_matrix.png, classification_report.txt                                  |

---

## 6. Dependencies

Thêm vào `requirements.txt`:

```
torch
transformers
shap
mlflow
matplotlib
```

> `scikit-learn` đã có từ data-preprocessing. `pandas` đã có.

---

## 7. Testing Strategy

### Test Files

| Test file          | Nội dung                                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `test_device.py`   | `get_device()` trả `torch.device` hợp lệ                                                                 |
| `test_baseline.py` | Mock model weights, test interface compliance: predict_single, predict_batch, properties, error handling |
| `test_evaluate.py` | Mock predictions, test metrics computation, required keys                                                |

### Unit Tests — Mock Approach

Tests **không download model thật** — dùng `unittest.mock` để mock `AutoTokenizer` và `AutoModelForSequenceClassification`. Đảm bảo:

- `predict_single` trả `PredictionResult` đúng schema
- `predict_batch` trả list đúng length
- `confidence` trong range [0, 1]
- `aspects == []` cho baseline
- `sarcasm_flag == False` cho baseline
- `UnsupportedLanguageError` raise khi `lang` không hỗ trợ
- `is_loaded == True` sau init
- `supported_languages` chứa `"en"`

### Integration Tests

```python
@pytest.mark.slow  # Chạy riêng, không chạy trong CI nhanh
class TestBaselineIntegration:
    def test_real_model_predict(self):
        """Smoke test: load model thật, predict 1 sample."""
        model = BaselineModelInference()
        result = model.predict_single("The food was great")
        assert result.sentiment in ("positive", "negative", "neutral")
```

### Coverage Target

- **Target:** ≥ 80% line coverage cho `src/model/`
- **Chạy:** `pytest tests/model/ --cov=src/model --cov-report=term-missing`
- **CI:** Unit tests chạy nhanh (mocked), integration tests chạy scheduled

---

## 8. Contract Alignment

| Interface method                   | Baseline behavior              | Handoff contract match      |
| ---------------------------------- | ------------------------------ | --------------------------- |
| `predict_single(text, lang)`       | Overall sentiment + confidence | ✅ `PredictionResult`       |
| `predict_batch(texts, lang)`       | Batch overall sentiment        | ✅ `list[PredictionResult]` |
| `get_shap_explanation(text, lang)` | Real SHAP values               | ✅ `SHAPResult`             |
| `supported_languages`              | `["en"]`                       | ✅                          |
| `is_loaded`                        | `True` sau init                | ✅                          |
| `aspects` field                    | `[]` (empty)                   | ✅ Contract cho phép rỗng   |
| `sarcasm_flag` field               | `False`                        | ✅ Phase 1 mock             |

### Backend Integration Path

Quân swap `MockModelInference` → `BaselineModelInference` trong FastAPI dependency:

```python
# Trước: model = MockModelInference()
# Sau:   model = BaselineModelInference()
# Không thay đổi API code — cùng interface
```

---

## 9. Scope Exclusions (sẽ làm ở sprint sau)

| Feature                | Sprint                       |
| ---------------------- | ---------------------------- |
| ABSA aspect detection  | B.3 (ABSA task riêng)        |
| Sarcasm detection      | B.4                          |
| PEFT/LoRA fine-tuning  | B.2                          |
| ONNX/Quantization      | B.5                          |
| Multi-language support | B.2 (fine-tune multilingual) |
| Fairness/Bias analysis | Responsible AI sprint        |

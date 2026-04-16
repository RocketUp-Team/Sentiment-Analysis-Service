# Zero-Shot ABSA Draft Model Design — Sentiment Analysis Service

> **Mục tiêu:** Tích hợp Zero-Shot ABSA vào `BaselineModelInference` để trả về `aspects` thật thay vì mảng rỗng, khai thông tích hợp cho Backend/Frontend.

## Bối cảnh

- **Phase:** Phase 1 — Khai thông tích hợp (Unblocking Integration)
- **Vấn đề:** `BaselineModelInference` hiện trả `aspects=[]`. Backend/Frontend không có dữ liệu aspect để hiển thị.
- **Phụ thuộc:** Baseline model ✅, Data preprocessing ✅, Contracts ✅
- **Output sẽ được dùng bởi:** Backend integration (Quân), Frontend ABSA UI (Long)
- **Tính tạm thời:** Đây là Draft Model — sẽ được thay thế bởi Fine-tuned ABSA Model sau khi fine-tune xong

## Quyết định đã chốt

| Quyết định           | Giá trị                                                                    | Lý do                                               |
| -------------------- | -------------------------------------------------------------------------- | --------------------------------------------------- |
| Phạm vi tích hợp     | Trực tiếp vào `baseline.py`                                                | Nhanh nhất cho Phase 1, không phá vỡ interface      |
| Aspect extraction    | Zero-Shot Classification (DeBERTa)                                         | Không cần train, accuracy tốt, confidence có sẵn    |
| Per-aspect sentiment | Zero-Shot Classification lần 2                                             | Chính xác hơn gán sentiment overall cho mỗi aspect  |
| Zero-Shot model      | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`                               | Nhẹ (~700MB), accuracy tốt hơn BART, phù hợp M1 Pro |
| Confidence threshold | 0.5                                                                        | Cân bằng precision/recall                           |
| Aspect categories    | 6 categories từ SemEval: food, service, ambiance, price, location, general | Khớp `params.yaml`                                  |

---

## 1. Kiến trúc & Luồng dữ liệu

### Pipeline ABSA 2 bước

```
Input text: "The food was amazing but the service was terrible"
    │
    ├──► [Bước 1] RoBERTa (existing)
    │         → overall sentiment: "positive", confidence: 0.72
    │
    ├──► [Bước 2] DeBERTa Zero-Shot — Aspect Extraction
    │         hypothesis_template: "This review is about {}."
    │         candidate_labels: ["food", "service", "ambiance", "price", "location", "general"]
    │         multi_label: True
    │         → scores: {food: 0.85, service: 0.78, ambiance: 0.12, price: 0.08, ...}
    │         → filter score > 0.5: [food, service]
    │
    └──► [Bước 3] DeBERTa Zero-Shot — Per-Aspect Sentiment (cho mỗi detected aspect)
              hypothesis_template: "The sentiment about {aspect} is {}."
              candidate_labels: ["positive", "negative", "neutral"]
              → food: positive (0.82)
              → service: negative (0.71)
    │
    ▼
PredictionResult(
    sentiment="positive",       # overall — from RoBERTa
    confidence=0.72,
    aspects=[
        AspectSentiment(aspect="food",    sentiment="positive", confidence=0.82),
        AspectSentiment(aspect="service", sentiment="negative", confidence=0.71),
    ],
    sarcasm_flag=False
)
```

### Nguyên tắc thiết kế

- **Lazy loading:** DeBERTa pipeline chỉ load khi gọi ABSA lần đầu (giống pattern SHAP pipeline)
- **Fallback an toàn:** Nếu DeBERTa load thất bại hoặc không tìm aspect nào, trả `aspects=[]` (không crash)
- **Configurable:** Model name, threshold, aspect categories đều nằm trong `ModelConfig`
- **Backward compatible:** Tất cả test hiện tại phải vẫn pass (trừ `test_aspects_empty_for_baseline` cần cập nhật)

---

## 2. Thay đổi code cụ thể

### File: `src/model/config.py`

Thêm 3 fields mới vào `ModelConfig`:

```python
@dataclass(frozen=True)
class ModelConfig:
    # ... existing fields giữ nguyên ...

    # ABSA Zero-Shot config (new)
    absa_model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    absa_threshold: float = 0.5
    absa_categories: tuple[str, ...] = (
        "food", "service", "ambiance", "price", "location", "general"
    )
```

### File: `src/model/baseline.py`

**Thêm:**

1. `_absa_pipeline` attribute (lazy-loaded, `None` ban đầu)
2. `_get_absa_pipeline()` method — lazy-init DeBERTa zero-shot-classification pipeline
3. `_extract_aspects(text: str) -> list[AspectSentiment]` method:
   - Bước 2: Gọi DeBERTa zero-shot với `multi_label=True` trên 6 categories
   - Lọc aspects có score > threshold
   - Bước 3: Cho mỗi aspect tìm được, gọi DeBERTa zero-shot lần 2 với labels `["positive", "negative", "neutral"]`
   - Trả về `list[AspectSentiment]`

**Sửa:**

4. `predict_single()`:

   ```python
   # Trước:
   aspects=[]
   # Sau:
   aspects=self._extract_aspects(text)
   ```

5. `predict_batch()`:
   ```python
   # Trước:
   aspects=[]
   # Sau:
   aspects=self._extract_aspects(texts[index])
   ```

### Pseudo-code cho `_extract_aspects`

```python
def _extract_aspects(self, text: str) -> list[AspectSentiment]:
    """Extract aspects and their sentiments using Zero-Shot Classification."""
    try:
        absa_pipe = self._get_absa_pipeline()

        # Bước 2: Aspect extraction
        aspect_result = absa_pipe(
            text,
            candidate_labels=list(self._config.absa_categories),
            hypothesis_template="This review is about {}.",
            multi_label=True,
        )

        detected_aspects = [
            (label, score)
            for label, score in zip(aspect_result["labels"], aspect_result["scores"])
            if score > self._config.absa_threshold
        ]

        if not detected_aspects:
            return []

        # Bước 3: Per-aspect sentiment
        aspects = []
        sentiment_labels = ["positive", "negative", "neutral"]
        for aspect_name, _ in detected_aspects:
            sent_result = absa_pipe(
                text,
                candidate_labels=sentiment_labels,
                hypothesis_template=f"The sentiment about {aspect_name} is {{}}.",
            )
            aspects.append(AspectSentiment(
                aspect=aspect_name,
                sentiment=sent_result["labels"][0],
                confidence=round(sent_result["scores"][0], 4),
            ))

        return aspects
    except Exception:
        # Fallback: nếu ABSA pipeline lỗi, trả mảng rỗng
        return []
```

---

## 3. Thay đổi Test

### File: `tests/model/test_baseline.py`

**Cập nhật:**

- `test_aspects_empty_for_baseline` → đổi thành `test_aspects_returned_from_absa`
  - Mock DeBERTa pipeline trả kết quả cố định
  - Assert `len(result.aspects) > 0`
  - Assert mỗi aspect có đúng schema: `aspect` in 6 categories, `sentiment` in 3 labels, `confidence` in [0, 1]

**Thêm tests mới:**

- `test_absa_pipeline_lazy_loaded`: Assert `_absa_pipeline is None` trước khi gọi predict
- `test_absa_fallback_on_error`: Mock pipeline raise Exception → aspects trả `[]`
- `test_absa_threshold_filters_low_scores`: Mock scores dưới threshold → aspects = `[]`
- `test_absa_per_aspect_sentiment`: Mock pipeline trả 2 aspects → verify mỗi aspect có sentiment riêng
- `test_predict_batch_includes_aspects`: Verify batch cũng trả aspects

### File: `tests/model/test_config.py`

**Thêm tests mới:**

- `test_default_absa_model_name`: Verify default = `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
- `test_default_absa_threshold`: Verify default = 0.5
- `test_default_absa_categories`: Verify 6 categories khớp project schema

---

## 4. Dependencies

Thêm vào `requirements.txt`:

```
# Không cần thêm dependency mới!
# transformers (đã có) bao gồm pipeline("zero-shot-classification")
# DeBERTa model sẽ tự download qua HuggingFace cache
```

> **Lưu ý:** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` cần `transformers>=4.30.0` (đã có trong requirements).

---

## 5. Performance Estimates (M1 Pro, 16GB)

| Tác vụ                           | Thời gian  | Ghi chú               |
| -------------------------------- | ---------- | --------------------- |
| Load DeBERTa (lần đầu, download) | ~10-30s    | Chỉ 1 lần, cache      |
| Load DeBERTa (từ cache)          | ~2-3s      | Lazy-loaded           |
| Aspect extraction / câu          | ~100-200ms | Zero-Shot inference   |
| Aspect sentiment / aspect        | ~80-150ms  | Zero-Shot inference   |
| Tổng 1 câu (2 aspects)           | ~350-600ms | Chấp nhận cho Phase 1 |
| Memory khi load cả 2 models      | ~1.5-2GB   | RoBERTa + DeBERTa     |

---

## 6. Contract Alignment

| Interface method                   | Behavior sau thay đổi                | Contract match              |
| ---------------------------------- | ------------------------------------ | --------------------------- |
| `predict_single(text, lang)`       | Overall sentiment + ABSA aspects     | ✅ `PredictionResult`       |
| `predict_batch(texts, lang)`       | Batch: overall + aspects per text    | ✅ `list[PredictionResult]` |
| `get_shap_explanation(text, lang)` | Không thay đổi                       | ✅ `SHAPResult`             |
| `supported_languages`              | Không thay đổi                       | ✅                          |
| `is_loaded`                        | Không thay đổi                       | ✅                          |
| `aspects` field                    | **[MỚI] list[AspectSentiment] thật** | ✅ Khớp contract            |
| `sarcasm_flag` field               | Không thay đổi (`False`)             | ✅                          |

---

## 7. Scope Exclusions

| Feature                             | Lý do                              |
| ----------------------------------- | ---------------------------------- |
| Fine-tuned ABSA                     | Sub-project B.3, sau Phase 1       |
| Aspect term extraction (exact span) | Chỉ cần category-level cho Phase 1 |
| Sarcasm detection                   | Sub-project B.4                    |
| SHAP cho ABSA                       | Phức tạp, sẽ thêm sau              |
| Multi-language ABSA                 | Chỉ hỗ trợ English                 |
| Batch optimization cho ABSA         | Có thể loop individual, tối ưu sau |

---

## 8. Verification Plan

### Automated Tests

```bash
# Chạy toàn bộ test suite (mocked, không cần download model)
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/ -v --tb=short

# Chạy riêng ABSA tests
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py -v -k "absa"

# Chạy config tests
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_config.py -v
```

### Integration Test (manual, cần download model)

```bash
# Smoke test: load model thật, verify aspects trả về
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -c "
from src.model.baseline import BaselineModelInference
model = BaselineModelInference()
result = model.predict_single('The food was amazing but service was slow')
print(f'Sentiment: {result.sentiment}')
print(f'Aspects: {result.aspects}')
assert len(result.aspects) > 0, 'ABSA should return at least 1 aspect'
for a in result.aspects:
    print(f'  {a.aspect}: {a.sentiment} ({a.confidence})')
    assert a.aspect in ('food','service','ambiance','price','location','general')
    assert a.sentiment in ('positive','negative','neutral')
print('PASS')
"
```

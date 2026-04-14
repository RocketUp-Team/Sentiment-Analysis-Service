# Handoff Package Design — Sentiment Analysis Service

> **Mục tiêu:** Trung (AI Core) chuẩn bị bộ tài liệu interface để Quân (Backend) và Long (Frontend) bắt tay code song song ngay, không cần chờ model thật.

## Bối cảnh

- **Dự án:** Sentiment Analysis Service (Topic 8, DDM501)
- **Team:** Trung (AI), Quân (Backend/DevOps), Long (Frontend/Docs)
- **Chiến lược:** MVP First — dựng 1 luồng chạy xuyên suốt UI → API → Model → UI trước, tối ưu model sau.

## Quyết định đã chốt

| Quyết định | Giá trị |
|------------|---------|
| Ngôn ngữ MVP | Tiếng Anh (giữ field `lang` để mở rộng) |
| Domain ABSA | Restaurant (Yelp) — mở rộng E-commerce (Amazon) sau |
| Batch processing | Async job-based (submit → `job_id` → poll → download) |
| Sarcasm detection | Có, trả `sarcasm_flag` trong response |
| Handoff approach | Shared Schemas + Mock Model + Sample Data |

---

## 1. API Contract

### Endpoints

| # | Method | Path | Mô tả |
|---|--------|------|--------|
| 1 | `POST` | `/api/v1/predict` | Real-time prediction 1 text |
| 2 | `POST` | `/api/v1/batch` | Upload CSV, tạo batch job |
| 3 | `GET` | `/api/v1/batch/{job_id}` | Check status batch job |
| 4 | `GET` | `/api/v1/batch/{job_id}/result` | Download kết quả CSV |
| 5 | `GET` | `/api/v1/health` | Health check + model status |

### 1.1. Real-time Predict

**`POST /api/v1/predict`**

Request:
```json
{
  "text": "The food was great but service was slow",
  "lang": "en"
}
```

Response:
```json
{
  "text": "The food was great but service was slow",
  "sentiment": "positive",
  "confidence": 0.72,
  "aspects": [
    {"aspect": "food", "sentiment": "positive", "confidence": 0.95},
    {"aspect": "service", "sentiment": "negative", "confidence": 0.88}
  ],
  "sarcasm_flag": false,
  "latency_ms": 45.2
}
```

- `sentiment`: `"positive"` | `"negative"` | `"neutral"`
- `confidence`: float 0.0–1.0, overall sentiment confidence
- `aspects`: list of aspect-level sentiments (có thể rỗng nếu không detect được)
- `sarcasm_flag`: bool
- `latency_ms`: float, thời gian inference thực tế (không tính network)

### 1.2. Batch Submit

**`POST /api/v1/batch`** (multipart/form-data)

Request: Upload file CSV với columns `[text, lang]`. `lang` optional, default `"en"`.

Response:
```json
{
  "job_id": "batch_abc123",
  "status": "pending",
  "total_items": 150,
  "created_at": "2026-04-14T10:30:00Z"
}
```

### 1.3. Batch Status

**`GET /api/v1/batch/{job_id}`**

Response:
```json
{
  "job_id": "batch_abc123",
  "status": "processing",
  "progress": 0.65,
  "total_items": 150,
  "processed_items": 97,
  "created_at": "2026-04-14T10:30:00Z",
  "completed_at": null
}
```

- `status`: `"pending"` | `"processing"` | `"completed"` | `"failed"`
- `progress`: float 0.0–1.0

### 1.4. Batch Result Download

**`GET /api/v1/batch/{job_id}/result`**

Response: File CSV download. Columns:
```
text,lang,sentiment,confidence,aspects_json,sarcasm_flag
"The food was great but...",en,positive,0.72,"[{""aspect"":""food"",""sentiment"":""positive"",""confidence"":0.95}]",false
```

Trả HTTP 404 nếu job chưa `completed`. Trả HTTP 410 nếu kết quả đã hết hạn.

### 1.5. Health Check

**`GET /api/v1/health`**

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0",
  "supported_languages": ["en"]
}
```

### ABSA Aspects (MVP — Restaurant domain)

Danh sách aspects cố định:
```
["food", "service", "ambiance", "price", "location", "general"]
```

`"general"` dùng khi không xác định được aspect cụ thể.

---

## 2. ModelInference Interface

### 2.1. Abstract Interface

```python
class ModelInference:
    def __init__(self, model_path: str | None = None):
        """Load model. Nếu model_path=None → dùng mock."""

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        """Predict 1 text. Raise ModelError nếu lỗi."""

    def predict_batch(self, texts: list[str], lang: str = "en") -> list[PredictionResult]:
        """Predict nhiều texts. Dùng cho batch job."""

    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        """Trả về SHAP values cho explainability. Long dùng hiển thị UI."""

    @property
    def supported_languages(self) -> list[str]:
        """Danh sách ngôn ngữ hỗ trợ. MVP: ['en']"""

    @property
    def is_loaded(self) -> bool:
        """Model đã load xong chưa? Quân dùng cho /health."""
```

### 2.2. Data Classes

```python
@dataclass
class AspectSentiment:
    aspect: str          # "food", "service", "ambiance", "price", "location", "general"
    sentiment: str       # "positive", "negative", "neutral"
    confidence: float    # 0.0 - 1.0

@dataclass
class PredictionResult:
    sentiment: str       # "positive", "negative", "neutral"
    confidence: float    # 0.0 - 1.0
    aspects: list[AspectSentiment]
    sarcasm_flag: bool

@dataclass
class SHAPResult:
    tokens: list[str]         # ["The", "food", "was", "great"]
    shap_values: list[float]  # [0.01, 0.45, 0.02, 0.52]
    base_value: float         # baseline prediction value
```

### 2.3. Error Classes

```python
class ModelError(Exception):
    """Lỗi chung từ model (load fail, predict fail)."""

class UnsupportedLanguageError(ModelError):
    """Ngôn ngữ chưa được hỗ trợ."""
```

**Mapping lỗi → HTTP:**
- `ModelError` → HTTP 500 Internal Server Error
- `UnsupportedLanguageError` → HTTP 400 Bad Request

### 2.4. Mock Implementation

`MockModelInference(ModelInference)` trả random data hợp lệ:
- Random sentiment từ `["positive", "negative", "neutral"]`
- Random 1-3 aspects từ danh sách restaurant
- `sarcasm_flag` = False (90%), True (10%)
- `time.sleep(0.03-0.08)` giả lập latency

---

## 3. Sample Data & Deliverables

### 3.1. File Structure

```
contracts/
├── schemas.py              # Pydantic models dùng chung
├── model_interface.py      # ModelInference abstract class
├── mock_model.py           # Mock implementation
├── errors.py               # ModelError, UnsupportedLanguageError
├── sample_batch_input.csv  # 20-30 dòng sample cho batch test
├── sample_responses.json   # Example API responses cho mỗi endpoint
└── README.md               # Hướng dẫn sử dụng cho Quân & Long
```

### 3.2. Ai nhận gì?

| Deliverable | Quân (Backend) | Long (Frontend) |
|-------------|:-:|:-:|
| `schemas.py` | ✅ Import vào FastAPI response models | ✅ Biết response format để parse |
| `model_interface.py` + `mock_model.py` | ✅ Plug vào API endpoints | — |
| `errors.py` | ✅ Catch errors → HTTP status codes | — |
| `sample_batch_input.csv` | ✅ Test batch endpoint | ✅ Test upload UI |
| `sample_responses.json` | — | ✅ Build UI components |
| SHAP example images | — | ✅ Design SHAP visualization |

### 3.3. Hướng dẫn sử dụng

**Cho Quân:**
1. Import `MockModelInference` từ `contracts/mock_model.py`
2. Gắn vào FastAPI dependency injection
3. Khi Trung train xong → swap bằng `ModelInference(model_path="path/to/weights")`
4. KHÔNG sửa file trong `contracts/` — báo Trung nếu cần thay đổi

**Cho Long:**
1. Đọc `sample_responses.json` để biết UI cần hiển thị gì
2. Dùng `sample_batch_input.csv` để test batch upload flow
3. SHAP visualization: render bar chart từ `SHAPResult.tokens` + `SHAPResult.shap_values`
4. KHÔNG sửa file trong `contracts/` — báo Trung nếu cần thay đổi

### 3.4. Quy tắc thay đổi Contract

- Chỉ Trung mới được sửa files trong `contracts/`
- Mọi thay đổi phải qua PR, Quân & Long review
- Breaking changes phải thông báo trước ≥ 1 ngày

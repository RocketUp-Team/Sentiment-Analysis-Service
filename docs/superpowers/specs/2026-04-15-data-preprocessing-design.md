# Data & Preprocessing Pipeline Design — Sentiment Analysis Service

> **Mục tiêu:** Xây dựng data pipeline cho Phase 1 (MVP) của AI Core, dùng SemEval-2014 restaurant dataset với kiến trúc modular để mở rộng cho Phase 2 (Yelp augmentation).

## Bối cảnh

- **Dự án:** Sentiment Analysis Service (Topic 8, DDM501)
- **Phạm vi:** Sub-project A trong AI Core & Modeling (Trung phụ trách)
- **Phụ thuộc:** Handoff package (contracts/) đã/đang được xây dựng song song
- **Output sẽ được dùng bởi:** Baseline Model & Fine-tuning (sub-project B)

## Quyết định đã chốt

| Quyết định | Giá trị | Lý do |
|------------|---------|-------|
| Dataset Phase 1 | SemEval-2014 Task 4 (Restaurants) | Có sẵn aspect-level annotations, match ABSA requirement. (Yelp/Amazon chỉ có overall sentiment, cần annotate thêm nếu làm Phase 1) |
| Dataset Phase 2 | Yelp subset (augmentation) | Tăng data volume, cải thiện F1-score |
| Preprocessing level | Modular pluggable pipeline | Đủ cho SemEval sạch, dễ mở rộng cho social media data |
| Data versioning | DVC (Data Version Control) | Rubric "Excellent" yêu cầu versioning, hỗ trợ pipeline DAG |
| Conflict labels | Drop (bỏ rows có label "conflict") | Giữ 3-class clean, match API schema, conflict < 5% data |
| Overall sentiment | Derived từ majority vote aspects bởi `SentimentDeriver` transform | SemEval chỉ có aspect-level labels; tách logic thành transform để testable |
| Train/val split | Stratified split từ train set, `validation_ratio=0.1`, `split_seed=42` | Đảm bảo phân phối class đồng đều ở val set |
| Sarcasm labels | Không có trong SemEval-2014, sarcasm detection không thuộc scope preprocessing | Mock implementation trả `sarcasm_flag=False` trong Phase 1 |
| **Môi trường thực thi** | **Local** (máy cá nhân) | Data nhỏ (~50MB), cần DVC tracking theo Git, dễ debug/test; Model training (Sub-project B) mới cần Colab GPU |
| **Python interpreter** | `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python` | Conda env riêng cho project, tránh conflict dependencies |

---

## 1. Kiến trúc tổng thể

### File Structure

```
Sentiment-Analysis-Service/
├── src/
│   └── data/
│       ├── __init__.py
│       ├── downloader.py          # Download & parse SemEval XML → raw CSV (validate schema sau parse)
│       ├── pipeline.py            # PreprocessingPipeline class
│       ├── validators.py          # DataQualityValidator class (exit non-zero nếu fail)
│       └── transforms/
│           ├── __init__.py
│           ├── base.py            # BaseTransform (abstract)
│           ├── text_cleaner.py    # Lowercase, strip, whitespace normalization
│           ├── label_mapper.py    # SemEval labels → project schema, drop conflict
│           ├── sentiment_deriver.py  # Derive overall sentiment từ aspect list
│           ├── splitter.py        # Stratified train/val split
│           ├── duplicate_remover.py  # Remove duplicate sentences (cascade to aspects)
│           └── length_filter.py   # Filter by text length bounds
├── data/
│   ├── raw/                       # DVC-tracked: SemEval gốc (parsed CSV, validated schema)
│   │   └── .gitkeep
│   ├── processed/                 # DVC-tracked: data sau preprocessing
│   │   └── .gitkeep
│   ├── manifest.yaml              # Dataset registry (metadata, expected counts)
│   └── reports/                   # Data quality reports (JSON)
│       └── .gitkeep
├── tests/
│   └── data/
│       ├── __init__.py
│       ├── test_transforms.py
│       ├── test_pipeline.py
│       ├── test_validators.py
│       └── test_downloader.py
├── dvc.yaml                       # Pipeline definition (3 stages)
├── params.yaml                    # Config cho pipeline
├── .dvc/                          # DVC internal config
└── .dvcignore
```

### Nguyên tắc thiết kế

- `src/data/` chứa **code** — được test, được version bằng Git
- `data/` chứa **artifacts** — được track bằng DVC, không commit vào Git
- `params.yaml` là **single source of truth** cho tất cả config
- Phase 2 mở rộng bằng cách thêm transform plugins + DVC stages mới

### Môi trường thực thi

| Thành phần | Giá trị |
|------------|--------|
| **Môi trường** | Local (máy cá nhân) |
| **Python interpreter** | `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python` |
| **Conda env name** | `sentiment_analysis_service` |
| **Kích hoạt env** | `conda activate sentiment_analysis_service` |
| **Chạy DVC pipeline** | `dvc repro` (từ root dự án) |
| **Chạy tests** | `/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/data/` |

> **Lưu ý:** Mọi lệnh `python -m ...` trong `dvc.yaml` phải dùng đúng interpreter này. Khi setup DVC stages, sử dụng path tuyệt đối hoặc đảm bảo env đã được activate trước khi chạy `dvc repro`.

### Boundary: Preprocessing → Model Training (Sub-project B)

> Ranh giới này phải rõ ràng vì hai sub-project phụ thuộc vào nhau.

- **Preprocessing outputs:** Text đã clean + labels trong CSV format. **Không tokenize**, không encode.
- **Model training inputs:** Đọc raw text từ `data/processed/` — model code tự xử lý tokenization.
- **Contract:** Model training chỉ import từ `data/processed/`, không chạm vào `src/data/`.
- **Schema guarantee:** Column names/types frozen sau khi Phase 1 pipeline chạy lần đầu thành công.
- **Tokenization:** Hoàn toàn thuộc trách nhiệm Sub-project B (model pipeline), không phải preprocessing.
- **Sarcasm labels:** Không có trong SemEval training data. `sarcasm_flag` trong API được xử lý bởi model riêng (out of scope preprocessing Phase 1).

---

## 2. Data Flow — DVC Pipeline

3 stages chạy tuần tự qua `dvc repro`:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  download    │───>│  preprocess   │───>│   validate   │
│             │    │              │    │              │
│ SemEval Hub │    │ Pipeline of  │    │ Quality      │
│ -> data/raw/│    │ Transforms   │    │ checks +     │
│ + schema    │    │ -> data/     │    │ report       │
│   validate  │    │   processed/ │    │ -> data/     │
│             │    │              │    │   reports/   │
│             │    │              │    │ exit(1) fail │
└─────────────┘    └──────────────┘    └──────────────┘
```

### `dvc.yaml`

```yaml
stages:
  download:
    cmd: python -m src.data.downloader
    params:
      - params.yaml:
          - data.dataset_name
          - data.splits
    outs:
      - data/raw/

  preprocess:
    cmd: python -m src.data.pipeline
    deps:
      - data/raw/
      - src/data/transforms/
    params:
      - params.yaml:
          - preprocessing
          - sentiment_derivation
          - label_mapping
          - data.validation_ratio
          - data.split_seed
    outs:
      - data/processed/

  validate:
    cmd: python -m src.data.validators
    deps:
      - data/processed/
    params:
      - params.yaml:
          - validation
    metrics:
      - data/reports/quality_report.json:
          cache: false
```

> **Lưu ý:** `validate` stage exit với `sys.exit(1)` nếu `checks_passed == false`. DVC sẽ đánh dấu stage fail và dừng pipeline. CI/CD sẽ fail tương ứng.

### `params.yaml`

```yaml
data:
  dataset_name: "semeval2014_restaurants"
  splits: ["train", "test"]
  validation_ratio: 0.1
  split_seed: 42
  max_text_length: 2000

preprocessing:
  lowercase: true
  strip_whitespace: true
  remove_duplicates: true
  drop_conflict_labels: true
  min_text_length: 3

sentiment_derivation:
  mixed_strategy: "negative_priority"

label_mapping:
  aspect_categories:
    "anecdotes/miscellaneous": "general"

validation:
  min_samples: 100
  max_null_ratio: 0.01
  expected_labels:
    sentiment: ["positive", "negative", "neutral"]
    aspect: ["food", "service", "ambiance", "price", "location", "general"]

mlflow:
  tracking_uri: "http://localhost:5000"   # Override bằng env var MLFLOW_TRACKING_URI khi chạy server
  experiment_name: "data_preprocessing"
```

---

## 3. Transform System — Pluggable Pipeline

### Thứ tự thực thi (bắt buộc, không hoán đổi)

> ⚠️ Transform order matters — đổi thứ tự sẽ gây kết quả sai:

| Bước | Transform | Lý do phải đứng vị trí này |
|------|-----------|---------------------------|
| 1 | `LabelMapper` | Drop conflict labels trước — tránh đếm them vào dedup/length filter |
| 2 | `SentimentDeriver` | Cần aspect labels còn đủ (sau LabelMapper) để derive overall sentiment đúng |
| 3 | `TextCleaner` | Normalize text trước khi dedup — "The food   was" và "The food was" phải coi là duplicate |
| 4 | `DuplicateRemover` | Chạy sau TextCleaner để normalized text matching chính xác |
| 5 | `LengthFilter` | Lọc cuối cùng dựa trên độ dài text đã normalized |
| 6 | `Splitter` | Tạo train/val split sau khi data đã sạch hoàn toàn |

### BaseTransform (abstract)

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseTransform(ABC):
    """Moi buoc preprocessing = 1 Transform.
    Moi transform nhan cap (sentences_df, aspects_df), tra ve cap da xu ly.
    Transform chi modify DataFrame thuoc trach nhiem cua minh,
    tra lai DataFrame kia nguyen ven.
    """

    # Subclass declares columns required on sentences_df after transform
    required_sentence_columns: list[str] = []
    # Subclass declares columns required on aspects_df after transform
    required_aspect_columns: list[str] = []

    @abstractmethod
    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Nhan cap DataFrames, tra cap DataFrames da xu ly."""

    def validate_output(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> None:
        """Post-condition check tren ca hai DataFrames. Dung pipeline neu bat thuong."""
        if sentences_df.empty:
            raise ValueError(f"{self.name} produced empty sentences DataFrame")
        if self.required_sentence_columns:
            missing = set(self.required_sentence_columns) - set(sentences_df.columns)
            if missing:
                raise ValueError(
                    f"{self.name} dropped required sentence columns: {missing}"
                )
        if self.required_aspect_columns:
            missing = set(self.required_aspect_columns) - set(aspects_df.columns)
            if missing:
                raise ValueError(
                    f"{self.name} dropped required aspect columns: {missing}"
                )

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

### Pipeline Logging & Execution

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("data_pipeline")

class PreprocessingPipeline:
    """Pipeline chay tren cap (sentences_df, aspects_df).
    Moi transform nhan ca hai DataFrames, tra ve ca hai.
    Transforms chi thay doi DataFrame thuoc trach nhiem cua minh,
    pipeline tu dong sync aspects khi sentences bi xoa.
    """
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def run(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        for t in self.transforms:
            before_s = len(sentences_df)
            sentences_df, aspects_df = t.transform(sentences_df, aspects_df)

            # Auto-cascade: remove orphan aspects after sentence-level transforms
            aspects_df = aspects_df[
                aspects_df["sentence_id"].isin(sentences_df["sentence_id"])
            ]

            # Validate both DataFrames — each transform declares its own requirements
            t.validate_output(sentences_df, aspects_df)
            after_s = len(sentences_df)
            logger.info(
                f"{t.name}: {before_s} -> {after_s} sentences, "
                f"{len(aspects_df)} aspects"
            )
        return sentences_df, aspects_df
```

### Pipeline Error Handling

| Scenario | Handling |
|----------|----------|
| **Download fails** | Retry với exponential backoff (3 attempts), sau đó raise `DownloadError` |
| **XML parsing error** | Log warning, skip malformed sentence, tiếp tục |
| **Raw CSV schema invalid** | `validate_raw_schema()` raise `SchemaError` ngay sau parse — dừng trước preprocess |
| **Empty output** | `validate_output` raises `ValueError` trước khi ghi file |
| **Validate stage fails** | `sys.exit(1)` — DVC đánh dấu stage fail, CI/CD pipeline dừng |
| **DVC remote không truy cập được** | DVC báo lỗi rõ ràng; data cached local vẫn dùng được |
| **Out of disk space** | `downloader.py` check disk space ước tính trước khi download (~50MB cho SemEval) |

### Phase 1 Transforms (MVP — SemEval)

| Transform | Chức năng | Modifies | `required_sentence_columns` | `required_aspect_columns` |
|-----------|-----------|----------|-----------------------------|---------------------------|
| `LabelMapper` | Map SemEval labels → schema chuẩn (như "anecdotes/miscellaneous" → "general"). Drop rows có label `conflict` từ aspects | `aspects_df` | `[]` | `["aspect_category", "sentiment"]` |
| `SentimentDeriver` | Group aspects by sentence_id, derive overall sentiment bằng negative_priority logic (xem Section 4), ghi vào `sentences_df["sentiment"]` | `sentences_df` (đọc `aspects_df`) | `["sentence_id", "sentiment"]` | `[]` |
| `TextCleaner` | Lowercase, strip whitespace, chuẩn hoá khoảng trắng thừa, loại bỏ rows rỗng | `sentences_df` | `["text"]` | `[]` |
| `DuplicateRemover` | Loại bỏ duplicate texts (cùng nội dung text, khác sentence_id), dùng `keep="first"`. Pipeline auto-cascade xoá aspects. | `sentences_df` | `["sentence_id", "text"]` | `[]` |
| `LengthFilter` | Bỏ text quá ngắn (< `min_text_length`) hoặc quá dài (> `max_text_length`). Pipeline auto-cascade xoá aspects. | `sentences_df` | `["text"]` | `[]` |
| `Splitter` | Stratified train/val split từ train set, dùng `split_seed` và `validation_ratio` | `sentences_df` | `["split", "sentiment"]` | `[]` |

### DuplicateRemover — Phạm vi và Cascade

Deduplication xảy ra ở **sentence level** dựa trên nội dung text (không phải aspect level), vì một câu có thể có nhiều aspects:

```python
class DuplicateRemover(BaseTransform):
    required_sentence_columns = ["sentence_id", "text"]

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Drop sentences có duplicate text content, giữ first occurrence
        deduped = sentences_df.drop_duplicates(subset=["text"], keep="first")
        # Pipeline.run() auto-cascade: xoá orphan aspects sau mỗi transform
        return deduped, aspects_df
```

> **Cascade:** `PreprocessingPipeline.run()` tự động filter `aspects_df` sau mỗi transform để loại bỏ aspects có `sentence_id` không còn tồn tại trong `sentences_df`. Không cần xử lý cascade trong từng transform.

### Splitter — Stratified Train/Val Split

```python
from sklearn.model_selection import train_test_split

class Splitter(BaseTransform):
    """Tao stratified val split tu train set.
    SemEval chi co train va test — val duoc tao tu train.
    """
    required_sentence_columns = ["split", "sentiment"]

    def __init__(self, validation_ratio: float, seed: int):
        self.validation_ratio = validation_ratio
        self.seed = seed

    def transform(
        self, sentences_df: pd.DataFrame, aspects_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_mask = sentences_df["split"] == "train"
        train_df = sentences_df[train_mask].copy()
        test_df = sentences_df[~train_mask].copy()

        train_final, val_df = train_test_split(
            train_df,
            test_size=self.validation_ratio,
            random_state=self.seed,
            stratify=train_df["sentiment"],  # Stratified — dam bao phan phoi class
        )
        train_final = train_final.copy()
        val_df = val_df.copy()
        val_df["split"] = "val"

        result = pd.concat([train_final, val_df, test_df], ignore_index=True)
        return result, aspects_df  # Splitter khong xoa sentences, aspects khong doi
```

### Phase 2 Transforms (thêm sau, không build trong scope này)

| Transform | Khi nào cần |
|-----------|-------------|
| `EmojiHandler` | Khi thêm Twitter/social media data |
| `SlangNormalizer` | Khi thêm social media data |
| `HashtagSplitter` | Khi thêm Twitter data |
| `UrlMentionRemover` | Khi thêm social media data |

---

## 4. SemEval Data Format & Output Schema

### SemEval Input (raw XML)

> *Lưu ý: Chúng ta dùng Subtask 2 của SemEval-2014 (Category Classification), vì nó cung cấp danh mục (categories) chuẩn cho dự án. Thuộc tính `aspectTerm` của Subtask 1 không phù hợp nếu muốn phân loại cố định.*

```xml
<sentence id="1">
  <text>The food was great but the service was slow.</text>
  <aspectCategories>
    <aspectCategory category="food" polarity="positive"/>
    <aspectCategory category="service" polarity="negative"/>
  </aspectCategories>
</sentence>
```

`downloader.py` parse XML → 2 CSV files chuẩn, sau đó chạy `validate_raw_schema()` ngay lập tức.

### Schema Validation Sau Download

```python
EXPECTED_RAW_SENTENCE_COLUMNS = {"sentence_id", "text", "split"}
EXPECTED_RAW_ASPECT_COLUMNS = {"sentence_id", "aspect_category", "sentiment"}

def validate_raw_schema(sentences_df: pd.DataFrame, aspects_df: pd.DataFrame) -> None:
    """Chay ngay sau parse XML, truoc khi ghi data/raw/. Raise SchemaError neu thieu column."""
    missing_s = EXPECTED_RAW_SENTENCE_COLUMNS - set(sentences_df.columns)
    missing_a = EXPECTED_RAW_ASPECT_COLUMNS - set(aspects_df.columns)
    if missing_s or missing_a:
        raise SchemaError(
            f"Raw data missing columns — sentences: {missing_s}, aspects: {missing_a}"
        )
    if sentences_df.empty or aspects_df.empty:
        raise SchemaError("Raw data is empty after parsing")
```

### Expected Data Volume (SemEval-2014 Restaurants)

| Split | Raw Samples | Sau Preprocessing (ước tính) | Ghi chú |
|-------|-------------|------------------------------|--------|
| Train (trước split) | ~3,041 sentences | ~2,900 sentences | Sau drop conflict, dedup, length filter |
| → Train (sau split) | — | ~2,610 sentences | 90% của ~2,900 |
| → Val (sau split) | — | ~290 sentences | 10% của ~2,900 (stratified) |
| Test | ~800 sentences | ~780 sentences | Không chia thêm |
| **Total** | **~3,841** | **~3,680** | |

> **Dùng làm sanity check:** Nếu processed data ít hơn 20% so với raw → pipeline có thể lỗi. `min_samples: 100` chỉ là hard floor, không phải target.

### Output Schema — 2 files CSV

**`sentences.csv`** — Overall sentiment per sentence

| column | type | description |
|--------|------|-------------|
| `sentence_id` | str | ID duy nhất |
| `text` | str | Text đã preprocessed |
| `sentiment` | str | `positive` / `negative` / `neutral` — derived bởi `SentimentDeriver` |
| `split` | str | `train` / `val` / `test` (val được tạo từ train bởi `Splitter`) |

**`aspects.csv`** — ABSA level (1 row = 1 aspect mention)

| column | type | description |
|--------|------|-------------|
| `sentence_id` | str | FK tới `sentences.csv` |
| `aspect_category` | str | Danh sách chuẩn: `food` / `service` / `ambiance` / `price` / `location` / `general` |
| `sentiment` | str | `positive` / `negative` / `neutral` |

> *Note: **Bỏ qua character offsets** (`char_start`, `char_end`) vì `TextCleaner` sẽ thay đổi độ dài chuỗi làm sai offsets. `confidence` không có trong training data vì là ground truth — field này chỉ xuất hiện trong API response từ model inference.*

### Contract Alignment với Handoff Package

| Field | Handoff API Contract | Training CSV | Ghi chú |
|-------|---------------------|--------------|---------|
| `confidence` | Có trong `PredictionResult` | Không có | Đúng — ground truth không có confidence. Model tự tính khi inference. |
| `sarcasm_flag` | Có trong `PredictionResult` | Không có | SemEval-2014 không có sarcasm labels. Phase 1 mock trả `false`. |
| `lang` | Required trong API request | Không có | MVP xử lý tiếng Anh. Preprocessing language-agnostic. |
| Aspect list | `["food","service","ambiance","price","location","general"]` | Cùng danh sách | ✅ Đồng nhất. |
| Sentiment labels | `["positive","negative","neutral"]` | Cùng danh sách | ✅ Đồng nhất. |

### Logic suy ra overall sentiment (SentimentDeriver)

Thuộc `src/data/transforms/sentiment_deriver.py`. Chạy sau `LabelMapper`, trước `TextCleaner`.

```python
def derive_overall_sentiment(aspects: list[str]) -> str:
    sentiments = set(aspects)
    if not sentiments:
        return "neutral"
    if len(sentiments) == 1:
        return sentiments.pop()
    # Mixed cases — prioritize negative (recall >= 90% target)
    if "negative" in sentiments:
        return "negative"
    # positive + neutral -> positive
    return "positive"
```

---

## 5. Data Quality Validation

### Validation Checks

| Check | Mô tả | Loại | Fail condition |
|-------|--------|------|----------------|
| **Null check** | Không có null/empty trong `text`, `sentiment` | ❌ Hard fail | null ratio > `max_null_ratio` (0.01) |
| **Label check** | Tất cả labels thuộc `["positive", "negative", "neutral"]` | ❌ Hard fail | Có label ngoài danh sách |
| **Aspect check** | Tất cả aspects thuộc danh sách restaurant | ❌ Hard fail | Có aspect ngoài danh sách |
| **Min samples** | Mỗi split phải có >= `min_samples` (100) rows | ❌ Hard fail | Quá ít data |
| **Referential integrity** | Mọi `sentence_id` trong `aspects.csv` tồn tại trong `sentences.csv` | ❌ Hard fail | Orphan aspect records |
| **Class distribution** | Tính % mỗi class, cảnh báo nếu imbalance | ⚠️ Warning | Bất kỳ class nào < 10% |
| **Text length stats** | Min/max/mean/median length | 📊 Report only | Không fail |

> **Hard fail** → ghi vào `report["errors"][]`, set `checks_passed = false`, pipeline exit non-zero.
> **Warning** → ghi vào `report["warnings"][]`, `checks_passed` vẫn `true`.

### Validate Stage Exit Behavior

```python
# src/data/validators.py
import sys

if __name__ == "__main__":
    params = load_params("params.yaml")
    validator = DataQualityValidator(params["validation"])
    report = validator.validate("data/processed/")
    save_report(report, "data/reports/quality_report.json")

    if not report["checks_passed"]:
        logger.error(f"Data quality checks FAILED: {report['errors']}")
        sys.exit(1)  # DVC marks stage as failed, stops pipeline

    if report["warnings"]:
        logger.warning(f"Warnings: {report['warnings']}")
    logger.info("Data quality checks passed.")
```

### Output: `quality_report.json`

Report tổng hợp tất cả splits, với breakdown per-split:

```json
{
  "total_samples": 3680,
  "splits": {
    "train": {
      "samples": 2610,
      "null_ratio": 0.0,
      "label_distribution": {
        "positive": 0.52,
        "negative": 0.31,
        "neutral": 0.17
      }
    },
    "val": {
      "samples": 290,
      "null_ratio": 0.0,
      "label_distribution": {
        "positive": 0.52,
        "negative": 0.31,
        "neutral": 0.17
      }
    },
    "test": {
      "samples": 780,
      "null_ratio": 0.0,
      "label_distribution": {
        "positive": 0.54,
        "negative": 0.29,
        "neutral": 0.17
      }
    }
  },
  "aspect_distribution": {
    "food": 0.35,
    "service": 0.28,
    "ambiance": 0.12,
    "price": 0.15,
    "location": 0.03,
    "general": 0.07
  },
  "text_length_stats": {
    "min": 5,
    "max": 487,
    "mean": 78.3,
    "median": 62
  },
  "checks_passed": true,
  "errors": [],
  "warnings": ["Class 'neutral' is underrepresented (17%) in split 'train'"]
}
```

### MLflow Integration

Trong bước `validate`, pipeline tự động log report lên MLflow để track version dữ liệu và chất lượng:

```python
import mlflow
import subprocess
import os

def log_quality_report_to_mlflow(report: dict) -> None:
    # Lấy git commit hash làm data version identifier
    dvc_version = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode().strip()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("data_preprocessing")

    with mlflow.start_run(run_name=f"data_v{dvc_version}"):
        mlflow.log_artifact("data/reports/quality_report.json")
        mlflow.log_metric("total_samples", report["total_samples"])
        mlflow.log_metric("passed_quality_checks", 1 if report["checks_passed"] else 0)
        # Log per-split metrics
        for split_name, split_data in report["splits"].items():
            mlflow.log_metric(f"{split_name}_samples", split_data["samples"])
            mlflow.log_metric(f"{split_name}_null_ratio", split_data["null_ratio"])
        mlflow.log_params({
            "dataset": "semeval2014",
            "git_version": dvc_version,
            "validation_ratio": 0.1,
            "drop_conflict": True,
        })
```

> **DVC vs MLflow roles:**
> - **DVC** track file artifacts (data CSVs, report JSON) theo content hash → `dvc metrics show/diff` so sánh chất lượng data giữa các *Git commits*.
> - **MLflow** track từng *pipeline run* theo thời gian → lịch sử chạy, param nào tạo ra data nào.

---

## 6. Dataset Manifest

`data/manifest.yaml` — Registry cho tất cả datasets, phục vụ Phase 2 extensibility và sanity checking:

```yaml
# data/manifest.yaml — DVC DOES NOT track this file (it's metadata, not data)
datasets:
  semeval2014_restaurants:
    display_name: "SemEval-2014 Task 4 (Restaurants)"
    version: "1.0"
    source_url: "https://huggingface.co/datasets/..."
    expected_samples:
      raw_train: 3041
      raw_test: 800
      processed_min: 3500   # Hard floor — actual expected ~3,680
    transforms:
      - LabelMapper
      - SentimentDeriver
      - TextCleaner
      - DuplicateRemover
      - LengthFilter
      - Splitter
    aspects: ["food", "service", "ambiance", "price", "location", "general"]
    sentiments: ["positive", "negative", "neutral"]

  yelp_restaurants:    # Phase 2 — chưa implement
    display_name: "Yelp Academic Dataset (Restaurants subset)"
    version: "TBD"
    source_url: "TBD"
    expected_samples: {}
    transforms: []
    notes: "Cần annotation ABSA trước khi dùng — chỉ có overall sentiment"
```

---

## 7. Testing Strategy

### Unit Tests (`tests/data/`)

| Test file | Nội dung |
|-----------|----------|
| `test_transforms.py` | Mỗi transform test riêng: TextCleaner xử lý whitespace, LabelMapper drop conflict, SentimentDeriver với mixed aspects, DuplicateRemover cascade, Splitter stratification, LengthFilter đúng threshold |
| `test_pipeline.py` | Pipeline chain nhiều transforms với đúng thứ tự, test pipeline rỗng, test cascade removes |
| `test_validators.py` | Validator detect null, label sai, orphan aspects, class imbalance. Test cả pass lẫn fail cases. Test exit code behavior. |
| `test_downloader.py` | Test parse XML → DataFrame đúng schema. Test `validate_raw_schema` với valid/invalid data. Dùng fixture XML nhỏ, **không** gọi network. |

### Data Quality Tests (chạy trong CI)

```python
def test_processed_data_quality():
    """Chay validator tren processed data, assert checks_passed == True."""
    report = DataQualityValidator(params).validate("data/processed/")
    assert report["checks_passed"] is True
    assert len(report["errors"]) == 0
    for split_name, split_data in report["splits"].items():
        assert split_data["null_ratio"] == 0.0, f"Null found in {split_name}"

def test_stratified_split_distribution():
    """Val set phai co du 3 classes (stratification working)."""
    val_df = pd.read_csv("data/processed/sentences.csv").query("split == 'val'")
    for label in ["positive", "negative", "neutral"]:
        assert (val_df["sentiment"] == label).sum() > 0, f"Missing class '{label}' in val set"

def test_referential_integrity():
    """Moi sentence_id trong aspects.csv phai co trong sentences.csv."""
    sentences = pd.read_csv("data/processed/sentences.csv")
    aspects = pd.read_csv("data/processed/aspects.csv")
    orphans = set(aspects["sentence_id"]) - set(sentences["sentence_id"])
    assert len(orphans) == 0, f"Orphan aspect records: {orphans}"
```

### Coverage Target

- **Target:** > 80% line coverage cho source `src/data/` (Theo rubric).
- **Tool:** `pytest --cov=src/data --cov-report=term-missing`
- **CI Integration:** Báo cáo coverage được upload làm GitHub Actions artifact, status ghi trong PR.

### CI/CD Notes

- CI chạy **unit tests** trên sample fixtures (nhanh, không cần download data)
- Lệnh `dvc repro` chạy **local** hoặc scheduled server bởi developer (cần tải raw data).
- Cấu hình remote storage (`.dvc/config`): `dvc remote add -d storage gdrive://<folder_id>`. Team dùng chung Google Drive với service account credentials (không dùng personal OAuth để tránh CI auth issue).
- Data quality tests có thể chạy trong CI sau lệnh `dvc pull` lấy dữ liệu đã cache.

---

## 8. Phase 2 Extension Plan (tham khảo, ngoài scope hiện tại)

Khi cần augment với Yelp data:

1. Thêm `src/data/transforms/emoji_handler.py`, `slang_normalizer.py`... vào `transforms/`
2. Cập nhật `data/manifest.yaml` với thông tin Yelp dataset
3. Thêm `download_yelp` stage mới vào `dvc.yaml`
4. Thêm `merge` stage để kết hợp SemEval + Yelp processed data
5. DVC tự quản dependency graph, chỉ chạy lại stages bị ảnh hưởng
6. `dvc metrics diff` so sánh quality report trước/sau augmentation

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
| Dataset Phase 1 | SemEval-2014 Task 4 (Restaurants) | Có sẵn aspect-level annotations, match ABSA requirement |
| Dataset Phase 2 | Yelp subset (augmentation) | Tăng data volume, cải thiện F1-score |
| Preprocessing level | Modular pluggable pipeline | Đủ cho SemEval sạch, dễ mở rộng cho social media data |
| Data versioning | DVC (Data Version Control) | Rubric "Excellent" yêu cầu versioning, hỗ trợ pipeline DAG |
| Conflict labels | Drop (bỏ rows có label "conflict") | Giữ 3-class clean, match API schema, conflict < 5% data |
| Overall sentiment | Derived từ majority vote aspects | SemEval chỉ có aspect-level labels |

---

## 1. Kiến trúc tổng thể

### File Structure

```
Sentiment-Analysis-Service/
├── src/
│   └── data/
│       ├── __init__.py
│       ├── downloader.py          # Download & parse SemEval datasets
│       ├── pipeline.py            # PreprocessingPipeline class
│       ├── validators.py          # DataQualityValidator class
│       └── transforms/
│           ├── __init__.py
│           ├── base.py            # BaseTransform (abstract)
│           ├── text_cleaner.py    # Lowercase, strip, whitespace normalization
│           ├── label_mapper.py    # SemEval labels → project schema
│           ├── duplicate_remover.py  # Remove duplicate texts
│           └── length_filter.py   # Filter by text length bounds
├── data/
│   ├── raw/                       # DVC-tracked: SemEval gốc (parsed CSV)
│   │   └── .gitkeep
│   ├── processed/                 # DVC-tracked: data sau preprocessing
│   │   └── .gitkeep
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

---

## 2. Data Flow — DVC Pipeline

3 stages chạy tuần tự qua `dvc repro`:

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  download    │───▶│  preprocess   │───▶│   validate   │
│             │    │              │    │              │
│ SemEval Hub │    │ Pipeline of  │    │ Quality      │
│ → data/raw/ │    │ Transforms   │    │ checks +     │
│             │    │ → data/      │    │ report       │
│             │    │   processed/ │    │ → data/      │
│             │    │              │    │   reports/   │
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

### `params.yaml`

```yaml
data:
  dataset_name: "semeval2014_restaurants"
  splits: ["train", "test"]
  max_text_length: 2000

preprocessing:
  lowercase: true
  strip_whitespace: true
  remove_duplicates: true
  min_text_length: 3

validation:
  min_samples: 100
  max_null_ratio: 0.01
  expected_labels:
    sentiment: ["positive", "negative", "neutral"]
    aspect: ["food", "service", "ambiance", "price", "location", "general"]
```

---

## 3. Transform System — Pluggable Pipeline

### BaseTransform (abstract)

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseTransform(ABC):
    """Mỗi bước preprocessing = 1 Transform."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nhận DataFrame, trả DataFrame đã xử lý."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

### PreprocessingPipeline

```python
class PreprocessingPipeline:
    def __init__(self, transforms: list[BaseTransform]):
        self.transforms = transforms

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transforms:
            before = len(df)
            df = t.transform(df)
            after = len(df)
            logger.info(f"{t.name}: {before} → {after} rows")
        return df
```

### Phase 1 Transforms (MVP — SemEval)

| Transform | Chức năng |
|-----------|-----------|
| `TextCleaner` | Lowercase, strip whitespace, chuẩn hóa khoảng trắng thừa, loại bỏ rows rỗng |
| `LabelMapper` | Map SemEval labels → schema chuẩn. Drop rows có label `conflict` |
| `DuplicateRemover` | Loại bỏ duplicate texts |
| `LengthFilter` | Bỏ text quá ngắn (< `min_text_length`) hoặc quá dài (> `max_text_length`) |

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

```xml
<sentence id="1">
  <text>The food was great but the service was slow.</text>
  <aspectTerms>
    <aspectTerm term="food" polarity="positive" from="4" to="8"/>
    <aspectTerm term="service" polarity="negative" from="26" to="33"/>
  </aspectTerms>
</sentence>
```

`downloader.py` parse XML → 2 CSV files chuẩn.

### Output Schema — 2 files CSV

**`sentences.csv`** — Overall sentiment per sentence

| column | type | description |
|--------|------|-------------|
| `sentence_id` | str | ID duy nhất |
| `text` | str | Text đã preprocessed |
| `sentiment` | str | `positive` / `negative` / `neutral` — derived từ aspects |
| `split` | str | `train` / `test` |

**`aspects.csv`** — ABSA level (1 row = 1 aspect mention)

| column | type | description |
|--------|------|-------------|
| `sentence_id` | str | FK tới `sentences.csv` |
| `aspect_term` | str | Từ gốc trong text ("food", "service"...) |
| `aspect_category` | str | Danh sách chuẩn: `food` / `service` / `ambiance` / `price` / `location` / `general` |
| `sentiment` | str | `positive` / `negative` / `neutral` |
| `char_start` | int | Vị trí bắt đầu trong text |
| `char_end` | int | Vị trí kết thúc trong text |

### Logic suy ra overall sentiment

Vì SemEval chỉ có aspect-level labels, overall sentiment được derived:

- **Tất cả aspects cùng sentiment** → dùng luôn sentiment đó
- **Mixed (có cả positive + negative)** → `negative` (ưu tiên recall negative ≥ 90% theo target)
- **Không có aspect nào** → `neutral`

---

## 5. Data Quality Validation

### Validation Checks

| Check | Mô tả | Fail condition |
|-------|--------|----------------|
| **Null check** | Không có null/empty trong `text`, `sentiment` | null ratio > `max_null_ratio` (0.01) |
| **Label check** | Tất cả labels thuộc `["positive", "negative", "neutral"]` | Có label ngoài danh sách |
| **Aspect check** | Tất cả aspects thuộc danh sách restaurant | Có aspect ngoài danh sách |
| **Min samples** | Mỗi split phải có ≥ `min_samples` (100) rows | Quá ít data |
| **Class distribution** | Tính % mỗi class, cảnh báo nếu imbalance | Bất kỳ class nào < 10% |
| **Text length stats** | Min/max/mean/median length | Report only (không fail) |

### Output: `quality_report.json`

```json
{
  "split": "train",
  "total_samples": 3041,
  "null_ratio": 0.0,
  "label_distribution": {
    "positive": 0.52,
    "negative": 0.31,
    "neutral": 0.17
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
  "warnings": ["Class 'neutral' is underrepresented (17%)"]
}
```

DVC track file này như metrics — `dvc metrics show` và `dvc metrics diff` sẽ hiển thị thay đổi giữa các version data.

---

## 6. Testing Strategy

### Unit Tests (`tests/data/`)

| Test file | Nội dung |
|-----------|----------|
| `test_transforms.py` | Mỗi transform test riêng: TextCleaner xử lý whitespace, LabelMapper bỏ conflict, DuplicateRemover, LengthFilter đúng threshold |
| `test_pipeline.py` | Pipeline chain nhiều transforms, test thứ tự thực thi, test pipeline rỗng |
| `test_validators.py` | Validator detect null, label sai, class imbalance. Test cả pass lẫn fail cases |
| `test_downloader.py` | Test parse XML → DataFrame đúng schema. Dùng fixture XML nhỏ, **không** gọi network |

### Data Quality Tests (chạy trong CI)

```python
def test_processed_data_quality():
    """Chạy validator trên processed data, assert checks_passed == True."""
    report = DataQualityValidator(params).validate("data/processed/")
    assert report["checks_passed"] is True
    assert report["null_ratio"] == 0.0
```

### CI/CD Notes

- CI chạy **unit tests** trên sample fixtures (nhanh, không cần download data)
- `dvc repro` chạy **local** bởi developer (cần download data thật)
- Data quality tests có thể chạy trong CI nếu có cached processed data

---

## 7. Phase 2 Extension Plan (tham khảo, ngoài scope hiện tại)

Khi cần augment với Yelp data:

1. Thêm `src/data/transforms/emoji_handler.py`, `slang_normalizer.py`... vào `transforms/`
2. Thêm `download_yelp` stage mới vào `dvc.yaml`
3. Thêm `merge` stage để kết hợp SemEval + Yelp processed data
4. DVC tự quản dependency graph, chỉ chạy lại stages bị ảnh hưởng
5. `dvc metrics diff` so sánh quality report trước/sau augmentation

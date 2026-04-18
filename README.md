# Sentiment Analysis Service

[![CI Pipeline](https://github.com/RocketUp-Team/Sentiment-Analysis-Service/actions/workflows/ci.yml/badge.svg)](https://github.com/RocketUp-Team/Sentiment-Analysis-Service/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/mlflow-2.x-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DDM501 – AI in Production: From Models to Systems**
> Final Project · FSB Institute of Management and Technology, FPT University
> Instructor: Huynh Cong Viet Ngu

A **production-grade NLP microservice** that performs real-time sentiment classification, Aspect-Based Sentiment Analysis (ABSA), and model explainability using a pre-trained RoBERTa transformer. Fully containerised, monitored, and tracked.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [ML Pipeline](#ml-pipeline)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Team](#team)

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Sentiment Classification** | Real-time positive / negative / neutral prediction via RoBERTa |
| 🧩 **Aspect-Based Sentiment (ABSA)** | Zero-shot aspect extraction across 6 categories (food, service, ambiance, price, location, general) |
| 💡 **Model Explainability** | Token-level SHAP attributions for every prediction |
| 📦 **Batch Processing** | Upload a CSV file and get bulk predictions via async job |
| 📊 **Full Observability** | Prometheus metrics + Grafana dashboards + alerting rules |
| 🧪 **Experiment Tracking** | MLflow integration — params, metrics, and artifact logging |
| 🔄 **Reproducible Pipeline** | DVC-managed data pipeline (download → preprocess → validate → evaluate) |
| 🚀 **CI/CD** | GitHub Actions: lint, test, coverage on every push |

---

## Architecture

```
User / Browser
      │
      ▼
Angular Frontend (:80)
      │
      ▼
FastAPI Application (:8000)
  ├── POST /predict   ──► RoBERTa Inference Engine
  ├── POST /explain   ──► SHAP Explainer
  ├── POST /batch_predict
  ├── GET  /health
  └── GET  /metrics   ──► Prometheus (:9091) ──► Grafana (:3000)
                              │
                          MLflow (:5005)
```

All services run on a shared Docker bridge network (`sentiment-network`) and are orchestrated by Docker Compose with resource limits.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python 3.11) + Uvicorn |
| ML Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| ABSA Model | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` |
| Explainability | SHAP ≥ 0.42 |
| Containerisation | Docker (multi-stage build) + Docker Compose |
| Monitoring | Prometheus v2.45 + Grafana 10.0 |
| Experiment Tracking | MLflow v2.x |
| Data Versioning | DVC v3 |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov + httpx |

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2 (included with Docker Desktop)

### Quick Start — Full Stack

```bash
# 1. Clone the repository
git clone https://github.com/RocketUp-Team/Sentiment-Analysis-Service.git
cd Sentiment-Analysis-Service

# 2. Launch all services (API + Frontend + Prometheus + Grafana + MLflow)
docker-compose up --build
```

### Service URLs

| Service | URL | Credentials |
|---|---|---|
| API (Swagger UI) | http://localhost:8000/docs | — |
| API (ReDoc) | http://localhost:8000/redoc | — |
| Prometheus | http://localhost:9091 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| MLflow | http://localhost:5005 | — |
| Frontend | http://localhost:80 | — |

### Local Development (without Docker)

```bash
# Create and activate a Python 3.11 environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model weights
python src/model/download_models.py

# Run the API
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /health`
Returns model load status and supported languages.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "supported_languages": ["en"]
}
```

### `POST /predict`
Single-text sentiment prediction with ABSA.

**Request**
```json
{ "text": "The food was amazing but the service was slow.", "lang": "en" }
```

**Response**
```json
{
  "text": "The food was amazing but the service was slow.",
  "sentiment": "positive",
  "confidence": 0.8732,
  "aspects": [
    { "aspect": "food",    "sentiment": "positive", "confidence": 0.92 },
    { "aspect": "service", "sentiment": "negative", "confidence": 0.87 }
  ],
  "sarcasm_flag": false,
  "latency_ms": 134.5
}
```

### `POST /explain`
Token-level SHAP attributions for a prediction.

**Request**
```json
{ "text": "The food was amazing!", "lang": "en" }
```

**Response**
```json
{
  "tokens": ["The", "food", "was", "amazing", "!"],
  "shap_values": [-0.02, 0.15, 0.01, 0.48, 0.09],
  "base_value": 0.33,
  "latency_ms": 820.1
}
```

### `POST /batch_predict`
Upload a CSV file with a `text` column for bulk inference.

```bash
curl -X POST http://localhost:8000/batch_predict \
  -F "file=@reviews.csv"
```

**Response**
```json
{ "job_id": "uuid-...", "status": "processing", "total_items": 500, "created_at": "..." }
```

### `GET /metrics`
Prometheus-compatible plain-text metrics endpoint.

---

## ML Pipeline

The data pipeline is managed by **DVC** and defined in `dvc.yaml`:

```
dvc repro
```

| Stage | Script | Output |
|---|---|---|
| `download` | `src/data/downloader.py` | `data/raw/` |
| `preprocess` | `src/data/pipeline.py` | `data/processed/` |
| `validate` | `src/data/validators.py` | `data/reports/quality_report.json` |
| `evaluate_baseline` | `src/model/evaluate.py` | `data/reports/baseline_metrics.json` + MLflow run |

### Dataset
- **SemEval-2014 Task 4** — Restaurant Reviews corpus
- Labels: `positive`, `negative`, `neutral`
- Aspect categories: food, service, ambiance, price, location, general

### Preprocessing Steps
1. Label normalisation (`LabelMapper`)
2. Sentence-level sentiment derivation with `negative_priority` strategy (`SentimentDeriver`)
3. Text cleaning — lowercase, strip whitespace (`TextCleaner`)
4. Duplicate removal (`DuplicateRemover`)
5. Length filtering — min 3 chars, max 2 000 chars (`LengthFilter`)
6. Train/val/test split — 10% validation, seed 42 (`Splitter`)

---

## Monitoring

### Prometheus Metrics

| Metric | Type | Labels |
|---|---|---|
| `api_requests_total` | Counter | `method`, `endpoint`, `http_status` |
| `api_request_latency_seconds` | Histogram | `method`, `endpoint` |
| `model_inference_latency_seconds` | Histogram | — |

### Alerting Rules (`infra/prometheus/alert_rules.yml`)

| Alert | Condition | Severity |
|---|---|---|
| `HighErrorRate` | 5xx rate > 10% for 1 min | critical |
| `HighInferenceLatency` | avg inference > 500 ms for 5 min | warning |

---

## Testing

```bash
# Run all tests with coverage report
pytest --cov=src tests/

# Run only unit tests
pytest tests/data/ tests/model/ tests/contracts/

# Run only integration tests
pytest tests/test_api.py
```

| Test Layer | Location | Coverage |
|---|---|---|
| Unit — Data transforms | `tests/data/` | TextCleaner, LengthFilter, DuplicateRemover, Splitter, SentimentDeriver |
| Unit — Model / Contracts | `tests/model/`, `tests/contracts/` | ModelConfig, Pydantic schemas, validators |
| Integration — API | `tests/test_api.py` | All 5 endpoints via FastAPI TestClient |

---

## Project Structure

```
Sentiment-Analysis-Service/
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── app/                         # Angular frontend chatbot
│   └── sentiment-analysis-chatbot/
├── contracts/                   # Shared interfaces & schemas
│   ├── model_interface.py       # Abstract ModelInference base class
│   ├── schemas.py               # Pydantic request/response models
│   └── errors.py                # Custom exception types
├── data/                        # DVC-managed data artefacts
│   ├── external/                # Source datasets
│   ├── raw/                     # Downloaded raw CSVs
│   ├── processed/               # Cleaned, split data
│   └── reports/                 # Quality & evaluation reports
├── docs/                        # Project documentation
│   ├── final_project_report.tex # LaTeX report source
│   ├── final_project_report.pdf # Compiled report
│   ├── ARCHITECTURE.md          # System design docs
│   └── plan.md                  # Project plan
├── infra/
│   ├── prometheus/              # prometheus.yml, alert_rules.yml
│   └── grafana/provisioning/    # Datasource + dashboard JSON
├── src/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── data/                    # Data pipeline & transforms
│   │   ├── downloader.py
│   │   ├── pipeline.py
│   │   ├── validators.py
│   │   └── transforms/
│   ├── model/                   # Inference engine
│   │   ├── baseline.py          # BaselineModelInference (RoBERTa)
│   │   ├── config.py            # ModelConfig dataclass
│   │   └── evaluate.py          # Offline evaluation + MLflow logging
│   └── monitoring/
│       └── metrics.py           # Prometheus metrics + ASGI middleware
├── tests/                       # pytest test suite
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # 5-service orchestration
├── dvc.yaml                     # Reproducible ML pipeline stages
├── params.yaml                  # All hyperparameters & config
└── requirements.txt             # Python dependencies
```

---

## Team

| # | Name | Role |
|---|---|---|
| 1 | Duong Hong Quan | ML Pipeline & Experiment Tracking (MLflow, DVC, `src/model/`) |
| 2 | Pham Duc Long | Backend API & DevOps (FastAPI, Docker, CI/CD) |
| 3 | Do Quoc Trung | Monitoring & Data Validation (Prometheus, Grafana, `src/data/`) |

**Course:** DDM501 – AI in Production: From Models to Systems
**Institution:** FSB Institute of Management and Technology, FPT University
**Instructor:** Huynh Cong Viet Ngu

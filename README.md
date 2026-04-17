# Sentiment Analysis Service

A production-ready NLP service for sentiment analysis and aspect detection.

## Features
- **Real-time Prediction**: Analyze text sentiment and detect aspects.
- **Model Explainability**: Get SHAP values for prediction transparency.
- **Batch Processing**: Upload CSV files for bulk analysis.
- **Full Observability**: Integrated Prometheus metrics and Grafana dashboards.
- **ML Tracking**: MLflow integration for model experiment tracking.

## Getting Started

### Prerequisites
- Docker and Docker Compose

### Local Python / DVC

- Use a Python 3.11+ environment with dependencies from `requirements.txt` installed (tests and `python -m src.model.evaluate` need **torch**, **transformers**, etc.).
- For **DVC** (`dvc repro`, `dvc repro evaluate_baseline`), stages invoke `python3`. Activate your project environment first so `python3` resolves to that interpreter (e.g. `conda activate sentiment_analysis_service`), or ensure `python3` on your `PATH` is the env where deps are installed.
- Optional **real-model ABSA smoke** (downloads HF weights): `ABSA_SCENARIOS_SMOKE=1 python tests/test_absa_scenarios.py` from the repo root after activating the same environment.

### Running the Service
```bash
docker-compose up --build
```

- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (User: `admin`, Pass: `admin`)
- **MLflow**: [http://localhost:5000](http://localhost:5000)

## API Endpoints
- `GET /health`: Check service status.
- `POST /predict`: Perform real-time sentiment analysis.
- `POST /explain`: Get SHAP explanation for a prediction.
- `POST /batch_predict`: Upload CSV for batch processing.
- `GET /metrics`: Prometheus metrics endpoint.

## Project Structure
- `src/`: Backend source code.
- `contracts/`: API schemas and model interfaces.
- `infra/`: Monitoring and infrastructure configuration.
- `tests/`: Unit and integration tests.
- `Dockerfile` & `docker-compose.yml`: Container configurations.
- `.github/workflows/`: CI/CD pipelines.

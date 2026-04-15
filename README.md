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

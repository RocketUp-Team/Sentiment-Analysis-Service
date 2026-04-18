from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Request, Response
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import time
import pandas as pd
import io
import uuid
from datetime import datetime

from contracts.schemas import (
    PredictRequest, PredictResponse, AspectSentimentOut,
    ExplainRequest, ExplainResponse, HealthResponse,
    BatchSubmitResponse, BatchStatusResponse
)
from contracts.mock_model import MockModelInference
from src.model.baseline import BaselineModelInference
from src.model.language_detector import LanguageDetectionResult, LanguageDetector
from contracts.model_interface import ModelInference
from src.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    MODEL_INFERENCE_LATENCY,
    monitor_middleware,
    normalize_language_label,
)
from contracts.errors import ModelError, UnsupportedLanguageError

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Global model instance initialized at startup
    global ml_model
    from src.model.config import ModelConfig
    import os
    mode = os.getenv("MODEL_MODE", "onnx")
    config = ModelConfig(mode=mode)
    ml_model = BaselineModelInference(config)
    ml_model.preload()
    yield
    # Clean up here if needed

app = FastAPI(
    title="Sentiment Analysis Service",
    description="API for sentiment analysis with aspect detection and model explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Exception handlers
@app.exception_handler(UnsupportedLanguageError)
async def unsupported_language_handler(request: Request, exc: UnsupportedLanguageError):
    return Response(
        status_code=400,
        content=f"Error: {str(exc)}",
        media_type="text/plain"
    )

@app.exception_handler(ModelError)
async def model_error_handler(request: Request, exc: ModelError):
    return Response(
        status_code=500,
        content=f"Model Error: {str(exc)}",
        media_type="text/plain"
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.middleware("http")(monitor_middleware)

# Global model instance initialized at startup
ml_model = None
language_detector = LanguageDetector()

# Dependency to get model inference instance
def get_model() -> ModelInference:
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")
    return ml_model


def resolve_request_language(request_lang: str | None, text: str) -> LanguageDetectionResult:
    if request_lang:
        return LanguageDetectionResult(lang=request_lang, confidence=1.0)
    return language_detector.detect(text)

@app.get("/health", response_model=HealthResponse)
async def health_check(model: ModelInference = Depends(get_model)):
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        version="1.0.0",
        supported_languages=model.supported_languages
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, model: ModelInference = Depends(get_model)):
    start_time = time.time()
    detected = resolve_request_language(request.lang, request.text)

    # Measure model inference latency
    inference_start = time.time()
    result = model.predict_single(request.text, detected.lang)
    inference_duration = (time.time() - inference_start) * 1000
    MODEL_INFERENCE_LATENCY.labels(
        lang=normalize_language_label(detected.lang)
    ).observe(inference_duration / 1000)

    latency_ms = (time.time() - start_time) * 1000

    return PredictResponse(
        text=request.text,
        sentiment=result.sentiment,
        confidence=result.confidence,
        aspects=[
            AspectSentimentOut(
                aspect=a.aspect,
                sentiment=a.sentiment,
                confidence=a.confidence
            ) for a in result.aspects
        ],
        sarcasm_flag=result.sarcasm_flag,
        detected_lang=detected.lang,
        lang_confidence=detected.confidence,
        latency_ms=latency_ms
    )

@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest, model: ModelInference = Depends(get_model)):
    start_time = time.time()
    detected = resolve_request_language(request.lang, request.text)

    result = model.get_shap_explanation(request.text, detected.lang)

    latency_ms = (time.time() - start_time) * 1000

    return ExplainResponse(
        tokens=result.tokens,
        shap_values=result.shap_values,
        base_value=result.base_value,
        latency_ms=latency_ms
    )

@app.post("/batch_predict", response_model=BatchSubmitResponse)
async def batch_predict(file: UploadFile = File(...), model: ModelInference = Depends(get_model)):
    # Simple implementation: read CSV and return mock job info
    # In a real app, this would be processed by a background worker (e.g., Celery)
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    job_id = str(uuid.uuid4())
    
    return BatchSubmitResponse(
        job_id=job_id,
        status="processing",
        total_items=len(df),
        created_at=datetime.now().isoformat()
    )

@app.get("/batch_status/{job_id}", response_model=BatchStatusResponse)
async def batch_status(job_id: str):
    # Mock status check
    return BatchStatusResponse(
        job_id=job_id,
        status="completed",
        progress=1.0,
        total_items=100,
        processed_items=100,
        created_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat()
    )

# Metrics endpoint for Prometheus
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

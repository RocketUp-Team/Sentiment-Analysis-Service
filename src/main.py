from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Request, Response
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
from contracts.model_interface import ModelInference
from src.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, MODEL_INFERENCE_LATENCY, monitor_middleware
from contracts.errors import ModelError, UnsupportedLanguageError

app = FastAPI(
    title="Sentiment Analysis Service",
    description="API for sentiment analysis with aspect detection and model explainability.",
    version="1.0.0"
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

# Dependency to get model inference instance
# In a real app, this would be a singleton or loaded at startup
def get_model() -> ModelInference:
    return MockModelInference()

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
    
    # Measure model inference latency
    inference_start = time.time()
    result = model.predict_single(request.text, request.lang)
    inference_duration = (time.time() - inference_start) * 1000
    MODEL_INFERENCE_LATENCY.observe(inference_duration / 1000)
    
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
        latency_ms=latency_ms
    )

@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest, model: ModelInference = Depends(get_model)):
    start_time = time.time()
    
    result = model.get_shap_explanation(request.text, request.lang)
    
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

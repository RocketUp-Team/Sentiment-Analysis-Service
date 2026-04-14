"""Pydantic schemas used directly by FastAPI and Swagger UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AspectSentimentOut(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str = "en"


class PredictResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    aspects: list[AspectSentimentOut] = Field(default_factory=list)
    sarcasm_flag: bool
    latency_ms: float


class ExplainRequest(PredictRequest):
    pass


class ExplainResponse(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    shap_values: list[float] = Field(default_factory=list)
    base_value: float
    latency_ms: float


class BatchSubmitResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    total_items: int
    created_at: str


class BatchStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: float = Field(ge=0.0, le=1.0)
    total_items: int
    processed_items: int
    created_at: str
    completed_at: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    version: str
    supported_languages: list[str] = Field(default_factory=list)


# Multipart upload for POST /api/v1/batch is handled by FastAPI UploadFile rather than a Pydantic request model.

"""Pydantic schemas used directly by FastAPI and Swagger UI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AspectSentimentOut(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str | None = None


class PredictResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    aspects: list[AspectSentimentOut] = Field(default_factory=list)
    sarcasm_flag: bool
    detected_lang: str
    lang_confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float


class ExplainRequest(PredictRequest):
    pass


class ExplainResponse(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    shap_values: list[float] = Field(default_factory=list)
    base_value: float
    latency_ms: float

    @model_validator(mode="after")
    def validate_token_and_shap_lengths(self) -> "ExplainResponse":
        if len(self.tokens) != len(self.shap_values):
            raise ValueError("tokens and shap_values must have the same length")
        return self


class BatchItemResult(BaseModel):
    row: int
    text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    aspects: list[AspectSentimentOut] = Field(default_factory=list)
    error: str | None = None


class BatchPredictResponse(BaseModel):
    total_items: int
    processed_items: int
    failed_items: int
    latency_ms: float
    results: list[BatchItemResult]


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


# Multipart upload for POST /batch_predict is handled by FastAPI UploadFile.

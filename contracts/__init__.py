"""Shared interface package with quick imports for Quan's FastAPI wiring."""

from contracts.errors import ModelError, UnsupportedLanguageError
from contracts.model_interface import AspectSentiment, ModelInference, PredictionResult, SHAPResult
from contracts.mock_model import MockModelInference
from contracts.schemas import (
    AspectSentimentOut,
    BatchStatusResponse,
    BatchSubmitResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

__all__ = [
    "ModelError",
    "UnsupportedLanguageError",
    "AspectSentiment",
    "ModelInference",
    "PredictionResult",
    "SHAPResult",
    "MockModelInference",
    "AspectSentimentOut",
    "BatchStatusResponse",
    "BatchSubmitResponse",
    "ExplainRequest",
    "ExplainResponse",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
]

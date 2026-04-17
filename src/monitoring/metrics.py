from prometheus_client import Counter, Histogram
from fastapi import Request
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    "api_requests_total", 
    "Total count of requests to the API",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests in seconds",
    ["method", "endpoint"]
)

MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Latency of model inference in seconds",
    ["lang"],
)


def normalize_language_label(lang: str) -> str:
    """Collapse free-form language codes into a low-cardinality allowlist."""
    return lang if lang in {"en", "vi"} else "other"

async def monitor_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Update metrics
    endpoint = request.url.path
    method = request.method
    status = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response

# ── Stage 1: Build Python dependencies ──────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .

# Optimize torch installation for CPU and clean cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Pull ONNX models from DagsHub ─────────────────────
FROM python:3.11-slim AS model-puller

WORKDIR /app

# Install only DVC (lightweight, no torch/transformers needed)
RUN pip install --no-cache-dir dvc

# Copy DVC configuration and lock file
COPY .dvc/ .dvc/
COPY dvc.yaml dvc.lock ./

# Receive credentials as build arguments
ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN

# Configure DVC remote auth and pull models
# Note: we only pull sentiment_fp32 and sarcasm_fp32 for the runtime API
RUN dvc config core.no_scm true && \
    dvc remote modify origin --local auth basic && \
    dvc remote modify origin --local user ${DAGSHUB_USERNAME} && \
    dvc remote modify origin --local password ${DAGSHUB_TOKEN} && \
    dvc pull models/onnx/sentiment_fp32 models/onnx/sarcasm_fp32 && \
    rm -f .dvc/config.local

# ── Stage 3: Final runtime image ───────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed dependencies from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code and contracts
COPY src/ ./src/
COPY contracts/ ./contracts/

# Copy models from model-puller
COPY --from=model-puller /app/models/onnx/ ./models/onnx/

# Environment variables for build & runtime
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Step 1: Download HuggingFace base model while online
RUN python src/model/download_models.py

# Step 2: Force offline mode for runtime (faster startup, no net checks)
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

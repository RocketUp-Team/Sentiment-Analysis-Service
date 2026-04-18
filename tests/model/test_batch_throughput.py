"""Throughput benchmark tests for predict_batch.

Run locally:   pytest tests/model/test_batch_throughput.py -m benchmark -v
Skip in CI:    pytest -m "not benchmark"

These tests load the real model — they are NOT mocked.
Requires: PYTHONPATH set to project root.
"""
import time
import pytest


@pytest.mark.benchmark
def test_sentiment_throughput_meets_target():
    """skip_absa=True on 1000 texts must achieve >= 100 samples/sec."""
    from src.model.baseline import BaselineModelInference
    from src.model.config import ModelConfig

    config = ModelConfig(batch_size=32)
    model = BaselineModelInference(config=config)

    texts = ["The food was absolutely amazing and I loved every bite!"] * 1000

    start = time.perf_counter()
    results = model.predict_batch(texts, skip_absa=True)
    elapsed = time.perf_counter() - start

    throughput = len(texts) / elapsed
    print(f"\nThroughput: {throughput:.1f} samples/sec ({elapsed:.2f}s for {len(texts)} texts)")

    assert len(results) == 1000, "Result count must match input count"
    assert throughput >= 100.0, (
        f"Throughput {throughput:.1f} samples/sec is below the 100 samples/sec target"
    )


@pytest.mark.benchmark
def test_full_pipeline_throughput():
    """skip_absa=False on 50 texts — log throughput, no hard assert."""
    from src.model.baseline import BaselineModelInference
    from src.model.config import ModelConfig

    config = ModelConfig(batch_size=32)
    model = BaselineModelInference(config=config)

    texts = [
        "The food was great but the service was slow.",
        "Absolutely terrible experience. Never coming back.",
        "Best ambiance in the city, fair prices.",
        "Average meal, nothing special.",
        "Outstanding in every way!",
    ] * 10  # 50 texts

    start = time.perf_counter()
    results = model.predict_batch(texts, skip_absa=False)
    elapsed = time.perf_counter() - start

    throughput = len(texts) / elapsed
    print(f"\nFull pipeline throughput: {throughput:.2f} samples/sec ({elapsed:.1f}s for {len(texts)} texts)")

    assert len(results) == 50, "Result count must match input count"
    # No hard throughput assert — ABSA is intentionally slow (~60-120s for 100 texts in CI)

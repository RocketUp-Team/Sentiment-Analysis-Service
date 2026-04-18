import argparse
import json
import logging
import time
from pathlib import Path

from src.model.baseline import BaselineModelInference
from src.model.config import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def benchmark_model(model_inference: BaselineModelInference, texts: list[str], runs: int = 5) -> dict:
    """Benchmark a loaded model and return metrics.

    skip_absa=True và skip_sarcasm=True là bắt buộc: nếu không, mỗi sample
    chạy thêm 2+ zero-shot forward passes (ABSA) + 1 sarcasm forward pass
    tuần tự → sai lệch throughput thực sự của sentiment model 10-50x.
    """
    # Warmup
    logger.info("Warming up model...")
    model_inference.predict_batch(texts[:10], skip_absa=True, skip_sarcasm=True)
    
    logger.info(f"Running benchmark ({runs} iterations)...")
    latencies = []
    
    for i in range(runs):
        start = time.perf_counter()
        model_inference.predict_batch(texts, skip_absa=True, skip_sarcasm=True)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        logger.info(f"  Run {i+1}/{runs}: {elapsed:.2f}s")
        
    avg_time = sum(latencies) / len(latencies)
    throughput = len(texts) / avg_time
    
    return {
        "avg_latency_seconds": avg_time,
        "throughput_samples_per_sec": throughput,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX models.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of dummy samples to benchmark.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for predict_batch.")
    parser.add_argument("--output", type=str, default="reports/onnx_benchmark.json", help="Output JSON path.")
    args = parser.parse_args()
    
    dummy_texts = ["This is a test sentence to benchmark model inference speed."] * args.samples
    
    metrics = {}
    
    # 1. PyTorch (Finetuned)
    logger.info("--- PyTorch (Finetuned) Benchmark ---")
    config_pt = ModelConfig(mode="finetuned", batch_size=args.batch_size)
    model_pt = BaselineModelInference(config_pt)
    metrics["pytorch"] = benchmark_model(model_pt, dummy_texts)
    del model_pt
    
    # 2. ONNX FP32
    logger.info("--- ONNX FP32 Benchmark ---")
    config_fp32 = ModelConfig(mode="onnx", batch_size=args.batch_size)
    model_fp32 = BaselineModelInference(config_fp32)
    metrics["onnx_fp32"] = benchmark_model(model_fp32, dummy_texts)
    del model_fp32
    
    # 3. ONNX INT8
    logger.info("--- ONNX INT8 Benchmark ---")
    config_int8 = ModelConfig(mode="onnx_int8", batch_size=args.batch_size)
    model_int8 = BaselineModelInference(config_int8)
    metrics["onnx_int8"] = benchmark_model(model_int8, dummy_texts)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Benchmark results saved to {output_path}")

if __name__ == "__main__":
    main()

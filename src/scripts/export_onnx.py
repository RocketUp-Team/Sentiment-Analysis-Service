import argparse
import logging
from pathlib import Path

from src.model.config import ModelConfig
from src.model.onnx_exporter import OnnxExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX formats.")
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="sentiment",
        choices=["sentiment", "sarcasm"],
        help="Adapter to merge and export.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Directory to save the exported ONNX models.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = ModelConfig(mode="finetuned")
    exporter = OnnxExporter(config)
    
    output_dir = Path(args.output_dir)
    fp32_path = output_dir / f"{args.adapter_name}_fp32"
    int8_path = output_dir / f"{args.adapter_name}_int8"
    
    logger.info(f"Starting ONNX export for adapter: {args.adapter_name}")
    
    try:
        exporter.export_fp32(fp32_path, args.adapter_name)
        exporter.export_int8(fp32_path, int8_path)
        logger.info("ONNX export pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise

if __name__ == "__main__":
    main()

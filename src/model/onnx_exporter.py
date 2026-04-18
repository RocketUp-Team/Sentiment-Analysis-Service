import logging
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from src.model.config import ModelConfig

logger = logging.getLogger(__name__)

class OnnxExporter:
    """Handles merging LoRA adapters and exporting to ONNX FP32 and INT8 formats."""

    def __init__(self, config: ModelConfig):
        self._config = config

    def export_fp32(self, output_path: str | Path, adapter_name: str) -> None:
        """Merge PEFT adapter into base model and export to FP32 ONNX."""
        logger.info(f"Starting FP32 ONNX export to {output_path} for adapter {adapter_name}")
        
        # 1. Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self._config.finetuned_model_name,
            num_labels=len(self._config.label_map),
            use_safetensors=True,
        )
        
        # 2. Load PEFT adapter
        adapter_path = (
            self._config.sentiment_adapter_path
            if adapter_name == "sentiment"
            else self._config.sarcasm_adapter_path
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 3. Merge weights
        merged_model = peft_model.merge_and_unload()
        
        # 4. Save intermediate merged model
        temp_dir = Path(output_path).parent / "temp_merged"
        merged_model.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(self._config.finetuned_model_name)
        tokenizer.save_pretrained(temp_dir)
        
        # 5. Export to ONNX using Optimum
        logger.info("Converting merged model to ONNX...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(temp_dir, export=True)
        ort_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info("FP32 ONNX export completed.")

    def export_int8(self, fp32_path: str | Path, int8_output_path: str | Path) -> None:
        """Quantize an existing FP32 ONNX model to INT8."""
        logger.info(f"Starting INT8 quantization from {fp32_path} to {int8_output_path}")
        
        ort_model = ORTModelForSequenceClassification.from_pretrained(fp32_path, export=False)
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        
        # Use AVX512 VNNI configuration for optimal CPU inference
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        
        quantizer.quantize(
            save_dir=int8_output_path,
            quantization_config=dqconfig,
        )
        
        # Copy tokenizer config
        tokenizer = AutoTokenizer.from_pretrained(fp32_path)
        tokenizer.save_pretrained(int8_output_path)
        logger.info("INT8 quantization completed.")

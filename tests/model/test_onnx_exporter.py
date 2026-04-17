from unittest.mock import MagicMock, patch
from pathlib import Path
from src.model.onnx_exporter import OnnxExporter
from src.model.config import ModelConfig

@patch("src.model.onnx_exporter.AutoModelForSequenceClassification")
@patch("src.model.onnx_exporter.PeftModel")
@patch("src.model.onnx_exporter.AutoTokenizer")
@patch("src.model.onnx_exporter.ORTModelForSequenceClassification")
@patch("src.model.onnx_exporter.ORTQuantizer")
@patch("src.model.onnx_exporter.AutoQuantizationConfig")
def test_export_fp32(mock_quant_config, mock_quantizer, mock_ort_model, mock_tokenizer, mock_peft, mock_base_model):
    # Setup mocks
    mock_base_model_instance = MagicMock()
    mock_base_model.from_pretrained.return_value = mock_base_model_instance
    
    mock_peft_instance = MagicMock()
    mock_peft.from_pretrained.return_value = mock_peft_instance
    mock_peft_instance.merge_and_unload.return_value = mock_base_model_instance
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    
    exporter = OnnxExporter(ModelConfig(mode="finetuned"))
    exporter.export_fp32("dummy_output_path", "sentiment")
    
    # Assertions
    mock_base_model.from_pretrained.assert_called_once()
    mock_peft.from_pretrained.assert_called_once()
    mock_peft_instance.merge_and_unload.assert_called_once()
    
    mock_base_model_instance.save_pretrained.assert_called_once()
    assert mock_tokenizer_instance.save_pretrained.call_count == 2
    
    mock_ort_model.from_pretrained.assert_called_once_with(Path("dummy_output_path").parent / "temp_merged", export=True)
    mock_ort_model.from_pretrained.return_value.save_pretrained.assert_called_once_with("dummy_output_path")

@patch("src.model.onnx_exporter.AutoModelForSequenceClassification")
@patch("src.model.onnx_exporter.PeftModel")
@patch("src.model.onnx_exporter.AutoTokenizer")
@patch("src.model.onnx_exporter.ORTModelForSequenceClassification")
@patch("src.model.onnx_exporter.ORTQuantizer")
@patch("src.model.onnx_exporter.AutoQuantizationConfig")
def test_export_int8(mock_quant_config, mock_quantizer, mock_ort_model, mock_tokenizer, mock_peft, mock_base_model):
    exporter = OnnxExporter(ModelConfig(mode="finetuned"))
    exporter.export_int8("dummy_fp32_path", "dummy_int8_path")
    
    mock_ort_model.from_pretrained.assert_called_once_with("dummy_fp32_path", export=False)
    mock_quantizer.from_pretrained.assert_called_once_with(mock_ort_model.from_pretrained.return_value)
    mock_quant_config.avx512_vnni.assert_called_once()
    mock_quantizer.from_pretrained.return_value.quantize.assert_called_once()

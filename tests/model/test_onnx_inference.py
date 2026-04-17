import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.model.onnx_inference import OnnxInferenceSession

@patch("src.model.onnx_inference.AutoTokenizer")
@patch("src.model.onnx_inference.ort.InferenceSession")
def test_onnx_inference_predict_probs(mock_ort_session, mock_auto_tokenizer):
    # Setup mocks
    mock_session_instance = MagicMock()
    mock_ort_session.return_value = mock_session_instance
    
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.return_value = {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]])
    }
    
    # Mock ONNX output: [batch_size, num_labels] logits
    mock_session_instance.run.return_value = [np.array([[0.1, 0.2, 0.7]])]
    mock_session_instance.get_inputs.return_value = [
        MagicMock(name="input_ids"), MagicMock(name="attention_mask")
    ]
    mock_session_instance.get_inputs()[0].name = "input_ids"
    mock_session_instance.get_inputs()[1].name = "attention_mask"
    
    session = OnnxInferenceSession("dummy_path", "dummy_tokenizer")
    probs = session.predict_probs(["test text"])
    
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1, 3)
    # Check if softmax was applied (sum to 1)
    np.testing.assert_allclose(np.sum(probs, axis=1), [1.0])

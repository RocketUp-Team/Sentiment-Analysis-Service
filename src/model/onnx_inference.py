import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.model.device import get_device

class OnnxInferenceSession:
    def __init__(self, model_path: str, tokenizer_name: str, max_length: int = 512):
        device = get_device()
        providers = ["CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        
        if device.type == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device.type == "mps" and "CoreMLExecutionProvider" in available_providers:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path, 
            providers=providers
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.input_names = [i.name for i in self.session.get_inputs()]
        
    def predict_probs(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        
        onnx_inputs = {name: inputs[name] for name in self.input_names if name in inputs}
        logits = self.session.run(None, onnx_inputs)[0]
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

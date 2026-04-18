import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import pytest
from contracts.model_interface import PredictionResult, AspectSentiment, SHAPResult

from src.scripts import generate_shap_plots

matplotlib.use("Agg")  # Use non-interactive backend for testing

def test_generate_plot_creates_file_with_correct_name(tmp_path):
    # Arrange
    output_dir = tmp_path / "plots"
    text = "The food was great but the service was terrible."
    
    # Mock inference result
    prediction = PredictionResult(
        sentiment="positive",
        confidence=0.85,
        aspects=[
            AspectSentiment(aspect="food", sentiment="positive", confidence=0.9),
            AspectSentiment(aspect="service", sentiment="negative", confidence=0.8)
        ],
        sarcasm_flag=False
    )
    
    shap_result = SHAPResult(
        tokens=[" The", " food", " was", " great", " but", " the", " service", " was", " terrible", "."],
        shap_values=[0.0, 0.1, 0.0, 0.5, 0.0, 0.0, -0.2, 0.0, -0.4, 0.0],
        base_value=0.1
    )
    
    mock_model = MagicMock()
    mock_model.predict_single.return_value = prediction
    mock_model.get_shap_explanation.return_value = shap_result

    # Act
    saved_path = generate_shap_plots.generate_plot_for_text(
        text=text,
        model=mock_model,
        output_dir=output_dir,
        file_prefix="sample_1"
    )

    # Assert
    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    # Filename should contain sentiment and aspect info
    assert "positive" in saved_path.name
    assert "food" in saved_path.name
    assert "service" in saved_path.name

def test_main_cli_execution(tmp_path, monkeypatch):
    # Arrange
    output_dir = tmp_path / "shap_plots"
    
    # Mock model to avoid loading real weights
    class FakeModel:
        def preload(self):
            pass
        def predict_single(self, text):
            return PredictionResult(sentiment="negative", confidence=0.9)
        def get_shap_explanation(self, text):
            return SHAPResult(tokens=["a", "bad", "test"], shap_values=[0.0, -0.5, 0.0], base_value=0.1)
    
    mock_generate = MagicMock()
    
    monkeypatch.setattr(generate_shap_plots, "BaselineModelInference", lambda config: FakeModel())
    monkeypatch.setattr(generate_shap_plots, "generate_plot_for_text", mock_generate)
    
    test_args = ["--output-dir", str(output_dir), "--texts", "a bad test", "another text"]
    
    # Act
    exit_code = generate_shap_plots.main(test_args)
    
    # Assert
    assert exit_code == 0
    assert mock_generate.call_count == 2

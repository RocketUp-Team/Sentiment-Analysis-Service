import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

from contracts.model_interface import ModelInference
from src.model.baseline import BaselineModelInference

logger = logging.getLogger(__name__)

def generate_plot_for_text(
    text: str, model: ModelInference, output_dir: Path, file_prefix: str
) -> Path:
    """Generate and save a SHAP plot for a single text."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get prediction to extract aspects and overall sentiment
    prediction = model.predict_single(text)
    
    # 2. Get SHAP explanation for the overall sentiment
    shap_result = model.get_shap_explanation(text)

    # 3. Construct filename based on predictions
    aspect_str = "_".join(a.aspect for a in prediction.aspects)
    if aspect_str:
        aspect_str = f"_aspects-{aspect_str}"
        
    filename = f"{file_prefix}_pred-{prediction.sentiment}{aspect_str}.png"
    output_path = output_dir / filename

    # 4. Reconstruct shap.Explanation
    exp = shap.Explanation(
        values=np.array(shap_result.shap_values),
        base_values=shap_result.base_value,
        data=np.array(shap_result.tokens),
        feature_names=shap_result.tokens,
    )

    # 5. Plot and save
    plt.figure(figsize=(10, 6))
    # We use waterfall plot. We must pass show=False to avoid blocking.
    shap.waterfall_plot(exp, show=False)
    
    title = f"Sentiment: {prediction.sentiment} (conf: {prediction.confidence:.2f})"
    if prediction.aspects:
        aspect_details = ", ".join(f"{a.aspect}:{a.sentiment}" for a in prediction.aspects)
        title += f"\nAspects: {aspect_details}"
    
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP plot to %s", output_path)
    return output_path

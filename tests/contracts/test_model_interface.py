from dataclasses import is_dataclass

import pytest

from contracts.model_interface import AspectSentiment, ModelInference, PredictionResult, SHAPResult


def test_aspect_sentiment_fields():
    aspect = AspectSentiment(aspect="service", sentiment="positive", confidence=0.91)

    assert aspect.aspect == "service"
    assert aspect.sentiment == "positive"
    assert aspect.confidence == 0.91


def test_aspect_sentiment_is_dataclass():
    assert is_dataclass(AspectSentiment)


def test_prediction_result_fields():
    result = PredictionResult(
        sentiment="negative",
        confidence=0.42,
        aspects=[AspectSentiment(aspect="speed", sentiment="negative", confidence=0.87)],
        sarcasm_flag=True,
    )

    assert result.sentiment == "negative"
    assert result.confidence == 0.42
    assert result.aspects[0].aspect == "speed"
    assert result.sarcasm_flag is True


def test_prediction_result_accepts_empty_aspects_list():
    result = PredictionResult(sentiment="neutral", confidence=0.5)

    assert result.aspects == []


def test_shap_result_fields():
    shap_result = SHAPResult(tokens=["great", "service"], shap_values=[0.2, 0.3], base_value=0.1)

    assert shap_result.tokens == ["great", "service"]
    assert shap_result.shap_values == [0.2, 0.3]
    assert shap_result.base_value == 0.1


def test_model_inference_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ModelInference()

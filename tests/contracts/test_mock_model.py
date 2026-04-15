from contracts.errors import UnsupportedLanguageError
from contracts.model_interface import AspectSentiment, ModelInference, PredictionResult, SHAPResult

from contracts.mock_model import MockModelInference


sentiments = {"positive", "negative", "neutral"}
aspects = {"food", "service", "ambiance", "price", "location", "general"}


def test_mock_model_is_model_inference_instance():
    assert isinstance(MockModelInference(), ModelInference)


def test_mock_model_is_loaded():
    assert MockModelInference().is_loaded is True


def test_mock_model_supports_english():
    assert "en" in MockModelInference().supported_languages


def test_predict_single_returns_prediction_result():
    result = MockModelInference().predict_single("Great food and service", lang="en")

    assert isinstance(result, PredictionResult)


def test_predict_single_returns_valid_sentiment():
    result = MockModelInference().predict_single("Great food and service", lang="en")

    assert result.sentiment in sentiments


def test_predict_single_returns_confidence_in_range():
    result = MockModelInference().predict_single("Great food and service", lang="en")

    assert 0.0 <= result.confidence <= 1.0


def test_predict_single_returns_valid_aspects():
    result = MockModelInference().predict_single("Great food and service", lang="en")

    assert 1 <= len(result.aspects) <= 3
    for aspect_sentiment in result.aspects:
        assert isinstance(aspect_sentiment, AspectSentiment)
        assert aspect_sentiment.aspect in aspects
        assert aspect_sentiment.sentiment in sentiments
        assert 0.0 <= aspect_sentiment.confidence <= 1.0


def test_predict_single_returns_sarcasm_flag_bool():
    result = MockModelInference().predict_single("Great food and service", lang="en")

    assert isinstance(result.sarcasm_flag, bool)


def test_predict_single_raises_for_unsupported_language():
    mock = MockModelInference()

    try:
        mock.predict_single("Bonjour", lang="fr")
    except UnsupportedLanguageError:
        assert True
    else:
        raise AssertionError("Expected UnsupportedLanguageError")


def test_predict_batch_returns_same_number_of_results_as_input():
    texts = ["Great food", "Slow service", "Nice ambiance"]

    results = MockModelInference().predict_batch(texts, lang="en")

    assert len(results) == len(texts)


def test_predict_batch_returns_prediction_result_items():
    results = MockModelInference().predict_batch(["Great food", "Slow service"], lang="en")

    assert all(isinstance(result, PredictionResult) for result in results)


def test_predict_batch_empty_returns_empty_list():
    assert MockModelInference().predict_batch([], lang="en") == []


def test_get_shap_explanation_returns_shap_result():
    result = MockModelInference().get_shap_explanation("Great food and service", lang="en")

    assert isinstance(result, SHAPResult)


def test_shap_tokens_and_values_have_same_length():
    result = MockModelInference().get_shap_explanation("Great food and service", lang="en")

    assert len(result.tokens) == len(result.shap_values)


def test_shap_tokens_are_nonempty():
    result = MockModelInference().get_shap_explanation("Great food and service", lang="en")

    assert all(token for token in result.tokens)


def test_shap_base_value_is_float():
    result = MockModelInference().get_shap_explanation("Great food and service", lang="en")

    assert isinstance(result.base_value, float)

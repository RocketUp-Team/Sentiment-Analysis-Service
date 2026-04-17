from dataclasses import FrozenInstanceError

import pytest

from src.model.config import ModelConfig


class TestModelConfig:
    def test_default_model_name(self):
        config = ModelConfig()
        assert config.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def test_default_max_length(self):
        config = ModelConfig()
        assert config.max_length == 512

    def test_default_language(self):
        config = ModelConfig()
        assert config.default_lang == "en"
        assert "en" in config.supported_languages

    def test_label_map_has_three_classes(self):
        config = ModelConfig()
        assert len(config.label_map) == 3
        assert set(config.label_map.values()) == {"negative", "neutral", "positive"}

    def test_label_map_indices(self):
        """cardiffnlp model: 0=negative, 1=neutral, 2=positive."""
        config = ModelConfig()
        assert config.label_map[0] == "negative"
        assert config.label_map[1] == "neutral"
        assert config.label_map[2] == "positive"

    def test_frozen_immutability(self):
        """Config should be immutable (frozen dataclass)."""
        config = ModelConfig()
        with pytest.raises(FrozenInstanceError):
            config.model_name = "other"

        try:
            config.label_map[0] = "changed"
            assert False, "Should have raised TypeError"
        except TypeError:
            pass  # Expected — read-only mapping

    def test_custom_model_name(self):
        config = ModelConfig(model_name="custom/model")
        assert config.model_name == "custom/model"

    def test_default_mode_is_baseline(self):
        assert ModelConfig().mode == "baseline"

    def test_finetuned_model_defaults_match_phase2_paths(self):
        config = ModelConfig()
        assert config.finetuned_model_name == "xlm-roberta-base"
        assert config.sentiment_adapter_path == "models/adapters/sentiment"
        assert config.sarcasm_adapter_path == "models/adapters/sarcasm"

    def test_label_map_matches_project_sentiment_labels(self):
        """Ensure config labels match the project's expected sentiment labels."""
        config = ModelConfig()
        project_labels = {"positive", "negative", "neutral"}
        assert set(config.label_map.values()) == project_labels


class TestModelConfigABSA:
    def test_default_absa_model_name(self):
        assert ModelConfig().absa_model_name == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

    def test_default_absa_threshold(self):
        assert ModelConfig().absa_threshold == 0.45

    def test_default_absa_categories(self):
        categories = ModelConfig().absa_categories
        assert set(categories) == {
            "food",
            "service",
            "ambiance",
            "price",
            "location",
            "general",
        }
        assert len(categories) == 6


def test_default_batch_size_is_32():
    config = ModelConfig()
    assert config.batch_size == 32


def test_batch_size_is_configurable():
    config = ModelConfig(batch_size=16)
    assert config.batch_size == 16


def test_supported_languages_include_english_and_vietnamese():
    config = ModelConfig()
    assert config.supported_languages == ("en", "vi")

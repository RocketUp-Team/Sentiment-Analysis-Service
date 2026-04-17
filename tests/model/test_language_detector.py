from src.model.language_detector import LanguageDetector, LanguageDetectionResult


def test_short_text_falls_back_to_default_lang():
    detector = LanguageDetector(default_lang="en")

    result = detector.detect("ok")

    assert result == LanguageDetectionResult(lang="en", confidence=0.0)


def test_detects_vietnamese_text_from_diacritics():
    detector = LanguageDetector(default_lang="en")

    result = detector.detect("Dịch vụ này rất tuyệt vời và đáng tiền")

    assert result.lang == "vi"
    assert result.confidence > 0.8


def test_detects_ascii_sentence_as_english():
    detector = LanguageDetector(default_lang="en")

    result = detector.detect("The service was good and delivery was fast")

    assert result.lang == "en"
    assert result.confidence > 0.5

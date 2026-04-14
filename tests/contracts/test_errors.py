import pytest


from contracts.errors import ModelError, UnsupportedLanguageError


def test_model_error_stringifies_message():
    error = ModelError("something broke")

    assert isinstance(error, Exception)
    assert str(error) == "something broke"


def test_unsupported_language_error_inherits_and_mentions_language():
    error = UnsupportedLanguageError("zh")

    assert isinstance(error, ModelError)
    assert isinstance(error, Exception)
    assert "zh" in str(error)


def test_model_error_is_caught_by_pytest_raises():
    with pytest.raises(ModelError):
        raise ModelError("something broke")


def test_unsupported_language_error_is_caught_by_pytest_raises_model_error():
    with pytest.raises(ModelError):
        raise UnsupportedLanguageError("zh")

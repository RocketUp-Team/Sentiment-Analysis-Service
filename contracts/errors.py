"""Error types for model-level failures."""


class ModelError(Exception):
    """Raised for model-level failures."""


class UnsupportedLanguageError(ModelError):
    def __init__(self, lang: str):
        super().__init__(f"Language not supported: '{lang}'")
        self.lang = lang

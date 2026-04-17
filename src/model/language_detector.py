"""Best-effort language detection with a safe local fallback."""

from __future__ import annotations

from dataclasses import dataclass
import re


_VIETNAMESE_MARKERS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Structured detection result returned by `LanguageDetector.detect()`."""

    lang: str
    confidence: float


class LanguageDetector:
    """Lightweight detector that can later wrap a stronger backend."""

    def __init__(
        self,
        *,
        default_lang: str = "en",
        minimum_tokens: int = 3,
    ):
        self.default_lang = default_lang
        self.minimum_tokens = minimum_tokens

    def detect(self, text: str) -> LanguageDetectionResult:
        stripped = text.strip()
        if not stripped:
            return LanguageDetectionResult(lang=self.default_lang, confidence=0.0)

        if len(stripped.split()) < self.minimum_tokens:
            return LanguageDetectionResult(lang=self.default_lang, confidence=0.0)

        if _VIETNAMESE_MARKERS.search(stripped):
            return LanguageDetectionResult(lang="vi", confidence=0.99)

        return LanguageDetectionResult(lang="en", confidence=0.75)

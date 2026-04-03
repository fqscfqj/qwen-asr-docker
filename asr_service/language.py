from __future__ import annotations

SUPPORTED_LANGUAGE_NAMES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
)

_NAME_BY_KEY = {name.casefold(): name for name in SUPPORTED_LANGUAGE_NAMES}
_LANGUAGE_ALIAS_TO_NAME = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "cmn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "yue": "Cantonese",
    "zh-yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
}


def normalize_language(language: str | None) -> str | None:
    """Normalize OpenAI-style language input into Qwen ASR accepted names."""
    if language is None:
        return None

    raw = language.strip()
    if not raw:
        return None

    name = _NAME_BY_KEY.get(raw.casefold())
    if name is not None:
        return name

    alias_key = raw.casefold().replace("_", "-")
    return _LANGUAGE_ALIAS_TO_NAME.get(alias_key, raw)

"""
Lang-Bridge Tool — AI Invoice Auditor
Translates extracted invoice text from any language to English.
Uses deep-translator (GoogleTranslator) as the primary provider.
Returns a confidence score and skips translation for English-language invoices.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

# Translation cache — avoids re-translating identical text (keyed by MD5 of text)
_CACHE_FILE = Path("./data/translation_cache.json")

# Languages that require no translation
_ENGLISH_CODES = {"en", "eng", "english"}

# Confidence scores by scenario
_CONFIDENCE_KNOWN_LANG = 0.85     # well-supported language (es, de, fr, etc.)
_CONFIDENCE_UNKNOWN_LANG = 0.65   # unsupported or ambiguous language
_CONFIDENCE_ENGLISH = 1.0         # no translation needed
_CONFIDENCE_CACHED = 0.90         # previously translated and cached
_LOW_CONFIDENCE_THRESHOLD = 0.75  # below this → flag for human review


def translate(
    text: str,
    source_language: str,
    use_cache: bool = True,
) -> dict:
    """
    Translate text to English if it is not already in English.

    Args:
        text:             The extracted invoice text to translate.
        source_language:  ISO 639-1 language code (e.g. "es", "de", "en").
        use_cache:        If True, check and populate the translation cache.

    Returns:
        {
            "translated_text":  str,    # English text (or original if already English)
            "source_language":  str,    # original detected language code
            "confidence":       float,  # 0.0 – 1.0
            "was_translated":   bool,   # True if translation was performed
            "error":            str | None
        }
    """
    if not text or not text.strip():
        return _result(text, source_language, _CONFIDENCE_ENGLISH, False)

    lang = (source_language or "").lower().strip()

    # Skip translation for English text
    if lang in _ENGLISH_CODES:
        logger.info("Text is already English — skipping translation")
        return _result(text, lang, _CONFIDENCE_ENGLISH, False)

    # Check cache
    if use_cache:
        cached = _get_cached(text)
        if cached:
            logger.info("Translation cache hit for lang=%s", lang)
            return _result(cached, lang, _CONFIDENCE_CACHED, True)

    # Perform translation
    try:
        translated = _call_translator(text, lang)
        confidence = _estimate_confidence(lang)

        if use_cache:
            _set_cached(text, translated)

        logger.info("Translated %s→en, confidence=%.2f, chars=%d", lang, confidence, len(translated))
        return _result(translated, lang, confidence, True)

    except Exception as exc:
        logger.error("Translation failed for lang=%s: %s", lang, exc)
        # Return original text with low confidence so pipeline continues
        return _result(text, lang, _CONFIDENCE_UNKNOWN_LANG, False, error=str(exc))


def is_low_confidence(confidence: float) -> bool:
    """Return True if confidence is below the human-review threshold."""
    return confidence < _LOW_CONFIDENCE_THRESHOLD


# ── Internal helpers ───────────────────────────────────────────────────────────

def _call_translator(text: str, source_lang: str) -> str:
    """Call GoogleTranslator via deep-translator."""
    from deep_translator import GoogleTranslator
    # GoogleTranslator handles long texts up to ~5000 chars per call
    # For longer texts, split and rejoin
    if len(text) <= 4500:
        return GoogleTranslator(source="auto", target="en").translate(text) or text

    chunks = _split_text(text, max_len=4500)
    translated_chunks = [
        GoogleTranslator(source="auto", target="en").translate(chunk) or chunk
        for chunk in chunks
    ]
    return "\n".join(translated_chunks)


def _split_text(text: str, max_len: int = 4500) -> list[str]:
    """Split text into chunks at sentence/newline boundaries."""
    lines = text.split("\n")
    chunks: list[str] = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 <= max_len:
            current = (current + "\n" + line).lstrip("\n")
        else:
            if current:
                chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks or [text]


def _estimate_confidence(lang: str) -> float:
    """Estimate confidence based on language support."""
    well_supported = {"es", "de", "fr", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ar"}
    return _CONFIDENCE_KNOWN_LANG if lang in well_supported else _CONFIDENCE_UNKNOWN_LANG


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_cached(text: str) -> Optional[str]:
    cache = _load_cache()
    return cache.get(_text_hash(text))


def _set_cached(text: str, translation: str) -> None:
    cache = _load_cache()
    cache[_text_hash(text)] = translation
    _save_cache(cache)


def _load_cache() -> dict:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _result(
    translated_text: str,
    source_language: str,
    confidence: float,
    was_translated: bool,
    error: Optional[str] = None,
) -> dict:
    return {
        "translated_text": translated_text,
        "source_language": source_language,
        "confidence": confidence,
        "was_translated": was_translated,
        "error": error,
    }

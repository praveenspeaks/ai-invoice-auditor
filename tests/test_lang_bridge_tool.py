"""
Unit tests for tools/lang_bridge_tool.py and agents/translation_agent.py
Target coverage: ≥ 85%
All network calls (GoogleTranslator) are mocked — no real API calls.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.lang_bridge_tool import (
    translate,
    is_low_confidence,
    _call_translator,
    _split_text,
    _estimate_confidence,
    _text_hash,
    _result,
    _CONFIDENCE_ENGLISH,
    _CONFIDENCE_KNOWN_LANG,
    _CONFIDENCE_UNKNOWN_LANG,
    _CONFIDENCE_CACHED,
    _LOW_CONFIDENCE_THRESHOLD,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_cache_file(tmp_path, monkeypatch):
    """Redirect cache file to tmp_path so tests don't pollute real cache."""
    import tools.lang_bridge_tool as mod
    monkeypatch.setattr(mod, "_CACHE_FILE", tmp_path / "translation_cache.json")


# ── is_low_confidence ──────────────────────────────────────────────────────────

class TestIsLowConfidence:
    def test_below_threshold_is_low(self):
        assert is_low_confidence(0.60) is True

    def test_at_threshold_is_not_low(self):
        assert is_low_confidence(_LOW_CONFIDENCE_THRESHOLD) is False

    def test_above_threshold_is_not_low(self):
        assert is_low_confidence(0.95) is False

    def test_zero_is_low(self):
        assert is_low_confidence(0.0) is True

    def test_one_is_not_low(self):
        assert is_low_confidence(1.0) is False


# ── _estimate_confidence ───────────────────────────────────────────────────────

class TestEstimateConfidence:
    def test_spanish_is_known(self):
        assert _estimate_confidence("es") == _CONFIDENCE_KNOWN_LANG

    def test_german_is_known(self):
        assert _estimate_confidence("de") == _CONFIDENCE_KNOWN_LANG

    def test_french_is_known(self):
        assert _estimate_confidence("fr") == _CONFIDENCE_KNOWN_LANG

    def test_unknown_lang_lower_confidence(self):
        assert _estimate_confidence("xx") == _CONFIDENCE_UNKNOWN_LANG

    def test_known_higher_than_unknown(self):
        assert _estimate_confidence("es") > _estimate_confidence("xx")


# ── _split_text ────────────────────────────────────────────────────────────────

class TestSplitText:
    def test_short_text_single_chunk(self):
        result = _split_text("short text", max_len=4500)
        assert result == ["short text"]

    def test_long_text_splits_at_newlines(self):
        lines = ["line"] * 100
        text = "\n".join(lines)
        chunks = _split_text(text, max_len=50)
        assert len(chunks) > 1
        assert all(len(c) <= 50 for c in chunks)

    def test_all_content_preserved(self):
        text = "line1\nline2\nline3\nline4\nline5"
        chunks = _split_text(text, max_len=15)
        rejoined = "\n".join(chunks)
        for word in ["line1", "line2", "line3"]:
            assert word in rejoined

    def test_empty_text_returns_list_with_empty(self):
        result = _split_text("", max_len=100)
        assert isinstance(result, list)

    def test_single_very_long_line_becomes_own_chunk(self):
        text = "A" * 200
        chunks = _split_text(text, max_len=100)
        assert len(chunks) >= 1


# ── _text_hash ─────────────────────────────────────────────────────────────────

class TestTextHash:
    def test_same_text_same_hash(self):
        assert _text_hash("hello") == _text_hash("hello")

    def test_different_text_different_hash(self):
        assert _text_hash("hello") != _text_hash("world")

    def test_returns_string(self):
        assert isinstance(_text_hash("test"), str)

    def test_hash_length_32(self):
        assert len(_text_hash("test")) == 32


# ── _result ────────────────────────────────────────────────────────────────────

class TestResult:
    def test_fields_populated(self):
        r = _result("text", "es", 0.85, True)
        assert r["translated_text"] == "text"
        assert r["source_language"] == "es"
        assert r["confidence"] == 0.85
        assert r["was_translated"] is True
        assert r["error"] is None

    def test_error_field_set(self):
        r = _result("text", "es", 0.65, False, error="timeout")
        assert r["error"] == "timeout"


# ── translate — English passthrough ───────────────────────────────────────────

class TestTranslateEnglish:
    def test_english_not_translated(self):
        result = translate("Invoice total: $500", "en")
        assert result["was_translated"] is False

    def test_english_confidence_is_1(self):
        result = translate("Invoice total: $500", "en")
        assert result["confidence"] == _CONFIDENCE_ENGLISH

    def test_english_text_unchanged(self):
        text = "Invoice for services rendered"
        result = translate(text, "en")
        assert result["translated_text"] == text

    def test_eng_variant_skipped(self):
        result = translate("Some text here", "eng")
        assert result["was_translated"] is False

    def test_empty_text_returns_empty(self):
        result = translate("", "es")
        assert result["translated_text"] == ""

    def test_whitespace_only_returns_whitespace(self):
        result = translate("   ", "es")
        assert result["translated_text"] == "   "


# ── translate — non-English with mock ─────────────────────────────────────────

class TestTranslateNonEnglish:
    def test_spanish_translated(self):
        with patch("tools.lang_bridge_tool._call_translator", return_value="Invoice total"):
            result = translate("Factura total", "es", use_cache=False)
        assert result["translated_text"] == "Invoice total"
        assert result["was_translated"] is True

    def test_german_translated(self):
        with patch("tools.lang_bridge_tool._call_translator", return_value="Transport boxes"):
            result = translate("Transportkisten", "de", use_cache=False)
        assert result["confidence"] == _CONFIDENCE_KNOWN_LANG

    def test_unknown_lang_lower_confidence(self):
        with patch("tools.lang_bridge_tool._call_translator", return_value="some text"):
            result = translate("some exotic text", "xx", use_cache=False)
        assert result["confidence"] == _CONFIDENCE_UNKNOWN_LANG

    def test_translation_error_returns_original(self):
        with patch("tools.lang_bridge_tool._call_translator", side_effect=Exception("API down")):
            result = translate("Factura total", "es", use_cache=False)
        assert result["translated_text"] == "Factura total"
        assert result["error"] is not None
        assert "API down" in result["error"]

    def test_translation_error_low_confidence(self):
        with patch("tools.lang_bridge_tool._call_translator", side_effect=Exception("timeout")):
            result = translate("text", "es", use_cache=False)
        assert is_low_confidence(result["confidence"]) is True


# ── translate — caching ────────────────────────────────────────────────────────

class TestTranslateCache:
    def test_cache_hit_avoids_translator_call(self):
        with patch("tools.lang_bridge_tool._call_translator", return_value="Invoice") as mock_t:
            # First call — populates cache
            translate("Factura", "es", use_cache=True)
            # Second call — should hit cache
            translate("Factura", "es", use_cache=True)
        # Translator should only be called once
        assert mock_t.call_count == 1

    def test_cache_hit_confidence_is_cached(self):
        with patch("tools.lang_bridge_tool._call_translator", return_value="Invoice"):
            translate("Factura", "es", use_cache=True)
            result = translate("Factura", "es", use_cache=True)
        assert result["confidence"] == _CONFIDENCE_CACHED

    def test_cache_disabled_always_calls_translator(self):
        with patch("tools.lang_bridge_tool._call_translator", return_value="Invoice") as mock_t:
            translate("Factura", "es", use_cache=False)
            translate("Factura", "es", use_cache=False)
        assert mock_t.call_count == 2


# ── _call_translator ───────────────────────────────────────────────────────────

class TestCallTranslator:
    def test_short_text_single_call(self):
        mock_gt = MagicMock()
        mock_gt.return_value.translate.return_value = "translated"
        with patch("deep_translator.GoogleTranslator", mock_gt):
            result = _call_translator("Hola mundo", "es")
        assert mock_gt.call_count == 1

    def test_long_text_split_into_chunks(self):
        long_text = "palabra\n" * 700   # ~5600 chars
        mock_gt = MagicMock()
        mock_gt.return_value.translate.return_value = "word"
        with patch("deep_translator.GoogleTranslator", mock_gt):
            result = _call_translator(long_text, "es")
        assert mock_gt.call_count > 1   # called multiple times for chunks

    def test_none_translation_falls_back_to_original(self):
        mock_gt = MagicMock()
        mock_gt.return_value.translate.return_value = None
        with patch("deep_translator.GoogleTranslator", mock_gt):
            result = _call_translator("original text", "es")
        assert result == "original text"


# ── translation_agent ──────────────────────────────────────────────────────────

class TestTranslationAgent:
    def test_english_state_confidence_1(self):
        from agents.translation_agent import translation_agent
        state = {
            "raw_text": "Invoice for services",
            "detected_language": "en",
            "errors": [],
            "human_review_required": False,
        }
        result = translation_agent(state)
        assert result["translation_confidence"] == 1.0
        assert result["human_review_required"] is False

    def test_spanish_translated(self):
        from agents.translation_agent import translation_agent
        with patch("agents.translation_agent.translate") as mock_translate:
            mock_translate.return_value = {
                "translated_text": "Invoice for boxes",
                "confidence": 0.85,
                "was_translated": True,
                "error": None,
            }
            state = {
                "raw_text": "Factura por cajas",
                "detected_language": "es",
                "errors": [],
                "human_review_required": False,
            }
            result = translation_agent(state)
        assert result["translated_text"] == "Invoice for boxes"
        assert result["translation_confidence"] == 0.85

    def test_low_confidence_sets_human_review(self):
        from agents.translation_agent import translation_agent
        with patch("agents.translation_agent.translate") as mock_translate, \
             patch("agents.translation_agent.is_low_confidence", return_value=True):
            mock_translate.return_value = {
                "translated_text": "some text",
                "confidence": 0.60,
                "was_translated": True,
                "error": None,
            }
            state = {
                "raw_text": "texto",
                "detected_language": "xx",
                "errors": [],
                "human_review_required": False,
            }
            result = translation_agent(state)
        assert result["human_review_required"] is True

    def test_translation_error_sets_human_review(self):
        from agents.translation_agent import translation_agent
        with patch("agents.translation_agent.translate") as mock_translate:
            mock_translate.return_value = {
                "translated_text": "original",
                "confidence": 0.65,
                "was_translated": False,
                "error": "API timeout",
            }
            state = {"raw_text": "texto", "detected_language": "es", "errors": []}
            result = translation_agent(state)
        assert result["human_review_required"] is True
        assert any("TRANSLATION_ERROR" in e for e in result["errors"])

    def test_empty_raw_text_returns_early(self):
        from agents.translation_agent import translation_agent
        state = {"raw_text": "", "detected_language": "es", "errors": []}
        result = translation_agent(state)
        assert result["translated_text"] == ""
        assert result["translation_confidence"] == 0.0

    def test_low_confidence_returns_only_new_error(self):
        """Agent returns only the new LOW_CONFIDENCE error; reducer merges old errors."""
        from agents.translation_agent import translation_agent
        with patch("agents.translation_agent.translate") as mock_translate, \
             patch("agents.translation_agent.is_low_confidence", return_value=True):
            mock_translate.return_value = {
                "translated_text": "text", "confidence": 0.60,
                "was_translated": True, "error": None,
            }
            state = {
                "raw_text": "texto",
                "detected_language": "xx",
                "errors": ["PRE_EXISTING_ERROR"],
            }
            result = translation_agent(state)
        # Only the new low-confidence error is returned; old errors NOT repeated
        assert any("LOW_TRANSLATION_CONFIDENCE" in e for e in result["errors"])
        assert "PRE_EXISTING_ERROR" not in result["errors"]

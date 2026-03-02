"""
Unit tests for:
  - tools/data_completeness_checker.py  (3-pass validation logic)
  - tools/field_extractor_tool.py       (LLM field extraction, mocked)
  - agents/data_validation_agent.py     (LangGraph node)

LLM calls are fully mocked — no real API usage.
Target coverage: ≥ 80%
"""

import pytest
from unittest.mock import patch, MagicMock

from tools.data_completeness_checker import (
    check,
    _check_required_fields,
    _check_data_types,
    _check_currency,
    _is_valid_date,
    _is_valid_float,
    _is_non_empty_str,
)
from tools.field_extractor_tool import extract_fields, _parse_json_response
from agents.data_validation_agent import data_validation_agent
from core.state import InvoiceState


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def valid_fields():
    """A fully populated, valid extracted_fields dict."""
    return {
        "invoice_no": "INV-1001",
        "invoice_date": "2025-03-14",
        "vendor_id": "VEND-001",
        "vendor_name": "Global Logistics Ltd",
        "po_number": "PO-1001",
        "currency": "USD",
        "total_amount": 3150.0,
        "line_items": [
            {"item_code": "SKU-001", "description": "Pallet Film",
             "qty": 50, "unit_price": 12.0, "total": 600.0},
        ],
    }


@pytest.fixture
def minimal_state():
    return InvoiceState(
        translated_text="Invoice text here",
        errors=[],
        discrepancies=[],
    )


# ── _is_valid_date ──────────────────────────────────────────────────────────

class TestIsValidDate:
    def test_iso_date_string(self):
        assert _is_valid_date("2025-03-14") is True

    def test_slash_format(self):
        assert _is_valid_date("03/14/2025") is True

    def test_invalid_string(self):
        assert _is_valid_date("not-a-date") is False

    def test_date_object(self):
        from datetime import date
        assert _is_valid_date(date(2025, 3, 14)) is True

    def test_none_is_invalid(self):
        assert _is_valid_date(None) is False

    def test_integer_is_invalid(self):
        assert _is_valid_date(20250314) is False


# ── _is_valid_float ─────────────────────────────────────────────────────────

class TestIsValidFloat:
    def test_float_value(self):
        assert _is_valid_float(3.14) is True

    def test_int_value(self):
        assert _is_valid_float(100) is True

    def test_string_float(self):
        assert _is_valid_float("3150.00") is True

    def test_comma_formatted(self):
        assert _is_valid_float("3,150.00") is True

    def test_invalid_string(self):
        assert _is_valid_float("abc") is False

    def test_none_is_invalid(self):
        assert _is_valid_float(None) is False


# ── _is_non_empty_str ───────────────────────────────────────────────────────

class TestIsNonEmptyStr:
    def test_non_empty(self):
        assert _is_non_empty_str("VEND-001") is True

    def test_empty_string(self):
        assert _is_non_empty_str("") is False

    def test_whitespace_only(self):
        assert _is_non_empty_str("   ") is False

    def test_non_string(self):
        assert _is_non_empty_str(123) is False


# ── _check_required_fields ──────────────────────────────────────────────────

class TestCheckRequiredFields:
    def test_all_present_no_missing(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        missing = _check_required_fields(valid_fields, rules)
        assert missing == []

    def test_missing_invoice_no(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["invoice_no"] = None
        missing = _check_required_fields(valid_fields, rules)
        assert any("invoice_no" in m for m in missing)

    def test_missing_vendor_id(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["vendor_id"] = ""
        missing = _check_required_fields(valid_fields, rules)
        assert any("vendor_id" in m for m in missing)

    def test_empty_line_items_flagged(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["line_items"] = []
        missing = _check_required_fields(valid_fields, rules)
        assert any("line_items" in m for m in missing)

    def test_line_item_missing_qty(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["line_items"][0]["qty"] = None
        missing = _check_required_fields(valid_fields, rules)
        assert any("qty" in m for m in missing)

    def test_multiple_missing_fields(self):
        from core.config import get_rules
        rules = get_rules()
        fields = {"line_items": []}
        missing = _check_required_fields(fields, rules)
        assert len(missing) >= 3  # invoice_no, invoice_date, vendor_id, currency, total_amount, line_items


# ── _check_data_types ───────────────────────────────────────────────────────

class TestCheckDataTypes:
    def test_valid_types_no_errors(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        errors = _check_data_types(valid_fields, rules)
        assert errors == []

    def test_invalid_date_format(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["invoice_date"] = "not-a-date"
        errors = _check_data_types(valid_fields, rules)
        assert any("invoice_date" in e for e in errors)

    def test_invalid_total_amount(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["total_amount"] = "three thousand"
        errors = _check_data_types(valid_fields, rules)
        assert any("total_amount" in e for e in errors)

    def test_line_item_invalid_qty(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["line_items"][0]["qty"] = "fifty"
        errors = _check_data_types(valid_fields, rules)
        assert any("qty" in e for e in errors)

    def test_none_value_skipped(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["total_amount"] = None  # None is not a type error (caught by Pass 1)
        errors = _check_data_types(valid_fields, rules)
        assert not any("total_amount" in e for e in errors)


# ── _check_currency ─────────────────────────────────────────────────────────

class TestCheckCurrency:
    def test_usd_accepted(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        status, code = _check_currency(valid_fields, rules)
        assert status == "ACCEPTED"
        assert code == "USD"

    def test_dollar_symbol_normalised(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["currency"] = "$"
        status, code = _check_currency(valid_fields, rules)
        assert status == "NORMALISED"
        assert code == "USD"

    def test_euro_symbol_normalised(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["currency"] = "€"
        status, code = _check_currency(valid_fields, rules)
        assert status == "NORMALISED"
        assert code == "EUR"

    def test_unknown_currency_rejected(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["currency"] = "XYZ"
        status, code = _check_currency(valid_fields, rules)
        assert status == "REJECTED"

    def test_missing_currency(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["currency"] = None
        status, code = _check_currency(valid_fields, rules)
        assert status == "MISSING"
        assert code is None

    def test_eur_accepted(self, valid_fields):
        from core.config import get_rules
        rules = get_rules()
        valid_fields["currency"] = "EUR"
        status, code = _check_currency(valid_fields, rules)
        assert status == "ACCEPTED"
        assert code == "EUR"


# ── check() integration ─────────────────────────────────────────────────────

class TestCheck:
    def test_valid_invoice_passes(self, valid_fields):
        result = check(valid_fields)
        assert result["status"] == "PASS"
        assert result["passed"] is True
        assert result["reject"] is False
        assert result["missing_fields"] == []
        assert result["type_errors"] == []

    def test_missing_fields_causes_fail(self, valid_fields):
        valid_fields["invoice_no"] = None
        result = check(valid_fields)
        assert result["status"] == "FAIL"
        assert result["passed"] is False
        assert any("invoice_no" in m for m in result["missing_fields"])

    def test_invalid_currency_causes_reject(self, valid_fields):
        valid_fields["currency"] = "BTC"
        result = check(valid_fields)
        assert result["reject"] is True
        assert result["currency_status"] == "REJECTED"

    def test_symbol_currency_normalised(self, valid_fields):
        valid_fields["currency"] = "$"
        result = check(valid_fields)
        assert result["currency"] == "USD"
        assert result["currency_status"] == "NORMALISED"

    def test_type_error_causes_fail(self, valid_fields):
        valid_fields["total_amount"] = "not-a-number"
        result = check(valid_fields)
        assert result["status"] == "FAIL"
        assert len(result["type_errors"]) > 0

    def test_result_has_all_keys(self, valid_fields):
        result = check(valid_fields)
        for key in ("status", "missing_fields", "type_errors",
                    "currency_status", "currency", "passed", "reject"):
            assert key in result


# ── _parse_json_response ────────────────────────────────────────────────────

class TestBuildLlm:
    def test_no_keys_raises(self, monkeypatch):
        from tools.field_extractor_tool import _build_llm
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(RuntimeError, match="No LLM API key"):
            _build_llm()

    def test_openai_key_returns_chat_openai(self, monkeypatch):
        from tools.field_extractor_tool import _build_llm
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = _build_llm()
        mock_cls.assert_called_once()

    def test_placeholder_key_skipped(self, monkeypatch):
        """Keys starting with 'your_' are treated as unset → fallback to next or raise."""
        from tools.field_extractor_tool import _build_llm
        monkeypatch.setenv("OPENAI_API_KEY", "your_openai_key_here")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(RuntimeError, match="No LLM API key"):
            _build_llm()


class TestParseJsonResponse:
    def test_plain_json(self):
        raw = '{"invoice_no": "INV-001", "total_amount": 100.0}'
        result = _parse_json_response(raw)
        assert result["invoice_no"] == "INV-001"

    def test_with_markdown_fence(self):
        raw = '```json\n{"invoice_no": "INV-001"}\n```'
        result = _parse_json_response(raw)
        assert result["invoice_no"] == "INV-001"

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_json_response("not json at all")


# ── extract_fields ──────────────────────────────────────────────────────────

class TestExtractFields:
    def test_empty_text_returns_error(self):
        result = extract_fields("")
        assert result["error"] is not None
        assert result["invoice_no"] is None

    def test_whitespace_only_returns_error(self):
        result = extract_fields("   ")
        assert result["error"] is not None

    def test_no_llm_configured_returns_error(self):
        with patch("tools.field_extractor_tool._build_llm",
                   side_effect=RuntimeError("No API key")):
            result = extract_fields("Invoice text")
        assert result["error"] is not None
        assert "No API key" in result["error"]

    def test_successful_extraction(self):
        mock_response = MagicMock()
        mock_response.content = '{"invoice_no": "INV-1001", "invoice_date": "2025-03-14", "vendor_id": "VEND-001", "vendor_name": "Global Logistics", "po_number": "PO-1001", "currency": "USD", "total_amount": 600.0, "line_items": [{"item_code": "SKU-001", "description": "Pallet Film", "qty": 50, "unit_price": 12.0, "total": 600.0}]}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("tools.field_extractor_tool._build_llm", return_value=mock_llm):
            result = extract_fields("Invoice No INV-1001 ...")

        assert result["invoice_no"] == "INV-1001"
        assert result["currency"] == "USD"
        assert result["error"] is None

    def test_llm_returns_malformed_json(self):
        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("tools.field_extractor_tool._build_llm", return_value=mock_llm):
            result = extract_fields("Some invoice text")

        assert result["error"] is not None
        assert result["invoice_no"] is None

    def test_line_items_default_to_empty_list(self):
        mock_response = MagicMock()
        mock_response.content = '{"invoice_no": "INV-1001", "line_items": null}'
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("tools.field_extractor_tool._build_llm", return_value=mock_llm):
            result = extract_fields("Invoice text")

        assert result["line_items"] == []


# ── data_validation_agent ───────────────────────────────────────────────────

def _mock_extraction(fields: dict):
    """Return an extract_fields-style result with no error."""
    return {**fields, "error": None}


class TestDataValidationAgent:
    def test_no_text_returns_fail_result(self):
        state = InvoiceState(translated_text="", raw_text="", errors=[], discrepancies=[])
        result = data_validation_agent(state)
        assert result["validation_result"]["status"] == "FAIL"
        assert result["extracted_fields"] == {}

    def test_valid_invoice_passes(self, valid_fields):
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={**valid_fields, "error": None}):
            state = InvoiceState(
                translated_text="Invoice INV-1001",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)
        assert result["validation_result"]["status"] == "PASS"
        assert result["validation_result"]["passed"] is True
        assert "recommendation" not in result or result.get("recommendation") != "REJECTED"

    def test_invalid_currency_sets_rejected(self, valid_fields):
        valid_fields["currency"] = "BTC"
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={**valid_fields, "error": None}):
            state = InvoiceState(
                translated_text="Invoice text",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)
        assert result["recommendation"] == "REJECTED"

    def test_extraction_error_added_to_errors(self):
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={"error": "LLM failed", "invoice_no": None,
                                 "invoice_date": None, "vendor_id": None,
                                 "vendor_name": None, "po_number": None,
                                 "currency": None, "total_amount": None,
                                 "line_items": []}):
            state = InvoiceState(
                translated_text="Invoice text",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)
        assert any("FIELD_EXTRACTION_ERROR" in e for e in result.get("errors", []))

    def test_extracted_fields_stored_in_state(self, valid_fields):
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={**valid_fields, "error": None}):
            state = InvoiceState(
                translated_text="Invoice text",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)
        assert result["extracted_fields"]["invoice_no"] == "INV-1001"

    def test_uses_translated_text_first(self):
        """Agent prefers translated_text over raw_text."""
        captured = {}

        def mock_extract(text):
            captured["text"] = text
            return {"error": "skip", "invoice_no": None, "invoice_date": None,
                    "vendor_id": None, "vendor_name": None, "po_number": None,
                    "currency": None, "total_amount": None, "line_items": []}

        with patch("agents.data_validation_agent.extract_fields", side_effect=mock_extract):
            state = InvoiceState(
                translated_text="Translated text",
                raw_text="Raw text",
                errors=[], discrepancies=[],
            )
            data_validation_agent(state)

        assert captured["text"] == "Translated text"

    def test_falls_back_to_raw_text(self):
        """Falls back to raw_text when translated_text is absent."""
        captured = {}

        def mock_extract(text):
            captured["text"] = text
            return {"error": "skip", "invoice_no": None, "invoice_date": None,
                    "vendor_id": None, "vendor_name": None, "po_number": None,
                    "currency": None, "total_amount": None, "line_items": []}

        with patch("agents.data_validation_agent.extract_fields", side_effect=mock_extract):
            state = InvoiceState(
                raw_text="Raw invoice text",
                errors=[], discrepancies=[],
            )
            data_validation_agent(state)

        assert captured["text"] == "Raw invoice text"

    def test_missing_fields_does_not_set_rejected(self, valid_fields):
        """Missing fields → FAIL but not REJECTED (policy is 'flag')."""
        valid_fields["invoice_no"] = None
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={**valid_fields, "error": None}):
            state = InvoiceState(
                translated_text="Invoice text",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)
        # Should not be REJECTED unless currency is invalid
        assert result.get("recommendation") != "REJECTED"

    def test_symbol_currency_normalised_into_extracted_fields(self, valid_fields):
        """Agent writes normalised ISO currency back into extracted_fields when it was a symbol."""
        valid_fields["currency"] = None  # extracted_fields has no currency
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={**valid_fields, "error": None}), \
             patch("agents.data_validation_agent.check",
                   return_value={"currency": "USD", "currency_status": "NORMALISED",
                                 "missing_fields": [], "type_errors": [],
                                 "passed": True, "reject": False, "status": "PASS"}):
            state = InvoiceState(
                translated_text="Invoice text",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)
        assert result["extracted_fields"].get("currency") == "USD"

    def test_returns_only_changed_fields(self):
        """Agent must NOT spread {**state} — returns only its own updates."""
        with patch("agents.data_validation_agent.extract_fields",
                   return_value={"error": None, "invoice_no": None,
                                 "invoice_date": None, "vendor_id": None,
                                 "vendor_name": None, "po_number": None,
                                 "currency": None, "total_amount": None,
                                 "line_items": []}):
            state = InvoiceState(
                translated_text="text",
                file_path="some/path.pdf",
                errors=[], discrepancies=[],
            )
            result = data_validation_agent(state)

        # Agent should NOT echo back file_path (it's not its field)
        assert "file_path" not in result
        # Agent SHOULD return extracted_fields and validation_result
        assert "extracted_fields" in result
        assert "validation_result" in result

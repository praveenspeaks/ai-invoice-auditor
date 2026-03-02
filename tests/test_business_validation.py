"""
Unit tests for:
  - tools/business_validation_tool.py
  - agents/business_validation_agent.py
  - tools/insight_reporter_tool.py
  - agents/reporting_agent.py

All HTTP calls and filesystem writes are mocked.
Target coverage: ≥ 80%
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.business_validation_tool import (
    validate,
    _pct_diff,
    _compare_field,
    _safe_float,
    _fetch_po,
    _compare_line_items,
)
from agents.business_validation_agent import business_validation_agent
from tools.insight_reporter_tool import generate_report, _build_field_rows
from agents.reporting_agent import reporting_agent
from core.state import InvoiceState


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_erp_po():
    return {
        "po_number": "PO-1001",
        "vendor_id": "VEND-001",
        "vendor_name": "Global Logistics Ltd",
        "currency": "USD",
        "line_items": [
            {"item_code": "SKU-001", "description": "Pallet Film",
             "qty": 50, "unit_price": 12.00, "currency": "USD"},
            {"item_code": "SKU-002", "description": "Industrial Gloves",
             "qty": 120, "unit_price": 3.00, "currency": "USD"},
        ],
        "subtotal": 960.0,
        "total_tax": 96.0,
        "grand_total": 1056.0,
    }


@pytest.fixture
def extracted_fields():
    return {
        "invoice_no": "INV-1001",
        "invoice_date": "2025-03-14",
        "vendor_id": "VEND-001",
        "vendor_name": "Global Logistics Ltd",
        "po_number": "PO-1001",
        "currency": "USD",
        "total_amount": 1056.0,
        "line_items": [
            {"item_code": "SKU-001", "description": "Pallet Film",
             "qty": 50, "unit_price": 12.00, "total": 600.0},
            {"item_code": "SKU-002", "description": "Industrial Gloves",
             "qty": 120, "unit_price": 3.00, "total": 360.0},
        ],
    }


@pytest.fixture
def full_state(extracted_fields, sample_erp_po):
    return InvoiceState(
        file_path="data/incoming/INV_EN_001.pdf",
        file_format="pdf",
        detected_language="en",
        translated_text="Invoice text",
        translation_confidence=0.97,
        extracted_fields=extracted_fields,
        validation_result={
            "status": "PASS", "missing_fields": [], "type_errors": [],
            "currency_status": "ACCEPTED", "currency": "USD",
            "passed": True, "reject": False,
        },
        erp_data=sample_erp_po,
        discrepancies=[],
        recommendation="AUTO_APPROVED",
        human_review_required=False,
        errors=[],
    )


# ── _pct_diff ──────────────────────────────────────────────────────────────

class TestPctDiff:
    def test_exact_match(self):
        assert _pct_diff(100.0, 100.0) == 0.0

    def test_ten_percent_over(self):
        assert abs(_pct_diff(110.0, 100.0) - 10.0) < 0.001

    def test_negative_diff(self):
        assert _pct_diff(90.0, 100.0) < 0

    def test_zero_erp_val_nonzero_invoice(self):
        assert _pct_diff(5.0, 0.0) == 100.0

    def test_both_zero(self):
        assert _pct_diff(0.0, 0.0) == 0.0


# ── _safe_float ────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_float_passthrough(self):
        assert _safe_float(3.14) == 3.14

    def test_int_converted(self):
        assert _safe_float(100) == 100.0

    def test_string_float(self):
        assert _safe_float("12.50") == 12.5

    def test_comma_formatted(self):
        assert _safe_float("1,200.00") == 1200.0

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_invalid_string_returns_none(self):
        assert _safe_float("abc") is None


# ── _compare_field ─────────────────────────────────────────────────────────

class TestCompareField:
    def test_exact_match(self):
        r = _compare_field("SKU-1", "qty", 50, 50, 0)
        assert r["status"] == "MATCH"

    def test_within_tolerance(self):
        r = _compare_field("SKU-1", "unit_price", 12.5, 12.0, 5)
        assert r["status"] == "WITHIN_TOLERANCE"

    def test_discrepancy(self):
        r = _compare_field("SKU-1", "unit_price", 15.0, 12.0, 5)
        assert r["status"] == "DISCREPANCY"

    def test_diff_pct_correct(self):
        r = _compare_field("SKU-1", "unit_price", 13.0, 12.0, 5)
        assert abs(r["diff_pct"] - 8.333) < 0.01

    def test_result_has_all_keys(self):
        r = _compare_field("SKU-1", "qty", 10, 10, 0)
        for k in ("item_code", "field", "invoice_val", "erp_val", "diff_pct", "status"):
            assert k in r


# ── _fetch_po ──────────────────────────────────────────────────────────────

class TestFetchPo:
    def test_404_returns_unregistered(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("httpx.get", return_value=mock_resp):
            data, error = _fetch_po("VEND-999", "PO-999")
        assert data is None
        assert error == "UNREGISTERED_INVOICE"

    def test_success_returns_data(self, sample_erp_po):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_erp_po
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            data, error = _fetch_po("VEND-001", "PO-1001")
        assert error is None
        assert data["po_number"] == "PO-1001"

    def test_timeout_returns_error(self):
        import httpx as httpx_mod
        with patch("httpx.get", side_effect=httpx_mod.TimeoutException("timeout")):
            data, error = _fetch_po("VEND-001", "PO-1001")
        assert data is None
        assert "TIMEOUT" in error

    def test_generic_exception(self):
        with patch("httpx.get", side_effect=Exception("network down")):
            data, error = _fetch_po("VEND-001", "PO-1001")
        assert "ERP_ERROR" in error


# ── _compare_line_items ────────────────────────────────────────────────────

class TestCompareLineItems:
    def test_matching_items_all_match(self, extracted_fields, sample_erp_po):
        from core.config import get_rules
        rules = get_rules()
        result = _compare_line_items(
            extracted_fields["line_items"],
            sample_erp_po["line_items"],
            rules.tolerances,
        )
        statuses = {r["status"] for r in result}
        assert "DISCREPANCY" not in statuses

    def test_price_discrepancy_detected(self, extracted_fields, sample_erp_po):
        from core.config import get_rules
        rules = get_rules()
        extracted_fields["line_items"][0]["unit_price"] = 15.0  # 25% over ERP 12.00
        result = _compare_line_items(
            extracted_fields["line_items"],
            sample_erp_po["line_items"],
            rules.tolerances,
        )
        disc = [r for r in result if r["field"] == "unit_price" and r["item_code"] == "SKU-001"]
        assert disc[0]["status"] == "DISCREPANCY"

    def test_unknown_item_code_flagged(self, sample_erp_po):
        from core.config import get_rules
        rules = get_rules()
        invoice_items = [{"item_code": "SKU-UNKNOWN", "qty": 10, "unit_price": 5.0, "total": 50.0}]
        result = _compare_line_items(invoice_items, sample_erp_po["line_items"], rules.tolerances)
        assert any(r["status"] == "DISCREPANCY" for r in result)

    def test_empty_invoice_items(self, sample_erp_po):
        from core.config import get_rules
        rules = get_rules()
        result = _compare_line_items([], sample_erp_po["line_items"], rules.tolerances)
        assert result == []


# ── validate() ────────────────────────────────────────────────────────────

class TestValidate:
    def test_missing_vendor_id_returns_error(self, extracted_fields):
        extracted_fields["vendor_id"] = ""
        result = validate(extracted_fields)
        assert result["error"] is not None
        assert "MISSING_IDS" in result["error"]

    def test_unregistered_invoice_returns_error(self, extracted_fields):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("httpx.get", return_value=mock_resp):
            result = validate(extracted_fields)
        assert result["error"] == "UNREGISTERED_INVOICE"

    def test_successful_validation(self, extracted_fields, sample_erp_po):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_erp_po
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = validate(extracted_fields)
        assert result["error"] is None
        assert isinstance(result["discrepancies"], list)

    def test_erp_error_propagated(self, extracted_fields):
        import httpx as httpx_mod
        with patch("httpx.get", side_effect=httpx_mod.TimeoutException("t/o")):
            result = validate(extracted_fields)
        assert "TIMEOUT" in result["error"]


# ── business_validation_agent ──────────────────────────────────────────────

class TestBusinessValidationAgent:
    def test_already_rejected_returns_empty(self):
        state = InvoiceState(
            recommendation="REJECTED",
            extracted_fields={}, validation_result={},
            translation_confidence=0.0, errors=[], discrepancies=[],
        )
        result = business_validation_agent(state)
        assert result == {}

    def test_auto_approved_when_all_ok(self, extracted_fields, sample_erp_po):
        with patch("agents.business_validation_agent.validate") as mock_v:
            mock_v.return_value = {"erp_data": sample_erp_po, "discrepancies": [], "error": None}
            state = InvoiceState(
                extracted_fields=extracted_fields,
                validation_result={"passed": True},
                translation_confidence=0.97,
                recommendation="",
                errors=[], discrepancies=[],
            )
            result = business_validation_agent(state)
        assert result["recommendation"] == "AUTO_APPROVED"
        assert result["human_review_required"] is False

    def test_manual_review_when_discrepancy(self, extracted_fields, sample_erp_po):
        discrepancy = {"item_code": "SKU-001", "field": "unit_price",
                       "invoice_val": 15.0, "erp_val": 12.0,
                       "diff_pct": 25.0, "status": "DISCREPANCY"}
        with patch("agents.business_validation_agent.validate") as mock_v:
            mock_v.return_value = {
                "erp_data": sample_erp_po,
                "discrepancies": [discrepancy],
                "error": None,
            }
            state = InvoiceState(
                extracted_fields=extracted_fields,
                validation_result={"passed": True},
                translation_confidence=0.97,
                recommendation="",
                errors=[], discrepancies=[],
            )
            result = business_validation_agent(state)
        assert result["recommendation"] == "MANUAL_REVIEW"
        assert result["human_review_required"] is True

    def test_manual_review_when_low_confidence(self, extracted_fields, sample_erp_po):
        with patch("agents.business_validation_agent.validate") as mock_v:
            mock_v.return_value = {"erp_data": sample_erp_po, "discrepancies": [], "error": None}
            state = InvoiceState(
                extracted_fields=extracted_fields,
                validation_result={"passed": True},
                translation_confidence=0.80,  # below 0.95 threshold
                recommendation="",
                errors=[], discrepancies=[],
            )
            result = business_validation_agent(state)
        assert result["recommendation"] == "MANUAL_REVIEW"

    def test_erp_error_adds_error_entry(self, extracted_fields):
        with patch("agents.business_validation_agent.validate") as mock_v:
            mock_v.return_value = {"erp_data": {}, "discrepancies": [], "error": "ERP_TIMEOUT"}
            state = InvoiceState(
                extracted_fields=extracted_fields,
                validation_result={"passed": True},
                translation_confidence=0.97,
                recommendation="",
                errors=[], discrepancies=[],
            )
            result = business_validation_agent(state)
        assert any("ERP_VALIDATION_ERROR" in e for e in result.get("errors", []))

    def test_returns_only_changed_fields(self, extracted_fields, sample_erp_po):
        with patch("agents.business_validation_agent.validate") as mock_v:
            mock_v.return_value = {"erp_data": sample_erp_po, "discrepancies": [], "error": None}
            state = InvoiceState(
                extracted_fields=extracted_fields,
                validation_result={"passed": True},
                translation_confidence=0.97,
                recommendation="",
                errors=[], discrepancies=[],
            )
            result = business_validation_agent(state)
        assert "file_path" not in result
        assert "recommendation" in result


# ── _build_field_rows ──────────────────────────────────────────────────────

class TestBuildFieldRows:
    def test_all_present_all_pass(self):
        fields = {
            "invoice_no": "INV-1001", "invoice_date": "2025-03-14",
            "vendor_id": "VEND-001", "vendor_name": "Acme",
            "po_number": "PO-1001", "total_amount": 100.0,
        }
        rows = _build_field_rows(fields, [])
        assert all(r["status"] == "PASS" for r in rows)

    def test_missing_invoice_no(self):
        rows = _build_field_rows({}, ["header.invoice_no"])
        inv_row = next(r for r in rows if r["name"] == "invoice_no")
        assert inv_row["status"] == "MISSING"


# ── generate_report ────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_generates_html_file(self, tmp_path, full_state):
        with patch.dict("os.environ", {"REPORTS_DIR": str(tmp_path)}):
            path = generate_report(dict(full_state))
        assert path != ""
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "INV-1001" in content

    def test_recommendation_in_report(self, tmp_path, full_state):
        with patch.dict("os.environ", {"REPORTS_DIR": str(tmp_path)}):
            path = generate_report(dict(full_state))
        content = Path(path).read_text(encoding="utf-8")
        assert "AUTO_APPROVED" in content

    def test_discrepancies_shown(self, tmp_path, full_state):
        full_state["discrepancies"] = [
            {"item_code": "SKU-001", "field": "unit_price",
             "invoice_val": 15.0, "erp_val": 12.0, "diff_pct": 25.0,
             "status": "DISCREPANCY"}
        ]
        with patch.dict("os.environ", {"REPORTS_DIR": str(tmp_path)}):
            path = generate_report(dict(full_state))
        content = Path(path).read_text(encoding="utf-8")
        assert "DISCREPANCY" in content

    def test_report_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "reports"
        state = {"recommendation": "MANUAL_REVIEW", "errors": []}
        with patch.dict("os.environ", {"REPORTS_DIR": str(new_dir)}):
            path = generate_report(state)
        assert new_dir.exists()

    def test_empty_state_generates_report(self, tmp_path):
        with patch.dict("os.environ", {"REPORTS_DIR": str(tmp_path)}):
            path = generate_report({})
        assert path != ""


# ── reporting_agent ────────────────────────────────────────────────────────

class TestReportingAgent:
    def test_sets_report_path(self, tmp_path, full_state):
        with patch("agents.reporting_agent.generate_report", return_value="/tmp/report.html"):
            result = reporting_agent(full_state)
        assert result["report_path"] == "/tmp/report.html"

    def test_failed_report_adds_error(self, full_state):
        with patch("agents.reporting_agent.generate_report", return_value=""):
            result = reporting_agent(full_state)
        assert result["report_path"] == ""
        assert any("REPORT_ERROR" in e for e in result.get("errors", []))

    def test_returns_only_changed_fields(self, full_state):
        with patch("agents.reporting_agent.generate_report", return_value="/tmp/r.html"):
            result = reporting_agent(full_state)
        assert "file_path" not in result
        assert "report_path" in result


# ── pipeline integration ───────────────────────────────────────────────────

class TestPipelineStubsRemoved:
    def test_pipeline_imports_real_agents(self):
        from workflows.invoice_pipeline import build_pipeline
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_business_stub_replaced(self):
        """Confirm stub is gone from pipeline module."""
        import workflows.invoice_pipeline as mod
        from agents.business_validation_agent import business_validation_agent as real
        # The pipeline should import from agents, not define its own stub
        assert not hasattr(mod, '_is_stub_business')

    def test_reporting_stub_replaced(self):
        import workflows.invoice_pipeline as mod
        assert not hasattr(mod, '_is_stub_reporting')

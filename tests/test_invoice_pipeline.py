"""
Unit tests for workflows/invoice_pipeline.py and agents/extractor_agent.py.
Tests verify graph structure, routing logic, and the extractor agent node.
LLM and heavy tool calls are mocked.
Target coverage: ≥ 80%
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.state import InvoiceState, initial_state
from agents.extractor_agent import extractor_agent


# ── Paths ──────────────────────────────────────────────────────────────────────

SAMPLE_PDF = str(Path(__file__).parent.parent / "data" / "incoming" / "INV_EN_001.pdf")


# ── extractor_agent ────────────────────────────────────────────────────────────

class TestExtractorAgent:
    def test_empty_file_path_returns_error(self):
        state = initial_state("", {}, "pdf")
        result = extractor_agent(state)
        assert any("file_path is empty" in e for e in result["errors"])

    def test_successful_extraction_sets_raw_text(self):
        with patch("agents.extractor_agent.harvest") as mock_h:
            mock_h.return_value = {
                "raw_text": "Invoice No: INV-001",
                "detected_language": "en",
                "file_format": "pdf",
                "error": None,
            }
            state = initial_state("invoice.pdf", {}, "pdf")
            result = extractor_agent(state)
        assert result["raw_text"] == "Invoice No: INV-001"

    def test_successful_extraction_sets_language(self):
        with patch("agents.extractor_agent.harvest") as mock_h:
            mock_h.return_value = {
                "raw_text": "Factura No: FAC-001",
                "detected_language": "es",
                "file_format": "pdf",
                "error": None,
            }
            state = initial_state("invoice.pdf", {}, "pdf")
            result = extractor_agent(state)
        assert result["detected_language"] == "es"

    def test_extraction_error_appended_to_errors(self):
        with patch("agents.extractor_agent.harvest") as mock_h:
            mock_h.return_value = {
                "raw_text": "",
                "detected_language": "en",
                "file_format": "unknown",
                "error": "File corrupted",
            }
            state = initial_state("bad.pdf", {}, "pdf")
            result = extractor_agent(state)
        assert any("EXTRACTION_ERROR" in e for e in result["errors"])

    def test_extraction_error_preserves_previous_errors(self):
        with patch("agents.extractor_agent.harvest") as mock_h:
            mock_h.return_value = {
                "raw_text": "", "detected_language": "en",
                "file_format": "unknown", "error": "oops"
            }
            state = initial_state("bad.pdf", {}, "pdf")
            state["errors"] = ["PRE_EXISTING"]
            result = extractor_agent(state)
        assert "PRE_EXISTING" in result["errors"]

    def test_real_pdf_extraction(self):
        """Integration: real PDF file, no mocks."""
        state = initial_state(SAMPLE_PDF, {}, "pdf")
        result = extractor_agent(state)
        assert result["raw_text"] != "" or result["errors"]  # either text or error
        assert result["detected_language"] in ("en", "unknown", "")

    def test_file_format_updated_from_harvest(self):
        with patch("agents.extractor_agent.harvest") as mock_h:
            mock_h.return_value = {
                "raw_text": "text", "detected_language": "en",
                "file_format": "docx", "error": None,
            }
            state = initial_state("inv.docx", {}, "")
            result = extractor_agent(state)
        assert result["file_format"] == "docx"


# ── Pipeline graph structure ───────────────────────────────────────────────────

class TestBuildPipeline:
    def test_pipeline_builds_without_error(self):
        from workflows.invoice_pipeline import build_pipeline
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_pipeline_has_invoke_method(self):
        from workflows.invoice_pipeline import build_pipeline
        pipeline = build_pipeline()
        assert hasattr(pipeline, "invoke")


# ── Routing functions ──────────────────────────────────────────────────────────

class TestRoutingAfterMonitor:
    def test_no_file_path_routes_to_end(self):
        from workflows.invoice_pipeline import _route_after_monitor
        from langgraph.graph import END
        state = InvoiceState(errors=[], discrepancies=[])
        assert _route_after_monitor(state) == END

    def test_with_file_path_routes_to_extract(self):
        from workflows.invoice_pipeline import _route_after_monitor
        state = InvoiceState(file_path="invoice.pdf", errors=[], discrepancies=[])
        assert _route_after_monitor(state) == "extract"

    def test_empty_string_file_path_routes_to_end(self):
        from workflows.invoice_pipeline import _route_after_monitor
        from langgraph.graph import END
        state = InvoiceState(file_path="", errors=[], discrepancies=[])
        assert _route_after_monitor(state) == END


class TestRoutingAfterValidation:
    def test_rejected_routes_to_report(self):
        from workflows.invoice_pipeline import _route_after_validation
        state = InvoiceState(recommendation="REJECTED", errors=[], discrepancies=[])
        assert _route_after_validation(state) == "report"

    def test_no_recommendation_routes_to_business_validation(self):
        from workflows.invoice_pipeline import _route_after_validation
        state = InvoiceState(recommendation="", errors=[], discrepancies=[])
        assert _route_after_validation(state) == "validate_business"

    def test_manual_review_routes_to_business_validation(self):
        from workflows.invoice_pipeline import _route_after_validation
        state = InvoiceState(recommendation="MANUAL_REVIEW", errors=[], discrepancies=[])
        assert _route_after_validation(state) == "validate_business"

    def test_auto_approved_routes_to_business_validation(self):
        from workflows.invoice_pipeline import _route_after_validation
        state = InvoiceState(recommendation="AUTO_APPROVED", errors=[], discrepancies=[])
        assert _route_after_validation(state) == "validate_business"


# ── Stub agents ────────────────────────────────────────────────────────────────

class TestStubAgents:
    def test_data_validation_stub_passes_state(self):
        from workflows.invoice_pipeline import data_validation_agent
        state = InvoiceState(errors=[], discrepancies=[], validation_result={"x": 1})
        result = data_validation_agent(state)
        assert result["validation_result"] == {"x": 1}

    def test_business_validation_stub_sets_discrepancies(self):
        from workflows.invoice_pipeline import business_validation_agent
        state = InvoiceState(errors=[], discrepancies=[])
        result = business_validation_agent(state)
        assert result["discrepancies"] == []

    def test_reporting_stub_sets_report_path(self):
        from workflows.invoice_pipeline import reporting_agent
        state = InvoiceState(errors=[], discrepancies=[])
        result = reporting_agent(state)
        assert "report_path" in result


# ── run_pipeline ───────────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_run_with_file_path_returns_state(self):
        from workflows.invoice_pipeline import run_pipeline
        with patch("agents.extractor_agent.harvest") as mock_h, \
             patch("agents.translation_agent.translate") as mock_t:
            mock_h.return_value = {
                "raw_text": "Invoice", "detected_language": "en",
                "file_format": "pdf", "error": None,
            }
            mock_t.return_value = {
                "translated_text": "Invoice", "confidence": 1.0,
                "was_translated": False, "error": None,
            }
            result = run_pipeline(SAMPLE_PDF)
        assert "file_path" in result

    def test_run_without_file_path_returns_state(self):
        from workflows.invoice_pipeline import run_pipeline
        # polling mode — inbox empty in test environment
        with patch("tools.invoice_watcher_tool.watch", return_value=[]):
            result = run_pipeline()
        assert isinstance(result, dict)

    def test_run_pipeline_errors_list_present(self):
        from workflows.invoice_pipeline import run_pipeline
        with patch("agents.extractor_agent.harvest") as mock_h, \
             patch("agents.translation_agent.translate") as mock_t:
            mock_h.return_value = {
                "raw_text": "text", "detected_language": "en",
                "file_format": "pdf", "error": None,
            }
            mock_t.return_value = {
                "translated_text": "text", "confidence": 1.0,
                "was_translated": False, "error": None,
            }
            result = run_pipeline(SAMPLE_PDF)
        assert "errors" in result

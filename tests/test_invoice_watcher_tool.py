"""
Unit tests for tools/invoice_watcher_tool.py and agents/invoice_monitor_agent.py
Target coverage: ≥ 85%
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from tools.invoice_watcher_tool import (
    watch,
    mark_processed,
    reset_registry,
    is_processed,
    _is_invoice_file,
    _load_meta,
    _get_format,
    _load_registry,
    _save_registry,
    INVOICE_EXTENSIONS,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def inbox(tmp_path):
    """Empty inbox directory."""
    d = tmp_path / "incoming"
    d.mkdir()
    return d


@pytest.fixture
def registry(tmp_path):
    """Registry file path inside tmp_path."""
    return str(tmp_path / "registry.json")


@pytest.fixture
def pdf_invoice(inbox):
    """Single PDF invoice with meta.json sidecar."""
    pdf = inbox / "INV_EN_001.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    meta = inbox / "INV_EN_001.meta.json"
    meta.write_text(json.dumps({
        "sender": "test@example.com",
        "subject": "Invoice INV-001",
        "received_timestamp": "2025-03-14T09:32:00Z",
        "language": "en",
        "attachments": ["INV_EN_001.pdf"]
    }))
    return pdf


@pytest.fixture
def es_invoice(inbox):
    """Spanish PDF invoice."""
    pdf = inbox / "INV_ES_003.pdf"
    pdf.write_bytes(b"%PDF-1.4 spanish")
    meta = inbox / "INV_ES_003.meta.json"
    meta.write_text(json.dumps({"language": "es", "sender": "vendor@es.com"}))
    return pdf


# ── _is_invoice_file ───────────────────────────────────────────────────────────

class TestIsInvoiceFile:
    def test_pdf_is_invoice(self, tmp_path):
        f = tmp_path / "inv.pdf"
        f.write_bytes(b"x")
        assert _is_invoice_file(f) is True

    def test_docx_is_invoice(self, tmp_path):
        f = tmp_path / "inv.docx"
        f.write_bytes(b"x")
        assert _is_invoice_file(f) is True

    def test_png_is_invoice(self, tmp_path):
        f = tmp_path / "inv.png"
        f.write_bytes(b"x")
        assert _is_invoice_file(f) is True

    def test_jpg_is_invoice(self, tmp_path):
        f = tmp_path / "inv.jpg"
        f.write_bytes(b"x")
        assert _is_invoice_file(f) is True

    def test_json_is_not_invoice(self, tmp_path):
        f = tmp_path / "meta.json"
        f.write_text("{}")
        assert _is_invoice_file(f) is False

    def test_txt_is_not_invoice(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        assert _is_invoice_file(f) is False

    def test_directory_is_not_invoice(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        assert _is_invoice_file(d) is False

    def test_uppercase_extension(self, tmp_path):
        f = tmp_path / "INV.PDF"
        f.write_bytes(b"x")
        assert _is_invoice_file(f) is True


# ── _load_meta ─────────────────────────────────────────────────────────────────

class TestLoadMeta:
    def test_loads_existing_meta(self, inbox, pdf_invoice):
        meta = _load_meta(pdf_invoice)
        assert meta["language"] == "en"
        assert meta["sender"] == "test@example.com"

    def test_returns_empty_dict_when_no_meta(self, inbox):
        f = inbox / "no_meta.pdf"
        f.write_bytes(b"x")
        assert _load_meta(f) == {}

    def test_returns_empty_dict_on_corrupt_json(self, inbox):
        f = inbox / "bad.pdf"
        f.write_bytes(b"x")
        meta_file = inbox / "bad.meta.json"
        meta_file.write_text("NOT VALID JSON {{")
        result = _load_meta(f)
        assert result == {}


# ── _get_format ────────────────────────────────────────────────────────────────

class TestGetFormat:
    def test_pdf(self, tmp_path):
        assert _get_format(tmp_path / "f.pdf") == "pdf"

    def test_docx(self, tmp_path):
        assert _get_format(tmp_path / "f.docx") == "docx"

    def test_doc(self, tmp_path):
        assert _get_format(tmp_path / "f.doc") == "docx"

    def test_png(self, tmp_path):
        assert _get_format(tmp_path / "f.png") == "image"

    def test_jpg(self, tmp_path):
        assert _get_format(tmp_path / "f.jpg") == "image"

    def test_tiff(self, tmp_path):
        assert _get_format(tmp_path / "f.tiff") == "image"


# ── _load_registry / _save_registry ───────────────────────────────────────────

class TestRegistry:
    def test_load_nonexistent_returns_empty(self, tmp_path):
        r = _load_registry(tmp_path / "nonexistent.json")
        assert r == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "reg.json"
        data = {"/some/path.pdf": True}
        _save_registry(path, data)
        loaded = _load_registry(path)
        assert loaded == data

    def test_load_corrupt_json_returns_empty(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{{invalid}}")
        assert _load_registry(path) == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "reg.json"
        _save_registry(path, {"key": True})
        assert path.exists()


# ── watch ──────────────────────────────────────────────────────────────────────

class TestWatch:
    def test_empty_inbox_returns_empty_list(self, inbox, registry):
        result = watch(str(inbox), registry)
        assert result == []

    def test_detects_single_pdf(self, inbox, registry, pdf_invoice):
        result = watch(str(inbox), registry)
        assert len(result) == 1
        assert result[0]["file_format"] == "pdf"

    def test_descriptor_has_file_path(self, inbox, registry, pdf_invoice):
        result = watch(str(inbox), registry)
        assert "file_path" in result[0]
        assert "INV_EN_001.pdf" in result[0]["file_path"]

    def test_descriptor_has_meta(self, inbox, registry, pdf_invoice):
        result = watch(str(inbox), registry)
        assert result[0]["meta"]["language"] == "en"

    def test_second_watch_skips_processed(self, inbox, registry, pdf_invoice):
        watch(str(inbox), registry)        # first call processes
        result = watch(str(inbox), registry)  # second call: already processed
        assert result == []

    def test_detects_multiple_invoices(self, inbox, registry, pdf_invoice, es_invoice):
        result = watch(str(inbox), registry)
        assert len(result) == 2

    def test_skips_meta_json_files(self, inbox, registry):
        # Only a meta.json, no actual invoice
        meta = inbox / "INV_EN_001.meta.json"
        meta.write_text(json.dumps({"language": "en"}))
        result = watch(str(inbox), registry)
        assert result == []

    def test_nonexistent_dir_returns_empty(self, tmp_path, registry):
        result = watch(str(tmp_path / "no_such_dir"), registry)
        assert result == []

    def test_docx_file_detected(self, inbox, registry):
        f = inbox / "INV_DE_004.docx"
        f.write_bytes(b"PK docx")
        result = watch(str(inbox), registry)
        assert result[0]["file_format"] == "docx"

    def test_image_file_detected(self, inbox, registry):
        f = inbox / "INV_EN_005_scan.png"
        f.write_bytes(b"\x89PNG")
        result = watch(str(inbox), registry)
        assert result[0]["file_format"] == "image"


# ── mark_processed / is_processed / reset_registry ────────────────────────────

class TestRegistryHelpers:
    def test_mark_processed_sets_file(self, inbox, registry, pdf_invoice):
        mark_processed(str(pdf_invoice), registry)
        assert is_processed(str(pdf_invoice), registry) is True

    def test_unregistered_file_not_processed(self, inbox, registry, pdf_invoice):
        assert is_processed(str(pdf_invoice), registry) is False

    def test_reset_clears_registry(self, inbox, registry, pdf_invoice):
        mark_processed(str(pdf_invoice), registry)
        reset_registry(registry)
        assert is_processed(str(pdf_invoice), registry) is False

    def test_watch_then_is_processed(self, inbox, registry, pdf_invoice):
        watch(str(inbox), registry)
        assert is_processed(str(pdf_invoice), registry) is True


# ── invoice_monitor_agent ──────────────────────────────────────────────────────

class TestInvoiceMonitorAgent:
    def test_manual_trigger_uses_existing_file_path(self, inbox, pdf_invoice):
        from agents.invoice_monitor_agent import invoice_monitor_agent
        state = {"file_path": str(pdf_invoice), "errors": []}
        result = invoice_monitor_agent(state)
        assert result["file_path"] == str(pdf_invoice)

    def test_manual_trigger_loads_meta(self, inbox, pdf_invoice):
        from agents.invoice_monitor_agent import invoice_monitor_agent
        state = {"file_path": str(pdf_invoice), "errors": []}
        result = invoice_monitor_agent(state)
        assert result["meta"].get("language") == "en"

    def test_manual_trigger_sets_format(self, inbox, pdf_invoice):
        from agents.invoice_monitor_agent import invoice_monitor_agent
        state = {"file_path": str(pdf_invoice), "errors": []}
        result = invoice_monitor_agent(state)
        assert result["file_format"] == "pdf"

    def test_polling_trigger_finds_new_invoice(self, inbox, registry, pdf_invoice, monkeypatch):
        from agents.invoice_monitor_agent import invoice_monitor_agent
        monkeypatch.setenv("INCOMING_DIR", str(inbox))
        monkeypatch.setenv("PROCESSED_REGISTRY", registry)
        # Reload env-dependent module values
        import importlib, agents.invoice_monitor_agent as mod
        mod._INCOMING_DIR = str(inbox)
        mod._REGISTRY_PATH = registry

        state = {"errors": []}
        result = invoice_monitor_agent(state)
        assert result["file_path"] is not None
        assert "INV_EN_001" in result["file_path"]

    def test_polling_no_invoices_returns_unchanged_state(self, inbox, registry, monkeypatch):
        from agents.invoice_monitor_agent import invoice_monitor_agent
        import agents.invoice_monitor_agent as mod
        mod._INCOMING_DIR = str(inbox)
        mod._REGISTRY_PATH = registry
        state = {"errors": ["pre-existing error"]}
        result = invoice_monitor_agent(state)
        # No new invoices → state unchanged
        assert result["errors"] == ["pre-existing error"]

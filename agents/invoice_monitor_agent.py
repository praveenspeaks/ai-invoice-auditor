"""
Invoice Monitor Agent — AI Invoice Auditor
LangGraph node that detects new invoices and initialises an InvoiceState
for each one. Used as the entry point of the invoice pipeline.
"""

import os
from pathlib import Path

from core.logger import get_logger
from core.state import InvoiceState, initial_state
from tools.invoice_watcher_tool import watch

logger = get_logger(__name__)

_INCOMING_DIR = os.environ.get("INCOMING_DIR", "./data/incoming")
_REGISTRY_PATH = os.environ.get("PROCESSED_REGISTRY", "./data/processed_registry.json")


def invoice_monitor_agent(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node — Invoice Monitor Agent.

    Scans the incoming directory for new invoice files and populates
    the InvoiceState with file_path, file_format, and meta from the
    .meta.json sidecar.

    If a file_path is already set in state (manual trigger via UI),
    the agent skips polling and uses that path directly.
    """
    # Manual trigger: file_path already provided (e.g. from Streamlit upload)
    if state.get("file_path"):
        logger.info("Manual trigger: %s", state["file_path"])
        return _enrich_state(state, state["file_path"])

    # Polling trigger: scan incoming directory
    new_invoices = watch(
        incoming_dir=_INCOMING_DIR,
        registry_path=_REGISTRY_PATH,
    )

    if not new_invoices:
        logger.info("No new invoices found in %s", _INCOMING_DIR)
        return state

    # Process the first detected invoice (pipeline is per-invoice)
    invoice = new_invoices[0]
    logger.info("Processing invoice: %s", invoice["file_path"])

    updated = initial_state(
        file_path=invoice["file_path"],
        meta=invoice["meta"],
        file_format=invoice["file_format"],
    )
    # Carry forward any existing errors from state
    updated["errors"] = list(state.get("errors", []))
    return updated


def _enrich_state(state: InvoiceState, file_path: str) -> InvoiceState:
    """Fill in meta and file_format for a manually triggered state."""
    from tools.invoice_watcher_tool import _load_meta, _get_format
    path = Path(file_path)
    if not state.get("meta"):
        state["meta"] = _load_meta(path)
    if not state.get("file_format"):
        state["file_format"] = _get_format(path)
    return state

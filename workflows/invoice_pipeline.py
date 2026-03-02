"""
Invoice Pipeline — AI Invoice Auditor
LangGraph StateGraph wiring all 6 core agents into a sequential pipeline:

    monitor → extract → translate → validate_data → validate_business → report

Stub nodes: business_validation_agent, reporting_agent (replaced in later tasks).
"""

import os
from typing import Literal

from langgraph.graph import StateGraph, END

from core.logger import get_logger
from core.state import InvoiceState, initial_state
from agents.invoice_monitor_agent import invoice_monitor_agent
from agents.extractor_agent import extractor_agent
from agents.translation_agent import translation_agent
from agents.data_validation_agent import data_validation_agent

logger = get_logger(__name__)


def business_validation_agent(state: InvoiceState) -> InvoiceState:
    """Stub — implemented in Task 2.2 (EP02-02)."""
    logger.debug("business_validation_agent: stub")
    return {**state, "erp_data": state.get("erp_data", {}), "discrepancies": []}


def reporting_agent(state: InvoiceState) -> InvoiceState:
    """Stub — implemented in Task 2.3 (EP02-03)."""
    logger.debug("reporting_agent: stub")
    return {**state, "report_path": ""}


# ── Routing ────────────────────────────────────────────────────────────────────

def _route_after_validation(state: InvoiceState) -> Literal["validate_business", "report"]:
    """
    After data_validation_agent: if recommendation is already REJECTED
    (e.g. missing required fields / bad currency), skip business validation
    and go straight to reporting.
    """
    if state.get("recommendation") == "REJECTED":
        return "report"
    return "validate_business"


def _route_after_monitor(state: InvoiceState) -> Literal["extract", END]:
    """If no file was found by the monitor, end the graph."""
    if not state.get("file_path"):
        return END
    return "extract"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construct and compile the invoice processing StateGraph."""
    graph = StateGraph(InvoiceState)

    # Register nodes
    graph.add_node("monitor",            invoice_monitor_agent)
    graph.add_node("extract",            extractor_agent)
    graph.add_node("translate",          translation_agent)
    graph.add_node("validate_data",      data_validation_agent)
    graph.add_node("validate_business",  business_validation_agent)
    graph.add_node("report",             reporting_agent)

    # Set entry point
    graph.set_entry_point("monitor")

    # Edges
    graph.add_conditional_edges("monitor", _route_after_monitor, {
        "extract": "extract",
        END: END,
    })
    graph.add_edge("extract",           "translate")
    graph.add_edge("translate",         "validate_data")
    graph.add_conditional_edges("validate_data", _route_after_validation, {
        "validate_business": "validate_business",
        "report":            "report",
    })
    graph.add_edge("validate_business", "report")
    graph.add_edge("report",            END)

    return graph.compile()


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_pipeline(file_path: str | None = None) -> InvoiceState:
    """
    Run the invoice pipeline for a single file.

    Args:
        file_path: If provided, bypasses the monitor and processes this
                   file directly (useful for UI uploads and testing).
    """
    pipeline = build_pipeline()

    if file_path:
        seed = initial_state(file_path=file_path, meta={}, file_format="")
    else:
        seed = InvoiceState(errors=[], discrepancies=[])

    logger.info("Pipeline started for: %s", file_path or "<polling>")
    final_state = pipeline.invoke(seed)
    logger.info("Pipeline finished. recommendation=%s", final_state.get("recommendation", "N/A"))
    return final_state


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_pipeline(fp)
    print("\n=== Pipeline Result ===")
    for key in ["file_path", "file_format", "detected_language",
                "translation_confidence", "recommendation", "errors"]:
        print(f"  {key}: {result.get(key)}")

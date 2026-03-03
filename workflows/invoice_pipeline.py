"""
Invoice Pipeline — AI Invoice Auditor
LangGraph StateGraph wiring all 6 core agents into a sequential pipeline:

    monitor → extract → translate → validate_data → validate_business → report

All agents are now real implementations (no stubs).
"""

from typing import Literal
from pathlib import Path

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from core.logger import get_logger
from core.state import InvoiceState, initial_state
from agents.invoice_monitor_agent import invoice_monitor_agent
from agents.extractor_agent import extractor_agent
from agents.translation_agent import translation_agent
from agents.data_validation_agent import data_validation_agent
from agents.business_validation_agent import business_validation_agent
from agents.reporting_agent import reporting_agent
from agents.rag.indexing_agent import indexing_agent

logger = get_logger(__name__)

# Load environment variables for CLI runs
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")


# ── Routing ────────────────────────────────────────────────────────────────────

def _route_after_validation(state: InvoiceState) -> Literal["validate_business", "report"]:
    """
    After data_validation_agent: if recommendation is already REJECTED
    (bad currency / missing required fields), skip business validation
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

    graph.add_node("monitor",           invoice_monitor_agent)
    graph.add_node("extract",           extractor_agent)
    graph.add_node("translate",         translation_agent)
    graph.add_node("index",             indexing_agent)
    graph.add_node("validate_data",     data_validation_agent)
    graph.add_node("validate_business", business_validation_agent)
    graph.add_node("report",            reporting_agent)

    graph.set_entry_point("monitor")

    graph.add_conditional_edges("monitor", _route_after_monitor, {
        "extract": "extract",
        END: END,
    })
    graph.add_edge("extract",           "translate")
    graph.add_edge("translate",         "index")
    graph.add_edge("index",             "validate_data")
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
                "translation_confidence", "recommendation", "report_path", "errors"]:
        print(f"  {key}: {result.get(key)}")

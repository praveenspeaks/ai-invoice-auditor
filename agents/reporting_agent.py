"""
Reporting Agent — AI Invoice Auditor
LangGraph node that generates the HTML audit report.

Reads:  full InvoiceState (extracted_fields, validation_result, discrepancies,
        recommendation, errors, etc.)
Writes: report_path
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.insight_reporter_tool import generate_report

logger = get_logger(__name__)


def reporting_agent(state: InvoiceState) -> dict:
    """
    LangGraph node — Reporting Agent.

    Always runs — generates a report regardless of recommendation outcome
    so every invoice has an auditable HTML record.
    """
    logger.info(
        "Generating report for %s (recommendation=%s)",
        state.get("file_path", "<unknown>"),
        state.get("recommendation", "N/A"),
    )

    report_path = generate_report(dict(state))

    if not report_path:
        logger.error("Report generation failed for %s", state.get("file_path"))
        return {
            "report_path": "",
            "errors": ["REPORT_ERROR: failed to generate HTML report"],
        }

    return {"report_path": report_path}

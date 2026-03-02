"""
Extractor Agent — AI Invoice Auditor
LangGraph node that extracts raw text, tables, and detected language
from an invoice file using the Data Harvester Tool.
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.data_harvester_tool import harvest, get_file_format

logger = get_logger(__name__)


def extractor_agent(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node — Extractor Agent.

    Reads:  file_path, file_format
    Writes: raw_text, detected_language, file_format (normalised)
    """
    file_path = state.get("file_path", "")
    if not file_path:
        return {**state, "errors": list(state.get("errors", [])) + ["EXTRACTOR: file_path is empty"]}

    result = harvest(file_path)

    if result["error"]:
        logger.error("Extraction failed for %s: %s", file_path, result["error"])
        return {
            **state,
            "raw_text": "",
            "detected_language": "en",
            "errors": list(state.get("errors", [])) + [f"EXTRACTION_ERROR: {result['error']}"],
        }

    logger.info("Extracted %d chars from %s (lang=%s)", len(result["raw_text"]),
                file_path, result["detected_language"])
    return {
        **state,
        "raw_text": result["raw_text"],
        "detected_language": result["detected_language"],
        "file_format": result["file_format"],
    }

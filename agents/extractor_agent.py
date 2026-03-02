"""
Extractor Agent — AI Invoice Auditor
LangGraph node that extracts raw text, tables, and detected language
from an invoice file using the Data Harvester Tool.
"""

from core.logger import get_logger
from core.observability import trace_agent
from core.rai_guardrails import check_injection
from core.state import InvoiceState
from tools.data_harvester_tool import harvest, get_file_format

logger = get_logger(__name__)


@trace_agent("extractor")
def extractor_agent(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node — Extractor Agent.

    Reads:  file_path, file_format, meta
    Writes: raw_text, detected_language, file_format (normalised)

    Language priority: meta["language"] (sidecar) > langdetect result.
    Returns only the fields it changes; the LangGraph reducer handles
    merging the errors list so agents must NOT include old errors in
    their return value.
    """
    file_path = state.get("file_path", "")
    if not file_path:
        return {"errors": ["EXTRACTOR: file_path is empty"]}

    result = harvest(file_path)

    if result["error"]:
        logger.error("Extraction failed for %s: %s", file_path, result["error"])
        return {
            "raw_text": "",
            "detected_language": "en",
            "errors": [f"EXTRACTION_ERROR: {result['error']}"],
        }

    # Prefer explicit language from .meta.json sidecar over langdetect
    meta_lang = state.get("meta", {}).get("language", "")
    detected_language = meta_lang or result["detected_language"]

    # RAI: prompt injection guard — flag and short-circuit before LLM extraction
    injection = check_injection(result["raw_text"])
    if injection["flagged"]:
        logger.warning("Prompt injection detected in %s — blocking LLM extraction", file_path)
        return {
            "raw_text": result["raw_text"],
            "detected_language": detected_language,
            "file_format": result["file_format"],
            "errors": [injection["reason"]],
        }

    logger.info("Extracted %d chars from %s (lang=%s)", len(result["raw_text"]),
                file_path, detected_language)
    return {
        "raw_text": result["raw_text"],
        "detected_language": detected_language,
        "file_format": result["file_format"],
    }

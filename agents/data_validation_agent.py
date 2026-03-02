"""
Data Validation Agent — AI Invoice Auditor
LangGraph node that:
  1. Calls field_extractor_tool to parse extracted_fields from translated_text
  2. Calls data_completeness_checker to validate those fields
  3. Sets recommendation = "REJECTED" if the checker flags a hard rejection
     (triggers early-exit routing in the pipeline)

Reads:   translated_text
Writes:  extracted_fields, validation_result, recommendation (if REJECTED),
         errors (new entries only — LangGraph reducer handles merging)
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.field_extractor_tool import extract_fields
from tools.data_completeness_checker import check

logger = get_logger(__name__)


def data_validation_agent(state: InvoiceState) -> dict:
    """
    LangGraph node — Data Validation Agent.

    Returns only the fields it changes (never {**state} — avoids reducer doubling).
    """
    translated_text = state.get("translated_text") or state.get("raw_text", "")

    if not translated_text.strip():
        logger.warning("data_validation_agent: no text to parse — skipping")
        return {
            "extracted_fields": {},
            "validation_result": {
                "status": "FAIL",
                "missing_fields": ["(no text available for extraction)"],
                "type_errors": [],
                "currency_status": "MISSING",
                "currency": None,
                "passed": False,
                "reject": False,
            },
        }

    # ── Step 1: LLM field extraction ──────────────────────────────────────
    extraction = extract_fields(translated_text)

    extract_error = extraction.pop("error", None)
    extracted_fields = extraction  # remaining keys are the field values

    updates: dict = {"extracted_fields": extracted_fields}

    if extract_error:
        logger.warning("Field extraction error: %s", extract_error)
        updates["errors"] = [f"FIELD_EXTRACTION_ERROR: {extract_error}"]

    # ── Step 2: Completeness / type / currency validation ─────────────────
    validation_result = check(extracted_fields)
    updates["validation_result"] = validation_result

    # Normalise currency back into extracted_fields if it was a symbol
    if validation_result.get("currency") and not extracted_fields.get("currency"):
        extracted_fields["currency"] = validation_result["currency"]

    # ── Step 3: Set recommendation if hard-rejected ────────────────────────
    if validation_result.get("reject"):
        updates["recommendation"] = "REJECTED"
        updates["human_review_required"] = False
        logger.warning(
            "Invoice REJECTED at data validation stage. "
            "currency_status=%s missing=%s",
            validation_result.get("currency_status"),
            validation_result.get("missing_fields"),
        )
    else:
        logger.info(
            "Data validation complete: status=%s missing=%d type_errors=%d",
            validation_result["status"],
            len(validation_result.get("missing_fields", [])),
            len(validation_result.get("type_errors", [])),
        )

    return updates

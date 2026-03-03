"""
Business Validation Agent — AI Invoice Auditor
LangGraph node that cross-checks invoice data against the ERP and
computes the final recommendation.

Reads:   extracted_fields, validation_result, translation_confidence
Writes:  erp_data, discrepancies (via reducer), recommendation,
         human_review_required
"""

from core.config import get_rules
from core.logger import get_logger
from core.observability import trace_agent
from core.state import InvoiceState
from tools.business_validation_tool import validate

logger = get_logger(__name__)


@trace_agent("business_validator")
def business_validation_agent(state: InvoiceState) -> dict:
    """
    LangGraph node — Business Validation Agent.

    Recommendation logic (in priority order):
      REJECTED      — already set by data_validation_agent (bad currency / missing fields)
      AUTO_APPROVED — translation_confidence ≥ 0.95 AND validation passed
                      AND no DISCREPANCY-status line items
      MANUAL_REVIEW — discrepancies found OR low translation confidence
    """
    extracted_fields: dict = state.get("extracted_fields") or {}
    validation_result: dict = state.get("validation_result") or {}
    translation_confidence: float = state.get("translation_confidence", 0.0)

    # If already rejected, skip ERP call
    if state.get("recommendation") == "REJECTED":
        logger.info("Invoice already REJECTED — skipping business validation")
        return {}

    # ── ERP validation ────────────────────────────────────────────────────
    result = validate(extracted_fields)
    erp_data = result["erp_data"]
    discrepancies: list = result["discrepancies"]
    erp_error = result.get("error")

    updates: dict = {
        "erp_data": erp_data,
        "discrepancies": discrepancies,
    }

    if erp_error:
        updates["errors"] = [f"ERP_VALIDATION_ERROR: {erp_error}"]

    # ── Recommendation ────────────────────────────────────────────────────
    rules = get_rules()
    threshold = rules.validation_policies.auto_approve_confidence_threshold

    has_discrepancies = any(
        d.get("status") == "DISCREPANCY" for d in discrepancies
    )
    validation_passed = validation_result.get("passed", False)
    confidence_ok = translation_confidence >= threshold

    if has_discrepancies or erp_error:
        recommendation = "MANUAL_REVIEW"
        human_review = True
    elif not validation_passed:
        recommendation = "MANUAL_REVIEW"
        human_review = True
    elif confidence_ok:
        recommendation = "AUTO_APPROVED"
        human_review = False
    else:
        recommendation = "MANUAL_REVIEW"
        human_review = True

    updates["recommendation"] = recommendation
    updates["human_review_required"] = human_review

    logger.info(
        "Business validation: recommendation=%s confidence=%.2f discrepancies=%d",
        recommendation, translation_confidence,
        sum(1 for d in discrepancies if d.get("status") == "DISCREPANCY"),
    )

    return updates

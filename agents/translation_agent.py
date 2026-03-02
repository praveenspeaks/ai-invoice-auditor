"""
Translation Agent — AI Invoice Auditor
LangGraph node that translates extracted invoice text to English.
Sets human_review_required=True for low-confidence translations.
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.lang_bridge_tool import translate, is_low_confidence

logger = get_logger(__name__)


def translation_agent(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node — Translation Agent.

    Reads: raw_text, detected_language
    Writes: translated_text, translation_confidence, human_review_required

    If translation fails or confidence is low, sets human_review_required=True
    and appends a warning to errors (but does NOT abort the pipeline).
    """
    raw_text = state.get("raw_text", "")
    detected_lang = state.get("detected_language", "en")

    if not raw_text:
        logger.warning("Translation agent: raw_text is empty — skipping")
        return {"translated_text": "", "translation_confidence": 0.0}

    result = translate(text=raw_text, source_language=detected_lang)

    translated_text = result["translated_text"]
    confidence = result["confidence"]
    error = result.get("error")
    was_translated = result["was_translated"]

    # Return only fields this agent changes.  Do NOT spread {**state} for
    # reducer fields (errors/discrepancies) — the LangGraph reducer merges
    # them automatically and spreading would cause duplicate entries.
    updates: dict = {
        "translated_text": translated_text,
        "translation_confidence": confidence,
    }

    if error:
        updates["errors"] = [f"TRANSLATION_ERROR: {error}"]
        updates["human_review_required"] = True
        logger.warning("Translation error — flagging for human review: %s", error)

    elif is_low_confidence(confidence):
        updates["human_review_required"] = True
        updates["errors"] = [
            f"LOW_TRANSLATION_CONFIDENCE: {confidence:.2f} (lang={detected_lang})"
        ]
        logger.warning(
            "Low translation confidence %.2f for lang=%s — flagging for human review",
            confidence, detected_lang
        )
    else:
        updates["human_review_required"] = state.get("human_review_required", False)

    if was_translated:
        logger.info(
            "Translation complete: %s→en, confidence=%.2f",
            detected_lang, confidence
        )
    else:
        logger.info("No translation needed (lang=%s)", detected_lang)

    return updates

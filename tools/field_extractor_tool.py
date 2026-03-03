"""
Field Extractor Tool — AI Invoice Auditor
LLM-assisted structured field extraction from translated invoice text.

Uses the configured LLM (OPENAI_API_KEY / ANTHROPIC_API_KEY) via LangChain
to parse free-form invoice text into a typed dict of invoice fields.

Returns a dict with keys:
    invoice_no, invoice_date, vendor_id, vendor_name,
    currency, total_amount, po_number, line_items (list)
Missing/unparseable fields are returned as None.
"""

import json
import os
import re
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)

# ── LLM prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an invoice data extraction assistant. "
    "Extract the requested fields from the invoice text accurately. "
    "Return ONLY a valid JSON object — no markdown, no explanation."
)

_USER_TEMPLATE = """\
Extract the following fields from this invoice text and return them as JSON.
Use null for any field that is missing or cannot be determined.

Fields to extract:
- invoice_no      (string, e.g. "INV-1001")
- invoice_date    (string ISO date, e.g. "2025-03-14")
- vendor_id       (string, e.g. "VEND-001" — look for vendor/supplier ID codes)
- vendor_name     (string, company name of the vendor/supplier)
- po_number       (string, purchase order number, e.g. "PO-1001")
- currency        (string, ISO 4217 code or symbol, e.g. "USD", "$", "EUR", "€")
- total_amount    (number, grand total of the invoice)
- line_items      (array of objects, each with:
                    item_code, description, qty, unit_price, total)

Invoice text:
\"\"\"
{text}
\"\"\"

Return ONLY a JSON object with exactly these keys."""


def _build_llm():
    """Instantiate the LLM from environment configuration."""
    # Priority: Azure OpenAI -> OpenAI -> Anthropic
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if (
        azure_key and not azure_key.startswith("your_")
        and azure_endpoint
        and azure_deployment
    ):
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
            deployment_name=azure_deployment,
            temperature=0,
        )

    if openai_key and not openai_key.startswith("your_"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, api_key=openai_key)

    if anthropic_key and not anthropic_key.startswith("your_"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            temperature=0,
            api_key=anthropic_key,
        )

    raise RuntimeError(
        "No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
    )


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    # Strip ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def extract_fields(text: str) -> dict[str, Any]:
    """
    Parse structured invoice fields from plain-English invoice text using LLM.

    Args:
        text: Translated (English) invoice text.

    Returns:
        dict with keys: invoice_no, invoice_date, vendor_id, vendor_name,
        po_number, currency, total_amount, line_items, error.
        All invoice fields default to None on failure; error is None on success.
    """
    _empty: dict[str, Any] = {
        "invoice_no": None,
        "invoice_date": None,
        "vendor_id": None,
        "vendor_name": None,
        "po_number": None,
        "currency": None,
        "total_amount": None,
        "line_items": [],
        "error": None,
    }

    if not text or not text.strip():
        return {**_empty, "error": "Empty text — nothing to extract"}

    try:
        llm = _build_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_TEMPLATE.format(text=text[:8000])),
        ]
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_response(raw)

        result = {**_empty}
        for key in ("invoice_no", "invoice_date", "vendor_id", "vendor_name",
                    "po_number", "currency", "total_amount"):
            result[key] = parsed.get(key)

        line_items = parsed.get("line_items")
        result["line_items"] = line_items if isinstance(line_items, list) else []

        logger.info(
            "Field extraction complete: invoice_no=%s vendor_id=%s currency=%s total=%s",
            result["invoice_no"], result["vendor_id"],
            result["currency"], result["total_amount"],
        )
        return result

    except RuntimeError as exc:
        # No LLM configured — return graceful fallback (test/offline mode)
        logger.warning("LLM not available: %s", exc)
        return {**_empty, "error": str(exc)}

    except Exception as exc:
        logger.error("Field extraction failed: %s", exc)
        return {**_empty, "error": str(exc)}

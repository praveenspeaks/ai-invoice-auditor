"""
Data Completeness Checker Tool — AI Invoice Auditor

Three-pass validation of extracted invoice fields against rules.yaml:

  Pass 1 — Required fields  (header + line-item level)
  Pass 2 — Data type check  (date, float, str)
  Pass 3 — Currency         (symbol normalisation + accepted_currencies list)

Returns a validation_result dict:
    {
        "status":          "PASS" | "FAIL",
        "missing_fields":  [str, ...],
        "type_errors":     [str, ...],
        "currency_status": "ACCEPTED" | "REJECTED" | "NORMALISED" | "MISSING",
        "currency":        str | None,   # normalised ISO code
        "passed":          bool,
        "reject":          bool,         # True → pipeline should REJECT
    }
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from core.config import get_rules
from core.logger import get_logger

logger = get_logger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _is_valid_date(value: Any) -> bool:
    """Accept ISO date strings (YYYY-MM-DD) and date objects."""
    if isinstance(value, (date, datetime)):
        return True
    if isinstance(value, str):
        try:
            datetime.strptime(value.strip(), "%Y-%m-%d")
            return True
        except ValueError:
            pass
        # Also accept common formats like MM/DD/YYYY, DD-MM-YYYY
        for fmt in ("%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y"):
            try:
                datetime.strptime(value.strip(), fmt)
                return True
            except ValueError:
                continue
    return False


def _is_valid_float(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.replace(",", "").strip())
            return True
        except ValueError:
            return False
    return False


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


_TYPE_VALIDATORS = {
    "date":  _is_valid_date,
    "float": _is_valid_float,
    "str":   _is_non_empty_str,
}


# ── Pass 1 — Required fields ───────────────────────────────────────────────

def _check_required_fields(
    fields: dict[str, Any],
    rules,
) -> list[str]:
    """Return list of missing required header + line-item fields."""
    missing: list[str] = []

    header_required = rules.required_fields.header
    for field in header_required:
        val = fields.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(f"header.{field}")

    line_items = fields.get("line_items") or []
    line_item_required = rules.required_fields.line_item

    if not line_items:
        missing.append("line_items (empty or missing)")
    else:
        for idx, item in enumerate(line_items):
            if not isinstance(item, dict):
                missing.append(f"line_items[{idx}] (not a dict)")
                continue
            for field in line_item_required:
                val = item.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    missing.append(f"line_items[{idx}].{field}")

    return missing


# ── Pass 2 — Data type validation ─────────────────────────────────────────

def _check_data_types(
    fields: dict[str, Any],
    rules,
) -> list[str]:
    """Return list of type-error descriptions for malformed field values."""
    type_errors: list[str] = []
    data_types: dict[str, str] = rules.data_types

    for field, expected_type in data_types.items():
        value = fields.get(field)
        if value is None:
            continue  # already caught in Pass 1 if required
        validator = _TYPE_VALIDATORS.get(expected_type)
        if validator is None:
            continue
        if not validator(value):
            type_errors.append(
                f"{field}: expected {expected_type}, got {type(value).__name__!r} ({value!r})"
            )

    # Validate line-item numeric fields
    line_items = fields.get("line_items") or []
    line_type_map = {k: v for k, v in data_types.items()
                     if k in ("qty", "unit_price", "total")}
    for idx, item in enumerate(line_items):
        if not isinstance(item, dict):
            continue
        for field, expected_type in line_type_map.items():
            value = item.get(field)
            if value is None:
                continue
            validator = _TYPE_VALIDATORS.get(expected_type)
            if validator and not validator(value):
                type_errors.append(
                    f"line_items[{idx}].{field}: expected {expected_type}, "
                    f"got {type(value).__name__!r} ({value!r})"
                )

    return type_errors


# ── Pass 3 — Currency normalisation & acceptance ───────────────────────────

def _check_currency(
    fields: dict[str, Any],
    rules,
) -> tuple[str, str | None]:
    """
    Returns (currency_status, normalised_iso_code_or_None).

    currency_status values:
        "MISSING"    — currency field absent
        "NORMALISED" — symbol was mapped to ISO code (e.g. $ → USD)
        "ACCEPTED"   — already a valid ISO code
        "REJECTED"   — not in accepted_currencies after normalisation
    """
    raw = fields.get("currency")
    if not raw or (isinstance(raw, str) and not raw.strip()):
        return "MISSING", None

    raw = str(raw).strip()
    symbol_map: dict[str, str] = rules.currency_symbol_map
    accepted: list[str] = rules.accepted_currencies

    # Attempt symbol → ISO normalisation
    normalised = symbol_map.get(raw, raw).upper()

    if raw in symbol_map:
        status = "NORMALISED"
    else:
        status = "ACCEPTED" if normalised in accepted else "REJECTED"

    if normalised not in accepted:
        status = "REJECTED"

    return status, normalised


# ── Main entry point ───────────────────────────────────────────────────────

def check(extracted_fields: dict[str, Any]) -> dict[str, Any]:
    """
    Run all three validation passes and return a validation_result dict.

    Args:
        extracted_fields: Output of field_extractor_tool.extract_fields()

    Returns:
        validation_result dict (see module docstring).
    """
    rules = get_rules()
    policies = rules.validation_policies

    # --- Pass 1: Required fields ------------------------------------------
    missing = _check_required_fields(extracted_fields, rules)

    # --- Pass 2: Data types -----------------------------------------------
    type_errors = _check_data_types(extracted_fields, rules)

    # --- Pass 3: Currency -------------------------------------------------
    currency_status, normalised_currency = _check_currency(extracted_fields, rules)

    # --- Apply policies ---------------------------------------------------
    reject = False

    # Currency rejection
    if currency_status == "REJECTED":
        if policies.invalid_currency_action == "reject":
            reject = True
            logger.warning("Invoice REJECTED: currency not accepted (%s)", normalised_currency)

    # Missing field policy (flag by default — does not immediately reject)
    if missing:
        logger.warning("Missing fields [action=%s]: %s", policies.missing_field_action, missing)

    # Determine overall pass/fail
    passed = not reject and not missing and not type_errors

    result = {
        "status": "PASS" if passed else "FAIL",
        "missing_fields": missing,
        "type_errors": type_errors,
        "currency_status": currency_status,
        "currency": normalised_currency,
        "passed": passed,
        "reject": reject,
    }

    logger.info(
        "Completeness check: status=%s missing=%d type_errors=%d currency=%s",
        result["status"], len(missing), len(type_errors), currency_status,
    )
    return result

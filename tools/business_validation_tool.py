"""
Business Validation Tool — AI Invoice Auditor

Compares extracted invoice line items against the ERP purchase-order record:
  1. Fetches PO from ERP: GET /erp/po/{vendor_id}/{po_number}
  2. Returns "UNREGISTERED_INVOICE" if 404
  3. For each invoice line item, locates the matching ERP item by item_code
  4. Compares qty (exact match), unit_price (±5%), total (±5%)
  5. Applies tolerance rules from rules.yaml

Returns a dict:
  {
    "erp_data":       dict,   # raw PO response from ERP
    "discrepancies":  list,   # list of discrepancy dicts
    "error":          str|None,
  }

Each discrepancy dict:
  {
    "item_code":    str,
    "field":        "qty" | "unit_price" | "total",
    "invoice_val":  float,
    "erp_val":      float,
    "diff_pct":     float,    # signed % difference
    "status":       "MATCH" | "WITHIN_TOLERANCE" | "DISCREPANCY",
  }
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from core.config import get_rules
from core.logger import get_logger

logger = get_logger(__name__)

_ERP_BASE_URL = os.getenv("ERP_BASE_URL", "http://localhost:8000")
_TIMEOUT = 10.0  # seconds


# ── helpers ────────────────────────────────────────────────────────────────

def _pct_diff(invoice_val: float, erp_val: float) -> float:
    """Signed percentage difference: (invoice - erp) / erp * 100."""
    if erp_val == 0:
        return 0.0 if invoice_val == 0 else 100.0
    return (invoice_val - erp_val) / abs(erp_val) * 100.0


def _compare_field(
    item_code: str,
    field: str,
    invoice_val: float,
    erp_val: float,
    tolerance_pct: float,
) -> dict[str, Any]:
    diff = _pct_diff(invoice_val, erp_val)
    abs_diff = abs(diff)

    if abs_diff == 0:
        status = "MATCH"
    elif abs_diff <= tolerance_pct:
        status = "WITHIN_TOLERANCE"
    else:
        status = "DISCREPANCY"

    return {
        "item_code":   item_code,
        "field":       field,
        "invoice_val": invoice_val,
        "erp_val":     erp_val,
        "diff_pct":    round(diff, 4),
        "status":      status,
    }


def _safe_float(val: Any) -> float | None:
    """Convert val to float, return None on failure."""
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ── ERP fetch ──────────────────────────────────────────────────────────────

def _fetch_po(vendor_id: str, po_number: str) -> tuple[dict | None, str | None]:
    """
    Fetch PO from ERP. Returns (po_data, error_string).
    po_data is None on failure.
    """
    url = f"{_ERP_BASE_URL}/erp/po/{vendor_id}/{po_number}"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
        if resp.status_code == 404:
            return None, "UNREGISTERED_INVOICE"
        resp.raise_for_status()
        return resp.json(), None
    except httpx.TimeoutException:
        return None, "ERP_TIMEOUT"
    except httpx.HTTPStatusError as exc:
        return None, f"ERP_HTTP_ERROR:{exc.response.status_code}"
    except Exception as exc:
        return None, f"ERP_ERROR:{exc}"


# ── Comparison logic ───────────────────────────────────────────────────────

def _compare_line_items(
    invoice_items: list[dict],
    erp_items: list[dict],
    tolerances,
) -> list[dict[str, Any]]:
    """Cross-check each invoice line item against its ERP counterpart."""
    erp_by_code: dict[str, dict] = {item["item_code"]: item for item in erp_items}
    discrepancies: list[dict] = []

    price_tol = tolerances.price_difference_percent
    qty_tol = tolerances.quantity_difference_percent  # 0 = exact match required

    for inv_item in invoice_items:
        if not isinstance(inv_item, dict):
            continue
        item_code = inv_item.get("item_code", "")
        if not item_code:
            continue

        erp_item = erp_by_code.get(item_code)
        if erp_item is None:
            discrepancies.append({
                "item_code":   item_code,
                "field":       "item_code",
                "invoice_val": item_code,
                "erp_val":     None,
                "diff_pct":    None,
                "status":      "DISCREPANCY",
            })
            continue

        inv_qty = _safe_float(inv_item.get("qty"))
        erp_qty = _safe_float(erp_item.get("qty"))
        if inv_qty is not None and erp_qty is not None:
            discrepancies.append(_compare_field(
                item_code, "qty", inv_qty, erp_qty, qty_tol
            ))

        inv_price = _safe_float(inv_item.get("unit_price"))
        erp_price = _safe_float(erp_item.get("unit_price"))
        if inv_price is not None and erp_price is not None:
            discrepancies.append(_compare_field(
                item_code, "unit_price", inv_price, erp_price, price_tol
            ))

        inv_total = _safe_float(inv_item.get("total"))
        if inv_total is not None and erp_qty is not None and erp_price is not None:
            erp_total = erp_qty * erp_price
            discrepancies.append(_compare_field(
                item_code, "total", inv_total, erp_total, price_tol
            ))

    return discrepancies


# ── Main entry point ───────────────────────────────────────────────────────

def validate(extracted_fields: dict[str, Any]) -> dict[str, Any]:
    """
    Run business validation against the ERP.

    Args:
        extracted_fields: Output of data_validation_agent (invoice fields).

    Returns:
        {"erp_data": dict, "discrepancies": list, "error": str|None}
    """
    vendor_id = extracted_fields.get("vendor_id") or ""
    po_number = extracted_fields.get("po_number") or ""

    if not vendor_id or not po_number:
        logger.warning(
            "Business validation skipped: vendor_id=%r po_number=%r",
            vendor_id, po_number,
        )
        return {
            "erp_data": {},
            "discrepancies": [],
            "error": f"MISSING_IDS: vendor_id={vendor_id!r} po_number={po_number!r}",
        }

    po_data, fetch_error = _fetch_po(vendor_id, po_number)

    if fetch_error:
        logger.warning("ERP fetch failed: %s (vendor=%s po=%s)", fetch_error, vendor_id, po_number)
        return {"erp_data": {}, "discrepancies": [], "error": fetch_error}

    rules = get_rules()
    invoice_items = extracted_fields.get("line_items") or []
    erp_items = po_data.get("line_items") or []

    discrepancies = _compare_line_items(invoice_items, erp_items, rules.tolerances)

    real_discrepancies = [d for d in discrepancies if d["status"] == "DISCREPANCY"]
    logger.info(
        "Business validation done: vendor=%s po=%s items=%d discrepancies=%d",
        vendor_id, po_number, len(invoice_items), len(real_discrepancies),
    )

    return {"erp_data": po_data, "discrepancies": discrepancies, "error": None}

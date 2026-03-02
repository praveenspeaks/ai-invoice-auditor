"""
Insight Reporter Tool — AI Invoice Auditor
Generates a self-contained HTML audit report from the fully-processed InvoiceState.

Report sections:
  1. Header        — invoice_no, vendor, date, run timestamp
  2. Extraction    — format, language, translation_confidence
  3. Data Validation — field table with PASS / FAIL / MISSING badges
  4. Business Validation — discrepancy table (colour-coded)
  5. Recommendation — large AUTO_APPROVED / MANUAL_REVIEW / REJECTED badge
  6. Errors         — any pipeline errors

Saves to: {output_dir}/report_{invoice_no}_{timestamp}.html
Returns:  absolute path string, or "" on failure (error appended to errors list).
"""

from __future__ import annotations

import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

from core.config import get_rules
from core.logger import get_logger

logger = get_logger(__name__)

# ── Jinja2 HTML template ───────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Invoice Report — {{ invoice_no }}</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #333; background: #f9f9f9; }
  h1 { color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }
  h2 { color: #283593; margin-top: 32px; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th { background: #283593; color: #fff; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #ddd; background: #fff; }
  tr:hover td { background: #f0f4ff; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; color: #fff; }
  .AUTO_APPROVED  { background: #2e7d32; }
  .MANUAL_REVIEW  { background: #f57c00; }
  .REJECTED       { background: #c62828; }
  .PASS           { background: #388e3c; }
  .FAIL, .MISSING { background: #d32f2f; }
  .MATCH          { background: #388e3c; }
  .WITHIN_TOLERANCE { background: #f57c00; }
  .DISCREPANCY    { background: #c62828; }
  .info { background: #e8eaf6; border-left: 4px solid #3949ab; padding: 10px 14px; margin: 8px 0; }
  .error-list li { color: #c62828; font-family: monospace; margin: 4px 0; }
  .meta { color: #666; font-size: 0.9em; }
</style>
</head>
<body>

<!-- ── 1. HEADER ─────────────────────────────────────────────── -->
<h1>Invoice Audit Report</h1>
<div class="info">
  <strong>Invoice No:</strong> {{ invoice_no }}<br>
  <strong>Vendor ID:</strong> {{ vendor_id }}<br>
  <strong>Vendor Name:</strong> {{ vendor_name }}<br>
  <strong>Invoice Date:</strong> {{ invoice_date }}<br>
  <strong>PO Number:</strong> {{ po_number }}<br>
  <strong>File:</strong> {{ file_path }}<br>
  <span class="meta">Report generated: {{ generated_at }}</span>
</div>

<!-- ── 2. EXTRACTION SUMMARY ─────────────────────────────────── -->
<h2>Extraction Summary</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>File Format</td><td>{{ file_format }}</td></tr>
  <tr><td>Detected Language</td><td>{{ detected_language }}</td></tr>
  <tr><td>Translation Confidence</td><td>{{ "%.0f%%"|format(translation_confidence * 100) }}</td></tr>
</table>

<!-- ── 3. DATA VALIDATION ────────────────────────────────────── -->
<h2>Data Validation</h2>
{% if missing_fields %}
<p><strong>Missing Fields:</strong></p>
<ul>{% for f in missing_fields %}<li>{{ f }}</li>{% endfor %}</ul>
{% endif %}
{% if type_errors %}
<p><strong>Type Errors:</strong></p>
<ul>{% for e in type_errors %}<li>{{ e }}</li>{% endfor %}</ul>
{% endif %}
<table>
  <tr><th>Field</th><th>Extracted Value</th><th>Status</th></tr>
  {% for row in field_rows %}
  <tr>
    <td>{{ row.name }}</td>
    <td>{{ row.value }}</td>
    <td><span class="badge {{ row.status }}">{{ row.status }}</span></td>
  </tr>
  {% endfor %}
</table>
<p><strong>Currency:</strong> {{ currency_raw }} →
  <span class="badge {{ currency_status }}">{{ currency_normalised }} ({{ currency_status }})</span>
</p>

<!-- ── 4. BUSINESS VALIDATION ────────────────────────────────── -->
<h2>Business Validation</h2>
{% if discrepancies %}
<table>
  <tr>
    <th>Item Code</th><th>Field</th>
    <th>Invoice Value</th><th>ERP Value</th>
    <th>Diff %</th><th>Status</th>
  </tr>
  {% for d in discrepancies %}
  <tr>
    <td>{{ d.item_code }}</td>
    <td>{{ d.field }}</td>
    <td>{{ d.invoice_val }}</td>
    <td>{{ d.erp_val }}</td>
    <td>{{ "%.2f%%"|format(d.diff_pct) if d.diff_pct is not none else "N/A" }}</td>
    <td><span class="badge {{ d.status }}">{{ d.status }}</span></td>
  </tr>
  {% endfor %}
</table>
{% else %}
<div class="info">No discrepancies found — all line items match ERP records.</div>
{% endif %}

<!-- ── 5. RECOMMENDATION ──────────────────────────────────────── -->
<h2>Final Recommendation</h2>
<div style="text-align:center; margin: 24px 0;">
  <span class="badge {{ recommendation }}" style="font-size:1.6em; padding:14px 32px;">
    {{ recommendation }}
  </span>
</div>

<!-- ── 6. ERRORS ─────────────────────────────────────────────── -->
{% if errors %}
<h2>Pipeline Errors</h2>
<ul class="error-list">
  {% for e in errors %}<li>{{ e }}</li>{% endfor %}
</ul>
{% endif %}

</body>
</html>"""

# ── row builder ────────────────────────────────────────────────────────────

_HEADER_FIELDS = [
    "invoice_no", "invoice_date", "vendor_id",
    "vendor_name", "po_number", "total_amount",
]


def _build_field_rows(extracted: dict, missing: list[str]) -> list[dict]:
    missing_names = {m.replace("header.", "") for m in missing}
    rows = []
    for field in _HEADER_FIELDS:
        val = extracted.get(field)
        if field in missing_names or val is None:
            status = "MISSING"
        else:
            status = "PASS"
        rows.append({"name": field, "value": str(val) if val is not None else "—", "status": status})
    return rows


# ── main ───────────────────────────────────────────────────────────────────

def generate_report(state: dict[str, Any]) -> str:
    """
    Render an HTML report from the pipeline state dict.

    Returns the absolute path to the saved report file, or "" on failure.
    """
    rules = get_rules()
    output_dir = Path(os.getenv("REPORTS_DIR", rules.reporting.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: dict = state.get("extracted_fields") or {}
    validation: dict = state.get("validation_result") or {}
    discrepancies: list = state.get("discrepancies") or []

    invoice_no: str = extracted.get("invoice_no") or "UNKNOWN"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_no = invoice_no.replace("/", "-").replace("\\", "-")
    filename = f"report_{safe_no}_{timestamp}.html"
    out_path = output_dir / filename

    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_HTML_TEMPLATE)

    context = {
        "invoice_no":             invoice_no,
        "vendor_id":              extracted.get("vendor_id") or "—",
        "vendor_name":            extracted.get("vendor_name") or "—",
        "invoice_date":           extracted.get("invoice_date") or "—",
        "po_number":              extracted.get("po_number") or "—",
        "file_path":              state.get("file_path") or "—",
        "generated_at":           datetime.now(UTC).isoformat(),
        "file_format":            state.get("file_format") or "—",
        "detected_language":      state.get("detected_language") or "—",
        "translation_confidence": state.get("translation_confidence") or 0.0,
        "missing_fields":         validation.get("missing_fields") or [],
        "type_errors":            validation.get("type_errors") or [],
        "field_rows":             _build_field_rows(extracted, validation.get("missing_fields") or []),
        "currency_raw":           extracted.get("currency") or "—",
        "currency_normalised":    validation.get("currency") or "—",
        "currency_status":        validation.get("currency_status") or "—",
        "discrepancies":          discrepancies,
        "recommendation":         state.get("recommendation") or "MANUAL_REVIEW",
        "errors":                 state.get("errors") or [],
    }

    try:
        html = tmpl.render(**context)
        out_path.write_text(html, encoding="utf-8")
        logger.info("Report saved: %s", out_path)
        return str(out_path)
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        return ""

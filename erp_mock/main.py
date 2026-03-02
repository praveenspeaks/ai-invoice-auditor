"""
Mock ERP FastAPI Server — AI Invoice Auditor
Simulates an enterprise ERP system exposing purchase order and vendor data
for invoice validation. Data is loaded from data/ERP_mockdata/*.json.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ── Data paths ─────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data" / "ERP_mockdata"
_PO_FILE = _DATA_DIR / "PO Records.json"
_VENDOR_FILE = _DATA_DIR / "vendors.json"
_SKU_FILE = _DATA_DIR / "sku_master.json"


# ── Response models ────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    item_code: str
    description: str
    qty: float
    unit_price: float
    total: float
    tax_rate: float
    tax_amount: float
    currency: str
    uom: Optional[str] = None
    category: Optional[str] = None


class POResponse(BaseModel):
    po_number: str
    vendor_id: str
    vendor_name: str
    currency: str
    line_items: list[LineItem]
    subtotal: float
    total_tax: float
    grand_total: float


class VendorResponse(BaseModel):
    vendor_id: str
    vendor_name: str
    country: str
    currency: str


class SKUResponse(BaseModel):
    item_code: str
    category: str
    uom: str
    gst_rate: float


class HealthResponse(BaseModel):
    status: str
    data_loaded: dict[str, int]


# ── Data loader ────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_indexes() -> tuple[dict, dict, dict]:
    """Build lookup dictionaries from the seed JSON files."""
    po_records: list[dict] = _load_json(_PO_FILE)
    vendors: list[dict] = _load_json(_VENDOR_FILE)
    skus: list[dict] = _load_json(_SKU_FILE)

    # vendor_id → vendor dict
    vendor_map = {v["vendor_id"]: v for v in vendors}
    # item_code → sku dict
    sku_map = {s["item_code"]: s for s in skus}
    # (vendor_id, po_number) → po dict
    po_map = {(p["vendor_id"], p["po_number"]): p for p in po_records}

    return po_map, vendor_map, sku_map


_PO_MAP, _VENDOR_MAP, _SKU_MAP = _build_indexes()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Invoice Auditor — Mock ERP API",
    description=(
        "Simulated ERP system for invoice validation. "
        "Exposes purchase orders, vendor records, and SKU master data."
    ),
    version="1.0.0",
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check — confirms server is up and data is loaded."""
    return HealthResponse(
        status="ok",
        data_loaded={
            "po_records": len(_PO_MAP),
            "vendors": len(_VENDOR_MAP),
            "skus": len(_SKU_MAP),
        },
    )


@app.get("/erp/vendors", response_model=list[VendorResponse], tags=["Vendors"])
def list_vendors() -> list[VendorResponse]:
    """Return all registered vendors."""
    return [VendorResponse(**v) for v in _VENDOR_MAP.values()]


@app.get("/erp/vendors/{vendor_id}", response_model=VendorResponse, tags=["Vendors"])
def get_vendor(vendor_id: str) -> VendorResponse:
    """Return a single vendor by ID."""
    vendor = _VENDOR_MAP.get(vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail=f"Vendor '{vendor_id}' not found")
    return VendorResponse(**vendor)


@app.get("/erp/po/{vendor_id}/{po_number}", response_model=POResponse, tags=["Purchase Orders"])
def get_po(vendor_id: str, po_number: str) -> POResponse:
    """
    Return purchase order line items for a given vendor and PO number.
    Used by the Business Validation Agent to cross-check invoice data.
    """
    po = _PO_MAP.get((vendor_id, po_number))
    if not po:
        raise HTTPException(
            status_code=404,
            detail=f"PO '{po_number}' for vendor '{vendor_id}' not found",
        )

    vendor = _VENDOR_MAP.get(vendor_id, {})
    currency = vendor.get("currency", "USD")

    line_items: list[LineItem] = []
    for item in po["line_items"]:
        sku = _SKU_MAP.get(item["item_code"], {})
        tax_rate = sku.get("gst_rate", 10.0)
        subtotal = round(item["qty"] * item["unit_price"], 2)
        tax_amount = round(subtotal * tax_rate / 100, 2)
        line_items.append(LineItem(
            item_code=item["item_code"],
            description=item["description"],
            qty=item["qty"],
            unit_price=item["unit_price"],
            total=subtotal,
            tax_rate=tax_rate,
            tax_amount=tax_amount,
            currency=item.get("currency", currency),
            uom=sku.get("uom"),
            category=sku.get("category"),
        ))

    subtotal = round(sum(li.total for li in line_items), 2)
    total_tax = round(sum(li.tax_amount for li in line_items), 2)

    return POResponse(
        po_number=po["po_number"],
        vendor_id=vendor_id,
        vendor_name=vendor.get("vendor_name", ""),
        currency=currency,
        line_items=line_items,
        subtotal=subtotal,
        total_tax=total_tax,
        grand_total=round(subtotal + total_tax, 2),
    )


@app.get("/erp/skus", response_model=list[SKUResponse], tags=["SKU Master"])
def list_skus(category: Optional[str] = Query(None, description="Filter by category")) -> list[SKUResponse]:
    """Return SKU master data, optionally filtered by category."""
    skus = list(_SKU_MAP.values())
    if category:
        skus = [s for s in skus if s["category"].lower() == category.lower()]
    return [SKUResponse(**s) for s in skus]


@app.get("/erp/skus/{item_code}", response_model=SKUResponse, tags=["SKU Master"])
def get_sku(item_code: str) -> SKUResponse:
    """Return a single SKU by item code."""
    sku = _SKU_MAP.get(item_code)
    if not sku:
        raise HTTPException(status_code=404, detail=f"SKU '{item_code}' not found")
    return SKUResponse(**sku)

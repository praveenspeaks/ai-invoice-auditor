"""
Unit tests for erp_mock/main.py — Mock ERP FastAPI server.
Uses FastAPI TestClient (in-process, no subprocess, no port binding).
Target coverage: erp_mock/main.py ≥ 85%
"""

import pytest
from fastapi.testclient import TestClient

from erp_mock.main import app, _PO_MAP, _VENDOR_MAP, _SKU_MAP

client = TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_data_loaded_keys_present(self):
        r = client.get("/health")
        data = r.json()["data_loaded"]
        assert "po_records" in data
        assert "vendors" in data
        assert "skus" in data

    def test_data_loaded_counts_positive(self):
        r = client.get("/health")
        data = r.json()["data_loaded"]
        assert data["po_records"] > 0
        assert data["vendors"] > 0
        assert data["skus"] > 0


# ── /erp/vendors ───────────────────────────────────────────────────────────────

class TestListVendors:
    def test_returns_200(self):
        r = client.get("/erp/vendors")
        assert r.status_code == 200

    def test_returns_list(self):
        r = client.get("/erp/vendors")
        assert isinstance(r.json(), list)

    def test_returns_all_vendors(self):
        r = client.get("/erp/vendors")
        assert len(r.json()) == len(_VENDOR_MAP)

    def test_vendor_has_required_fields(self):
        r = client.get("/erp/vendors")
        vendor = r.json()[0]
        assert "vendor_id" in vendor
        assert "vendor_name" in vendor
        assert "country" in vendor
        assert "currency" in vendor


class TestGetVendor:
    def test_known_vendor_returns_200(self):
        r = client.get("/erp/vendors/VEND-001")
        assert r.status_code == 200

    def test_known_vendor_correct_name(self):
        r = client.get("/erp/vendors/VEND-001")
        assert r.json()["vendor_name"] == "Global Logistics Ltd"

    def test_known_vendor_currency(self):
        r = client.get("/erp/vendors/VEND-001")
        assert r.json()["currency"] == "USD"

    def test_unknown_vendor_returns_404(self):
        r = client.get("/erp/vendors/VEND-UNKNOWN")
        assert r.status_code == 404

    def test_404_detail_message(self):
        r = client.get("/erp/vendors/VEND-UNKNOWN")
        assert "not found" in r.json()["detail"].lower()

    @pytest.mark.parametrize("vendor_id", ["VEND-001", "VEND-002", "VEND-003", "VEND-004"])
    def test_all_known_vendors_return_200(self, vendor_id):
        r = client.get(f"/erp/vendors/{vendor_id}")
        assert r.status_code == 200


# ── /erp/po/{vendor_id}/{po_number} ───────────────────────────────────────────

class TestGetPO:
    def test_known_po_returns_200(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        assert r.status_code == 200

    def test_po_has_required_fields(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        data = r.json()
        for field in ["po_number", "vendor_id", "vendor_name", "currency",
                      "line_items", "subtotal", "total_tax", "grand_total"]:
            assert field in data, f"Missing field: {field}"

    def test_po_number_matches(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        assert r.json()["po_number"] == "PO-1001"

    def test_vendor_id_matches(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        assert r.json()["vendor_id"] == "VEND-001"

    def test_line_items_not_empty(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        assert len(r.json()["line_items"]) > 0

    def test_line_item_has_required_fields(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        item = r.json()["line_items"][0]
        for field in ["item_code", "description", "qty", "unit_price",
                      "total", "tax_rate", "tax_amount", "currency"]:
            assert field in item, f"Missing field: {field}"

    def test_line_item_total_computed_correctly(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        item = r.json()["line_items"][0]
        expected_total = round(item["qty"] * item["unit_price"], 2)
        assert item["total"] == expected_total

    def test_tax_amount_computed_correctly(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        item = r.json()["line_items"][0]
        expected_tax = round(item["total"] * item["tax_rate"] / 100, 2)
        assert item["tax_amount"] == expected_tax

    def test_grand_total_equals_subtotal_plus_tax(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        data = r.json()
        expected = round(data["subtotal"] + data["total_tax"], 2)
        assert data["grand_total"] == expected

    def test_subtotal_equals_sum_of_line_totals(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        data = r.json()
        expected = round(sum(li["total"] for li in data["line_items"]), 2)
        assert data["subtotal"] == expected

    def test_unknown_vendor_returns_404(self):
        r = client.get("/erp/po/VEND-UNKNOWN/PO-1001")
        assert r.status_code == 404

    def test_unknown_po_returns_404(self):
        r = client.get("/erp/po/VEND-001/PO-UNKNOWN")
        assert r.status_code == 404

    def test_404_detail_message(self):
        r = client.get("/erp/po/VEND-999/PO-999")
        assert "not found" in r.json()["detail"].lower()

    def test_euro_vendor_currency(self):
        r = client.get("/erp/po/VEND-003/PO-1003")
        assert r.json()["currency"] == "EUR"

    def test_inr_vendor_currency(self):
        r = client.get("/erp/po/VEND-006/PO-1006")
        assert r.json()["currency"] == "INR"

    def test_sku_uom_enriched(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        item = r.json()["line_items"][0]
        assert item.get("uom") is not None

    def test_sku_category_enriched(self):
        r = client.get("/erp/po/VEND-001/PO-1001")
        item = r.json()["line_items"][0]
        assert item.get("category") is not None

    @pytest.mark.parametrize("vendor_id,po_number", [
        ("VEND-001", "PO-1001"),
        ("VEND-002", "PO-1002"),
        ("VEND-003", "PO-1003"),
        ("VEND-004", "PO-1004"),
        ("VEND-005", "PO-1005"),
        ("VEND-006", "PO-1006"),
    ])
    def test_all_six_pos_return_200(self, vendor_id, po_number):
        r = client.get(f"/erp/po/{vendor_id}/{po_number}")
        assert r.status_code == 200


# ── /erp/skus ──────────────────────────────────────────────────────────────────

class TestListSKUs:
    def test_returns_200(self):
        r = client.get("/erp/skus")
        assert r.status_code == 200

    def test_returns_list(self):
        r = client.get("/erp/skus")
        assert isinstance(r.json(), list)

    def test_returns_all_skus(self):
        r = client.get("/erp/skus")
        assert len(r.json()) == len(_SKU_MAP)

    def test_sku_has_required_fields(self):
        r = client.get("/erp/skus")
        sku = r.json()[0]
        assert "item_code" in sku
        assert "category" in sku
        assert "uom" in sku
        assert "gst_rate" in sku

    def test_filter_by_category(self):
        r = client.get("/erp/skus?category=Packaging")
        result = r.json()
        assert all(s["category"] == "Packaging" for s in result)

    def test_filter_case_insensitive(self):
        r_upper = client.get("/erp/skus?category=Safety")
        r_lower = client.get("/erp/skus?category=safety")
        assert len(r_upper.json()) == len(r_lower.json())

    def test_filter_no_match_returns_empty(self):
        r = client.get("/erp/skus?category=NonExistentCategory")
        assert r.json() == []


class TestGetSKU:
    def test_known_sku_returns_200(self):
        r = client.get("/erp/skus/SKU-001")
        assert r.status_code == 200

    def test_known_sku_correct_category(self):
        r = client.get("/erp/skus/SKU-001")
        assert r.json()["category"] == "Packaging"

    def test_unknown_sku_returns_404(self):
        r = client.get("/erp/skus/SKU-UNKNOWN")
        assert r.status_code == 404

"""
Unit tests for core/config.py — RulesConfig loader.
Target coverage: core/config.py ≥ 85%
"""

import textwrap
from pathlib import Path

import pytest
import yaml

from core.config import (
    RulesConfig,
    RequiredFieldsConfig,
    TolerancesConfig,
    ValidationPoliciesConfig,
    ReportingConfig,
    LoggingConfig,
    get_rules,
    invalidate_rules_cache,
    reload_rules,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear LRU cache before and after every test."""
    invalidate_rules_cache()
    yield
    invalidate_rules_cache()


@pytest.fixture
def rules_yaml_path():
    """Return path to the real rules.yaml."""
    return str(Path(__file__).parent.parent / "config" / "rules.yaml")


@pytest.fixture
def minimal_yaml(tmp_path):
    """Write a minimal valid rules.yaml to a temp file."""
    content = textwrap.dedent("""\
        required_fields:
          header:
            - invoice_no
          line_item:
            - item_code
        accepted_currencies:
          - USD
        validation_policies:
          missing_field_action: flag
          total_mismatch_action: manual_review
          invalid_currency_action: reject
          auto_approve_confidence_threshold: 0.95
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(content)
    return str(p)


@pytest.fixture
def invalid_policy_yaml(tmp_path):
    content = textwrap.dedent("""\
        validation_policies:
          missing_field_action: explode
          total_mismatch_action: manual_review
          invalid_currency_action: reject
          auto_approve_confidence_threshold: 0.95
    """)
    p = tmp_path / "bad_rules.yaml"
    p.write_text(content)
    return str(p)


# ── get_rules — happy path ─────────────────────────────────────────────────────

class TestGetRules:
    def test_loads_real_rules_yaml(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert isinstance(rules, RulesConfig)

    def test_required_header_fields_present(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert "invoice_no" in rules.required_fields.header
        assert "vendor_id" in rules.required_fields.header

    def test_required_line_item_fields_present(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert "item_code" in rules.required_fields.line_item
        assert "qty" in rules.required_fields.line_item

    def test_tolerances_price_percent(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.tolerances.price_difference_percent == 5.0

    def test_tolerances_quantity_percent(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.tolerances.quantity_difference_percent == 0.0

    def test_accepted_currencies_contains_usd(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert "USD" in rules.accepted_currencies

    def test_currency_symbol_map_dollar(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.currency_symbol_map.get("$") == "USD"

    def test_currency_symbol_map_euro(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.currency_symbol_map.get("€") == "EUR"

    def test_auto_approve_threshold(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.validation_policies.auto_approve_confidence_threshold == 0.95

    def test_report_format_html(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.reporting.report_format == "HTML"

    def test_logging_level_info(self, rules_yaml_path):
        rules = get_rules(rules_yaml_path)
        assert rules.logging.log_level == "INFO"

    def test_result_is_cached(self, rules_yaml_path):
        r1 = get_rules(rules_yaml_path)
        r2 = get_rules(rules_yaml_path)
        assert r1 is r2

    def test_minimal_yaml_loads(self, minimal_yaml):
        rules = get_rules(minimal_yaml)
        assert "USD" in rules.accepted_currencies

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_rules(str(tmp_path / "nonexistent.yaml"))


# ── invalidate_rules_cache ─────────────────────────────────────────────────────

class TestInvalidateCache:
    def test_cache_cleared_returns_new_object(self, rules_yaml_path):
        r1 = get_rules(rules_yaml_path)
        invalidate_rules_cache()
        r2 = get_rules(rules_yaml_path)
        # New object after cache clear
        assert r1 == r2  # same values
        assert r1 is not r2  # but different instance


# ── reload_rules ───────────────────────────────────────────────────────────────

class TestReloadRules:
    def test_reload_returns_rules_config(self, rules_yaml_path):
        rules = reload_rules(rules_yaml_path)
        assert isinstance(rules, RulesConfig)

    def test_reload_reflects_file_changes(self, tmp_path):
        p = tmp_path / "rules.yaml"
        p.write_text("accepted_currencies:\n  - USD\n")
        r1 = get_rules(str(p))
        assert r1.accepted_currencies == ["USD"]

        p.write_text("accepted_currencies:\n  - USD\n  - GBP\n")
        r2 = reload_rules(str(p))
        assert "GBP" in r2.accepted_currencies


# ── Pydantic sub-models ────────────────────────────────────────────────────────

class TestSubModels:
    def test_required_fields_defaults(self):
        m = RequiredFieldsConfig()
        assert m.header == []
        assert m.line_item == []

    def test_tolerances_defaults(self):
        t = TolerancesConfig()
        assert t.price_difference_percent == 5.0
        assert t.quantity_difference_percent == 0.0
        assert t.tax_difference_percent == 2.0

    def test_validation_policies_defaults(self):
        p = ValidationPoliciesConfig()
        assert p.missing_field_action == "flag"
        assert p.total_mismatch_action == "manual_review"
        assert p.invalid_currency_action == "reject"

    def test_reporting_defaults(self):
        r = ReportingConfig()
        assert r.include_translation_confidence is True
        assert r.report_format == "HTML"

    def test_logging_defaults(self):
        lg = LoggingConfig()
        assert lg.log_level == "INFO"
        assert lg.enable_audit_log is True

    def test_invalid_policy_action_raises(self, invalid_policy_yaml):
        with pytest.raises(Exception):
            get_rules(invalid_policy_yaml)

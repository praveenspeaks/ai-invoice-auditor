"""
Configuration loader for rules.yaml.
Provides typed access to all validation rules, thresholds and policies.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Sub-models ────────────────────────────────────────────────────────────────

class RequiredFieldsConfig(BaseModel):
    header: list[str] = Field(default_factory=list)
    line_item: list[str] = Field(default_factory=list)


class TolerancesConfig(BaseModel):
    price_difference_percent: float = 5.0
    quantity_difference_percent: float = 0.0
    tax_difference_percent: float = 2.0


class ValidationPoliciesConfig(BaseModel):
    missing_field_action: str = "flag"
    total_mismatch_action: str = "manual_review"
    invalid_currency_action: str = "reject"
    auto_approve_confidence_threshold: float = 0.95


class ReportingConfig(BaseModel):
    include_translation_confidence: bool = True
    include_discrepancy_summary: bool = True
    report_format: str = "HTML"
    output_dir: str = "./outputs/reports"


class LoggingConfig(BaseModel):
    enable_audit_log: bool = True
    log_file: str = "./logs/invoice_auditor.log"
    log_level: str = "INFO"


# ── Root config model ──────────────────────────────────────────────────────────

class RulesConfig(BaseModel):
    required_fields: RequiredFieldsConfig = Field(default_factory=RequiredFieldsConfig)
    data_types: dict[str, str] = Field(default_factory=dict)
    tolerances: TolerancesConfig = Field(default_factory=TolerancesConfig)
    accepted_currencies: list[str] = Field(default_factory=list)
    currency_symbol_map: dict[str, str] = Field(default_factory=dict)
    validation_policies: ValidationPoliciesConfig = Field(default_factory=ValidationPoliciesConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("validation_policies")
    @classmethod
    def validate_policies(cls, v: ValidationPoliciesConfig) -> ValidationPoliciesConfig:
        valid_actions = {"flag", "manual_review", "reject"}
        for action in [v.missing_field_action, v.total_mismatch_action, v.invalid_currency_action]:
            if action not in valid_actions:
                raise ValueError(f"Invalid policy action '{action}'. Must be one of {valid_actions}")
        return v


# ── Loader ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "rules.yaml"


@lru_cache(maxsize=1)
def get_rules(config_path: str | None = None) -> RulesConfig:
    """
    Load and parse rules.yaml into a typed RulesConfig.
    Result is cached — call invalidate_rules_cache() to force reload.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Rules config not found at: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return RulesConfig(**raw)


def invalidate_rules_cache() -> None:
    """Force reload of rules.yaml on next get_rules() call."""
    get_rules.cache_clear()


def reload_rules(config_path: str | None = None) -> RulesConfig:
    """Clear cache and reload rules.yaml. Useful for dynamic rule updates."""
    invalidate_rules_cache()
    return get_rules(config_path)

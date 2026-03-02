"""
RAI Guardrails — AI Invoice Auditor
Responsible AI checks applied before LLM processing:

  1. Prompt Injection Detection
     Scans invoice text for patterns that attempt to hijack LLM behaviour.
     If detected → flag in errors, skip LLM extraction.

  2. PII Sensitivity Warning
     Detects clear PII patterns (SSN, credit card, passport numbers) that
     should not be passed to external LLM APIs.

  3. RAG Domain Restriction
     Enforced at the system-prompt level in response_synthesizer_tool.py.

Public API:
    check_injection(text) -> GuardrailResult
    check_pii(text)       -> GuardrailResult

GuardrailResult = {"flagged": bool, "reason": str, "patterns_found": list[str]}
"""

from __future__ import annotations

import re
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)


# ── Prompt Injection Patterns ─────────────────────────────────────────────────

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", "ignore previous instructions"),
    (r"you\s+are\s+now\s+", "you are now"),
    (r"\bdisregard\b.{0,40}\binstructions?\b", "disregard instructions"),
    (r"system\s+prompt", "system prompt reference"),
    (r"\bact\s+as\b.{0,30}\b(AI|assistant|GPT|Claude)\b", "act as AI persona"),
    (r"forget\s+(everything|all)\s+(you|I)\s+", "forget everything"),
    (r"new\s+role[:\s]", "new role assignment"),
    (r"DAN\s+mode", "DAN jailbreak"),
    (r"jailbreak", "jailbreak attempt"),
    (r"override\s+(safety|ethical|content)\s+(filter|guidelines?|policy)", "safety override"),
    (r"<\s*/?system\s*>", "system tag injection"),
    (r"\[\s*INST\s*\]", "instruction tag injection"),
]

# ── PII Patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "US Social Security Number"),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "Visa card number"),
    (r"\b5[1-5][0-9]{14}\b", "Mastercard number"),
    (r"\b3[47][0-9]{13}\b", "Amex card number"),
    (r"\b[A-Z]{1,2}[0-9]{6,9}\b", "Passport number"),
    (r"\b\d{16}\b", "16-digit card number"),
]


# ── Result type ───────────────────────────────────────────────────────────────

def _make_result(flagged: bool, reason: str, patterns: list[str]) -> dict[str, Any]:
    return {"flagged": flagged, "reason": reason, "patterns_found": patterns}


# ── Public API ─────────────────────────────────────────────────────────────────

def check_injection(text: str) -> dict[str, Any]:
    """
    Scan invoice text for prompt injection patterns.

    Returns:
        {"flagged": bool, "reason": str, "patterns_found": list[str]}
    """
    if not text or not text.strip():
        return _make_result(False, "", [])

    found = []
    for pattern, label in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)

    if found:
        reason = f"SECURITY: Prompt injection attempt detected — {', '.join(found)}"
        logger.warning(reason)
        return _make_result(True, reason, found)

    return _make_result(False, "", [])


def check_pii(text: str) -> dict[str, Any]:
    """
    Scan invoice text for sensitive PII that should not go to external LLMs.

    Returns:
        {"flagged": bool, "reason": str, "patterns_found": list[str]}
    """
    if not text or not text.strip():
        return _make_result(False, "", [])

    found = []
    for pattern, label in _PII_PATTERNS:
        if re.search(pattern, text):
            found.append(label)

    if found:
        reason = f"PII_WARNING: Sensitive data detected — {', '.join(found)}"
        logger.warning(reason)
        return _make_result(True, reason, found)

    return _make_result(False, "", [])


def run_all_checks(text: str) -> list[str]:
    """
    Run all guardrail checks and return a list of error/warning strings.
    Returns an empty list if no issues are found.
    """
    issues: list[str] = []

    injection = check_injection(text)
    if injection["flagged"]:
        issues.append(injection["reason"])

    pii = check_pii(text)
    if pii["flagged"]:
        issues.append(pii["reason"])

    return issues

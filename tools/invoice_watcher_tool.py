"""
Invoice-Watcher Tool — AI Invoice Auditor
Polls a directory for new invoice files and their .meta.json sidecars.
Maintains a processed-files registry to prevent re-processing.
"""

import json
import os
from pathlib import Path
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)

# Extensions treated as invoice files
INVOICE_EXTENSIONS = {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".tiff"}

# Default registry file path
DEFAULT_REGISTRY = Path("./data/processed_registry.json")


# ── Public API ─────────────────────────────────────────────────────────────────

def watch(incoming_dir: str, registry_path: Optional[str] = None) -> list[dict]:
    """
    Scan incoming_dir for new, unprocessed invoice files.

    Returns a list of invoice descriptors:
        [{"file_path": str, "meta": dict, "file_format": str}, ...]

    Each returned file is immediately registered as processed so subsequent
    calls to watch() will not return the same file again.

    Args:
        incoming_dir:   Directory to poll for new invoices.
        registry_path:  Path to the JSON registry file.
                        Defaults to DEFAULT_REGISTRY.
    """
    dir_path = Path(incoming_dir)
    if not dir_path.exists():
        logger.warning("Incoming directory does not exist: %s", incoming_dir)
        return []

    reg_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY
    registry = _load_registry(reg_path)
    new_invoices: list[dict] = []

    for file in sorted(dir_path.iterdir()):
        if not _is_invoice_file(file):
            continue
        key = str(file.resolve())
        if key in registry:
            continue  # already processed

        meta = _load_meta(file)
        file_format = _get_format(file)
        descriptor = {
            "file_path": str(file),
            "meta": meta,
            "file_format": file_format,
        }
        new_invoices.append(descriptor)
        registry[key] = True
        logger.info("New invoice detected: %s (format=%s, lang=%s)",
                    file.name, file_format, meta.get("language", "?"))

    _save_registry(reg_path, registry)
    return new_invoices


def mark_processed(file_path: str, registry_path: Optional[str] = None) -> None:
    """Manually mark a single file as processed in the registry."""
    reg_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY
    registry = _load_registry(reg_path)
    registry[str(Path(file_path).resolve())] = True
    _save_registry(reg_path, registry)


def reset_registry(registry_path: Optional[str] = None) -> None:
    """Clear the processed-files registry (useful for re-processing in tests/demo)."""
    reg_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY
    _save_registry(reg_path, {})
    logger.info("Processed registry cleared: %s", reg_path)


def is_processed(file_path: str, registry_path: Optional[str] = None) -> bool:
    """Return True if the file has already been registered as processed."""
    reg_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY
    registry = _load_registry(reg_path)
    return str(Path(file_path).resolve()) in registry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_invoice_file(path: Path) -> bool:
    """Return True if the path is a regular file with a supported invoice extension."""
    return path.is_file() and path.suffix.lower() in INVOICE_EXTENSIONS


def _load_meta(invoice_path: Path) -> dict:
    """Load the .meta.json sidecar for an invoice file, or return empty dict."""
    meta_path = invoice_path.with_suffix("").parent / (invoice_path.stem + ".meta.json")
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load meta for %s: %s", invoice_path.name, exc)
    return {}


def _get_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".docx", ".doc"}:
        return "docx"
    return "image"


def _load_registry(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

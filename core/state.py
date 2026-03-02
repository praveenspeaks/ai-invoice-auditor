"""
Shared LangGraph state for the AI Invoice Auditor pipeline.
All agents read from and write to this typed state dictionary.
"""

from typing import Annotated
from typing_extensions import TypedDict
import operator


def _merge_lists(a: list, b: list) -> list:
    """Merge lists by appending — used for errors and discrepancies."""
    return a + b


class InvoiceState(TypedDict, total=False):
    # --- Ingestion ---
    file_path: str
    file_format: str              # "pdf" | "docx" | "image"
    meta: dict                    # parsed .meta.json sidecar

    # --- Extraction ---
    raw_text: str
    detected_language: str        # ISO 639-1 code e.g. "en", "es", "de"

    # --- Translation ---
    translated_text: str
    translation_confidence: float  # 0.0 – 1.0

    # --- Field Extraction ---
    extracted_fields: dict        # invoice_no, vendor_id, line_items, etc.

    # --- Data Validation ---
    validation_result: dict       # missing_fields, type_errors, currency_status

    # --- Business Validation ---
    erp_data: dict                # raw ERP API response
    discrepancies: Annotated[list, _merge_lists]  # list of discrepancy dicts

    # --- Decision ---
    recommendation: str           # "AUTO_APPROVED" | "MANUAL_REVIEW" | "REJECTED"
    human_review_required: bool

    # --- Reporting ---
    report_path: str

    # --- RAG ---
    rag_indexed: bool
    rag_query: str                # populated when running RAG subgraph
    rag_chunks: list              # retrieved chunks
    rag_answer: str               # generated answer
    rag_sources: list             # source invoice references
    rag_scores: dict              # context_relevance, groundedness, answer_relevance

    # --- Audit ---
    pipeline_start_time: str
    errors: Annotated[list, _merge_lists]


def initial_state(file_path: str, meta: dict, file_format: str) -> InvoiceState:
    """Create a fresh InvoiceState for a newly detected invoice."""
    import datetime
    return InvoiceState(
        file_path=file_path,
        file_format=file_format,
        meta=meta,
        raw_text="",
        detected_language="",
        translated_text="",
        translation_confidence=0.0,
        extracted_fields={},
        validation_result={},
        erp_data={},
        discrepancies=[],
        recommendation="",
        human_review_required=False,
        report_path="",
        rag_indexed=False,
        rag_query="",
        rag_chunks=[],
        rag_answer="",
        rag_sources=[],
        rag_scores={},
        pipeline_start_time=datetime.datetime.now(datetime.UTC).isoformat(),
        errors=[],
    )

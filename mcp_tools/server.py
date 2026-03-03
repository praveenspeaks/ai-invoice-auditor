"""
FastMCP Tool Registry — AI Invoice Auditor
Exposes all 10 pipeline tools as MCP-compatible tools that can be
discovered and invoked by any MCP-compatible host (Claude Desktop, etc.).

Run:
    python mcp/server.py           # stdio transport (for Claude Desktop)
    python mcp/server.py --sse     # SSE transport (HTTP, for testing)

Lists tools:
    mcp dev mcp/server.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Ensure project root is on the path when run directly
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

mcp = FastMCP(
    "AI Invoice Auditor Tool Registry",
    instructions=(
        "Tools for processing, validating, and querying business invoices. "
        "The pipeline ingests PDF/DOCX/image invoices, extracts structured fields, "
        "translates non-English documents, validates against ERP data, "
        "generates audit reports, and answers natural-language questions via RAG."
    ),
)


# ── 1. Invoice Watcher Tool ────────────────────────────────────────────────────

@mcp.tool()
def invoice_watcher(incoming_dir: str = "./data/incoming") -> list[dict]:
    """
    Monitor an incoming directory for new unprocessed invoice files.

    Returns a list of new invoice records: {file_path, meta_path, file_format}.
    Maintains a processed registry to avoid re-processing.
    """
    from tools.invoice_watcher_tool import watch
    return watch(incoming_dir)


# ── 2. Data Harvester Tool ─────────────────────────────────────────────────────

@mcp.tool()
def data_harvester(file_path: str) -> dict:
    """
    Extract raw text from a PDF, DOCX, or image (OCR) invoice file.

    Returns: {raw_text, file_format, detected_language, error}
    """
    from tools.data_harvester_tool import harvest
    return harvest(file_path)


# ── 3. Lang-Bridge Tool ────────────────────────────────────────────────────────

@mcp.tool()
def lang_bridge(text: str, source_language: str = "auto") -> dict:
    """
    Translate invoice text to English.
    If source_language is 'en', returns the original text unchanged (confidence=1.0).

    Returns: {translated_text, source_language, confidence, error}
    """
    from tools.lang_bridge_tool import translate
    return translate(text, source_language)


# ── 4. Data Completeness Checker ──────────────────────────────────────────────

@mcp.tool()
def data_completeness_checker(extracted_fields: dict) -> dict:
    """
    Validate extracted invoice fields against rules.yaml (3-pass validation).

    Pass 1: Required fields (header + line-item level)
    Pass 2: Data type validation (date, float, str)
    Pass 3: Currency normalisation + accepted_currencies check

    Returns: {status, missing_fields, type_errors, currency_status, currency, passed, reject}
    """
    from tools.data_completeness_checker import check
    return check(extracted_fields)


# ── 5. Business Validation Tool ───────────────────────────────────────────────

@mcp.tool()
def business_validation(
    vendor_id: str,
    po_number: str,
    extracted_fields: dict,
) -> dict:
    """
    Cross-reference invoice line items against the ERP purchase order.

    Compares qty (exact), unit_price (±5%), and total (±5%) for each line item.
    Returns: {erp_data, discrepancies, error}
    """
    from tools.business_validation_tool import validate
    fields = {**extracted_fields, "vendor_id": vendor_id, "po_number": po_number}
    return validate(fields)


# ── 6. Insight Reporter Tool ──────────────────────────────────────────────────

@mcp.tool()
def insight_reporter(state: dict) -> str:
    """
    Generate an HTML audit report from the full invoice pipeline state.

    Report includes: extraction summary, data validation, business validation,
    final recommendation badge, and audit trail.

    Returns: absolute path to the generated HTML report file.
    """
    from tools.insight_reporter_tool import generate_report
    return generate_report(state)


# ── 7. Vector Indexer Tool ─────────────────────────────────────────────────────

@mcp.tool()
def vector_indexer(
    invoice_no: str,
    text: str,
    metadata: dict | None = None,
) -> dict:
    """
    Chunk, embed, and index invoice text into the ChromaDB vector store.
    Uses sentence-transformers (all-MiniLM-L6-v2) — fully local, no API key.

    Returns: {indexed_chunks, error}
    """
    from tools.vector_indexer_tool import index_invoice
    return index_invoice(invoice_no, text, metadata)


# ── 8. Semantic Retriever Tool ─────────────────────────────────────────────────

@mcp.tool()
def semantic_retriever(
    query: str,
    top_k: int = 5,
    invoice_no_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve the top-k most semantically similar invoice chunks for a query.

    Args:
        query:             Natural-language question about invoices.
        top_k:             Number of chunks to return (default 5).
        invoice_no_filter: Restrict results to a single invoice number.

    Returns: list of {text, metadata, distance, score}
    """
    from tools.semantic_retriever_tool import retrieve
    return retrieve(query=query, top_k=top_k, invoice_no_filter=invoice_no_filter)


# ── 9. Chunk Ranker Tool ──────────────────────────────────────────────────────

@mcp.tool()
def chunk_ranker(chunks: list[dict], threshold: float = 0.25) -> list[dict]:
    """
    Filter retrieved chunks below a similarity threshold and sort by score.

    Args:
        chunks:    Output of semantic_retriever (list of chunk dicts).
        threshold: Minimum cosine similarity score to keep (default 0.25).

    Returns: Filtered and sorted list (highest score first).
    """
    from tools.chunk_ranker_tool import rerank
    return rerank(chunks=chunks, threshold=threshold)


# ── 10. Response Synthesizer Tool ─────────────────────────────────────────────

@mcp.tool()
def response_synthesizer(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded English answer from ranked invoice chunks using an LLM.
    The answer cites source invoice numbers and stays strictly within context.

    Returns: {answer, sources, error}
    """
    from tools.response_synthesizer_tool import synthesize
    return synthesize(query=query, chunks=chunks)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    mcp.run(transport=transport)

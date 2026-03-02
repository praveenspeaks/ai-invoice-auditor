"""
Indexing Agent — AI Invoice Auditor RAG
LangGraph node that indexes translated invoice text into ChromaDB.
Runs after translation in the core pipeline (or as a standalone RAG step).

Reads:  translated_text, extracted_fields (for metadata), file_path
Writes: rag_indexed
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.vector_indexer_tool import index_invoice

logger = get_logger(__name__)


def indexing_agent(state: InvoiceState) -> dict:
    """LangGraph node — Indexing Agent."""
    text = state.get("translated_text") or state.get("raw_text", "")
    if not text.strip():
        logger.warning("indexing_agent: no text to index")
        return {"rag_indexed": False}

    extracted = state.get("extracted_fields") or {}
    invoice_no = extracted.get("invoice_no") or state.get("file_path", "unknown")

    metadata = {
        "file_path":    state.get("file_path", ""),
        "file_format":  state.get("file_format", ""),
        "language":     state.get("detected_language", ""),
        "vendor_id":    extracted.get("vendor_id", ""),
        "invoice_date": extracted.get("invoice_date", ""),
    }

    result = index_invoice(invoice_no, text, metadata)

    if result["error"]:
        logger.error("Indexing failed: %s", result["error"])
        return {
            "rag_indexed": False,
            "errors": [f"RAG_INDEX_ERROR: {result['error']}"],
        }

    logger.info("Indexed %d chunks for %s", result["indexed_chunks"], invoice_no)
    return {"rag_indexed": True}

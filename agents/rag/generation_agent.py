"""
Generation Agent — AI Invoice Auditor RAG
LangGraph node that synthesizes a grounded answer from ranked chunks.

Reads:  rag_query, rag_chunks
Writes: rag_answer, rag_sources
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.response_synthesizer_tool import synthesize

logger = get_logger(__name__)


def generation_agent(state: InvoiceState) -> dict:
    """LangGraph node — Generation Agent."""
    query = state.get("rag_query", "")
    chunks = state.get("rag_chunks") or []

    result = synthesize(query=query, chunks=chunks)

    updates: dict = {
        "rag_answer":  result["answer"],
        "rag_sources": result["sources"],
    }

    if result.get("error"):
        updates["errors"] = [f"RAG_GENERATION_ERROR: {result['error']}"]

    logger.info(
        "Generation complete. answer_len=%d sources=%s",
        len(result["answer"]), result["sources"],
    )
    return updates

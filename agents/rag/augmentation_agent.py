"""
Augmentation Agent — AI Invoice Auditor RAG
LangGraph node that filters and reranks retrieved chunks.

Reads:  rag_chunks
Writes: rag_chunks (filtered + sorted)
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.chunk_ranker_tool import rerank

logger = get_logger(__name__)


def augmentation_agent(state: InvoiceState) -> dict:
    """LangGraph node — Augmentation Agent."""
    chunks = state.get("rag_chunks") or []
    ranked = rerank(chunks)
    logger.info("Augmentation: %d → %d chunks after reranking", len(chunks), len(ranked))
    return {"rag_chunks": ranked}

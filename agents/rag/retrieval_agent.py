"""
Retrieval Agent — AI Invoice Auditor RAG
LangGraph node that retrieves semantically similar chunks for a query.

Reads:  rag_query, rag_query_filter (optional invoice_no)
Writes: rag_chunks
"""

from core.logger import get_logger
from core.state import InvoiceState
from tools.semantic_retriever_tool import retrieve

logger = get_logger(__name__)


def retrieval_agent(state: InvoiceState) -> dict:
    """LangGraph node — Retrieval Agent."""
    query = state.get("rag_query", "")
    if not query:
        logger.warning("retrieval_agent: rag_query is empty")
        return {"rag_chunks": []}

    invoice_filter = state.get("rag_query_filter")  # optional field
    chunks = retrieve(query=query, top_k=5, invoice_no_filter=invoice_filter)
    logger.info("Retrieved %d chunks for query: %r", len(chunks), query[:60])
    return {"rag_chunks": chunks}

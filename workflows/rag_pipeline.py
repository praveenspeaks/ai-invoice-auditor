"""
RAG Pipeline — AI Invoice Auditor
LangGraph StateGraph wiring the 4 RAG agents into a sequential sub-pipeline:

    retrieval → augmentation → generation → reflection

Can be run standalone (inject rag_query) or composed into the main pipeline
after the indexing step.

Public API:
    build_rag_pipeline()   -> CompiledStateGraph
    run_rag_query(query, invoice_no_filter) -> InvoiceState
"""

from langgraph.graph import StateGraph, END

from core.logger import get_logger
from core.state import InvoiceState
from agents.rag.retrieval_agent import retrieval_agent
from agents.rag.augmentation_agent import augmentation_agent
from agents.rag.generation_agent import generation_agent
from agents.rag.reflection_agent import reflection_agent

logger = get_logger(__name__)


def build_rag_pipeline():
    """Construct and compile the RAG sub-pipeline StateGraph."""
    graph = StateGraph(InvoiceState)

    graph.add_node("retrieve",   retrieval_agent)
    graph.add_node("augment",    augmentation_agent)
    graph.add_node("generate",   generation_agent)
    graph.add_node("reflect",    reflection_agent)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "augment")
    graph.add_edge("augment",  "generate")
    graph.add_edge("generate", "reflect")
    graph.add_edge("reflect",  END)

    return graph.compile()


def run_rag_query(
    query: str,
    invoice_no_filter: str | None = None,
    seed_state: InvoiceState | None = None,
) -> InvoiceState:
    """
    Run a RAG query against the indexed invoice vector store.

    Args:
        query:             Natural-language question about invoices.
        invoice_no_filter: Optional invoice number to restrict retrieval scope.
        seed_state:        Optional base state to merge into (for composability).

    Returns:
        Final InvoiceState with rag_answer, rag_sources, and rag_scores populated.
    """
    pipeline = build_rag_pipeline()

    base: InvoiceState = dict(seed_state) if seed_state else {}  # type: ignore[assignment]
    base["rag_query"] = query
    base["errors"] = base.get("errors") or []
    base["discrepancies"] = base.get("discrepancies") or []

    if invoice_no_filter:
        base["rag_query_filter"] = invoice_no_filter

    logger.info("RAG query started: %r (filter=%s)", query[:80], invoice_no_filter)
    final_state = pipeline.invoke(base)

    scores = final_state.get("rag_scores", {})
    logger.info(
        "RAG query finished. low_quality=%s answer_len=%d",
        scores.get("low_quality"),
        len(final_state.get("rag_answer", "")),
    )
    return final_state


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What invoices are available?"
    result = run_rag_query(q)
    print("\n=== RAG Result ===")
    print(f"Query:   {q}")
    print(f"Answer:  {result.get('rag_answer', '')}")
    print(f"Sources: {result.get('rag_sources', [])}")
    print(f"Scores:  {result.get('rag_scores', {})}")
    if result.get("errors"):
        print(f"Errors:  {result['errors']}")

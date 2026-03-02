"""
Response Synthesizer Tool — AI Invoice Auditor
Generates a grounded English answer from ranked invoice chunks using an LLM.

Public API:
    synthesize(query, chunks) -> {"answer": str, "sources": list[str], "error": str|None}
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger
from tools.field_extractor_tool import _build_llm   # shared LLM builder

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an invoice auditing assistant. "
    "Answer questions ONLY based on the provided invoice context. "
    "If the answer is not in the context, say: "
    "'I cannot find this information in the available invoices.' "
    "Do not make up any information. Always cite the source invoice number."
)


def _build_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        meta = c.get("metadata", {})
        inv_no = meta.get("invoice_no", "unknown")
        idx = meta.get("chunk_index", "?")
        parts.append(f"[Source: {inv_no}, Chunk {idx}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def synthesize(
    query: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate a grounded answer from ranked invoice chunks.

    Args:
        query:  Natural-language question.
        chunks: Output of chunk_ranker_tool.rerank() (must be non-empty).

    Returns:
        {"answer": str, "sources": list[str], "error": str|None}
    """
    if not chunks:
        return {
            "answer":  "No relevant invoice context found to answer this question.",
            "sources": [],
            "error":   None,
        }

    context = _build_context(chunks)
    sources = sorted({c.get("metadata", {}).get("invoice_no", "") for c in chunks} - {""})

    try:
        llm = _build_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        logger.info("Synthesized answer for: %r (sources=%s)", query[:60], sources)
        return {"answer": answer, "sources": sources, "error": None}

    except RuntimeError as exc:
        logger.warning("LLM not available for synthesis: %s", exc)
        # Graceful fallback: return the best chunk as the answer
        best = chunks[0]["text"][:500]
        return {
            "answer":  f"[LLM unavailable] Best matching context: {best}",
            "sources": sources,
            "error":   str(exc),
        }

    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        return {"answer": "", "sources": sources, "error": str(exc)}

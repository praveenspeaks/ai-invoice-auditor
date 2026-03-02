"""
Reflection Agent — AI Invoice Auditor RAG
LangGraph node that scores RAG answer quality using the RAG Triad:

  1. context_relevance  — mean similarity score of retrieved chunks (0–1)
  2. groundedness       — is the answer supported by the provided context? (0–1)
  3. answer_relevance   — does the answer actually address the question? (0–1)

Reads:  rag_query, rag_chunks, rag_answer
Writes: rag_scores
"""

from __future__ import annotations

from core.logger import get_logger
from core.state import InvoiceState

logger = get_logger(__name__)

_LOW_QUALITY_THRESHOLD = 0.6

_GROUNDEDNESS_PROMPT = """\
Given the following CONTEXT and ANSWER, score how well the answer is supported \
by the context on a scale from 0.0 to 1.0.

Rules:
- 1.0 = every claim in the answer is explicitly supported by the context
- 0.5 = the answer is partially supported
- 0.0 = the answer contradicts or ignores the context entirely

Respond with a single float only, no explanation.

CONTEXT:
{context}

ANSWER:
{answer}

Score (0.0–1.0):"""

_RELEVANCE_PROMPT = """\
Given the following QUESTION and ANSWER, score how well the answer addresses \
the question on a scale from 0.0 to 1.0.

Rules:
- 1.0 = the answer directly and completely addresses the question
- 0.5 = the answer is partially relevant
- 0.0 = the answer is completely off-topic

Respond with a single float only, no explanation.

QUESTION:
{query}

ANSWER:
{answer}

Score (0.0–1.0):"""


# ── scoring helpers ────────────────────────────────────────────────────────────

def _score_context_relevance(chunks: list[dict]) -> float:
    """Mean cosine similarity score of retrieved + ranked chunks."""
    if not chunks:
        return 0.0
    scores = [float(c.get("score", 0.0)) for c in chunks]
    return round(sum(scores) / len(scores), 4)


def _score_with_llm(prompt: str) -> float:
    """Call LLM and parse a float from its response. Returns -1.0 on failure."""
    try:
        from tools.field_extractor_tool import _build_llm  # shared builder
        from langchain_core.messages import HumanMessage

        llm = _build_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        return max(0.0, min(1.0, float(raw.strip())))
    except (RuntimeError, ValueError, TypeError):
        return -1.0  # sentinel — LLM unavailable or bad parse
    except Exception as exc:
        logger.warning("LLM scoring failed: %s", exc)
        return -1.0


def _heuristic_groundedness(answer: str, chunks: list[dict]) -> float:
    """
    Fallback when LLM is not available.
    Checks whether key n-grams from the answer appear in any chunk text.
    Simple but reasonable for CI / offline environments.
    """
    if not answer or not chunks:
        return 0.0
    context_blob = " ".join(c.get("text", "") for c in chunks).lower()
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return 0.5
    hits = sum(1 for w in words if w in context_blob)
    return round(hits / len(words), 4)


def _heuristic_answer_relevance(query: str, answer: str) -> float:
    """
    Fallback when LLM is not available.
    Checks keyword overlap between question and answer.
    """
    if not query or not answer:
        return 0.0
    q_words = {w.lower() for w in query.split() if len(w) > 3}
    a_words = {w.lower() for w in answer.split() if len(w) > 3}
    if not q_words:
        return 0.5
    overlap = len(q_words & a_words)
    return round(min(1.0, overlap / len(q_words)), 4)


# ── main node ──────────────────────────────────────────────────────────────────

def reflection_agent(state: InvoiceState) -> dict:
    """LangGraph node — Reflection Agent (RAG Triad scoring)."""
    query   = state.get("rag_query", "")
    chunks  = state.get("rag_chunks") or []
    answer  = state.get("rag_answer", "")

    # 1. Context relevance — always computable from chunk scores
    context_relevance = _score_context_relevance(chunks)

    # 2. Groundedness — LLM preferred, heuristic fallback
    context_text = "\n\n".join(c.get("text", "")[:300] for c in chunks[:5])
    groundedness = _score_with_llm(
        _GROUNDEDNESS_PROMPT.format(context=context_text, answer=answer)
    )
    if groundedness < 0:
        groundedness = _heuristic_groundedness(answer, chunks)
        logger.info("Groundedness: using heuristic fallback (%.2f)", groundedness)

    # 3. Answer relevance — LLM preferred, heuristic fallback
    answer_relevance = _score_with_llm(
        _RELEVANCE_PROMPT.format(query=query, answer=answer)
    )
    if answer_relevance < 0:
        answer_relevance = _heuristic_answer_relevance(query, answer)
        logger.info("Answer relevance: using heuristic fallback (%.2f)", answer_relevance)

    rag_scores = {
        "context_relevance": context_relevance,
        "groundedness":      groundedness,
        "answer_relevance":  answer_relevance,
        "low_quality":       any(
            s < _LOW_QUALITY_THRESHOLD
            for s in [context_relevance, groundedness, answer_relevance]
        ),
    }

    logger.info(
        "RAG Triad scores — context=%.2f groundedness=%.2f relevance=%.2f low_quality=%s",
        context_relevance, groundedness, answer_relevance, rag_scores["low_quality"],
    )

    return {"rag_scores": rag_scores}

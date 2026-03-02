"""
Chunk Ranker Tool — AI Invoice Auditor
Reranks and filters retrieved chunks by cosine similarity score.

Public API:
    rerank(chunks, threshold) -> list[dict]

Chunks below the threshold are discarded; survivors are sorted
descending by score (most relevant first).
"""

from __future__ import annotations

from core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_THRESHOLD = 0.25  # cosine similarity threshold (0–1)


def rerank(
    chunks: list[dict],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    Filter chunks below the similarity threshold and sort by score.

    Args:
        chunks:    Output of semantic_retriever_tool.retrieve().
        threshold: Minimum similarity score (0–1) to keep a chunk.

    Returns:
        Filtered and re-sorted list (highest score first).
    """
    if not chunks:
        return []

    kept = [c for c in chunks if c.get("score", 0.0) >= threshold]
    ranked = sorted(kept, key=lambda c: c.get("score", 0.0), reverse=True)

    logger.info(
        "Chunk ranker: %d/%d chunks kept (threshold=%.2f)",
        len(ranked), len(chunks), threshold,
    )
    return ranked

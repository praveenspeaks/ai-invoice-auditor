"""
Semantic Retriever Tool — AI Invoice Auditor
Encodes a query and retrieves the top-k most similar chunks from ChromaDB.

Public API:
    retrieve(query, top_k, invoice_no_filter) -> list[dict]

Each result dict:
    {"text": str, "metadata": dict, "distance": float, "score": float}
    score = 1 - distance  (cosine similarity, higher is better)
"""

from __future__ import annotations

from typing import Any

from core.logger import get_logger
from tools.vector_indexer_tool import get_collection, _get_embed_model

logger = get_logger(__name__)


def retrieve(
    query: str,
    top_k: int = 5,
    invoice_no_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most semantically similar chunks for a query.

    Args:
        query:              Natural-language question.
        top_k:              Number of chunks to return.
        invoice_no_filter:  If set, restrict results to a single invoice.

    Returns:
        List of result dicts sorted by similarity (highest first).
        Empty list on error or empty index.
    """
    if not query or not query.strip():
        logger.warning("retrieve() called with empty query")
        return []

    try:
        model = _get_embed_model()
        collection = get_collection()

        if collection.count() == 0:
            logger.warning("ChromaDB collection is empty — nothing to retrieve")
            return []

        query_embedding = model.encode([query], convert_to_numpy=True)[0]

        where = {"invoice_no": invoice_no_filter} if invoice_no_filter else None

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append({
                "text":     doc,
                "metadata": meta,
                "distance": dist,
                "score":    round(1.0 - dist, 6),
            })

        # Already sorted by distance (ascending = best first in cosine space)
        logger.info("Retrieved %d chunks for query: %r", len(chunks), query[:60])
        return chunks

    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []

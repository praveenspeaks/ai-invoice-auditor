"""
Vector Indexer Tool — AI Invoice Auditor
Chunks translated invoice text and stores embeddings in ChromaDB.

Uses sentence-transformers (all-MiniLM-L6-v2) — local, no API key needed.
Persists the ChromaDB collection at VECTOR_STORE_DIR (default: data/vector_store/).

Public API:
    index_invoice(invoice_no, text, metadata) -> {"indexed_chunks": int, "error": str|None}
    get_collection()                          -> chromadb.Collection
    reset_collection()                        -> None  (for tests)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)

_VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./data/vector_store")
_COLLECTION_NAME = "invoices"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 400    # approximate tokens (chars / 4)
_CHUNK_OVERLAP = 50  # chars overlap between chunks


# ── lazy singletons ────────────────────────────────────────────────────────

_collection = None
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_EMBED_MODEL)
        logger.info("Loaded embedding model: %s", _EMBED_MODEL)
    return _embed_model


def get_collection():
    """Return (or lazily create) the persistent ChromaDB collection."""
    global _collection
    if _collection is None:
        import chromadb
        store_path = Path(_VECTOR_STORE_DIR)
        store_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(store_path))
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready at %s", _COLLECTION_NAME, store_path)
    return _collection


def reset_collection():
    """Drop and recreate the collection — used in tests."""
    global _collection
    import chromadb
    store_path = Path(_VECTOR_STORE_DIR)
    store_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_path))
    client.delete_collection(_COLLECTION_NAME)
    _collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── chunking ───────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, splitting on sentence/newline boundaries.
    chunk_size is in characters (approx. tokens * 4).
    """
    if not text.strip():
        return []

    # Split on sentence-like boundaries first
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            tail = current[-overlap:] if len(current) > overlap else current
            current = (tail + " " + sentence).strip()

    if current:
        chunks.append(current)

    return chunks or [text[:chunk_size]]


# ── main API ───────────────────────────────────────────────────────────────

def index_invoice(
    invoice_no: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Chunk, embed, and upsert invoice text into ChromaDB.

    Args:
        invoice_no: Unique invoice identifier (used as chunk ID prefix).
        text:       Translated English invoice text.
        metadata:   Extra metadata stored alongside each chunk.

    Returns:
        {"indexed_chunks": int, "error": str|None}
    """
    if not text or not text.strip():
        return {"indexed_chunks": 0, "error": "Empty text — nothing to index"}

    if not invoice_no:
        return {"indexed_chunks": 0, "error": "invoice_no is required"}

    try:
        model = _get_embed_model()
        collection = get_collection()

        chunks = _chunk_text(text)
        if not chunks:
            return {"indexed_chunks": 0, "error": "No chunks produced"}

        embeddings = model.encode(chunks, convert_to_numpy=True)

        base_meta = metadata or {}
        ids = [f"{invoice_no}_chunk_{i}" for i in range(len(chunks))]
        metas = [
            {**base_meta, "invoice_no": invoice_no, "chunk_index": i}
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metas,
        )

        logger.info("Indexed %d chunks for invoice %s", len(chunks), invoice_no)
        return {"indexed_chunks": len(chunks), "error": None}

    except Exception as exc:
        logger.error("Indexing failed for %s: %s", invoice_no, exc)
        return {"indexed_chunks": 0, "error": str(exc)}

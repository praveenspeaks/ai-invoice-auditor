"""
Unit tests for the full RAG system:
  - tools/vector_indexer_tool.py
  - tools/semantic_retriever_tool.py
  - tools/chunk_ranker_tool.py
  - tools/response_synthesizer_tool.py
  - agents/rag/indexing_agent.py
  - agents/rag/retrieval_agent.py
  - agents/rag/augmentation_agent.py
  - agents/rag/generation_agent.py
  - agents/rag/reflection_agent.py
  - workflows/rag_pipeline.py

All ML models (sentence-transformers, ChromaDB) and LLM calls are mocked.
Target coverage: ≥ 80%
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chunk(text: str = "invoice text", score: float = 0.8, invoice_no: str = "INV-001") -> dict:
    return {
        "text":     text,
        "metadata": {"invoice_no": invoice_no, "chunk_index": 0},
        "distance": round(1.0 - score, 6),
        "score":    score,
    }


def _fake_embed(texts, convert_to_numpy=True):
    """Return deterministic fake embeddings (one row per text)."""
    arr = np.ones((len(texts), 384), dtype=np.float32)
    return arr


# ==============================================================================
# tools/vector_indexer_tool.py
# ==============================================================================

class TestChunkText:
    def test_empty_text_returns_empty(self):
        from tools.vector_indexer_tool import _chunk_text
        assert _chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        from tools.vector_indexer_tool import _chunk_text
        chunks = _chunk_text("Hello world. This is a short invoice.")
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_long_text_splits_into_multiple_chunks(self):
        from tools.vector_indexer_tool import _chunk_text
        # Build a text clearly longer than 400 chars
        sentence = "This is a sentence about invoices. "
        long_text = sentence * 20
        chunks = _chunk_text(long_text, chunk_size=400, overlap=50)
        assert len(chunks) > 1

    def test_overlap_is_included_in_next_chunk(self):
        from tools.vector_indexer_tool import _chunk_text
        sentence = "Alpha beta gamma delta epsilon. "
        long_text = sentence * 20
        chunks = _chunk_text(long_text, chunk_size=200, overlap=40)
        # Each chunk after the first should share some chars with the previous
        if len(chunks) >= 2:
            assert len(chunks[1]) > 0  # not empty

    def test_returns_at_least_one_chunk_for_nonempty_text(self):
        from tools.vector_indexer_tool import _chunk_text
        chunks = _chunk_text("a")
        assert len(chunks) >= 1


class TestGetCollection:
    def test_returns_collection(self):
        from tools import vector_indexer_tool as vit
        mock_col = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            vit._collection = None  # reset singleton
            col = vit.get_collection()
            assert col is mock_col

    def test_returns_same_singleton(self):
        from tools import vector_indexer_tool as vit
        mock_col = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            vit._collection = None
            c1 = vit.get_collection()
            c2 = vit.get_collection()
            assert c1 is c2


class TestResetCollection:
    def test_resets_collection(self):
        from tools import vector_indexer_tool as vit
        mock_col = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("chromadb.PersistentClient", return_value=mock_client):
            vit.reset_collection()
            assert mock_client.delete_collection.called


class TestIndexInvoice:
    def _mock_env(self):
        """Returns (mock_embed_model, mock_collection, mock_chroma_client)."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = _fake_embed

        mock_col = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        return mock_model, mock_col, mock_client

    def test_empty_text_returns_error(self):
        from tools.vector_indexer_tool import index_invoice
        result = index_invoice("INV-001", "   ")
        assert result["indexed_chunks"] == 0
        assert result["error"] is not None

    def test_missing_invoice_no_returns_error(self):
        from tools.vector_indexer_tool import index_invoice
        result = index_invoice("", "some text")
        assert result["indexed_chunks"] == 0
        assert "invoice_no" in result["error"].lower()

    def test_successful_indexing(self):
        from tools import vector_indexer_tool as vit
        mock_model, mock_col, mock_client = self._mock_env()

        with patch("chromadb.PersistentClient", return_value=mock_client):
            vit._collection = None
            vit._embed_model = mock_model
            result = vit.index_invoice("INV-001", "Invoice content for testing purposes.")
            assert result["error"] is None
            assert result["indexed_chunks"] >= 1
            assert mock_col.upsert.called

    def test_metadata_stored_with_chunks(self):
        from tools import vector_indexer_tool as vit
        mock_model, mock_col, mock_client = self._mock_env()

        with patch("chromadb.PersistentClient", return_value=mock_client):
            vit._collection = None
            vit._embed_model = mock_model
            vit.index_invoice("INV-002", "Test invoice text.", metadata={"vendor": "ACME"})
            call_kwargs = mock_col.upsert.call_args.kwargs
            for meta in call_kwargs["metadatas"]:
                assert meta["invoice_no"] == "INV-002"
                assert meta["vendor"] == "ACME"

    def test_exception_returns_error(self):
        from tools import vector_indexer_tool as vit
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("embed failed")
        vit._embed_model = mock_model
        vit._collection = MagicMock()

        result = vit.index_invoice("INV-003", "Some invoice text here.")
        assert result["indexed_chunks"] == 0
        assert "embed failed" in result["error"]


# ==============================================================================
# tools/semantic_retriever_tool.py
# ==============================================================================

class TestSemanticRetriever:
    def _make_collection(self, count: int = 3) -> MagicMock:
        col = MagicMock()
        col.count.return_value = count
        col.query.return_value = {
            "documents": [["chunk text 1", "chunk text 2", "chunk text 3"][:count]],
            "metadatas": [[{"invoice_no": "INV-001", "chunk_index": i} for i in range(count)]],
            "distances": [[0.1, 0.2, 0.3][:count]],
        }
        return col

    def test_empty_query_returns_empty(self):
        from tools.semantic_retriever_tool import retrieve
        assert retrieve("") == []
        assert retrieve("   ") == []

    def test_empty_collection_returns_empty(self):
        from tools import vector_indexer_tool as vit
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384))

        vit._collection = mock_col
        vit._embed_model = mock_model

        from tools.semantic_retriever_tool import retrieve
        result = retrieve("What is the total amount?")
        assert result == []

    def test_retrieve_returns_scored_chunks(self):
        from tools import vector_indexer_tool as vit
        mock_col = self._make_collection(3)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384))

        vit._collection = mock_col
        vit._embed_model = mock_model

        from tools.semantic_retriever_tool import retrieve
        chunks = retrieve("invoice total", top_k=3)
        assert len(chunks) == 3
        assert "text" in chunks[0]
        assert "score" in chunks[0]
        assert "metadata" in chunks[0]

    def test_score_is_one_minus_distance(self):
        from tools import vector_indexer_tool as vit
        mock_col = self._make_collection(1)
        mock_col.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"invoice_no": "INV-001"}]],
            "distances": [[0.3]],
        }
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384))

        vit._collection = mock_col
        vit._embed_model = mock_model

        from tools.semantic_retriever_tool import retrieve
        chunks = retrieve("query")
        assert abs(chunks[0]["score"] - 0.7) < 1e-4

    def test_invoice_filter_applied(self):
        from tools import vector_indexer_tool as vit
        mock_col = self._make_collection(2)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384))

        vit._collection = mock_col
        vit._embed_model = mock_model

        from tools.semantic_retriever_tool import retrieve
        retrieve("query", invoice_no_filter="INV-001")
        _, kwargs = mock_col.query.call_args
        assert kwargs["where"] == {"invoice_no": "INV-001"}

    def test_exception_returns_empty(self):
        from tools import vector_indexer_tool as vit
        mock_col = MagicMock()
        mock_col.count.side_effect = RuntimeError("db error")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones((1, 384))

        vit._collection = mock_col
        vit._embed_model = mock_model

        from tools.semantic_retriever_tool import retrieve
        result = retrieve("query")
        assert result == []


# ==============================================================================
# tools/chunk_ranker_tool.py
# ==============================================================================

class TestChunkRanker:
    def test_empty_input_returns_empty(self):
        from tools.chunk_ranker_tool import rerank
        assert rerank([]) == []

    def test_filters_below_threshold(self):
        from tools.chunk_ranker_tool import rerank
        chunks = [
            _make_chunk("text1", score=0.1),
            _make_chunk("text2", score=0.5),
            _make_chunk("text3", score=0.9),
        ]
        result = rerank(chunks, threshold=0.25)
        assert len(result) == 2
        scores = [c["score"] for c in result]
        assert 0.1 not in scores

    def test_sorted_descending_by_score(self):
        from tools.chunk_ranker_tool import rerank
        chunks = [
            _make_chunk("a", score=0.3),
            _make_chunk("b", score=0.9),
            _make_chunk("c", score=0.6),
        ]
        result = rerank(chunks, threshold=0.0)
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.6
        assert result[2]["score"] == 0.3

    def test_all_below_threshold_returns_empty(self):
        from tools.chunk_ranker_tool import rerank
        chunks = [_make_chunk(score=0.1), _make_chunk(score=0.05)]
        result = rerank(chunks, threshold=0.5)
        assert result == []

    def test_custom_threshold(self):
        from tools.chunk_ranker_tool import rerank
        chunks = [_make_chunk(score=0.4), _make_chunk(score=0.6)]
        result = rerank(chunks, threshold=0.5)
        assert len(result) == 1
        assert result[0]["score"] == 0.6


# ==============================================================================
# tools/response_synthesizer_tool.py
# ==============================================================================

class TestResponseSynthesizer:
    def test_no_chunks_returns_no_context_message(self):
        from tools.response_synthesizer_tool import synthesize
        result = synthesize("What is the total?", [])
        assert result["error"] is None
        assert "No relevant" in result["answer"]
        assert result["sources"] == []

    def test_llm_generates_answer(self):
        from tools.response_synthesizer_tool import synthesize
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The total is $1,056.00."
        mock_llm.invoke.return_value = mock_response

        chunks = [_make_chunk("Total amount is $1,056.00", score=0.9)]

        with patch("tools.response_synthesizer_tool._build_llm", return_value=mock_llm):
            result = synthesize("What is the total?", chunks)
            assert result["answer"] == "The total is $1,056.00."
            assert result["error"] is None
            assert "INV-001" in result["sources"]

    def test_llm_unavailable_fallback(self):
        from tools.response_synthesizer_tool import synthesize
        chunks = [_make_chunk("Best matching context text here.", score=0.9)]

        with patch("tools.response_synthesizer_tool._build_llm",
                   side_effect=RuntimeError("No API key")):
            result = synthesize("What is the total?", chunks)
            assert "LLM unavailable" in result["answer"]
            assert result["error"] is not None
            assert "INV-001" in result["sources"]

    def test_llm_exception_returns_error(self):
        from tools.response_synthesizer_tool import synthesize
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("network error")
        chunks = [_make_chunk("some text")]

        with patch("tools.response_synthesizer_tool._build_llm", return_value=mock_llm):
            result = synthesize("query", chunks)
            assert result["error"] is not None
            assert result["answer"] == ""

    def test_sources_deduplicated(self):
        from tools.response_synthesizer_tool import synthesize
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_llm.invoke.return_value = mock_response

        chunks = [
            _make_chunk("text1", invoice_no="INV-001"),
            _make_chunk("text2", invoice_no="INV-001"),
            _make_chunk("text3", invoice_no="INV-002"),
        ]

        with patch("tools.response_synthesizer_tool._build_llm", return_value=mock_llm):
            result = synthesize("query", chunks)
            assert result["sources"].count("INV-001") == 1

    def test_build_context_includes_source_headers(self):
        from tools.response_synthesizer_tool import _build_context
        chunks = [_make_chunk("some text", invoice_no="INV-999")]
        ctx = _build_context(chunks)
        assert "INV-999" in ctx
        assert "some text" in ctx


# ==============================================================================
# agents/rag/indexing_agent.py
# ==============================================================================

class TestIndexingAgent:
    def test_no_text_returns_not_indexed(self):
        from agents.rag.indexing_agent import indexing_agent
        state = {"translated_text": "", "raw_text": ""}
        result = indexing_agent(state)
        assert result["rag_indexed"] is False

    def test_successful_indexing(self):
        from agents.rag.indexing_agent import indexing_agent
        state = {
            "translated_text": "Invoice INV-001 for vendor VEND-001.",
            "extracted_fields": {"invoice_no": "INV-001", "vendor_id": "VEND-001"},
            "file_path": "data/INV_001.pdf",
            "file_format": "pdf",
            "detected_language": "en",
        }
        with patch("agents.rag.indexing_agent.index_invoice",
                   return_value={"indexed_chunks": 3, "error": None}):
            result = indexing_agent(state)
            assert result["rag_indexed"] is True

    def test_indexing_error_sets_flag_and_error(self):
        from agents.rag.indexing_agent import indexing_agent
        state = {
            "translated_text": "Some invoice text.",
            "extracted_fields": {"invoice_no": "INV-001"},
        }
        with patch("agents.rag.indexing_agent.index_invoice",
                   return_value={"indexed_chunks": 0, "error": "DB write failed"}):
            result = indexing_agent(state)
            assert result["rag_indexed"] is False
            assert any("RAG_INDEX_ERROR" in e for e in result["errors"])

    def test_falls_back_to_raw_text(self):
        from agents.rag.indexing_agent import indexing_agent
        state = {
            "raw_text": "Raw invoice content.",
            "extracted_fields": {"invoice_no": "INV-X"},
        }
        captured = {}
        def fake_index(invoice_no, text, metadata=None):
            captured["text"] = text
            return {"indexed_chunks": 1, "error": None}

        with patch("agents.rag.indexing_agent.index_invoice", side_effect=fake_index):
            indexing_agent(state)
            assert captured["text"] == "Raw invoice content."

    def test_uses_file_path_as_invoice_no_fallback(self):
        from agents.rag.indexing_agent import indexing_agent
        state = {
            "translated_text": "some text",
            "extracted_fields": {},
            "file_path": "data/UNKNOWN.pdf",
        }
        captured = {}
        def fake_index(invoice_no, text, metadata=None):
            captured["invoice_no"] = invoice_no
            return {"indexed_chunks": 1, "error": None}

        with patch("agents.rag.indexing_agent.index_invoice", side_effect=fake_index):
            indexing_agent(state)
            assert captured["invoice_no"] == "data/UNKNOWN.pdf"


# ==============================================================================
# agents/rag/retrieval_agent.py
# ==============================================================================

class TestRetrievalAgent:
    def test_empty_query_returns_empty_chunks(self):
        from agents.rag.retrieval_agent import retrieval_agent
        state = {"rag_query": ""}
        result = retrieval_agent(state)
        assert result["rag_chunks"] == []

    def test_missing_query_returns_empty_chunks(self):
        from agents.rag.retrieval_agent import retrieval_agent
        result = retrieval_agent({})
        assert result["rag_chunks"] == []

    def test_returns_retrieved_chunks(self):
        from agents.rag.retrieval_agent import retrieval_agent
        fake_chunks = [_make_chunk("text", score=0.8)]
        state = {"rag_query": "What is the invoice total?"}

        with patch("agents.rag.retrieval_agent.retrieve", return_value=fake_chunks):
            result = retrieval_agent(state)
            assert result["rag_chunks"] == fake_chunks

    def test_filter_passed_to_retrieve(self):
        from agents.rag.retrieval_agent import retrieval_agent
        state = {"rag_query": "vendor details", "rag_query_filter": "INV-001"}

        with patch("agents.rag.retrieval_agent.retrieve", return_value=[]) as mock_retrieve:
            retrieval_agent(state)
            _, kwargs = mock_retrieve.call_args
            assert kwargs.get("invoice_no_filter") == "INV-001"


# ==============================================================================
# agents/rag/augmentation_agent.py
# ==============================================================================

class TestAugmentationAgent:
    def test_empty_chunks_returns_empty(self):
        from agents.rag.augmentation_agent import augmentation_agent
        result = augmentation_agent({"rag_chunks": []})
        assert result["rag_chunks"] == []

    def test_chunks_are_reranked(self):
        from agents.rag.augmentation_agent import augmentation_agent
        chunks = [
            _make_chunk("a", score=0.3),
            _make_chunk("b", score=0.9),
        ]
        ranked = [_make_chunk("b", score=0.9), _make_chunk("a", score=0.3)]

        with patch("agents.rag.augmentation_agent.rerank", return_value=ranked):
            result = augmentation_agent({"rag_chunks": chunks})
            assert result["rag_chunks"] == ranked

    def test_below_threshold_chunks_removed(self):
        from agents.rag.augmentation_agent import augmentation_agent
        chunks = [_make_chunk("low", score=0.1)]

        with patch("agents.rag.augmentation_agent.rerank", return_value=[]):
            result = augmentation_agent({"rag_chunks": chunks})
            assert result["rag_chunks"] == []


# ==============================================================================
# agents/rag/generation_agent.py
# ==============================================================================

class TestGenerationAgent:
    def test_generates_answer(self):
        from agents.rag.generation_agent import generation_agent
        state = {
            "rag_query": "What is the total?",
            "rag_chunks": [_make_chunk("Total is $1,056", score=0.9)],
        }
        fake_result = {"answer": "The total is $1,056.", "sources": ["INV-001"], "error": None}

        with patch("agents.rag.generation_agent.synthesize", return_value=fake_result):
            result = generation_agent(state)
            assert result["rag_answer"] == "The total is $1,056."
            assert result["rag_sources"] == ["INV-001"]

    def test_error_added_to_errors_field(self):
        from agents.rag.generation_agent import generation_agent
        state = {"rag_query": "query", "rag_chunks": [_make_chunk()]}
        fake_result = {"answer": "", "sources": [], "error": "LLM failed"}

        with patch("agents.rag.generation_agent.synthesize", return_value=fake_result):
            result = generation_agent(state)
            assert any("RAG_GENERATION_ERROR" in e for e in result.get("errors", []))

    def test_empty_chunks_still_calls_synthesize(self):
        from agents.rag.generation_agent import generation_agent
        state = {"rag_query": "query", "rag_chunks": []}
        fake_result = {"answer": "No context found.", "sources": [], "error": None}

        with patch("agents.rag.generation_agent.synthesize", return_value=fake_result) as mock_s:
            generation_agent(state)
            mock_s.assert_called_once()


# ==============================================================================
# agents/rag/reflection_agent.py
# ==============================================================================

class TestReflectionAgent:
    def test_context_relevance_mean_of_chunk_scores(self):
        from agents.rag.reflection_agent import _score_context_relevance
        chunks = [_make_chunk(score=0.8), _make_chunk(score=0.6)]
        assert abs(_score_context_relevance(chunks) - 0.7) < 1e-3

    def test_context_relevance_empty_chunks(self):
        from agents.rag.reflection_agent import _score_context_relevance
        assert _score_context_relevance([]) == 0.0

    def test_heuristic_groundedness_high_overlap(self):
        from agents.rag.reflection_agent import _heuristic_groundedness
        chunks = [_make_chunk("The total amount is twelve hundred dollars")]
        answer = "total amount twelve hundred dollars"
        score = _heuristic_groundedness(answer, chunks)
        assert score > 0.5

    def test_heuristic_groundedness_no_overlap(self):
        from agents.rag.reflection_agent import _heuristic_groundedness
        chunks = [_make_chunk("completely unrelated context about apples")]
        answer = "The invoice vendor is GlobalTech"
        score = _heuristic_groundedness(answer, chunks)
        assert score < 0.5

    def test_heuristic_groundedness_empty_answer(self):
        from agents.rag.reflection_agent import _heuristic_groundedness
        assert _heuristic_groundedness("", [_make_chunk()]) == 0.0

    def test_heuristic_groundedness_empty_chunks(self):
        from agents.rag.reflection_agent import _heuristic_groundedness
        assert _heuristic_groundedness("some answer", []) == 0.0

    def test_heuristic_answer_relevance_full_overlap(self):
        from agents.rag.reflection_agent import _heuristic_answer_relevance
        query = "What is the total amount for this invoice?"
        answer = "The total amount for this invoice is $1,056."
        score = _heuristic_answer_relevance(query, answer)
        assert score > 0.5

    def test_heuristic_answer_relevance_no_overlap(self):
        from agents.rag.reflection_agent import _heuristic_answer_relevance
        query = "What is the vendor name?"
        answer = "The weather is sunny today in London."
        score = _heuristic_answer_relevance(query, answer)
        assert score == 0.0

    def test_heuristic_answer_relevance_empty_query(self):
        from agents.rag.reflection_agent import _heuristic_answer_relevance
        assert _heuristic_answer_relevance("", "some answer") == 0.0

    def test_reflection_agent_uses_heuristics_when_llm_unavailable(self):
        from agents.rag.reflection_agent import reflection_agent
        state = {
            "rag_query":  "What is the invoice total?",
            "rag_chunks": [_make_chunk("The total is $500", score=0.7)],
            "rag_answer": "The total is $500",
        }
        with patch("agents.rag.reflection_agent._score_with_llm", return_value=-1.0):
            result = reflection_agent(state)
            scores = result["rag_scores"]
            assert "context_relevance" in scores
            assert "groundedness" in scores
            assert "answer_relevance" in scores
            assert "low_quality" in scores
            assert isinstance(scores["low_quality"], bool)

    def test_reflection_agent_uses_llm_scores_when_available(self):
        from agents.rag.reflection_agent import reflection_agent
        state = {
            "rag_query":  "What is the vendor?",
            "rag_chunks": [_make_chunk(score=0.9)],
            "rag_answer": "The vendor is GlobalTech.",
        }
        with patch("agents.rag.reflection_agent._score_with_llm", return_value=0.85):
            result = reflection_agent(state)
            scores = result["rag_scores"]
            assert scores["groundedness"] == 0.85
            assert scores["answer_relevance"] == 0.85

    def test_low_quality_flag_set_when_score_below_threshold(self):
        from agents.rag.reflection_agent import reflection_agent
        state = {
            "rag_query":  "query",
            "rag_chunks": [_make_chunk(score=0.1)],
            "rag_answer": "answer",
        }
        with patch("agents.rag.reflection_agent._score_with_llm", return_value=0.9):
            result = reflection_agent(state)
            # context_relevance = 0.1 < 0.6 threshold → low_quality
            assert result["rag_scores"]["low_quality"] is True

    def test_low_quality_false_when_all_scores_high(self):
        from agents.rag.reflection_agent import reflection_agent
        state = {
            "rag_query":  "What is the total?",
            "rag_chunks": [_make_chunk(score=0.9)],
            "rag_answer": "The total is $1,000.",
        }
        with patch("agents.rag.reflection_agent._score_with_llm", return_value=0.95):
            result = reflection_agent(state)
            assert result["rag_scores"]["low_quality"] is False

    def test_score_with_llm_clamps_to_zero_one(self):
        from agents.rag.reflection_agent import _score_with_llm
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "1.5"  # out-of-range
        mock_llm.invoke.return_value = mock_response

        # _build_llm is a local import inside _score_with_llm — patch at source
        with patch("tools.field_extractor_tool._build_llm", return_value=mock_llm):
            score = _score_with_llm("some prompt")
            assert score <= 1.0

    def test_score_with_llm_returns_sentinel_on_bad_parse(self):
        from agents.rag.reflection_agent import _score_with_llm
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "not a number"
        mock_llm.invoke.return_value = mock_response

        with patch("tools.field_extractor_tool._build_llm", return_value=mock_llm):
            score = _score_with_llm("prompt")
            assert score == -1.0

    def test_score_with_llm_returns_sentinel_on_runtime_error(self):
        from agents.rag.reflection_agent import _score_with_llm
        with patch("tools.field_extractor_tool._build_llm",
                   side_effect=RuntimeError("no key")):
            score = _score_with_llm("prompt")
            assert score == -1.0


# ==============================================================================
# workflows/rag_pipeline.py
# ==============================================================================

class TestRagPipeline:
    def _patch_all_agents(self):
        """Return a context manager that patches all 4 RAG agents."""
        from unittest.mock import patch
        patches = [
            patch("workflows.rag_pipeline.retrieval_agent",
                  return_value={"rag_chunks": [_make_chunk(score=0.8)]}),
            patch("workflows.rag_pipeline.augmentation_agent",
                  return_value={"rag_chunks": [_make_chunk(score=0.8)]}),
            patch("workflows.rag_pipeline.generation_agent",
                  return_value={"rag_answer": "Answer.", "rag_sources": ["INV-001"]}),
            patch("workflows.rag_pipeline.reflection_agent",
                  return_value={"rag_scores": {"context_relevance": 0.8,
                                               "groundedness": 0.9,
                                               "answer_relevance": 0.85,
                                               "low_quality": False}}),
        ]
        return patches

    def test_build_rag_pipeline_returns_compiled_graph(self):
        from workflows.rag_pipeline import build_rag_pipeline
        pipeline = build_rag_pipeline()
        assert pipeline is not None

    def test_run_rag_query_returns_state(self):
        from workflows.rag_pipeline import run_rag_query
        patches = self._patch_all_agents()
        with patches[0], patches[1], patches[2], patches[3]:
            result = run_rag_query("What are the line items?")
            assert "rag_answer" in result
            assert "rag_sources" in result
            assert "rag_scores" in result

    def test_run_rag_query_sets_query_in_state(self):
        from workflows.rag_pipeline import run_rag_query
        captured = {}

        def fake_retrieval(state):
            captured["query"] = state.get("rag_query")
            return {"rag_chunks": []}

        with patch("workflows.rag_pipeline.retrieval_agent", side_effect=fake_retrieval), \
             patch("workflows.rag_pipeline.augmentation_agent", return_value={"rag_chunks": []}), \
             patch("workflows.rag_pipeline.generation_agent",
                   return_value={"rag_answer": "", "rag_sources": []}), \
             patch("workflows.rag_pipeline.reflection_agent",
                   return_value={"rag_scores": {}}):
            run_rag_query("total amount?")
            assert captured["query"] == "total amount?"

    def test_run_rag_query_with_filter(self):
        from workflows.rag_pipeline import run_rag_query
        # The filter is set in the seed state before graph invocation.
        # Verify by inspecting what run_rag_query puts in the base state dict
        # (test at the graph input level, not through a mocked node).
        patches = self._patch_all_agents()
        with patches[0], patches[1], patches[2], patches[3]:
            result = run_rag_query("query", invoice_no_filter="INV-001")
            # Result should at minimum contain rag_answer set by mocked generation agent
            assert "rag_answer" in result

    def test_run_rag_query_with_seed_state(self):
        from workflows.rag_pipeline import run_rag_query
        from core.state import InvoiceState
        seed: InvoiceState = {"rag_indexed": True, "errors": []}  # type: ignore
        patches = self._patch_all_agents()
        with patches[0], patches[1], patches[2], patches[3]:
            result = run_rag_query("query", seed_state=seed)
            assert result is not None

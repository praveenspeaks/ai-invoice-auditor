"""
Unit tests for core/state.py — InvoiceState and initial_state factory.
Target coverage: core/state.py ≥ 80%
"""

import datetime
import pytest
from core.state import InvoiceState, initial_state, _merge_lists


# ── _merge_lists ───────────────────────────────────────────────────────────────

class TestMergeLists:
    def test_merges_two_non_empty_lists(self):
        assert _merge_lists([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_merges_with_empty_left(self):
        assert _merge_lists([], [1]) == [1]

    def test_merges_with_empty_right(self):
        assert _merge_lists([1], []) == [1]

    def test_both_empty(self):
        assert _merge_lists([], []) == []

    def test_preserves_order(self):
        result = _merge_lists(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]

    def test_handles_dicts_in_lists(self):
        a = [{"key": "v1"}]
        b = [{"key": "v2"}]
        assert _merge_lists(a, b) == [{"key": "v1"}, {"key": "v2"}]


# ── initial_state ──────────────────────────────────────────────────────────────

class TestInitialState:
    def test_sets_file_path(self):
        state = initial_state("data/incoming/INV_EN_001.pdf", {}, "pdf")
        assert state["file_path"] == "data/incoming/INV_EN_001.pdf"

    def test_sets_file_format(self):
        state = initial_state("file.docx", {}, "docx")
        assert state["file_format"] == "docx"

    def test_sets_meta(self):
        meta = {"sender": "test@test.com", "language": "en"}
        state = initial_state("file.pdf", meta, "pdf")
        assert state["meta"] == meta

    def test_raw_text_empty_string(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["raw_text"] == ""

    def test_detected_language_empty(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["detected_language"] == ""

    def test_translated_text_empty(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["translated_text"] == ""

    def test_translation_confidence_zero(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["translation_confidence"] == 0.0

    def test_extracted_fields_empty_dict(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["extracted_fields"] == {}

    def test_validation_result_empty_dict(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["validation_result"] == {}

    def test_erp_data_empty_dict(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["erp_data"] == {}

    def test_discrepancies_empty_list(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["discrepancies"] == []

    def test_recommendation_empty(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["recommendation"] == ""

    def test_human_review_required_false(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["human_review_required"] is False

    def test_report_path_empty(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["report_path"] == ""

    def test_rag_indexed_false(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["rag_indexed"] is False

    def test_rag_chunks_empty_list(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["rag_chunks"] == []

    def test_rag_sources_empty_list(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["rag_sources"] == []

    def test_rag_scores_empty_dict(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["rag_scores"] == {}

    def test_errors_empty_list(self):
        state = initial_state("f.pdf", {}, "pdf")
        assert state["errors"] == []

    def test_pipeline_start_time_is_iso_string(self):
        state = initial_state("f.pdf", {}, "pdf")
        ts = state["pipeline_start_time"]
        # Must parse as ISO datetime without raising
        parsed = datetime.datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime.datetime)

    def test_pipeline_start_time_is_recent(self):
        before = datetime.datetime.now(datetime.UTC)
        state = initial_state("f.pdf", {}, "pdf")
        after = datetime.datetime.now(datetime.UTC)
        ts = datetime.datetime.fromisoformat(state["pipeline_start_time"])
        assert before <= ts <= after

    def test_meta_empty_dict_default(self):
        state = initial_state("f.pdf", {}, "image")
        assert state["meta"] == {}

    def test_all_required_keys_present(self):
        state = initial_state("f.pdf", {}, "pdf")
        required_keys = [
            "file_path", "file_format", "meta", "raw_text", "detected_language",
            "translated_text", "translation_confidence", "extracted_fields",
            "validation_result", "erp_data", "discrepancies", "recommendation",
            "human_review_required", "report_path", "rag_indexed", "rag_query",
            "rag_chunks", "rag_answer", "rag_sources", "rag_scores",
            "pipeline_start_time", "errors",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    @pytest.mark.parametrize("fmt", ["pdf", "docx", "image"])
    def test_all_file_formats_accepted(self, fmt):
        state = initial_state(f"file.{fmt}", {}, fmt)
        assert state["file_format"] == fmt

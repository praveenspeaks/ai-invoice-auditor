# AI Invoice Auditor

An agentic AI system that ingests invoices (PDF/DOCX/image), validates them against ERP data,
generates audit reports, and answers natural-language questions via a RAG pipeline.

Built with **LangGraph**, **ChromaDB**, **FastMCP**, **LangFuse**, and **Streamlit**.

---

## Architecture

```text
Invoice (PDF/DOCX/PNG)
        |
        v
[Invoice Monitor Agent]     <- polls data/incoming/
        |
        v
[Extractor Agent]           <- pdfplumber / python-docx / pytesseract + RAI injection check
        |
        v
[Translation Agent]         <- GoogleTranslator (non-English -> English)
        |
        v
[Data Validation Agent]     <- field extraction (LLM) + 3-pass completeness check
        |
        v
[Business Validation Agent] <- ERP cross-reference via FastAPI mock
        |
        v
[Reporting Agent]           <- Jinja2 HTML report (AUTO_APPROVED / MANUAL_REVIEW / REJECTED)

RAG Subgraph (runs in parallel / on-demand):
  [Indexing Agent]    -> ChromaDB (sentence-transformers)
  [Retrieval Agent]   -> semantic search
  [Augmentation Agent]-> chunk reranking
  [Generation Agent]  -> LLM-grounded answer
  [Reflection Agent]  -> RAG Triad quality scores
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- `.venv` already created — all dependencies pre-installed

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your API keys:
#   OPENAI_API_KEY=sk-...          (or ANTHROPIC_API_KEY)
#   LANGFUSE_SECRET_KEY=...        (optional — tracing disabled if absent)
#   LANGFUSE_PUBLIC_KEY=...        (optional)
```

### 3. Run the full demo

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Start everything: ERP mock server + pipeline + Streamlit UI
python main.py
```

The demo will:

1. Start the Mock ERP server on `http://localhost:8000`
2. Run the invoice pipeline on all 6 sample invoices in `data/incoming/`
3. Launch the Streamlit dashboard at `http://localhost:8501`

### Demo runner notes

`main.py` is the one-command launcher. It starts the ERP mock server in the background,
runs the invoice pipeline across all files in `data/incoming/`, and then opens the UI.

Flags:

```bash
python main.py --pipeline   # run pipeline only
python main.py --ui         # UI only (no pipeline run)
python main.py --mcp        # FastMCP tool registry (stdio)
```

### 4. Individual components

```bash
# Pipeline only (no UI)
python main.py --pipeline

# Streamlit UI only (use existing reports)
python main.py --ui

# FastMCP tool registry (stdio — for Claude Desktop integration)
python main.py --mcp

# RAG query from CLI
python workflows/rag_pipeline.py "What is the total amount on the Spanish invoice?"

# ERP server only
uvicorn erp_mock.main:app --port 8000
```

---

## Project Structure

```text
agents/                  # 11 LangGraph agent nodes
  invoice_monitor_agent.py
  extractor_agent.py
  translation_agent.py
  data_validation_agent.py
  business_validation_agent.py
  reporting_agent.py
  rag/
    indexing_agent.py
    retrieval_agent.py
    augmentation_agent.py
    generation_agent.py
    reflection_agent.py

tools/                   # 10 reusable tool modules
  invoice_watcher_tool.py
  data_harvester_tool.py
  lang_bridge_tool.py
  data_completeness_checker.py
  business_validation_tool.py
  insight_reporter_tool.py
  vector_indexer_tool.py
  semantic_retriever_tool.py
  chunk_ranker_tool.py
  response_synthesizer_tool.py

workflows/
  invoice_pipeline.py    # LangGraph core pipeline (6 agents)
  rag_pipeline.py        # LangGraph RAG subgraph (4 agents)

core/
  state.py               # InvoiceState TypedDict
  config.py              # rules.yaml loader (Pydantic)
  logger.py              # structured logging
  observability.py       # LangFuse @trace_agent decorator
  rai_guardrails.py      # prompt injection + PII detection

mcp_tools/
  server.py              # FastMCP registry — all 10 tools

erp_mock/
  main.py                # FastAPI mock ERP server

ui/
  app.py                 # Streamlit dashboard + RAG chat

data/
  incoming/              # 6 sample invoices (EN/ES/DE, PDF/DOCX/PNG)
  vector_store/          # ChromaDB persistent index

outputs/
  reports/               # Generated HTML audit reports

tests/
  test_e2e.py            # E2E integration tests (33 tests)
  test_rag_system.py     # RAG unit tests (66 tests)
  test_business_validation.py
  test_data_validation_agent.py
  test_invoice_pipeline.py

config/
  rules.yaml             # Validation rules, tolerances, currency config
```

---

## Sample Invoices

| File                     | Language | Format    | Expected Result               |
| ------------------------ | -------- | --------- | ----------------------------- |
| INV_EN_001.pdf           | English  | PDF       | AUTO_APPROVED                 |
| INV_EN_002.pdf           | English  | PDF       | AUTO_APPROVED or MANUAL_REVIEW|
| INV_ES_003.pdf           | Spanish  | PDF       | translated -> validated       |
| INV_DE_004.pdf           | German   | PDF       | translated -> validated       |
| INV_EN_005_scan.png      | English  | PNG (OCR) | extracted via Tesseract       |
| INV_EN_006_malformed.pdf | English  | PDF       | REJECTED (missing fields)     |

---

## Running Tests

```bash
# Full test suite (no network, no API keys required)
uv run --no-sync pytest

# E2E tests only
uv run --no-sync pytest tests/test_e2e.py -v

# RAG system tests
uv run --no-sync pytest tests/test_rag_system.py -v
```

Current: **495 tests passing**, 2 skipped.

---

## LangFuse Observability

Set `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` in `.env` to enable tracing.
Without keys, all agents run normally with tracing silently disabled.

Each pipeline run creates a trace per agent:
`invoice_pipeline.extractor`, `invoice_pipeline.translator`, etc.

---

## FastMCP Integration (Claude Desktop)

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "invoice-auditor": {
      "command": "python",
      "args": ["path/to/ai-invoice-auditor/main.py", "--mcp"]
    }
  }
}
```

This exposes all 10 pipeline tools to Claude as callable MCP tools.

---

## RAI Guardrails

- **Prompt injection detection**: regex patterns scan invoice text before LLM calls.
  Flagged invoices are marked with `SECURITY:` errors and skip LLM extraction.
- **PII detection**: SSN, credit card, passport patterns trigger `PII_WARNING` errors.
- **RAG domain restriction**: system prompt restricts answers to indexed invoice context only.

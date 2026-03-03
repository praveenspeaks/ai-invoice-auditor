## AI Invoice Auditor

Agentic AI-powered multilingual invoice validation system. It ingests invoices from a simulated inbox, extracts and translates data, validates against ERP records and rules, generates audit reports, and supports RAG-based Q&A with a human-in-the-loop UI.

### Requirements vs. What Is Achieved

**Required** (from capstone) and **Implemented** (in this repo):

- **Monitor inbox** (simulated via `data/incoming`) -> Invoice watcher + monitor agent.
- **Extract data from PDF/DOCX/images** -> Data harvester tool (pdfplumber, python-docx, pytesseract).
- **Translate to English** -> Language bridge + translation agent.
- **Validate against ERP** -> Business validation tool + agent (mock ERP).
- **Dynamic validation rules** -> Data completeness checker + rules in `config/rules.yaml`.
- **Generate discrepancy reports** -> Insight reporter tool (HTML reports).
- **RAG Q&A** -> Indexing, retrieval, augmentation, generation, reflection agents + tools.
- **Human-in-the-loop** -> Streamlit UI with approve/reject decisions for manual review.
- **RAI guardrails** -> Prompt injection and PII checks before LLM extraction.
- **MCP support** -> FastMCP tool registry exposing pipeline tools.
- **Observability** -> LangFuse tracing when keys are provided.

### End-to-End Flow

1. **Monitor** incoming invoices (file system inbox).
2. **Extract** text/tables from PDF, DOCX, or image OCR.
3. **Translate** to English (if needed).
4. **Index** invoice text into vector store for RAG.
5. **Validate data** for completeness, types, and currency rules.
6. **Validate business** data against ERP (line-item comparisons).
7. **Report** with discrepancies, missing fields, and recommendation.
8. **Q&A** via RAG in the UI.
9. **Human review** for manual approvals or rejections.

### Key Functionality and Where It Lives

**Core agents (invoice pipeline)**
- Invoice monitor: [agents/invoice_monitor_agent.py](agents/invoice_monitor_agent.py)
- Extractor: [agents/extractor_agent.py](agents/extractor_agent.py)
- Translation: [agents/translation_agent.py](agents/translation_agent.py)
- Data validation: [agents/data_validation_agent.py](agents/data_validation_agent.py)
- Business validation: [agents/business_validation_agent.py](agents/business_validation_agent.py)
- Reporting: [agents/reporting_agent.py](agents/reporting_agent.py)

**Pipeline wiring**
- Invoice pipeline: [workflows/invoice_pipeline.py](workflows/invoice_pipeline.py)
- RAG pipeline: [workflows/rag_pipeline.py](workflows/rag_pipeline.py)

**Tools (agent capabilities)**
- Invoice watcher: [tools/invoice_watcher_tool.py](tools/invoice_watcher_tool.py)
- Data harvester: [tools/data_harvester_tool.py](tools/data_harvester_tool.py)
- Language bridge: [tools/lang_bridge_tool.py](tools/lang_bridge_tool.py)
- Data completeness checker: [tools/data_completeness_checker.py](tools/data_completeness_checker.py)
- Business validation: [tools/business_validation_tool.py](tools/business_validation_tool.py)
- Insight reporter: [tools/insight_reporter_tool.py](tools/insight_reporter_tool.py)

**RAG agents and tools**
- Indexing agent: [agents/rag/indexing_agent.py](agents/rag/indexing_agent.py)
- Retrieval agent: [agents/rag/retrieval_agent.py](agents/rag/retrieval_agent.py)
- Augmentation agent: [agents/rag/augmentation_agent.py](agents/rag/augmentation_agent.py)
- Generation agent: [agents/rag/generation_agent.py](agents/rag/generation_agent.py)
- Reflection agent: [agents/rag/reflection_agent.py](agents/rag/reflection_agent.py)
- Vector indexer: [tools/vector_indexer_tool.py](tools/vector_indexer_tool.py)
- Semantic retriever: [tools/semantic_retriever_tool.py](tools/semantic_retriever_tool.py)
- Chunk ranker: [tools/chunk_ranker_tool.py](tools/chunk_ranker_tool.py)
- Response synthesizer: [tools/response_synthesizer_tool.py](tools/response_synthesizer_tool.py)

**RAI guardrails**
- Prompt injection + PII checks: [core/rai_guardrails.py](core/rai_guardrails.py)

**Observability**
- LangFuse tracing wrapper: [core/observability.py](core/observability.py)

**MCP tool registry**
- FastMCP server: [mcp_tools/server.py](mcp_tools/server.py)

**UI and human-in-the-loop**
- Streamlit app: [ui/app.py](ui/app.py)
- Decisions log: `logs/human_decisions.json`

### Local Setup

1) Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Configure `.env`
```bash
OPENAI_API_KEY=...
# or ANTHROPIC_API_KEY=...
LLM_MODEL=gpt-4o-mini
INCOMING_DIR=./data/incoming
REPORTS_DIR=./outputs/reports
VECTOR_STORE_DIR=./data/vector_store
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

### Run Locally

**Run the invoice pipeline (single file):**
```bash
python -m workflows.invoice_pipeline data/incoming/INV_EN_001.pdf
```

**Run the invoice pipeline (polling):**
```bash
python -m workflows.invoice_pipeline
```

**Run the RAG Q&A pipeline directly:**
```bash
python -m workflows.rag_pipeline "What is the invoice total?"
```

**Run the Streamlit UI:**
```bash
streamlit run ui/app.py
```

**Run the MCP server:**
```bash
python mcp_tools/server.py
```

### End-to-End Testing

Run the full test suite:
```bash
pytest -q
```

### Notes

- The pipeline processes **one invoice per run** by design. Use the reset registry button in the UI to reprocess.
- If invoices do not appear in RAG Q&A, ensure they have been indexed by running the pipeline after enabling the indexing step.
- OCR requires Tesseract installed and `TESSERACT_CMD` set on Windows.

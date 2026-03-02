# AI Invoice Auditor — 3-Day Compressed Execution Plan
**Deadline:** 3 Days from Start
**Status:** Dependencies already installed (.venv ready)
**Evaluation Weight Guide:** Functional (35%) → RAG/Agentic (25%) → UI/HITL (15%) → Code Quality (15%) → Agile (10%)

---

## What We Cut vs Keep (Scope Decision)

| Component | Decision | Reason |
|---|---|---|
| All 6 Core Agents in LangGraph | **KEEP — full** | Core evaluation criterion |
| All 5 RAG Agents | **KEEP — simplified** | 25% of marks |
| Mock ERP FastAPI | **KEEP — full** | Required for business validation |
| Streamlit UI + HITL | **KEEP — essential screens only** | 15% of marks |
| HTML Reports | **KEEP — full** | Part of functional completeness |
| rules.yaml enforcement | **KEEP — full** | Already configured |
| FastMCP tool registry | **KEEP — simplified** | Required by spec |
| LangFuse tracing | **KEEP — decorator-based** | Required by spec |
| RAI Guardrails | **KEEP — system prompt only** | Required by spec |
| Cross-encoder reranking | **SIMPLIFY → cosine score** | Too slow to tune in 3 days |
| RAG Triad full scoring | **SIMPLIFY → 3 basic metrics** | Keep structure, simplify math |
| Unit tests per tool | **CUT → E2E test only** | Time constraint |
| Docker-compose | **CUT** | Not in evaluation criteria |
| Field-level human correction | **CUT → approve/reject only** | Complex UI, low mark weight |

---

## Project Root Structure to Build

```
d:/My Professional Projects/AI Invoice Auditor/
├── agents/
│   ├── __init__.py
│   ├── invoice_monitor_agent.py
│   ├── extractor_agent.py
│   ├── translation_agent.py
│   ├── data_validation_agent.py
│   ├── business_validation_agent.py
│   ├── reporting_agent.py
│   └── rag/
│       ├── __init__.py
│       ├── indexing_agent.py
│       ├── retrieval_agent.py
│       ├── augmentation_agent.py
│       ├── generation_agent.py
│       └── reflection_agent.py
├── tools/
│   ├── __init__.py
│   ├── invoice_watcher_tool.py
│   ├── data_harvester_tool.py
│   ├── lang_bridge_tool.py
│   ├── data_completeness_checker.py
│   ├── business_validation_tool.py
│   ├── insight_reporter_tool.py
│   ├── vector_indexer_tool.py
│   ├── semantic_retriever_tool.py
│   ├── chunk_ranker_tool.py
│   └── response_synthesizer_tool.py
├── workflows/
│   ├── __init__.py
│   ├── invoice_pipeline.py          ← LangGraph core pipeline
│   └── rag_pipeline.py              ← LangGraph RAG subgraph
├── erp_mock/
│   ├── __init__.py
│   ├── main.py                      ← FastAPI server
│   └── erp_seed.json                ← Mock ERP data
├── mcp/
│   ├── __init__.py
│   └── server.py                    ← FastMCP tool registry
├── ui/
│   └── app.py                       ← Streamlit frontend
├── core/
│   ├── __init__.py
│   ├── state.py                     ← InvoiceState Pydantic model
│   ├── config.py                    ← rules.yaml loader
│   └── logger.py                    ← Centralized logging
├── data/
│   ├── incoming/                    ← Simulated inbox (6 invoices)
│   └── vector_store/                ← ChromaDB persisted index
├── outputs/
│   └── reports/                     ← Generated HTML reports
├── logs/
│   └── invoice_auditor.log
├── tests/
│   └── test_e2e.py
├── config/
│   └── rules.yaml                   ← Already exists ✓
├── .env                             ← API keys + LangFuse config
├── main.py                          ← Entry point (pipeline runner)
└── requirements.txt                 ← Already exists ✓
```

---

# DAY 1 — Foundation + Ingestion + Extraction + Translation
**Target by end of Day 1:** Drop an invoice into `data/incoming/`, pipeline runs, extracts text, translates if non-English, prints structured fields to console.

---

## Morning Block (Hours 1–4): Core Foundation

### Task 1.1 — Folder Scaffold + .env (30 min)
Create all folders and `__init__.py` files. Create `.env`:
```
OPENAI_API_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_HOST=https://cloud.langfuse.com
ERP_BASE_URL=http://localhost:8000
INCOMING_DIR=./data/incoming
VECTOR_STORE_DIR=./data/vector_store
REPORTS_DIR=./outputs/reports
POLL_INTERVAL_SECONDS=5
```

**Done when:** `python -c "from dotenv import load_dotenv; load_dotenv(); print('OK')"` passes.

---

### Task 1.2 — Core State + Config + Logger (45 min)
**File:** `core/state.py`
```python
# InvoiceState — shared LangGraph state
class InvoiceState(TypedDict):
    file_path: str
    file_format: str          # pdf | docx | image
    meta: dict                # from .meta.json
    raw_text: str
    detected_language: str
    translated_text: str
    translation_confidence: float
    extracted_fields: dict    # invoice_no, vendor_id, line_items, etc.
    validation_result: dict   # missing fields, type errors, currency issues
    erp_data: dict            # response from mock ERP
    discrepancies: list       # list of {field, invoice_val, erp_val, status}
    recommendation: str       # AUTO_APPROVED | MANUAL_REVIEW | REJECTED
    report_path: str
    rag_indexed: bool
    human_review_required: bool
    errors: list[str]
    pipeline_start_time: str
```

**File:** `core/config.py` — loads `rules.yaml` into typed dict, exposes `get_rules()`.
**File:** `core/logger.py` — configures Python logging to file + console, level from rules.yaml.

**Done when:** `from core.state import InvoiceState; from core.config import get_rules; print(get_rules())` works.

---

### Task 1.3 — Mock ERP FastAPI Server (60 min)
**File:** `erp_mock/erp_seed.json` — 6 vendor records matching invoice filenames:
```json
{
  "VENDOR-001": {
    "INV-1001": {
      "vendor_name": "GlobalTech Supplies",
      "po_number": "PO-1001",
      "line_items": [
        {"item_code": "ITEM-A1", "description": "Logistics Container 20ft", "qty": 10, "unit_price": 250.00, "total": 2500.00, "tax": 125.00},
        {"item_code": "ITEM-A2", "description": "Packing Material", "qty": 100, "unit_price": 5.00, "total": 500.00, "tax": 25.00}
      ],
      "currency": "USD",
      "total_amount": 3000.00
    }
  }
}
```
Create similar records for VENDOR-002 (Spanish), VENDOR-003 (German), with one having a deliberate price mismatch for testing.

**File:** `erp_mock/main.py` — FastAPI with:
- `GET /erp/invoice/{vendor_id}/{invoice_no}` → returns line items
- `GET /erp/vendors` → list all vendors
- `GET /health` → ping

**Done when:** `uvicorn erp_mock.main:app --port 8000` starts and `GET /erp/invoice/VENDOR-001/INV-1001` returns JSON.

---

### Task 1.4 — Sample Invoice Files (45 min)
Create 6 synthetic invoice files in `data/incoming/` + their `.meta.json` sidecars:
- `INV_EN_001.pdf` — English, valid, 2 line items (use `reportlab` or pre-made PDF)
- `INV_EN_002.pdf` — English, valid, different vendor
- `INV_ES_003.pdf` — Spanish invoice text
- `INV_DE_004.docx` — German invoice as Word doc
- `INV_EN_005_scan.png` — Scanned image (screenshot of invoice text)
- `INV_EN_006_malformed.pdf` — Missing invoice_no and currency

> **Shortcut:** Use Python `reportlab` to generate PDFs programmatically, or create a `scripts/generate_samples.py` script.

**Done when:** All 6 files + 6 `.meta.json` sidecars exist in `data/incoming/`.

---

## Afternoon Block (Hours 5–8): Extraction + Translation Pipeline

### Task 1.5 — Data Harvester Tool (90 min)
**File:** `tools/data_harvester_tool.py`

Three extraction paths:
```
PDF → pdfplumber → raw text + tables
DOCX → python-docx → paragraphs + tables
PNG/JPG → PIL + pytesseract → OCR text
```
After extraction → `langdetect.detect()` for language detection.
Then LLM-assisted field extraction prompt:
```
Given this invoice text, extract these fields as JSON:
invoice_no, invoice_date, vendor_id, vendor_name, currency,
total_amount, line_items (list of: item_code, description, qty, unit_price, total)
Return ONLY valid JSON. If a field is missing, use null.
```

**Done when:** Running the tool on all 6 files returns raw text + detected language.

---

### Task 1.6 — Invoice Monitor Agent + Invoice-Watcher Tool (45 min)
**File:** `tools/invoice_watcher_tool.py` — Polls `data/incoming/` every N seconds.
Maintains `data/processed_registry.json` to track processed files.
Returns list of new `(file_path, meta_path)` tuples.

**File:** `agents/invoice_monitor_agent.py` — LangGraph node that:
1. Calls `invoice_watcher_tool`
2. For each new file, initializes an `InvoiceState`
3. Emits state to next node

**Done when:** Dropping a file triggers detection within 5 seconds.

---

### Task 1.7 — Lang-Bridge Tool + Translation Agent (60 min)
**File:** `tools/lang_bridge_tool.py`

Logic:
```python
if detected_language == "en":
    return text, confidence=1.0   # skip translation
else:
    # Use deep-translator (GoogleTranslator) as primary
    translated = GoogleTranslator(source='auto', target='en').translate(text)
    confidence = estimate_confidence(detected_language, translated)
    return translated, confidence
```

Confidence estimation: Use LLM to score translation quality OR use a simple heuristic (known language → 0.85, unknown → 0.65).

**File:** `agents/translation_agent.py` — LangGraph node calling Lang-Bridge Tool, sets `translated_text` and `translation_confidence` in state. If `confidence < 0.75`, sets `human_review_required = True`.

**Done when:** `INV_ES_003` and `INV_DE_004` produce English translated text with confidence scores.

---

### Task 1.8 — LangGraph Pipeline Skeleton (30 min)
**File:** `workflows/invoice_pipeline.py`

Wire all nodes built so far (monitor → extract → translate) with stub nodes for remaining agents. Run the pipeline and verify state flows correctly through all three completed nodes.

**Done when:** `python workflows/invoice_pipeline.py` processes a test invoice through monitor → extract → translate and prints the state.

---

### Day 1 Checkpoint ✓
- [ ] All folders and `__init__.py` files created
- [ ] `.env` configured with all keys
- [ ] `InvoiceState`, `RulesConfig`, logger all working
- [ ] Mock ERP server running on port 8000
- [ ] All 6 sample invoices + meta.json files exist
- [ ] PDF/DOCX/OCR extraction working on all 6 files
- [ ] Language detection working
- [ ] Translation working for Spanish and German invoices
- [ ] LangGraph pipeline runs through 3 nodes end-to-end

---

# DAY 2 — Validation + Reporting + Full RAG System
**Target by end of Day 2:** Full pipeline runs end-to-end producing HTML reports. RAG system indexes processed invoices and answers natural language questions.

---

## Morning Block (Hours 1–4): Validation + Reporting

### Task 2.1 — Data Completeness Checker + Data Validation Agent (75 min)
**File:** `tools/data_completeness_checker.py`

Three validation passes from `rules.yaml`:
```
Pass 1: Required fields check (header + line_item level)
         → missing fields → action per validation_policies.missing_field_action
Pass 2: Data type validation (date, float, str)
         → type errors → flag
Pass 3: Currency normalization + accepted_currencies check
         → normalize symbols using currency_symbol_map
         → reject if not in accepted_currencies
```
Output: `validation_result = {status, missing_fields[], type_errors[], currency_status, passed}`

**File:** `agents/data_validation_agent.py` — LangGraph node. Calls checker. Routes: if `rejected` → jump to reporting with `REJECTED` recommendation. Else continue.

**Done when:** `INV_EN_006_malformed` produces 2 missing field flags. EUR symbol normalized to EUR.

---

### Task 2.2 — Business Validation Tool + Business Validation Agent (75 min)
**File:** `tools/business_validation_tool.py`

Steps:
```
1. Call GET /erp/invoice/{vendor_id}/{invoice_no} via httpx
2. If 404 → flag as "UNREGISTERED_INVOICE"
3. For each invoice line item, find matching ERP line item by item_code
4. Compare: qty (exact), unit_price (±5%), total (±5%), tax (±2%)
5. Apply tolerances from rules.yaml
6. Return list of discrepancies with: field, invoice_val, erp_val, diff_pct, status
```

**File:** `agents/business_validation_agent.py` — LangGraph node. Calls tool. Computes final recommendation:
```
AUTO_APPROVED: confidence ≥ 0.95 AND no discrepancies AND no missing fields
REJECTED: any required field missing OR currency invalid
MANUAL_REVIEW: discrepancies found OR confidence < 0.95
```

**Done when:** Deliberate 6% price mismatch in ERP seed data creates a DISCREPANCY flag.

---

### Task 2.3 — Insight Reporter Tool + Reporting Agent (60 min)
**File:** `tools/insight_reporter_tool.py`

Generates HTML report using Python `string.Template` or `jinja2`:

Report sections:
```
1. Header — invoice_no, vendor, date, pipeline_run_id, generated_at
2. Extraction Summary — format, language, translation_confidence
3. Data Validation — table of all fields with PASS/FAIL/MISSING badges
4. Business Validation — discrepancy table (green/yellow/red rows)
5. Final Recommendation — large badge: AUTO_APPROVED / MANUAL_REVIEW / REJECTED
6. Audit Trail — timestamps for each pipeline stage
```

Saves to `outputs/reports/report_{invoice_no}_{timestamp}.html`

**File:** `agents/reporting_agent.py` — LangGraph node. Calls tool. Sets `report_path` in state.

**Done when:** HTML reports generated for all 6 invoices. `INV_EN_001` shows AUTO_APPROVED in green.

---

### Task 2.4 — Complete Core LangGraph Pipeline (30 min)
**File:** `workflows/invoice_pipeline.py` — Connect all 6 core agent nodes:
```
monitor → extract → translate → validate_data → validate_business → report
```
Add conditional edge after `validate_data`:
```python
graph.add_conditional_edges(
    "validate_data",
    lambda state: "report" if state["recommendation"] == "REJECTED" else "validate_business"
)
```

**Done when:** Running pipeline on all 6 invoices produces 6 HTML reports with correct statuses.

---

## Afternoon Block (Hours 5–9): RAG System

### Task 2.5 — Vector Indexer Tool + Indexing Agent (75 min)
**File:** `tools/vector_indexer_tool.py`

```python
# Uses sentence-transformers for embeddings + ChromaDB for storage
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, local, no API needed

def index_invoice(invoice_no, translated_text, metadata):
    chunks = chunk_text(translated_text, max_tokens=400, overlap=50)
    embeddings = model.encode(chunks)
    collection.upsert(
        ids=[f"{invoice_no}_chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))]
    )
```

ChromaDB persisted at `data/vector_store/`.

**File:** `agents/rag/indexing_agent.py` — LangGraph node triggered after translation. Sets `rag_indexed = True`.

**Done when:** After running pipeline, ChromaDB contains chunks for all 6 invoices. Verify with `collection.count()`.

---

### Task 2.6 — Semantic Retriever Tool + Retrieval Agent (45 min)
**File:** `tools/semantic_retriever_tool.py`

```python
def retrieve(query: str, top_k: int = 5, filter_invoice: str = None):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"invoice_no": filter_invoice} if filter_invoice else None
    )
    return [{"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(results['documents'][0],
                                       results['metadatas'][0],
                                       results['distances'][0])]
```

**File:** `agents/rag/retrieval_agent.py` — LangGraph RAG node. Accepts query + optional invoice filter.

**Done when:** Query "total amount on INV-1001" returns chunks containing total amount.

---

### Task 2.7 — Chunk Ranker Tool + Augmentation Agent (30 min)
**File:** `tools/chunk_ranker_tool.py`

Simplified reranking using distance score (ChromaDB returns cosine distance):
```python
def rerank(chunks: list, threshold: float = 0.7):
    # Convert distance to similarity score (1 - distance for cosine)
    scored = [(1 - c["distance"], c) for c in chunks]
    # Filter below threshold, sort descending
    filtered = [(score, c) for score, c in scored if score >= threshold]
    return [c for _, c in sorted(filtered, reverse=True)]
```

**File:** `agents/rag/augmentation_agent.py` — Filters and reranks retrieved chunks.

**Done when:** Low-relevance chunks filtered out, relevant chunks sorted by score.

---

### Task 2.8 — Response Synthesizer Tool + Generation Agent (45 min)
**File:** `tools/response_synthesizer_tool.py`

```python
SYSTEM_PROMPT = """You are an invoice auditing assistant. Answer questions ONLY
based on the provided invoice context. If the answer is not in the context,
say 'I cannot find this information in the available invoices.'
Do not make up information. Always cite the source invoice."""

def synthesize(query: str, chunks: list) -> dict:
    context = "\n\n---\n\n".join([
        f"[Source: {c['metadata']['invoice_no']}, Chunk {c['metadata']['chunk_index']}]\n{c['text']}"
        for c in chunks
    ])
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ])
    sources = list({c['metadata']['invoice_no'] for c in chunks})
    return {"answer": response.content, "sources": sources}
```

**File:** `agents/rag/generation_agent.py` — Calls synthesizer, returns answer + sources.

**Done when:** "What is the unit price of ITEM-A1?" returns correct value with source citation.

---

### Task 2.9 — Reflection Agent (30 min)
**File:** `agents/rag/reflection_agent.py`

Computes simplified RAG Triad scores:
```python
# Context Relevance: avg similarity score of retrieved chunks
context_relevance = mean([chunk["score"] for chunk in chunks])

# Groundedness: ask LLM "Is this answer supported by the context? Score 0-1"
groundedness = llm_score(answer, context)

# Answer Relevance: ask LLM "Does this answer address the question? Score 0-1"
answer_relevance = llm_score(answer, query)

# Flag if any score < 0.6
low_quality = any(s < 0.6 for s in [context_relevance, groundedness, answer_relevance])
```

**Done when:** Each RAG response includes all 3 scores. Off-topic query gets `low_quality=True`.

---

### Task 2.10 — Wire RAG LangGraph Pipeline (15 min)
**File:** `workflows/rag_pipeline.py`

```
retrieval_agent → augmentation_agent → generation_agent → reflection_agent
```

**Done when:** `rag_pipeline.invoke({"query": "...", "invoice_filter": None})` returns answer with scores.

---

### Day 2 Checkpoint ✓
- [ ] Data Validation Agent flags missing fields and bad currencies
- [ ] Business Validation Agent detects price/qty discrepancies against ERP
- [ ] HTML reports generated for all 6 invoices with correct recommendations
- [ ] Core LangGraph pipeline runs end-to-end (all 6 agents)
- [ ] ChromaDB indexed with all 6 invoice chunks
- [ ] RAG pipeline answers invoice questions with source citations
- [ ] RAG Triad scores computed per response
- [ ] All 5 RAG agents wired in LangGraph

---

# DAY 3 — MCP + LangFuse + RAI + Streamlit UI + Integration
**Target by end of Day 3:** Full working demo — UI shows reports, chat answers questions, pipeline is observable in LangFuse, all tools registered in MCP.

---

## Morning Block (Hours 1–4): MCP + LangFuse + RAI

### Task 3.1 — FastMCP Server (60 min)
**File:** `mcp/server.py`

Register all 10 tools as FastMCP tools:
```python
from fastmcp import FastMCP

mcp = FastMCP("AI Invoice Auditor Tool Registry")

@mcp.tool()
def invoice_watcher(incoming_dir: str) -> list[dict]:
    """Monitors incoming directory for new invoice files."""
    from tools.invoice_watcher_tool import watch
    return watch(incoming_dir)

@mcp.tool()
def data_harvester(file_path: str) -> dict:
    """Extracts text and tables from PDF, DOCX, or image invoices."""
    from tools.data_harvester_tool import harvest
    return harvest(file_path)

# ... register all 10 tools the same way

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or "sse" for HTTP
```

**Done when:** `python mcp/server.py` starts and lists all 10 tools.

---

### Task 3.2 — LangFuse Tracing Integration (60 min)
**File:** `core/observability.py`

```python
from langfuse import Langfuse
from functools import wraps

langfuse = Langfuse()

def trace_agent(agent_name: str):
    """Decorator to wrap any agent node with LangFuse tracing."""
    def decorator(func):
        @wraps(func)
        def wrapper(state: dict) -> dict:
            trace = langfuse.trace(name=f"invoice_pipeline.{agent_name}",
                                   input={"invoice": state.get("file_path")})
            span = trace.span(name=agent_name, input=state)
            try:
                result = func(state)
                span.end(output=result)
                return result
            except Exception as e:
                span.end(output={"error": str(e)}, level="ERROR")
                raise
        return wrapper
    return decorator
```

Apply `@trace_agent("extractor")` decorator to each agent node function.

**Done when:** Processing one invoice creates a visible trace in LangFuse dashboard with child spans for each agent.

---

### Task 3.3 — RAI Guardrails (45 min)
**File:** `core/rai_guardrails.py`

Two guardrails:

**1. Prompt Injection Detection** (applied during extraction):
```python
INJECTION_PATTERNS = [
    r"ignore (all )?(previous |prior )?instructions",
    r"you are now",
    r"disregard",
    r"system prompt",
    r"act as",
]

def check_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)
```
If detected → set `errors.append("SECURITY: Prompt injection attempt detected")`, skip LLM processing.

**2. RAG Domain Restriction** — already handled by system prompt in `response_synthesizer_tool.py`.

Apply `check_injection` in `extractor_agent.py` before LLM field extraction call.

**Done when:** Invoice text containing "ignore previous instructions" is flagged and blocked.

---

## Afternoon Block (Hours 5–9): Streamlit UI + Integration + Demo

### Task 3.4 — Streamlit Dashboard + Report Viewer (90 min)
**File:** `ui/app.py`

**Page 1 — Invoice Dashboard (sidebar nav)**
```
┌─────────────────────────────────────────────────────────────────┐
│  AI Invoice Auditor                               [Run Pipeline] │
├──────────────────────┬──────────────────────────────────────────┤
│  Invoice List        │  Report Viewer                           │
│  ─────────────────   │  ─────────────────────────────────────   │
│  ✅ INV-1001 AUTO    │  [HTML report displayed here via         │
│  ✅ INV-1002 AUTO    │   st.components.html()]                  │
│  ⚠️  INV-1003 REVIEW │                                          │
│  ⚠️  INV-1004 REVIEW │  [APPROVE] [REJECT]  ← HITL buttons     │
│  ❌ INV-1005 REJECT  │  (only shown for MANUAL_REVIEW status)   │
│  ❌ INV-1006 REJECT  │                                          │
└──────────────────────┴──────────────────────────────────────────┘
```

Key features:
- Load all reports from `outputs/reports/`
- Display status badges (green/yellow/red)
- Click invoice → show HTML report in right panel
- APPROVE/REJECT buttons for MANUAL_REVIEW invoices → write decision to `logs/human_decisions.json`
- "Run Pipeline" button → trigger pipeline on `data/incoming/`

---

### Task 3.5 — RAG Chat Interface (60 min)
**Page 2 — Invoice Q&A Chat**
```
┌─────────────────────────────────────────────────────────────────┐
│  Invoice Q&A Assistant                                          │
│                                                                 │
│  Filter by Invoice: [All Invoices ▼]                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🤖 Hello! Ask me anything about the processed invoices. │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 👤 What is the total on the Spanish invoice?            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🤖 The total amount on INV_ES_003 is €4,250.00...       │   │
│  │    Sources: INV_ES_003 (chunk 2)                        │   │
│  │    📊 Context: 0.82 | Grounded: 0.91 | Relevance: 0.88 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Type your question...]                         [Send]         │
└─────────────────────────────────────────────────────────────────┘
```

Use `st.chat_message` and `st.chat_input`. Display RAG Triad scores below each answer.

---

### Task 3.6 — E2E Integration Test (45 min)
**File:** `tests/test_e2e.py`

```python
def test_full_pipeline():
    # 1. Start ERP mock server (subprocess)
    # 2. Run pipeline on all 6 invoices
    # 3. Assert 6 reports exist in outputs/reports/
    # 4. Assert INV_EN_001 → AUTO_APPROVED
    # 5. Assert INV_EN_006 → REJECTED with missing fields
    # 6. Assert ChromaDB has > 20 chunks indexed
    # 7. Run RAG query, assert answer contains expected value

def test_rag_quality():
    # 5 benchmark queries → assert RAG Triad scores ≥ 0.6
```

**Done when:** `pytest tests/test_e2e.py -v` passes all assertions.

---

### Task 3.7 — Demo Script + README (30 min)
**File:** `main.py` — Update to be the one-command demo runner:
```python
# Starts ERP server, runs pipeline on all 6 invoices, launches Streamlit UI
```

**File:** `README.md` — Update with:
- Setup instructions (venv already exists, just `source .venv/bin/activate`)
- How to set `.env` keys
- How to run: `python main.py`
- Architecture diagram (text-based)

---

### Day 3 Checkpoint ✓
- [ ] FastMCP server lists all 10 tools
- [ ] LangFuse dashboard shows traces for pipeline runs
- [ ] RAI prompt injection detection working
- [ ] Streamlit dashboard shows all 6 invoices with status
- [ ] HTML report viewer embedded in UI
- [ ] APPROVE/REJECT buttons working for MANUAL_REVIEW invoices
- [ ] RAG chat interface answers questions with Triad scores
- [ ] E2E tests passing
- [ ] `python main.py` demo script launches everything

---

## Daily Hour Budget

| Day | Hours | Focus |
|---|---|---|
| Day 1 | 8 hrs | Foundation (2h) + ERP Mock (1h) + Sample Data (1h) + Extraction (2h) + Translation (2h) |
| Day 2 | 9 hrs | Validation (2.5h) + Reporting (1h) + Pipeline wiring (0.5h) + Full RAG (5h) |
| Day 3 | 8 hrs | MCP (1h) + LangFuse (1h) + RAI (0.75h) + UI (2.5h) + Tests (0.75h) + Demo (1h) |
| **Total** | **25 hrs** | **Achievable in 3 focused work days** |

---

## Coding Order (Dependency Chain)

```
Day 1:  core/ → erp_mock/ → data/incoming/ → tools/data_harvester → tools/invoice_watcher
        → agents/invoice_monitor → tools/lang_bridge → agents/translation
        → workflows/invoice_pipeline (stub all, fill 3 nodes)

Day 2:  tools/data_completeness_checker → agents/data_validation
        → tools/business_validation → agents/business_validation
        → tools/insight_reporter → agents/reporting
        → workflows/invoice_pipeline (complete all 6 nodes)
        → tools/vector_indexer → agents/rag/indexing
        → tools/semantic_retriever → agents/rag/retrieval
        → tools/chunk_ranker → agents/rag/augmentation
        → tools/response_synthesizer → agents/rag/generation
        → agents/rag/reflection → workflows/rag_pipeline

Day 3:  mcp/server.py → core/observability.py (LangFuse)
        → core/rai_guardrails.py → ui/app.py
        → tests/test_e2e.py → main.py (demo runner)
```

---

## LLM API Key Decision

| If you have... | Use |
|---|---|
| OpenAI API key | `langchain-openai` with `gpt-4o-mini` (fast, cheap) |
| Anthropic API key | `langchain-anthropic` with `claude-haiku-4-5` (fast, cheap) |
| No API key | `langchain-community` with `Ollama` (local `llama3.2`) |

Set in `.env` and pass to all LLM-using tools through `core/config.py`.

---

## Risk Mitigation for 3-Day Timeline

| Risk | Fast Fix |
|---|---|
| pytesseract not finding Tesseract binary | Set `pytesseract.pytesseract.tesseract_cmd` to full path in `.env` |
| deep-translator rate limit | Cache translations in a local `.json` dict keyed by text hash |
| LLM field extraction returning bad JSON | Wrap in `try/except`, fall back to regex-based extraction |
| ChromaDB slow on first embedding batch | Pre-warm model at startup, use `all-MiniLM-L6-v2` (fastest local model) |
| LangFuse credentials not available | Comment out tracing decorator, system still works |
| OpenAI API quota | Switch to `claude-haiku-4-5` or Ollama immediately |

---

*3-Day Sprint Plan v1.0 — Start coding immediately after reading this document.*

## AI Invoice Auditor

End-to-end invoice auditing pipeline built with LangGraph. It ingests invoices (PDF, DOCX, images), extracts fields, validates against business rules and ERP records, generates audit reports, and supports RAG Q&A over indexed invoices.

### Full Application Flow (Start to End)

1. **Monitor incoming invoices**
	- Watches the incoming directory for new files and skips already-processed items.
2. **Extract raw text and tables**
	- PDF: `pdfplumber`
	- DOCX: `python-docx`
	- Images: `pytesseract` OCR
3. **Detect language and translate**
	- Non-English text is translated to English for consistent extraction.
4. **Extract structured fields**
	- Uses LLM-backed parsing when keys are available, with graceful fallback.
5. **Validate data completeness**
	- Required fields, type checks, and currency normalization/acceptance.
6. **Business validation against ERP**
	- Compares invoice line items vs. ERP PO data with tolerance checks.
7. **Generate HTML report**
	- Report includes extraction summary, validation status, discrepancies, and recommendation.
8. **Index for RAG and answer questions**
	- Chunks and indexes invoice text; answers questions grounded in invoice data.

### Key Functionality

- **Invoice pipeline**: Sequential LangGraph pipeline for extraction, validation, and reporting.
- **RAG Q&A**: Semantic retrieval and grounded answer generation across invoices.
- **Streamlit UI**: Dashboard for reports and an invoice Q&A chat.
- **MCP tools**: Exposes pipeline tools via FastMCP for external clients.
- **Guardrails**: Prompt injection and PII detection before LLM extraction.

### Repository Map (What Runs What)

- **Invoice pipeline**: [workflows/invoice_pipeline.py](workflows/invoice_pipeline.py)
- **RAG pipeline**: [workflows/rag_pipeline.py](workflows/rag_pipeline.py)
- **UI**: [ui/app.py](ui/app.py)
- **MCP server**: [mcp_tools/server.py](mcp_tools/server.py)
- **Rules/config**: [config/rules.yaml](config/rules.yaml)

### Local Setup

1. Create and activate a virtual environment
	```bash
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

2. Install dependencies
	```bash
	pip install -r requirements.txt
	```

3. Configure environment variables
	- Create a `.env` file with your keys and paths.
	- Minimum recommended:
	  ```bash
	  OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY
	  TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
	  INCOMING_DIR=./data/incoming
	  REPORTS_DIR=./outputs/reports
	  ```

### Run Locally

#### Run the invoice pipeline
Process a specific file:
```bash
python workflows/invoice_pipeline.py data/incoming/INV_EN_001.pdf
```

Run with directory monitoring (polling):
```bash
python workflows/invoice_pipeline.py
```

#### Run the RAG pipeline
```bash
python workflows/rag_pipeline.py "What is the invoice total?"
```

#### Run the Streamlit UI
```bash
streamlit run ui/app.py
```

#### Run the MCP server
Stdio transport:
```bash
python mcp_tools/server.py
```

SSE transport:
```bash
python mcp_tools/server.py --sse
```

### How to Test Locally

Run the full test suite:
```bash
pytest -q
```

Tip: Some tests mock LLM/ERP calls to avoid network dependencies.

### Sample Data

- Incoming invoices are in [data/incoming](data/incoming)
- Mock ERP data is in [data/ERP_mockdata](data/ERP_mockdata)
- Reports are generated in [outputs/reports](outputs/reports)

### Notes

- LLM keys improve extraction quality and RAG answer generation. The pipeline still runs without them, but results may be reduced.
- OCR requires Tesseract installed locally and `TESSERACT_CMD` set on Windows.
- `main.py` is currently a minimal placeholder; use the pipeline/UI entry points above for execution.

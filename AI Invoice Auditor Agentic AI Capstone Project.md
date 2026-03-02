# AI Invoice Auditor

## An Agentic AI-Powered Multilingual Invoice Validation System

---

## Project Overview

### Business Context

A global logistics firm receives hundreds of vendor invoices daily from suppliers around the world — in different formats (PDF, DOCX, scanned images) and languages (English, Spanish, German, etc.).

Currently, invoices are manually reviewed and validated against internal ERP records to identify discrepancies, missing details, or fraudulent submissions.

This manual process is:
- **Time-consuming**, especially with multilingual data
- **Error-prone**, due to human oversight
- **Costly**, skilled auditors are needed to understand foreign languages

### Solution Requirements

The company seeks an **Agentic AI-powered system** that can:

1. Automatically ingest invoices received via email
2. Extract key fields from any format or language
3. Translate them into English
4. Validate extracted details against ERP records
5. Generate discrepancy reports
6. Allow human agents to query invoices through an RAG-based (Retrieval-Augmented Generation) QA system

---

## Project Objective

Design and implement an end-to-end Agentic AI system using open source or Azure-provided models that:

- Ingests multilingual invoices from a simulated mailbox
- Extracts and translates invoice data to English
- Validates invoice line-items against a mock ERP system
- Indexes invoice data for RAG-based question answering
- Generates validation reports with audit trails
- Supports human-in-the-loop feedback and corrections

---

## Project Description

An end-to-end **Agentic AI-powered system** for automated multilingual invoice processing and validation. The solution uses AI Agents to extract, translate, validate, and audit invoices received via email in various formats and languages. It integrates Retrieval-Augmented Generation (RAG) and human-in-the-loop feedback for enhanced accuracy and reliability.

---

## Problem Statement

Design and implement an end-to-end Agentic AI system called **AI Invoice Auditor** that automates the end-to-end lifecycle of multilingual invoice auditing using open-source frameworks such as **LangGraph**, **LangChain**, **FastAPI**, and **Pydantic**, also by incorporating the standard protocols using **MCP**.

### The System Must:

1. **Monitor** an email inbox for incoming invoices (simulate incoming invoices via a folder `/data/incoming` that acts as an email inbox)
2. **Extract** invoice data (text, tables, key fields) from attachments which can be in any language and format (format and language agnostic solution)
3. **Translate** extracted data to English
4. **Validate** invoice data against records in an enterprise system (e.g., ERP, database, or mock data) – The call to the enterprise system can be mocked for capstone implementation. The mock API call should return the data from ERP system to be validated against the data received at line item level in customer invoice
5. **Generate** a detailed validation report highlighting:
   - Discrepancies (data mismatch between invoice vs ERP system data)
   - Missing fields from invoice (invoice validation should be performed against dynamic rules configured)
   - Translation confidence
6. **Provide** question answering capabilities (implemented using vector-based RAG) on the invoices to the support agent (Human Agent)

### RAG Implementation (Agentic AI)

RAG should be implemented using agentic AI with the following agent roles:

| Agent | Role |
|-------|------|
| **Indexing Agent** | Parses and indexes invoice documents in a vector DB |
| **Retrieval Agent** | Accepts user query, generates query embedding, and performs similarity search |
| **Augmentation Agent** | Performs reranking of chunks based on similarity score |
| **Generation Agent** | Uses LLM to generate contextual responses |
| **Reflection Agent** | Evaluates response |

### RAG Tools

As part of the RAG Implementation, the following tools need to be leveraged:

| Tool | Description |
|------|-------------|
| **Vector-Indexer Tool** | Converts documents into embeddings and stores them in a vector database for efficient retrieval |
| **Semantic-Retriever Tool** | Performs semantic search using vector similarity to find contextually relevant data |
| **Chunk-Ranker Tool** | Re-ranks retrieved chunks based on relevance scores to improve context quality |
| **Response-Synthesizer Tool** | Generates natural language answers using retrieved context and user queries |

Implements an agentic workflow using open-source **LangGraph** framework, with RAG, prompt engineering, and human-in-the-loop feedback.

---

## Invoice Auditor Solution Agents

The overall Invoice Auditor solution should be implemented using agentic AI with the following agent roles:

| Agent | Role |
|-------|------|
| **Invoice-Monitor Agent** | Continuously monitors the mailbox for emails with invoice attachments (can read documents from file system rather than live mailbox) |
| **Extractor Agent** | Extracts data from various document formats |
| **Translation Agent** | Translates extracted invoice data from different languages into English |
| **Invoice Data Validation Agent** | Validates missing information based on defined rules |
| **Business Validation Agent** | Validates extracted invoice data with enterprise system data to identify discrepancies |
| **Reporting Agent** | Generates detailed reports highlighting discrepancies, missing fields, translation confidence, and final recommendation |

### Invoice Auditor Tools

The above-mentioned agents should have the appropriate tools to accomplish the assigned task:

| Tool | Description |
|------|-------------|
| **Invoice-Watcher Tool** | Monitors a designated mailbox or file system to detect and retrieve emails containing invoice attachments (can read from file system) |
| **Data Harvester Tool** | Extracts data from a wide range of document formats, enabling downstream processing |
| **Lang-Bridge Tool** | Converts extracted invoice content from various languages into English for standardized interpretation |
| **Data Completeness Checker Tool** | Validates invoice data for completeness and accuracy based on predefined business rules (rules.yml) |
| **Business Validation Tool** | Cross-verifies extracted invoice data against enterprise systems to identify mismatches or inconsistencies (dummy API as enterprise system) |
| **Insight Reporter Tool** | Generates comprehensive reports highlighting data gaps, validation results, translation confidence, and final recommendations |

---

## Additional Requirements

- Implements **RAI (Responsible AI)** guardrails across agents and tools wherever applicable, ensuring ethical, secure, and compliant use of AI components throughout the solution
- The solution should utilize the **Model Context Protocol (MCP)** to dynamically discover, access, and invoke the appropriate tools required for task execution
- The solution should integrate observability tools such as **LangFuse** to enable real-time tracing, monitoring, and debugging of the agentic AI workflows, ensuring transparency and operational reliability throughout the system

---

## Non-Functional Requirements

| Requirement | Description |
|-------------|-------------|
| **Performance** | Efficient processing of large invoice batches |
| **Scalability** | Modular agent design allows horizontal scaling |
| **Security** | Secure email access and document handling |
| **Reliability** | Robust error handling and fallback mechanisms |
| **Usability** | Intuitive UI for support agents |
| **Maintainability** | Modular codebase with documentation |

---

## Implementation Details

### Technology Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| **Language** | Python 3.11+ |
| **Framework** | LangGraph, LangChain |
| **Backend (ERP)** | FastAPI / Flask |
| **Frontend (UI)** | Streamlit |
| **OCR / Image** | Pytesseract |
| **Document Parsing** | pdfplumber, python-docx |
| **Vector DB** | FAISS / Qdrant / Chroma |
| **Schema Validation** | Pydantic |
| **Data Source** | `/data/incoming/` (simulated inbox) |
| **RAG Evaluation** | RAG Triad Scoring |
| **Communication Protocol** | Model Context Protocol (Fast MCP) |

---

## Deliverables

1. Source code (GitHub repo - Optional if Trainees can get access to repo)
2. Demo (video or live)
3. Presentation slides – not exceeding 5 slides

---

## Evaluation Criteria (for 50 Marks)

| Component | Weightage |
|-----------|-----------|
| Functional completeness | 35% |
| RAG & Agentic integration | 25% |
| UI & Human-in-the-loop | 15% |
| Agile Methodology | 10% |
| Code quality & modularity | 15% |

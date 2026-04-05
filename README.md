# Biomedical RAG Assistant with PubMed Documents

An OCR-powered Retrieval-Augmented Generation (RAG) assistant for biomedical research papers. The application lets you upload PubMed PDFs, extract text from scanned pages, index chunks in ChromaDB, retrieve relevant context, and generate concise answers with inline source citations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Features](#core-features)
3. [Architecture and Flow](#architecture-and-flow)
4. [Tech Stack](#tech-stack)
5. [Repository Structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Local Setup](#local-setup)
8. [Environment Variables](#environment-variables)
9. [Run the App](#run-the-app)
10. [Docker Deployment](#docker-deployment)
11. [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
12. [Testing](#testing)
13. [Current Limitations](#current-limitations)
14. [Troubleshooting](#troubleshooting)
15. [Future Improvements](#future-improvements)

## Project Overview

This project implements a biomedical assistant that combines:

- OCR-based PDF text extraction for scanned and image-heavy research papers.
- Vector search using sentence-transformer embeddings and ChromaDB.
- Context-grounded answer generation through a configurable LLM layer.
- Inline citation attachment to improve traceability of generated statements.
- A Streamlit chat UI for upload + ask workflows.

The system is intended as a capstone-style end-to-end RAG pipeline with practical components for ingestion, indexing, retrieval, and answer generation.

## Core Features

- PDF upload from the Streamlit sidebar.
- OCR extraction and chunking of uploaded biomedical PDFs.
- Incremental indexing into persistent ChromaDB storage.
- Top-k retrieval for user biomedical questions.
- LLM answer generation with prompt contract enforcement.
- Sentence-level citation attachment based on embedding similarity.
- Configurable LLM provider:
	- NVIDIA-compatible OpenAI client
	- Groq
- Unit tests for prompt contract, LLM manager, and core RAG utility behavior.
- Docker and Docker Compose support.

## Architecture and Flow

### High-Level Flow

1. User uploads PDF in Streamlit sidebar.
2. OCR pipeline extracts text elements from the document.
3. Text is chunked and embedded.
4. Chunks are appended to ChromaDB.
5. User asks a question in chat.
6. Retriever fetches relevant chunks.
7. LLM generates answer from retrieved context.
8. Sentences are post-processed with inline citations.

### Runtime Data Path

User PDF -> OCR extraction -> Chunking -> Embeddings -> ChromaDB

User question -> Retriever -> Context assembly -> LLM -> Sentence split -> Citation attach -> Chat response

## Tech Stack

- UI: Streamlit
- RAG and orchestration: LangChain components
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector database: ChromaDB
- OCR and PDF parsing:
	- unstructured.partition.pdf (OCR strategy)
	- pytesseract
	- pdf2image
- LLM access:
	- NVIDIA-compatible OpenAI client mode
	- Groq client mode
- Testing: pytest
- Containerization: Docker, Docker Compose

## Repository Structure

```
.
|-- app.py                    # Streamlit UI and chat workflow
|-- rag_groq.py              # Main RAG execution pipeline
|-- test.py                  # Simple CLI runner for run_rag()
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- pytest.ini
|-- src/
|   |-- config.py            # Environment and model/runtime configuration
|   |-- llm_manager.py       # Provider abstraction, retries, streaming, token budgeting
|   |-- prompt_contract.py   # Prompt/message construction and answer extraction
|   |-- retrieval.py         # Retriever loading helpers
|   |-- ocr_loader.py        # OCR loading for PDFs from data/pdfs
|   |-- ocr_chunking.py      # Chunking strategy for OCR documents
|   |-- ocr_indexer.py       # Upload-time OCR -> chunk -> embed -> index
|   |-- embeddings.py        # Embedding utility module
|   |-- vector_store.py      # Chroma vector store build helpers
|-- tests/
|   |-- test_llm_manager.py
|   |-- test_prompt_contract.py
|   |-- test_rag_groq.py
|-- data/
|   `-- pdfs/                # Optional local PDF dataset directory
`-- chroma_db/               # Persistent local vector store
```

## Prerequisites

- Python 3.11+
- pip
- OCR system dependencies:
	- Tesseract OCR
	- Poppler utilities

Notes:

- On Docker, OCR dependencies are already installed via Dockerfile.
- On local Windows/macOS/Linux, ensure Tesseract and Poppler are installed and available in PATH.

## Local Setup

1. Clone the repository:

```bash
git clone https://github.com/purva-2006-uhm/RAG.git
cd RAG
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` with your real API key(s).

## Environment Variables

The project loads `.env` from the repository root using `src/config.py`.

| Variable | Default | Required | Description |
|---|---|---|---|
| LLM_PROVIDER | nvidia | Yes | LLM backend. Options: `nvidia` or `groq`. |
| NVIDIA_API_KEY | - | If provider is nvidia | API key for NVIDIA-hosted compatible endpoint. |
| LLM_BASE_URL | https://integrate.api.nvidia.com/v1 | If provider is nvidia | Base URL for NVIDIA-compatible OpenAI client. |
| GROQ_API_KEY | - | If provider is groq | API key for Groq. |
| LLM_MODEL | openai/gpt-oss-120b | Yes | Model identifier sent to provider. |
| LLM_MAX_OUTPUT_TOKENS | 2048 | No | Requested max completion tokens. |
| LLM_TIMEOUT_SECONDS | 30 | No | Request timeout. |
| LLM_CONTEXT_WINDOW | 12000 | No | Used for output-budget checks. |
| LLM_TEMPERATURE | 0.2 | No | Sampling temperature. |
| LLM_MAX_RETRIES | 2 | No | Retry count on transient failures. |
| LLM_RESPONSE_MODE | structured_reasoning | No | Prompt response style. |

## Run the App

Start Streamlit:

```bash
streamlit run app.py
```

Open:

http://localhost:8501

Usage:

1. Upload a biomedical PDF in the sidebar.
2. Wait for OCR + indexing completion message.
3. Ask a biomedical question in chat input.
4. Inspect answer lines and expandable Sources section.

## Docker Deployment

### Option A: Docker Compose (recommended)

```bash
docker compose up --build
```

App URL:

http://localhost:8501

Compose mounts:

- `./chroma_db:/app/chroma_db` for persistent vectors.
- `./data:/app/data` for local dataset access.

### Option B: Docker CLI

```bash
docker build -t biomedical-rag .
docker run --rm -p 8501:8501 --env-file .env biomedical-rag
```

## How the RAG Pipeline Works

### 1) Ingestion and OCR

- `app.py` accepts uploaded PDF files.
- `src/ocr_indexer.py` saves each upload to a temporary file, extracts OCR text with `partition_pdf(..., strategy="ocr_only")`, and converts elements to LangChain `Document` objects.

### 2) Chunking

- Upload pipeline uses `RecursiveCharacterTextSplitter` with:
	- chunk_size: 800
	- chunk_overlap: 150
- Metadata includes filename and category to preserve source context.

### 3) Embeddings and Vector Index

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are normalized for cosine-like similarity behavior.
- Chunks are appended to ChromaDB collection `biomedical_rag` in `chroma_db`.

### 4) Retrieval

- `rag_groq.py` loads retriever from ChromaDB with configurable `k`.
- Retrieved chunks are joined into a context block separated by boundaries.

### 5) Prompt Contract and LLM Completion

- `src/prompt_contract.py` builds system + user messages and can extract text after `[FINAL ANSWER]` marker.
- `src/llm_manager.py` handles:
	- provider routing
	- token budget checks
	- retries with backoff
	- optional streaming and usage metadata extraction

### 6) Citation Attachment

- Generated response is split into sentences.
- Each sentence is matched against retrieved docs via embedding similarity.
- Best matching source is appended inline, for example:
	- `[paper.pdf]`
	- `[paper.pdf, p.4]`

## Testing

Run test suite:

```bash
pytest -q
```

Current test coverage includes:

- `tests/test_prompt_contract.py`
	- Empty input handling
	- Direct mode instruction handling
	- Final-answer extraction
- `tests/test_llm_manager.py`
	- Token estimation and output-budget logic
	- Completion output parsing
	- Stream chunk handling
- `tests/test_rag_groq.py`
	- Sentence filtering
	- Citation formatting
	- No-document fallback behavior

## Current Limitations

- OCR quality depends on scan quality and language clarity.
- Citation mapping is sentence-to-document similarity based (not strict quote alignment).
- Uploaded-document metadata may not always include page number for every extracted element.
- There is no reranker stage beyond embedding similarity.

## Troubleshooting

### API key errors

- If using NVIDIA mode and key is missing, requests fail.
- If using Groq mode and key is missing, requests fail.
- Confirm `.env` values and `LLM_PROVIDER` selection.

### OCR errors locally

- Ensure `tesseract` and `poppler` are installed and in PATH.
- Docker image includes these dependencies by default.

### Empty retrieval / weak answers

- Upload documents first so chunks exist in ChromaDB.
- Increase retrieval parameter `k` in `run_rag(query, k=...)`.
- Ask narrower, domain-specific questions.

## Future Improvements

- Add semantic and metadata-aware chunking variants.
- Add reranking for improved citation precision.
- Add stronger evaluation metrics and benchmark scripts.
- Expand test coverage to ingestion/indexing integration tests.
- Add observability for latency and token-cost analytics.

## License

Add a LICENSE file for your preferred open-source license before production use.

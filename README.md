# Document Q&A: End-to-End Multi-Tenant RAG Pipeline

```
DATA → CHUNK → EMBED → INDEX → UNDERSTAND QUERY → RETRIEVE → FUSE → RERANK
     → BUILD CONTEXT → GENERATE → VERIFY → CITE → RESPOND
```
Memory, caching, security, evaluation, and observability wrap this pipeline.

---

## 📁 Project Structure

```
RAG-MultiFile-QA/
├── backend/
│   ├── rag/                       # Core RAG engine modules
│   │   ├── builtin/               # System built-in documents (HOW_TO_USE.md)
│   │   ├── chunking.py            # Structure-aware parent-child chunker
│   │   ├── ingestion.py           # Multi-format doc parsing & sanitizer
│   │   ├── store.py               # FAISS dense + BM25 sparse hybrid index
│   │   ├── query.py               # Intent classifier & query rewriter
│   │   ├── retrieval.py           # Hybrid retrieval + RRF + Cross-Encoder reranker
│   │   ├── context.py             # Context assembler & token budget optimizer
│   │   ├── generation.py          # LLM generator with streaming gate
│   │   ├── verification.py        # Claim-level NLI verifier & citation checker
│   │   ├── memory.py              # Short-term / long-term memory management
│   │   ├── cache.py               # SQLite-backed semantic & KV caching
│   │   ├── security.py            # Injection detection, PII redaction, rate-limiter
│   │   ├── evaluation.py          # Golden dataset evaluation metrics
│   │   ├── observability.py       # Distributed tracing & metrics logging
│   │   ├── pipeline.py            # End-to-end RAG pipeline coordinator
│   │   ├── cli.py                 # Backend CLI interface
│   │   └── config.py              # Centralized configuration dataclass
│   ├── data/                      # Tenant data, indices, and cache persistence
│   ├── logs/                      # Observability traces and metrics logs
│   ├── requirements.txt           # Backend-specific dependencies
│   └── __init__.py
├── frontend/
│   ├── app.py                     # Streamlit frontend application
│   ├── ui/                        # UI stylesheet and custom assets
│   │   └── style.css
│   ├── legacy/                    # Legacy single-file application archive
│   │   └── main_legacy.py
│   ├── requirements.txt           # Frontend dependencies
│   └── __init__.py
├── tests/
│   ├── test_rag.py                # Unit and integration test suite
│   ├── conftest.py                # Pytest configuration and path resolution
│   ├── test_data/                 # Fixtures and evaluation datasets
│   │   ├── eval.jsonl
│   │   └── widgets.md
│   └── __init__.py
├── docs/
│   ├── rag_end2end_notes.md       # Architecture & engineering notes
│   └── rag_end2end_notes.pdf      # PDF export of engineering notes
├── .github/
│   └── workflows/
│       └── rag_test.yml           # CI workflow (pytest across Python versions)
├── main.py                        # Unified launcher entry point
├── requirements.txt               # Unified project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

Using `uv` (recommended):
```bash
uv sync
```
Or using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application

```bash
# Using uv:
uv run streamlit run frontend/app.py

# Or using python3 / virtualenv:
python3 -m streamlit run frontend/app.py

# Offline demo mode (model-free):
RAG_OFFLINE=1 uv run streamlit run frontend/app.py
```

### 3. Backend CLI

```bash
# Ingest documents
uv run python -m backend.rag.cli --offline ingest tests/test_data/widgets.md

# Ask questions
uv run python -m backend.rag.cli --offline ask "How long is the warranty period?"

# Run evaluation suite
uv run python -m backend.rag.cli --offline eval tests/test_data/eval.jsonl --name base
```

### 4. Running Tests

```bash
uv run pytest -v
```

---

## 🏗️ Architecture & Layer Map

| Layer | Module | Description |
|---|---|---|
| **1 Ingestion** | `ingestion.py`, `chunking.py`, `embeddings.py`, `store.py` | PDF/DOCX/TXT/MD/CSV parsing, boilerplate stripping, metadata extraction. **Structure-aware parent-child chunking**: sections form parents, sentence-packed children embedded with heading paths. FAISS dense + BM25 sparse hybrid index. |
| **2 Query Understanding** | `query.py`, `retrieval.py` | Normalization, conversational query rewriting, sub-query decomposition, metadata filter extraction, RRF (Reciprocal Rank Fusion), and Cross-Encoder reranking. |
| **3 Context Engine** | `context.py` | Deduplication (Exact / Cosine / Jaccard) → MMR diversity → parent expansion → extractive compression → token budget optimization → `[S1]..[Sn]` citation labeling. |
| **4 Generation** | `generation.py`, `llm.py` | XML-delimited prompt formatting, canary token protection, `NO_ANSWER` abstention, and token-streaming gate. |
| **5 Verification** | `verification.py` | Atomic claim extraction, evidence matching, NLI (Natural Language Inference) entailment checking, numeric consistency check, and groundedness scoring. |
| **6 Memory** | `memory.py` | Short-term conversation history, semantic recall of older turns, rolling summarization, and PII-redacted long-term fact extraction. |
| **7 Caching** | `cache.py` | SQLite-backed LRU cache for embeddings, retrieval results, and LLM responses, plus semantic answer caching. |
| **8 Security** | `security.py` | Upload file validation (magic bytes, active content, zip bomb checks), prompt injection neutralization, PII redaction, and rate-limiting. |
| **9 Evaluation** | `evaluation.py`, `cli.py` | Recall, Precision, MRR, NDCG, groundedness, citation accuracy, and automated regression comparison. |
| **10 Observability** | `observability.py` | Per-request tracing (spans, events, timings), stage latency metrics, and PII-safe JSONL log export. |

---

## ⚙️ Configuration

All configuration settings are defined in [`backend/rag/config.py`](backend/rag/config.py) (`RAGConfig`) and can be dynamically overridden via environment variables prefixed with `RAG_` (e.g. `RAG_LLM_MODEL`, `RAG_MAX_CONTEXT_ITEMS`, `RAG_DENSE_K`).

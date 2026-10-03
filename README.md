# Document Q&A: an end-to-end RAG pipeline

```
DATA → CHUNK → EMBED → INDEX → UNDERSTAND QUERY → RETRIEVE → FUSE → RERANK
     → BUILD CONTEXT → GENERATE → VERIFY → CITE → RESPOND
```
Memory, caching, security, evaluation and observability wrap this chain.

## Run

```bash
pip install -r requirements.txt
export HUGGINGFACE_HUB_TOKEN=hf_...        # or .env / st.secrets
streamlit run main.py

RAG_OFFLINE=1 streamlit run main.py        # stub embedder + extractive LLM: no models, no key
python -m rag.cli --offline ingest tests/data/widgets.md
python -m rag.cli --offline ask "How long is the warranty period?"
python -m rag.cli --offline eval tests/data/eval.jsonl --name base
python -m rag.cli --offline eval tests/data/eval.jsonl --name new --baseline reports/base.json   # exit 1 on regression
pytest -q
```

## Layer map

| Layer | Module | What it does |
|---|---|---|
| 1 Ingestion | `ingestion.py`, `chunking.py`, `embeddings.py`, `store.py` | pdf/docx/txt/md/csv parsing, boilerplate removal, metadata (title, year, tenant, owner, roles, version), heading-tree extraction. **Structure-aware parent-child chunking**: sections become parents (tiny siblings merged, huge ones split); sentence-packed children (~500 chars, 1-sentence overlap; tables/code atomic) are embedded as `Section > Path` + text; the LLM receives parents. FAISS (numpy fallback) + BM25, incremental and versioned per tenant. |
| 2 Query understanding | `query.py`, `retrieval.py` | normalization, follow-up rewriting from memory, decomposition, metadata-filter extraction (relaxed if it matches nothing), dense + BM25 run concurrently, RRF fusion, cross-encoder rerank, answerability gate. |
| 3 Context | `context.py` | dedup (exact / cosine / Jaccard) → MMR diversity → parent expansion → extractive compression → token budget → ordering (document / edges / relevance) → `S1..Sn` labels. |
| 4 Generation | `generation.py`, `llm.py` | XML-delimited prompt, canary token, `NO_ANSWER` abstention, streaming with a gate so abstentions never flash, citation parsing/validation (unknown labels stripped). |
| 5 Verification | `verification.py` | atomic claim extraction, sentence-level evidence matching, NLI (lexical+embedding fallback), numeric check, citation accuracy, groundedness, one self-correction retry, abstain below threshold. |
| 7 Memory | `memory.py` | short-term turns, semantic recall of older turns, rolling summarization with archive, long-term facts (PII-redacted, injection-filtered); isolated per (tenant, user). |
| 8 Caching | `cache.py` | sqlite-backed LRU for embeddings, retrieval and LLM calls, plus a semantic answer cache scoped by tenant+roles+filters. Versioned keys (index version, embedder, prompt, LLM, memory facts) invalidate automatically. |
| 9 Security | `security.py` | upload validation (magic bytes, PDF active content, zip-bomb ratio, macros), prompt and indirect-injection scan/neutralize, PII redaction, tenant isolation + role ACL, constant-time token auth, token-bucket rate limits. |
| 10 Evaluation | `evaluation.py`, `cli.py` | Recall/Precision/Hit@K, MRR, NDCG; context precision/recall/redundancy; correctness, groundedness, citation accuracy, abstention accuracy; latency/cost. JSON report with config snapshot; `compare_reports` for regression checks. |
| 11 Observability | `observability.py` | per-request trace (spans, counters, events, errors), per-stage latency, retrieval logging, cost, failure rate; `MetricsStore` writes PII-redacted jsonl. |

## Configuration
All knobs live in `rag/config.py` (`RAGConfig`) and can be overridden with `RAG_<FIELD>` environment variables (models, chunk sizes, k's, thresholds, token budgets, rate limits).

## Auth
Without `RAG_AUTH_TOKENS` the UI runs as a single local user. To require login, set
`RAG_AUTH_TOKENS='{"<token>": {"tenant": "acme", "user": "alice", "roles": ["member"]}}'`. Authentication proves who you are; authorization (tenant + role ACL) is enforced inside the store on every query.

## Eval dataset format (JSONL)
```json
{"id":"q1","question":"...","expected_answer":"...","answerable":true,
 "relevant":[{"filename":"paper.pdf","page":12,"contains":"Adam","grade":3}]}
```
Set `"answerable": false` to test abstention.

## Notes
- The offline hash embedder and `FakeLLM` (`rag/testing.py`) are test doubles. Quality numbers from `--offline` runs only check the plumbing.
- `main_legacy.py` is the original single-file app, kept for reference.

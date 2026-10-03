"""Single place for every tunable. Snapshot of this goes into eval reports (versioning)."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class RAGConfig:
    # models 
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    use_reranker: bool = True
    use_nli: bool = True
    hf_token_env: str = "HUGGINGFACE_HUB_TOKEN"

    # storage 
    data_dir: str = "data"
    log_dir: str = "logs"

    # ingestion / chunking 
    max_file_mb: int = 25
    max_pages: int = 500
    max_csv_rows: int = 100_000
    max_zip_ratio: float = 100.0
    redact_pii_in_index: bool = False
    injection_policy: str = "neutralize"  # neutralize | drop | flag
    child_chars: int = 500
    child_overlap_sentences: int = 1
    parent_max_chars: int = 3000
    parent_min_chars: int = 300

    # query / retrieval
    max_subqueries: int = 4
    dense_k: int = 30
    bm25_k: int = 30
    rrf_k: int = 60
    candidates: int = 40          # fused candidates sent to the reranker
    rerank_keep: int = 12         # reranked candidates handed to context stage
    min_rerank_score: float = 0.01  # abstain gate (sigmoid cross-encoder score)
    min_dense_sim: float = 0.20     # abstain gate when no reranker is available

    # context 
    max_context_items: int = 6
    mmr_lambda: float = 0.7
    near_dup_sim: float = 0.95
    expand_parent_max_chars: int = 2400
    compress_min_chars: int = 600
    compress_keep_ratio: float = 0.6
    context_window: int = 8192
    answer_reserve_tokens: int = 700
    memory_budget_tokens: int = 600
    order_strategy: str = "auto"  # auto | relevance | document | edges

    # generation
    max_new_tokens: int = 600
    temperature: float = 0.2

    # verification
    claim_extraction: str = "llm"   # llm | sentence
    min_groundedness: float = 0.7
    abstain_groundedness: float = 0.4
    max_retries: int = 1

    # memory
    short_term_turns: int = 3
    summarize_after_turns: int = 8
    keep_recent_turns: int = 4
    memory_top_k: int = 3
    memory_threshold: float = 0.35

    # caching 
    cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    embed_batch_size: int = 64

    # security 
    query_rate_per_min: int = 30
    upload_rate_per_min: int = 10
    max_question_chars: int = 2000

    #cost
    cost_per_1k_in: float = 0.0
    cost_per_1k_out: float = 0.0

    def snapshot(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        cfg = cls()
        for key, val in os.environ.items():
            if key.startswith("RAG_") and hasattr(cfg, key[4:].lower()):
                name = key[4:].lower()
                cur = getattr(cfg, name)
                if isinstance(cur, bool):
                    setattr(cfg, name, val.lower() in ("1", "true", "yes"))
                elif isinstance(cur, int):
                    setattr(cfg, name, int(val))
                elif isinstance(cur, float):
                    setattr(cfg, name, float(val))
                else:
                    setattr(cfg, name, val)
        return cfg
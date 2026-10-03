"""Layer 3 (part 2): dense + BM25 (concurrent) -> metadata filter -> RRF -> cross-encoder rerank."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .cache import KVCache, make_key
from .observability import ctx_submit, current
from .types import Hit, Principal, QueryPlan
from .utils import sigmoid


def rrf(rankings: list, k: int = 60, weights: list | None = None) -> dict:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d))."""
    scores: dict = defaultdict(float)
    for r, ranking in enumerate(rankings):
        w = weights[r] if weights else 1.0
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += w / (k + rank)
    return dict(scores)


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16):
        self.name, self.batch_size = model_name, batch_size
        self._model, self._failed = None, False

    @property
    def available(self) -> bool:
        if self._model is None and not self._failed:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.name)
            except Exception as e:  # offline / model missing -> degrade to RRF order
                self._failed = True
                current().fail("reranker_load", e)
        return self._model is not None

    def score(self, query: str, texts: list) -> np.ndarray | None:
        if not self.available:
            return None
        raw = np.asarray(self._model.predict([(query, t) for t in texts], batch_size=self.batch_size), dtype=np.float32)
        if raw.ndim > 1:
            raw = raw[:, -1]
        if raw.min() < 0 or raw.max() > 1:
            raw = 1.0 / (1.0 + np.exp(-raw))
        return raw


@dataclass
class RetrievalResult:
    hits: list
    strength: float = 0.0
    kind: str = "none"        # rerank | dense | none
    filter_relaxed: bool = False
    from_cache: bool = False


class Retriever:
    def __init__(self, cfg, embedder, store, reranker=None, cache: KVCache | None = None):
        self.cfg, self.embedder, self.store, self.reranker, self.cache = cfg, embedder, store, reranker, cache
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retr")

    def is_answerable(self, r: RetrievalResult) -> bool:
        if not r.hits:
            return False
        return r.strength >= (self.cfg.min_rerank_score if r.kind == "rerank" else self.cfg.min_dense_sim)

    def retrieve(self, principal: Principal, plan: QueryPlan) -> RetrievalResult:
        tr, cfg = current(), self.cfg
        index = self.store.get(principal.tenant_id)
        if len(index) == 0:
            return RetrievalResult([])
        queries = plan.all_queries[: 1 + cfg.max_subqueries]
        rr_name = self.reranker.name if self.reranker else None
        # key = query + everything that can change the result (index/model/params) + who is asking (ACL)
        key = make_key("ret", principal.tenant_id, sorted(principal.roles), queries, plan.filters, index.version,
                       self.embedder.name, cfg.dense_k, cfg.bm25_k, cfg.rrf_k, cfg.candidates, cfg.rerank_keep, rr_name)
        if self.cache:
            c = self.cache.get_json("retrieval", key)
            if c is not None:
                return RetrievalResult([Hit(**h) for h in c["hits"]], c["strength"], c["kind"], c["relaxed"], True)

        with tr.span("retrieval"):
            hits, relaxed = self._search(principal, index, queries, plan.filters)
        with tr.span("rerank"):
            hits = self._rerank(index, queries, hits)
        hits = hits[: cfg.rerank_keep]

        if any(h.rerank is not None for h in hits):
            strength, kind = max(h.rerank for h in hits), "rerank"
        else:
            strength, kind = (max((h.dense for h in hits), default=0.0), "dense")
        tr.log("retrieval", queries=queries, filters=plan.filters, relaxed=relaxed, strength=round(strength, 4), kind=kind,
               hits=[{"id": h.chunk_id, "fused": round(h.fused, 5), "dense": round(h.dense, 3),
                      "rerank": None if h.rerank is None else round(h.rerank, 4)} for h in hits])
        res = RetrievalResult(hits, float(strength), kind, relaxed)
        if self.cache:
            self.cache.set_json("retrieval", key, {"hits": [asdict(h) for h in hits], "strength": res.strength,
                                                   "kind": kind, "relaxed": relaxed})
        return res

    #  internals
    def _search(self, principal, index, queries, flt):
        cfg = self.cfg
        relaxed = False
        mask = index.mask(principal, flt)
        if flt and not mask.any():           # filter matched nothing -> relax rather than answer from nothing
            mask, flt, relaxed = index.mask(principal, None), None, True
        if not mask.any():
            return [], relaxed
        qvecs = self.embedder.embed_documents(queries)   # one batched call for all (sub)queries

        jobs = []
        for qi, q in enumerate(queries):
            w = 1.0 if qi == 0 else 0.8
            jobs.append((ctx_submit(self.pool, index.dense_search, qvecs[qi], cfg.dense_k, mask), True, w))
            jobs.append((ctx_submit(self.pool, index.bm25_search, q, cfg.bm25_k, mask), False, w))
        rankings, weights = [], []
        for fut, _is_dense, w in jobs:        # dense and BM25 ran concurrently
            rankings.append([index.children[i].chunk_id for i, _ in fut.result()])
            weights.append(w)
        fused = rrf(rankings, cfg.rrf_k, weights)
        top = sorted(fused.items(), key=lambda x: -x[1])[: cfg.candidates]
        hits = [Hit(cid, f, float(max(qv @ index.vector(cid) for qv in qvecs))) for cid, f in top]
        return hits, relaxed

    def _rerank(self, index, queries, hits):
        if not hits or not self.reranker or not self.reranker.available:
            return hits
        texts = [index.child(h.chunk_id).embed_text for h in hits]
        best = np.full(len(hits), -1.0, dtype=np.float32)
        for q in queries:                      # max-pool over the question and its sub-questions
            s = self.reranker.score(q, texts)
            if s is None:
                return hits
            best = np.maximum(best, s)
        for h, s in zip(hits, best):
            h.rerank = float(s)
        return sorted(hits, key=lambda h: -h.rerank)

"""Layer 10: evaluation. Retrieval, context, generation and system are measured separately; the same
dataset is re-run after every change (regression testing).

Dataset: JSONL, one object per line:
  {"id": "q1", "question": "...", "expected_answer": "...", "answerable": true,
   "relevant": [{"filename": "paper.pdf", "page": 12, "contains": "Adam", "grade": 3}]}
A chunk matches a `relevant` spec if every given field matches (filename, page, case-insensitive substring).
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter

import numpy as np

from .types import Principal
from .utils import tokenize


# ------------------------------------------------------------------ primitives
def spec_matches(spec: dict, filename: str, page, text: str) -> bool:
    if spec.get("filename") and spec["filename"] != filename:
        return False
    if spec.get("page") is not None and spec["page"] != page:
        return False
    if spec.get("contains") and spec["contains"].lower() not in text.lower():
        return False
    return True


def grade_of(specs: list, filename: str, page, text: str) -> int:
    return max((int(s.get("grade", 1)) for s in specs if spec_matches(s, filename, page, text)), default=0)


def covered_specs(specs: list, chunks: list) -> int:
    return sum(any(spec_matches(s, c[0], c[1], c[2]) for c in chunks) for s in specs)


def precision_at_k(grades: list, k: int) -> float:
    top = grades[:k]
    return sum(g > 0 for g in top) / k if k else 0.0


def reciprocal_rank(grades: list) -> float:
    for i, g in enumerate(grades, start=1):
        if g > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(grades: list, ideal: list, k: int) -> float:
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / idcg if idcg else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(tokenize(pred)), Counter(tokenize(gold))
    common = sum((p & g).values())
    if not common:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _mean(xs: list):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def load_dataset(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ evaluation
def evaluate(pipe, principal: Principal, dataset: list, ks=(1, 3, 5, 10), generate: bool = True,
             name: str = "run", out_dir: str = "reports") -> dict:
    index = pipe.store.get(principal.tenant_id)
    emb = pipe.embedder
    rows, lat = [], []
    for item in dataset:
        q, specs = item["question"], item.get("relevant", [])
        answerable = item.get("answerable", True)
        row = {"id": item.get("id", q[:30]), "question": q}

        # ---- retrieval (scored on the reranked child chunks) ----
        _, res = pipe.retrieve_only(principal, q)
        chunks = [(index.child(h.chunk_id).meta["filename"], index.child(h.chunk_id).meta.get("page"),
                   index.child(h.chunk_id).text) for h in res.hits]
        grades = [grade_of(specs, *c) for c in chunks]
        if answerable and specs:
            ideal = [int(s.get("grade", 1)) for s in specs]
            for k in ks:
                row[f"hit@{k}"] = float(any(g > 0 for g in grades[:k]))
                row[f"recall@{k}"] = covered_specs(specs, chunks[:k]) / len(specs)
                row[f"precision@{k}"] = precision_at_k(grades, k)
                row[f"ndcg@{k}"] = ndcg_at_k(grades, ideal, k)
            row["mrr"] = reciprocal_rank(grades)

        # ---- generation + context ----
        if generate:
            t0 = time.perf_counter()
            ans = pipe.ask(principal, q, session_id=f"eval-{uuid.uuid4().hex[:8]}")
            lat.append((time.perf_counter() - t0) * 1000)
            row["abstained"] = ans.abstained
            row["abstain_correct"] = float(ans.abstained == (not answerable))
            if answerable and not ans.abstained:
                exp = item.get("expected_answer", "")
                if exp:
                    row["correctness_f1"] = token_f1(ans.text, exp)
                    row["correctness_sim"] = float(emb.embed_documents([ans.text])[0] @ emb.embed_documents([exp])[0])
                row["groundedness"] = ans.groundedness
                row["citation_accuracy"] = ans.citation_accuracy
                row["answer_relevance"] = float(emb.embed_documents([q])[0] @ emb.embed_documents([ans.text])[0])
            if answerable and specs and ans.context:
                items = ans.context
                igr = []
                cchunks = []
                for it in items:
                    ch = [index.child(cid) for cid in it["chunk_ids"] if cid in index.row]
                    cchunks += [(c.meta["filename"], c.meta.get("page"), c.text) for c in ch]
                    igr.append(max((grade_of(specs, c.meta["filename"], c.meta.get("page"), c.text) for c in ch), default=0))
                row["context_precision"] = sum(g > 0 for g in igr) / len(igr)
                row["context_recall"] = covered_specs(specs, cchunks) / len(specs)
                if len(items) > 1:
                    v = emb.embed_documents([it["text"] for it in items])
                    sim = v @ v.T
                    row["context_redundancy"] = float((sim.sum() - len(items)) / (len(items) * (len(items) - 1)))
        rows.append(row)

    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    def group(k: str) -> str:
        if k.split("@")[0] in ("hit", "recall", "precision", "ndcg", "mrr"):
            return "retrieval"
        return "context" if k.startswith("context_") and k not in ("context_precision_",) and k in (
            "context_precision", "context_recall", "context_redundancy") else "generation"

    metrics = {f"{group(k)}.{k}": _mean([r.get(k) for r in rows]) for k in keys}
    sysm = pipe.metrics.summary()
    if lat:
        metrics["system.latency_p50_ms"] = round(float(np.percentile(lat, 50)), 1)
        metrics["system.latency_p95_ms"] = round(float(np.percentile(lat, 95)), 1)
    for k in ("tokens_in", "tokens_out", "cost_usd", "cache_hit_rate", "failure_rate"):
        if k in sysm:
            metrics[f"system.{k}"] = sysm[k]
    report = {"name": name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(rows),
              "config": pipe.cfg.snapshot(), "index_version": index.version, "metrics": metrics, "per_item": rows}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


# ------------------------------------------------------------ regression testing
_LOWER_IS_BETTER = ("latency", "tokens", "cost", "failure", "redundancy")


def compare_reports(base: dict, new: dict, tol: float = 0.02, rel_tol: float = 0.25) -> list:
    """Flag metrics that got worse: quality drops > tol (absolute); latency/cost/tokens rise > rel_tol (relative)."""
    out = []
    for k, b in base["metrics"].items():
        n = new["metrics"].get(k)
        if b is None or n is None:
            continue
        if any(w in k for w in _LOWER_IS_BETTER):
            if b > 0 and (n - b) / b > rel_tol:
                out.append({"metric": k, "baseline": b, "new": n, "kind": "worse (higher)"})
        elif b - n > tol:
            out.append({"metric": k, "baseline": b, "new": n, "kind": "worse (lower)"})
    return out

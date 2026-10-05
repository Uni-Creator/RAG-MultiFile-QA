"""Layer 4: context engineering.

reranked children -> dedup -> diversity (MMR) -> expand to parent -> compress -> token budget -> order -> [S1..Sn]

Budgeting runs before the final ordering on purpose: if we ordered first and then truncated, a
"lost-in-the-middle" layout would cut off the second-best chunk sitting at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .observability import current
from .security import can_read
from .types import ContextItem, Principal, QueryPlan
from .utils import count_tokens, jaccard, split_sentences, tokenize, truncate_to_tokens


@dataclass
class _Cand:
    child: object
    vec: np.ndarray
    score: float
    rel: float = 1.0


@dataclass
class ContextResult:
    items: list
    tokens: int = 0
    budget: int = 0
    dropped: int = 0
    stats: dict = field(default_factory=dict)


class ContextBuilder:
    def __init__(self, cfg, embedder, store):
        self.cfg, self.embedder, self.store = cfg, embedder, store

    def build(self, principal: Principal, plan: QueryPlan, hits: list, budget_tokens: int, wide: bool = False) -> ContextResult:
        tr, cfg = current(), self.cfg
        with tr.span("context"):
            index = self.store.get(principal.tenant_id)
            # Also load system index so cross-tenant hits (HOW_TO_USE.md etc.) can be resolved
            from .pipeline import SYSTEM_TENANT  # deferred import
            system_index = self.store.get(SYSTEM_TENANT)

            def _resolve(chunk_id: str):
                """Return (child, vector, src_index) from whichever index holds this chunk."""
                if chunk_id in index.row:
                    return index.child(chunk_id), index.vector(chunk_id), index
                if chunk_id in system_index.row:
                    return system_index.child(chunk_id), system_index.vector(chunk_id), system_index
                return None, None, None

            cands = []
            for h in hits:
                child, vec, src_index = _resolve(h.chunk_id)
                if child is None:
                    continue
                # ACL: user chunks need normal check; system chunks are always readable
                if src_index is not system_index and not can_read(principal, child.meta):
                    continue
                c = _Cand(child, vec, h.score)
                c._src_index = src_index  # stash for parent lookup
                cands.append(c)

            n0 = len(cands)
            if not cands:
                return ContextResult([], 0, budget_tokens)
            lo, hi = min(c.score for c in cands), max(c.score for c in cands)
            for c in cands:
                c.rel = (c.score - lo) / (hi - lo) if hi > lo else 1.0

            cands = self._dedup(cands)
            n_dedup = len(cands)
            cands = self._mmr(cands, cfg.max_context_items + (4 if wide else 0))
            items = self._expand_multi(system_index, cands)
            if not wide:
                items = self._compress(items, plan)
            items, dropped = self._budget(items, budget_tokens)
            items = self._order(items, plan.intent)
            for i, it in enumerate(items, start=1):
                it.label = f"S{i}"
            used = sum(it.tokens for it in items)
            stats = {"candidates": n0, "after_dedup": n_dedup, "after_mmr": len(cands), "items": len(items), "dropped_for_budget": dropped}
            tr.log("context", tokens=used, budget=budget_tokens, **stats,
                   selected=[{"label": it.label, "chunk": it.chunk_id, "score": round(it.score, 4), "tokens": it.tokens} for it in items])
            return ContextResult(items, used, budget_tokens, dropped, stats)

    # -------------------------------------------------------------- 1. dedup
    def _dedup(self, cands: list) -> list:
        kept, seen_text, toks = [], set(), []
        for c in sorted(cands, key=lambda c: -c.score):
            norm = " ".join(c.child.text.lower().split())
            if norm in seen_text:
                continue
            ts = set(tokenize(norm))
            if any(float(c.vec @ k.vec) >= self.cfg.near_dup_sim or jaccard(ts, t) >= 0.85 for k, t in zip(kept, toks)):
                continue
            seen_text.add(norm)
            kept.append(c)
            toks.append(ts)
        return kept

    # ----------------------------------------------------------- 2. diversity
    def _mmr(self, cands: list, k: int) -> list:
        lam = self.cfg.mmr_lambda
        remaining = sorted(cands, key=lambda c: -c.score)
        selected = [remaining.pop(0)]
        while remaining and len(selected) < k:
            best, best_val = None, -1e9
            for c in remaining:
                val = lam * c.rel - (1 - lam) * max(float(c.vec @ s.vec) for s in selected)
                if val > best_val:
                    best, best_val = c, val
            selected.append(best)
            remaining.remove(best)
        return selected

    # ------------------------------------------------- 3. parent expansion
    def _expand(self, index, cands: list) -> list:
        """Single-index expand (legacy path, kept for test compatibility)."""
        for c in cands:
            c._src_index = index  # ensure attribute exists
        return self._expand_multi(index, cands)

    def _expand_multi(self, system_index, cands: list) -> list:
        """Expand to parents, routing each candidate to its originating index."""
        groups: dict = {}
        for c in cands:
            groups.setdefault(c.child.parent_id, []).append(c)
        items = []
        for pid, group in groups.items():
            src_index = getattr(group[0], "_src_index", None)
            if src_index is None or pid not in src_index.parents:
                # fallback: try user index first, then system
                src_index = system_index if pid in system_index.parents else src_index
            if src_index is None or pid not in src_index.parents:
                continue
            parent = src_index.parent(pid)
            group.sort(key=lambda c: c.child.meta.get("chunk_index", 0))
            anchors = [c.child.text for c in group]
            text = parent.text if len(parent.text) <= self.cfg.expand_parent_max_chars else "\n…\n".join(anchors)
            m = group[0].child.meta
            pages = [c.child.meta.get("page") for c in group if c.child.meta.get("page") is not None]
            meta = {"filename": m["filename"], "doc_id": m["doc_id"], "section_path": m["section_path"],
                    "page": min(pages) if pages else None, "chunk_index": m.get("chunk_index", 0),
                    "chunk_ids": [c.child.chunk_id for c in group], "anchors": anchors, "compressed": False}
            it = ContextItem("", group[0].child.chunk_id, pid, text, max(c.score for c in group), meta,
                             suspicious=any(c.child.meta.get("suspicious") for c in group))
            items.append(it)
        return sorted(items, key=lambda i: -i.score)

    # ---------------------------------------------------------- 4. compression
    def _compress(self, items: list, plan: QueryPlan) -> list:
        cfg = self.cfg
        qvecs = None
        for it in items:
            t = it.text
            if len(t) < cfg.compress_min_chars or "|---" in t or "```" in t:
                continue
            sents = split_sentences(t)
            if len(sents) < 4:
                continue
            if qvecs is None:
                qvecs = self.embedder.embed_documents(plan.all_queries)
            sv = self.embedder.embed_documents(sents)
            sims = (sv @ qvecs.T).max(axis=1)
            anchor_blob = " ".join(it.meta["anchors"])
            keep = {i for i, s in enumerate(sents) if s in anchor_blob}   # the retrieved child is always kept
            target = cfg.compress_keep_ratio * len(t)
            size = sum(len(sents[i]) for i in keep)
            for i in np.argsort(-sims):
                if size >= target:
                    break
                if i not in keep:
                    keep.add(int(i))
                    size += len(sents[i])
            if len(keep) == len(sents):
                continue
            out, prev = [], -1
            for i in sorted(keep):
                out.append(("… " if prev != -1 and i != prev + 1 else "") + sents[i])
                prev = i
            it.text = " ".join(out)
            it.meta["compressed"] = True
        return items

    # ----------------------------------------------------------- 5. budgeting
    def _budget(self, items: list, budget: int) -> tuple:
        kept, used, dropped = [], 0, 0
        for it in sorted(items, key=lambda i: -i.score):
            cost = count_tokens(it.text) + 40   # + tag overhead
            if used + cost <= budget:
                it.tokens = cost
            elif budget - used >= 120:           # truncate the last item rather than lose it entirely
                it.text = truncate_to_tokens(it.text, budget - used - 40)
                it.meta["truncated"] = True
                it.tokens = count_tokens(it.text) + 40
            else:
                dropped += 1
                continue
            used += it.tokens
            kept.append(it)
        return kept, dropped

    # -------------------------------------------------------------- 6. ordering
    def _order(self, items: list, intent: str) -> list:
        strat = self.cfg.order_strategy
        if strat == "auto":
            strat = "document" if intent in ("summarization", "multi_step") else ("edges" if len(items) >= 4 else "relevance")
        if strat == "document":
            return sorted(items, key=lambda i: (i.meta["filename"], i.meta.get("page") or 0, i.meta.get("chunk_index", 0)))
        ranked = sorted(items, key=lambda i: -i.score)
        if strat == "edges":   # best first, 2nd best last, ... weakest in the middle (lost-in-the-middle)
            front, back = ranked[0::2], ranked[1::2]
            return front + back[::-1]
        return ranked

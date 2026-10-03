"""Layer 2: per-tenant stores - dense vectors (FAISS when available), BM25, parent/child document store.

One TenantIndex per tenant => tenant isolation is physical (separate files, separate indexes),
and ACL role checks are applied inside the search mask as defence in depth.
"""
from __future__ import annotations

import heapq
import json
import math
import threading
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .security import can_read, validate_id
from .types import Child, Parent, Principal
from .utils import sha, tokenize

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = 0
        self.inv: dict = defaultdict(list)
        self.dl = np.zeros(0)
        self.avg = 1.0

    def build(self, token_lists: list):
        self.n = len(token_lists)
        self.inv = defaultdict(list)
        self.dl = np.array([len(t) for t in token_lists], dtype=np.float32)
        self.avg = float(self.dl.mean()) if self.n else 1.0
        for i, toks in enumerate(token_lists):
            tf: dict = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, c in tf.items():
                self.inv[t].append((i, c))

    def search(self, q_tokens: list, k: int, mask: np.ndarray | None = None) -> list:
        scores: dict = defaultdict(float)
        for t in set(q_tokens):
            postings = self.inv.get(t)
            if not postings:
                continue
            idf = math.log(1 + (self.n - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, tf in postings:
                if mask is not None and not mask[i]:
                    continue
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


class TenantIndex:
    def __init__(self, path: Path, embed_name: str, dim: int, cfg):
        self.path, self.embed_name, self.dim, self.cfg = path, embed_name, dim, cfg
        self.lock = threading.RLock()
        self.docs: dict = {}
        self.parents: dict = {}
        self.children: list = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.row: dict = {}
        self.bm25 = BM25()
        self._faiss = None
        self._rebuild()

    #  versions
    @property
    def version(self) -> str:
        c = self.cfg
        return sha(self.embed_name, c.child_chars, c.parent_max_chars, c.child_overlap_sentences,
                   *sorted(self.docs), n=12)

    def __len__(self) -> int:
        return len(self.children)

    #  mutation
    def _rebuild(self):
        self.row = {c.chunk_id: i for i, c in enumerate(self.children)}
        self.bm25.build([tokenize(c.embed_text) for c in self.children])
        if faiss is not None and len(self.children):
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(np.ascontiguousarray(self.vectors))
        else:
            self._faiss = None

    def add_document(self, meta: dict, parents: list, children: list, vectors: np.ndarray):
        with self.lock:
            self.docs[meta["doc_id"]] = meta
            self.parents.update({p.parent_id: p for p in parents})
            self.children.extend(children)
            self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) else vectors.astype(np.float32)
            self._rebuild()  # vectors of existing docs are NOT recomputed: incremental indexing
            self.save()

    def remove_document(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            keep = [i for i, c in enumerate(self.children) if c.doc_id != doc_id]
            self.children = [self.children[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
            self.parents = {k: p for k, p in self.parents.items() if p.doc_id != doc_id}
            del self.docs[doc_id]
            self._rebuild()
            self.save()
            return True

    def find_by_filename(self, filename: str):
        return next((d for d in self.docs.values() if d["filename"] == filename), None)

    #  search
    def mask(self, principal: Principal, flt: dict | None) -> np.ndarray:
        return np.array([can_read(principal, c.meta) and matches(c.meta, flt) for c in self.children], dtype=bool)

    def dense_search(self, qvec: np.ndarray, k: int, mask: np.ndarray) -> list:
        n = len(self.children)
        if not n or not mask.any():
            return []
        k = min(k, int(mask.sum()))
        if self._faiss is not None and mask.all():
            D, I = self._faiss.search(qvec[None].astype(np.float32), k)
            return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        s = self.vectors @ qvec
        s = np.where(mask, s, -np.inf)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx if np.isfinite(s[i])]

    def bm25_search(self, query: str, k: int, mask: np.ndarray) -> list:
        return self.bm25.search(tokenize(query), k, mask)

    def child(self, chunk_id: str) -> Child:
        return self.children[self.row[chunk_id]]

    def vector(self, chunk_id: str) -> np.ndarray:
        return self.vectors[self.row[chunk_id]]

    def parent(self, parent_id: str) -> Parent:
        return self.parents[parent_id]

    def known_meta(self, principal: Principal) -> dict:
        docs = [d for d in self.docs.values() if can_read(principal, d)]
        return {"filenames": [d["filename"] for d in docs],
                "years": {d["year"] for d in docs if d.get("year")},
                "doc_types": {d["doc_type"] for d in docs}}

    #  persistence
    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        payload = {"embed_name": self.embed_name, "dim": self.dim, "docs": self.docs,
                   "parents": [asdict(p) for p in self.parents.values()],
                   "children": [asdict(c) for c in self.children]}
        tmp = self.path / "index.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path / "index.json")
        np.save(self.path / "vectors.npy", self.vectors)

    @classmethod
    def load_or_new(cls, path: Path, embed_name: str, dim: int, cfg) -> "TenantIndex":
        f = path / "index.json"
        if not f.exists():
            return cls(path, embed_name, dim, cfg)
        payload = json.loads(f.read_text(encoding="utf-8"))
        if payload["embed_name"] != embed_name or payload["dim"] != dim:
            # embedding versioning: vectors from another model are meaningless -> start clean, caller re-ingests
            return cls(path, embed_name, dim, cfg)
        idx = cls(path, embed_name, dim, cfg)
        idx.docs = payload["docs"]
        idx.parents = {p["parent_id"]: Parent(**p) for p in payload["parents"]}
        idx.children = [Child(**c) for c in payload["children"]]
        idx.vectors = np.load(path / "vectors.npy")
        idx._rebuild()
        return idx


def matches(meta: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    for key, cond in flt.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            if "gte" in cond and (val is None or val < cond["gte"]):
                return False
            if "lte" in cond and (val is None or val > cond["lte"]):
                return False
        elif isinstance(cond, (list, tuple, set)):
            if val not in cond:
                return False
        elif val != cond:
            return False
    return True


class Store:
    def __init__(self, cfg, embedder):
        self.cfg, self.embedder = cfg, embedder
        self.root = Path(cfg.data_dir)
        self._idx: dict = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> TenantIndex:
        validate_id(tenant_id, "tenant_id")
        with self._lock:
            if tenant_id not in self._idx:
                self._idx[tenant_id] = TenantIndex.load_or_new(
                    self.root / tenant_id / "index", self.embedder.name, self.embedder.dim, self.cfg)
            return self._idx[tenant_id]

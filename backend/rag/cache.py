"""Layer 8: four independent caches. Every key carries the versions that can change the result."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict

import numpy as np

from .observability import current
from .utils import cosine, sha, stable_json


def make_key(*parts) -> str:
    return sha(stable_json(list(parts)), n=32)


class KVCache:
    """sqlite-backed (survives restarts) with an in-memory LRU in front. JSON/bytes only - no pickle."""

    def __init__(self, path: str | None = None, mem_items: int = 4096):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v BLOB, ts REAL, PRIMARY KEY(ns,k))")
        self.lock = threading.Lock()
        self.mem: OrderedDict = OrderedDict()
        self.mem_items = mem_items

    def _get(self, ns: str, key: str, ttl: float | None):
        mk = (ns, key)
        with self.lock:
            if mk in self.mem:
                self.mem.move_to_end(mk)
                v, ts = self.mem[mk]
            else:
                row = self.db.execute("SELECT v, ts FROM kv WHERE ns=? AND k=?", mk).fetchone()
                if not row:
                    current().count(f"cache.{ns}.miss")
                    return None
                v, ts = row
                self.mem[mk] = (v, ts)
                if len(self.mem) > self.mem_items:
                    self.mem.popitem(last=False)
        if ttl and time.time() - ts > ttl:
            current().count(f"cache.{ns}.miss")
            return None
        current().count(f"cache.{ns}.hit")
        return v

    def _set(self, ns: str, key: str, value: bytes):
        ts = time.time()
        with self.lock:
            self.mem[(ns, key)] = (value, ts)
            if len(self.mem) > self.mem_items:
                self.mem.popitem(last=False)
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?,?)", (ns, key, value, ts))
            self.db.commit()

    def get_json(self, ns, key, ttl=None):
        v = self._get(ns, key, ttl)
        return None if v is None else json.loads(v.decode("utf-8"))

    def set_json(self, ns, key, obj):
        self._set(ns, key, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def get_vec(self, ns, key):
        v = self._get(ns, key, None)
        return None if v is None else np.frombuffer(v, dtype=np.float32).copy()

    def set_vec(self, ns, key, vec: np.ndarray):
        self._set(ns, key, np.asarray(vec, dtype=np.float32).tobytes())

    def clear(self):
        with self.lock:
            self.mem.clear()
            self.db.execute("DELETE FROM kv")
            self.db.commit()


class SemanticCache:
    """Similar queries share an answer - but only inside the same (tenant, roles, filters) scope and
    only if every version in `versions` matches. Process-lifetime (not persisted) on purpose."""

    def __init__(self, threshold: float = 0.95, max_per_scope: int = 300):
        self.threshold = threshold
        self.max = max_per_scope
        self.store: dict = {}
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.store.clear()

    def get(self, scope: str, qvec: np.ndarray, versions: str):
        with self.lock:
            best, best_sim = None, 0.0
            for e in self.store.get(scope, []):
                if e["versions"] != versions:
                    continue
                s = cosine(qvec, e["vec"])
                if s > best_sim:
                    best, best_sim = e, s
        if best is not None and best_sim >= self.threshold:
            current().count("cache.semantic.hit")
            return best["payload"]
        current().count("cache.semantic.miss")
        return None

    def set(self, scope: str, qvec: np.ndarray, versions: str, payload: dict):
        with self.lock:
            lst = self.store.setdefault(scope, [])
            lst.append({"vec": qvec, "versions": versions, "payload": payload})
            del lst[: max(0, len(lst) - self.max)]

    def invalidate_scope_prefix(self, prefix: str):
        with self.lock:
            for k in [k for k in self.store if k.startswith(prefix)]:
                del self.store[k]

"""Layer 11: request tracing, retrieval logging, latency/cost/failure monitoring."""
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager

from .security import redact_pii


class Trace:
    def __init__(self, name: str = "request", principal=None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.user = principal.key if principal else None
        self.t0 = time.perf_counter()
        self.started = time.time()
        self.spans: dict = defaultdict(float)     # stage -> ms
        self.counters: dict = defaultdict(float)
        self.events: list = []
        self.errors: list = []
        self._lock = threading.Lock()

    @contextmanager
    def span(self, stage: str):
        t = time.perf_counter()
        try:
            yield
        except Exception as e:
            self.fail(stage, e)
            raise
        finally:
            with self._lock:
                self.spans[stage] += (time.perf_counter() - t) * 1000

    def count(self, key: str, n: float = 1):
        with self._lock:
            self.counters[key] += n

    def log(self, _event: str, **data):
        with self._lock:
            self.events.append({"event": _event, **data})

    def fail(self, stage: str, exc: Exception):
        with self._lock:
            self.errors.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}"[:300]})

    def finish(self) -> dict:
        return {
            "trace_id": self.id, "name": self.name, "user": self.user, "ts": self.started,
            "total_ms": round((time.perf_counter() - self.t0) * 1000, 1),
            "spans_ms": {k: round(v, 1) for k, v in self.spans.items()},
            "counters": dict(self.counters), "events": self.events, "errors": self.errors,
        }


_NULL = Trace("null")
_current: contextvars.ContextVar = contextvars.ContextVar("rag_trace", default=None)


def set_trace(t):
    _current.set(t)


def current() -> Trace:
    return _current.get() or _NULL


def span(stage: str):
    return current().span(stage)


def ctx_submit(pool, fn, *args, **kwargs):
    """Submit to a thread pool while keeping the current trace visible in the worker."""
    ctx = contextvars.copy_context()
    return pool.submit(ctx.run, fn, *args, **kwargs)


def _pct(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 1)


class MetricsStore:
    """Appends finished traces to logs/traces.jsonl (PII-redacted) and aggregates them."""

    def __init__(self, log_dir: str, cfg=None):
        self.path = os.path.join(log_dir, "traces.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        self.recent: deque = deque(maxlen=1000)
        self.cfg = cfg
        self._lock = threading.Lock()

    def record(self, trace: Trace) -> dict:
        d = trace.finish()
        c = d["counters"]
        if self.cfg:
            d["cost_usd"] = round(c.get("llm.tokens_in", 0) / 1000 * self.cfg.cost_per_1k_in
                                  + c.get("llm.tokens_out", 0) / 1000 * self.cfg.cost_per_1k_out, 6)
        with self._lock:
            self.recent.append(d)
            try:
                line, _ = redact_pii(json.dumps(d, default=str, ensure_ascii=False))
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        return d

    def summary(self) -> dict:
        rows = [r for r in self.recent if r["name"] == "ask"]
        if not rows:
            return {"requests": 0}
        stages: dict = defaultdict(list)
        for r in rows:
            for k, v in r["spans_ms"].items():
                stages[k].append(v)
        hits = sum(v for r in rows for k, v in r["counters"].items() if k.startswith("cache.") and k.endswith(".hit"))
        miss = sum(v for r in rows for k, v in r["counters"].items() if k.startswith("cache.") and k.endswith(".miss"))
        return {
            "requests": len(rows),
            "latency_ms": {"p50": _pct([r["total_ms"] for r in rows], 0.5), "p95": _pct([r["total_ms"] for r in rows], 0.95)},
            "stage_p50_ms": {k: _pct(v, 0.5) for k, v in stages.items()},
            "tokens_in": sum(r["counters"].get("llm.tokens_in", 0) for r in rows),
            "tokens_out": sum(r["counters"].get("llm.tokens_out", 0) for r in rows),
            "embedding_texts": sum(r["counters"].get("embed.texts", 0) for r in rows),
            "cost_usd": round(sum(r.get("cost_usd", 0) for r in rows), 5),
            "cache_hit_rate": round(hits / (hits + miss), 3) if hits + miss else 0.0,
            "failure_rate": round(sum(1 for r in rows if r["errors"]) / len(rows), 3),
        }

"""Layer 7: memory is a separate retrieval path from documents; both feed the context.

short-term   last N turns as real chat messages (resolves "its")
semantic     embed older turns, fetch only the relevant ones
summary      old turns are folded into a rolling summary (+ archived for semantic recall)
long-term    user facts/preferences persisted across sessions
isolation    files are keyed by validated (tenant, user); every API takes a Principal
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .generation import _CITE
from .observability import current
from .security import redact_pii, scan_injection, validate_id
from .types import Principal
from .utils import sha, truncate_to_tokens

SUMMARY_SYS = ("Summarize the conversation so far in at most 120 words, keeping names, decisions and stated preferences. "
               "Output only the summary.")
_FACT_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?remember\s+(?:that\s+)?(?P<f>.+)", re.I),
    re.compile(r"\b(?P<f>my name is [\w .'-]{2,40})", re.I),
    re.compile(r"\b(?P<f>call me [\w .'-]{2,30})", re.I),
    re.compile(r"\b(?P<f>i (?:prefer|like|want) (?:you to )?[^.?!]{3,120})", re.I),
    re.compile(r"\b(?P<f>(?:from now on|always|never)[^.?!]{3,120})", re.I),
    re.compile(r"\b(?P<f>we decided [^.?!]{3,120})", re.I),
]


class MemoryContext:
    def __init__(self, summary: str = "", relevant: list | None = None, facts: list | None = None):
        self.summary, self.relevant, self.facts = summary, relevant or [], facts or []

    @property
    def facts_hash(self) -> str:
        return sha(*self.facts, n=8)

    def render(self, max_tokens: int) -> str:
        parts = []
        if self.facts:
            parts.append("User preferences and notes:\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.summary:
            parts.append("Summary of earlier conversation:\n" + self.summary)
        if self.relevant:
            parts.append("Possibly relevant earlier exchanges:\n" + "\n\n".join(self.relevant))
        return truncate_to_tokens("\n\n".join(parts), max_tokens) if parts else ""


class MemoryManager:
    def __init__(self, cfg, embedder, llm=None):
        self.cfg, self.embedder, self.llm = cfg, embedder, llm
        self._lock = threading.RLock()
        self._cache: dict = {}

    # ------------------------------------------------------------------ storage
    def _path(self, p: Principal) -> Path:
        validate_id(p.tenant_id, "tenant_id")
        validate_id(p.user_id, "user_id")
        return Path(self.cfg.data_dir) / p.tenant_id / "memory" / f"{p.user_id}.json"

    def _load(self, p: Principal) -> dict:
        with self._lock:
            if p.key not in self._cache:
                f = self._path(p)
                self._cache[p.key] = json.loads(f.read_text("utf-8")) if f.exists() else {"facts": [], "sessions": {}}
            return self._cache[p.key]

    def _save(self, p: Principal):
        f = self._path(p)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache[p.key], ensure_ascii=False), "utf-8")
        tmp.replace(f)

    @staticmethod
    def _session(data: dict, sid: str) -> dict:
        return data["sessions"].setdefault(sid, {"turns": [], "archive": [], "summary": ""})

    # ---------------------------------------------------------------- short-term
    def recent_turns(self, p: Principal, sid: str) -> list:
        with self._lock:
            turns = self._session(self._load(p), sid)["turns"][-self.cfg.short_term_turns:]
        out = []
        for t in turns:
            out += [{"role": "user", "content": t["q"]}, {"role": "assistant", "content": t["a"]}]
        return out

    # ----------------------------------------------------------------- retrieval
    @staticmethod
    def _turn_text(t: dict) -> str:
        return f"Q: {t['q']}\nA: {t['a']}"[:600]

    def retrieve(self, p: Principal, sid: str, query: str) -> MemoryContext:
        cfg = self.cfg
        with current().span("memory"):
            with self._lock:
                data = self._load(p)
                sess = self._session(data, sid)
                older = sess["archive"] + sess["turns"][:-cfg.short_term_turns]
                facts = [f["text"] for f in data["facts"]]
                summary = sess["summary"]
            relevant: list = []
            qv = None
            if older or len(facts) > 5:
                qv = self.embedder.embed_query(query)
            if older:
                sims = self.embedder.embed_documents([self._turn_text(t) for t in older]) @ qv
                for i in np.argsort(-sims)[: cfg.memory_top_k]:
                    if sims[i] >= cfg.memory_threshold:
                        relevant.append(self._turn_text(older[int(i)])[:400])
            if len(facts) > 5:  # keep the 5 most relevant long-term facts
                fs = self.embedder.embed_documents(facts) @ qv
                facts = [facts[int(i)] for i in np.argsort(-fs)[:5]]
            ctx = MemoryContext(summary, relevant, facts)
            current().log("memory", facts=len(facts), relevant_turns=len(relevant), has_summary=bool(summary))
            return ctx

    # ------------------------------------------------------------------- writing
    def record_turn(self, p: Principal, sid: str, question: str, answer: str):
        answer = _CITE.sub("", answer).strip()[:1200]
        with self._lock:
            data = self._load(p)
            sess = self._session(data, sid)
            sess["turns"].append({"q": question[:800], "a": answer, "ts": time.time()})
            if len(sess["turns"]) > self.cfg.summarize_after_turns:
                self._fold(sess)
            self._save(p)

    def _fold(self, sess: dict):
        keep = self.cfg.keep_recent_turns
        old, sess["turns"] = sess["turns"][:-keep], sess["turns"][-keep:]
        text = (f"Previous summary: {sess['summary']}\n\n" if sess["summary"] else "") + "\n\n".join(self._turn_text(t) for t in old)
        summary = ""
        if self.llm is not None:
            try:
                summary = self.llm.complete([{"role": "system", "content": SUMMARY_SYS},
                                             {"role": "user", "content": text}], max_tokens=220, temperature=0.0).strip()
            except Exception as e:
                current().fail("summarize", e)
        sess["summary"] = summary or (sess["summary"] + " " + " | ".join(t["q"][:80] for t in old)).strip()[:900]
        sess["archive"] = (sess["archive"] + old)[-200:]

    def detect_facts(self, text: str) -> list:
        out = []
        for pat in _FACT_PATTERNS:
            m = pat.search(text)
            if m:
                out.append(m.group("f").strip(" .")[:200])
        return out

    def remember(self, p: Principal, text: str) -> bool:
        text, _ = redact_pii(text.strip())   # never persist PII
        if not text or scan_injection(text):  # never persist instruction-like text
            return False
        with self._lock:
            data = self._load(p)
            if any(f["text"].lower() == text.lower() for f in data["facts"]):
                return False
            data["facts"].append({"id": uuid.uuid4().hex[:8], "text": text, "ts": time.time()})
            data["facts"] = data["facts"][-100:]
            self._save(p)
        return True

    def maybe_store_facts(self, p: Principal, question: str) -> list:
        return [f for f in self.detect_facts(question) if self.remember(p, f)]

    def facts(self, p: Principal) -> list:
        with self._lock:
            return [dict(f) for f in self._load(p)["facts"]]

    def forget_fact(self, p: Principal, fact_id: str):
        with self._lock:
            data = self._load(p)
            data["facts"] = [f for f in data["facts"] if f["id"] != fact_id]
            self._save(p)

    def clear_session(self, p: Principal, sid: str):
        with self._lock:
            self._load(p)["sessions"].pop(sid, None)
            self._save(p)

    def forget_all(self, p: Principal):
        with self._lock:
            self._cache[p.key] = {"facts": [], "sessions": {}}
            self._save(p)

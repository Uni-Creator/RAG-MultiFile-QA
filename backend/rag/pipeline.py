"""Orchestrator: wires all 11 layers.

DATA -> CHUNK -> EMBED -> INDEX -> UNDERSTAND QUERY -> RETRIEVE -> FUSE -> RERANK
     -> BUILD CONTEXT -> GENERATE -> VERIFY -> CITE -> RESPOND
Memory, caching, security, evaluation and observability sit around that chain.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from .cache import KVCache, SemanticCache
from .chunking import chunk_document
from .config import RAGConfig
from .context import ContextBuilder
from .embeddings import CachedEmbedder, SentenceTransformerEmbedder
from .generation import PROMPT_VERSION, Generator, StreamGate
from .ingestion import parse_document
from .llm import CachedLLM, HFChatLLM
from .memory import MemoryManager
from .observability import MetricsStore, Trace, set_trace
from .query import QueryUnderstanding, normalize_query
from .retrieval import CrossEncoderReranker, Retriever
from .security import RateLimiter, SecurityError, can_read
from .store import Store
from .types import Answer, Principal
from .utils import sha, stable_json
from .verification import NLIModel, Verifier

BUILTIN_DIR = Path(__file__).parent / "builtin"
SYSTEM_TENANT = "__system__"

# Small-talk intent -> direct reply (never goes through retrieval)
_SMALL_TALK_REPLIES: dict[str, str] = {
    "greeting": "Hello! How can I help you with your documents?",
    "casual": "I'm doing well, thanks! What would you like to know about your documents?",
    "gratitude": "You're welcome!",
    "farewell": "Goodbye! Feel free to come back whenever you have questions.",
}


class RAGPipeline:
    def __init__(self, cfg: RAGConfig | None = None, llm=None, embedder=None, reranker=None, nli=None):
        cfg = cfg or RAGConfig.from_env()
        self.cfg = cfg
        os.makedirs(cfg.data_dir, exist_ok=True)
        self.cache = KVCache(os.path.join(cfg.data_dir, "cache.sqlite")) if cfg.cache_enabled else None
        self.sem_cache = SemanticCache(cfg.semantic_cache_threshold) if cfg.cache_enabled else None
        self.embedder = CachedEmbedder(embedder or SentenceTransformerEmbedder(cfg.embedding_model, cfg.embed_batch_size),
                                       self.cache, cfg.embed_batch_size)
        self.llm = CachedLLM(llm or HFChatLLM(cfg.llm_model, os.getenv(cfg.hf_token_env)), self.cache)
        if reranker is None and cfg.use_reranker:
            reranker = CrossEncoderReranker(cfg.reranker_model)
        if nli is None and cfg.use_nli:
            nli = NLIModel(cfg.nli_model)
        self.store = Store(cfg, self.embedder)
        self.retriever = Retriever(cfg, self.embedder, self.store, reranker or None, self.cache)
        self.query = QueryUnderstanding(self.llm, cfg)
        self.ctx = ContextBuilder(cfg, self.embedder, self.store)
        self.gen = Generator(self.llm, cfg)
        self.verifier = Verifier(self.llm, self.embedder, nli or None, cfg)
        self.memory = MemoryManager(cfg, self.embedder, self.llm)
        self.limiter = RateLimiter({"query": cfg.query_rate_per_min, "upload": cfg.upload_rate_per_min})
        self.metrics = MetricsStore(cfg.log_dir, cfg)
        self.ensure_builtin_documents()

    # builtin system documents
    def ensure_builtin_documents(self) -> None:
        """Index built-in documents (e.g. HOW_TO_USE.md) under the __system__ tenant.
        Idempotent: only re-indexes when file content changes (SHA-256 hash check)."""
        if not BUILTIN_DIR.exists():
            return
        system_index = self.store.get(SYSTEM_TENANT)
        for path in BUILTIN_DIR.glob("*.md"):
            try:
                content = path.read_bytes()
                content_hash = hashlib.sha256(content).hexdigest()
                # Check if already indexed with same hash
                existing = system_index.find_by_filename(path.name)
                if existing and existing.get("content_hash") == content_hash:
                    continue  # unchanged
                self._ingest_system_document(path.name, content, content_hash)
                logging.getLogger("rag").info("builtin: indexed %s (hash %s)", path.name, content_hash[:12])
            except Exception as e:
                logging.getLogger("rag").warning("builtin: failed to index %s: %s", path.name, e)

    def _ingest_system_document(self, filename: str, content: bytes, content_hash: str) -> None:
        """Ingest a system document under the __system__ tenant with public visibility."""
        system_principal = Principal(SYSTEM_TENANT, "system", ("system",))
        tr = Trace("ingest_builtin", system_principal)
        set_trace(tr)
        try:
            parsed, _ = parse_document(filename, content, system_principal, self.cfg)
            # Override metadata: system doc, visible to all users
            parsed.meta.update(
                source_type="system",
                visibility="public",
                content_hash=content_hash,
            )
            index = self.store.get(SYSTEM_TENANT)
            old = index.find_by_filename(filename)
            parents, children = chunk_document(parsed, self.cfg)
            version = (old.get("version", 1) + 1) if old else 1
            parsed.meta.update(version=version, n_chunks=len(children), n_parents=len(parents))
            for c in children:
                c.meta.update(version=version, source_type="system", visibility="public")
            vecs = self.embedder.embed_documents([c.embed_text for c in children])
            if old:
                index.remove_document(old["doc_id"])
            index.add_document(parsed.meta, parents, children, vecs)
        finally:
            set_trace(None)

    # ingest
    def ingest(self, principal: Principal, files: list, allowed_roles: list | None = None) -> list:
        """files: [(filename, bytes)]. Returns one report per file. Unchanged files are skipped (incremental)."""
        self.limiter.check(principal, "upload")
        index = self.store.get(principal.tenant_id)
        reports = []
        for name, data in files:
            tr = Trace("ingest", principal)
            set_trace(tr)
            rep = {"filename": name, "status": "failed"}
            try:
                if sha(data, n=16) in index.docs:
                    rep.update(status="skipped", detail="unchanged (same content already indexed)")
                    continue
                with tr.span("parse"):
                    parsed, stats = parse_document(name, data, principal, self.cfg, allowed_roles)
                old = index.find_by_filename(parsed.filename)
                with tr.span("chunk"):
                    parents, children = chunk_document(parsed, self.cfg)
                version = (old.get("version", 1) + 1) if old else 1
                parsed.meta.update(version=version, n_chunks=len(children), n_parents=len(parents))
                for c in children:
                    c.meta["version"] = version
                with tr.span("embed"):
                    vecs = self.embedder.embed_documents([c.embed_text for c in children])
                if old:  # new version of the same filename replaces the old one
                    index.remove_document(old["doc_id"])
                index.add_document(parsed.meta, parents, children, vecs)
                rep.update(status="indexed", doc_id=parsed.meta["doc_id"], version=version, chunks=len(children),
                           parents=len(parents), **stats)
            except SecurityError as e:
                tr.fail("ingest", e)
                rep.update(status="rejected", detail=str(e))
            except Exception as e:
                tr.fail("ingest", e)
                rep.update(status="failed", detail=f"{type(e).__name__}: {e}")
            finally:
                reports.append(rep)
                self.metrics.record(tr)
                set_trace(None)
        return reports

    def list_documents(self, principal: Principal) -> list:
        idx = self.store.get(principal.tenant_id)
        return [d for d in idx.docs.values() if can_read(principal, d)]

    def delete_document(self, principal: Principal, doc_id: str) -> bool:
        idx = self.store.get(principal.tenant_id)
        doc = idx.docs.get(doc_id)
        if not doc or not can_read(principal, doc):
            return False
        return idx.remove_document(doc_id)

    def stats(self, principal: Principal) -> dict:
        idx = self.store.get(principal.tenant_id)
        return {"documents": len(self.list_documents(principal)), "chunks": len(idx), "parents": len(idx.parents),
                "index_version": idx.version, "metrics": self.metrics.summary()}

    def get_facts(self, principal: Principal) -> list[str]:
        return [f["text"] for f in self.memory.facts(principal)]

    def clear_facts(self, principal: Principal) -> None:
        self.memory.clear(principal)

    def clear_memory(self, principal: Principal, session_id: str | None = None) -> None:
        self.memory.clear(principal)

    def clear_caches(self) -> None:
        if self.cache:
            self.cache.clear()
        if self.sem_cache:
            self.sem_cache.clear()

    #  retrieval only
    def retrieve_only(self, principal: Principal, question: str, history: list | None = None):
        """Plan + retrieve (no generation). Used by evaluation to score retrieval in isolation."""
        tr = Trace("retrieve", principal)
        set_trace(tr)
        try:
            idx = self.store.get(principal.tenant_id)
            plan = self.query.plan(question, history or [], idx.known_meta(principal))
            return plan, self.retriever.retrieve(principal, plan)
        finally:
            set_trace(None)

    #  ask / stream
    def ask(self, principal: Principal, question: str, session_id: str = "default", filters: dict | None = None) -> Answer:
        final = None
        for ev in self.ask_stream(principal, question, session_id, filters):
            if ev["type"] == "final":
                final = ev["answer"]
        return final

    def ask_stream(self, principal: Principal, question: str, session_id: str = "default",
                   filters: dict | None = None) -> Iterator[dict]:
        """Events: status | plan | token | replace | final. The last event is always `final` with a structured Answer."""
        cfg = self.cfg
        tr = Trace("ask", principal)
        set_trace(tr)
        try:
            ans = None
            try:
                ans = yield from self._ask(principal, question, session_id, filters, tr)
            except SecurityError as e:
                tr.fail("ask", e)
                ans = Answer(f"Request refused: {e}", abstained=True, abstain_reason=str(e))
            except Exception as e:  # never leak internals to the user; the trace has the detail
                tr.fail("ask", e)
                logging.getLogger("rag").exception("ask failed (trace %s)", tr.id)   # full traceback in the terminal
                detail = f"{type(e).__name__}: {e}"[:300] if os.getenv("RAG_DEBUG") else type(e).__name__
                ans = Answer("Something went wrong while answering. Please try again.", abstained=True,
                             abstain_reason="internal error", warnings=[detail])
            ans.trace_id = tr.id
            ans.timings_ms = {k: round(v, 1) for k, v in tr.spans.items()}
            yield {"type": "final", "answer": ans}
        finally:
            self.metrics.record(tr)
            set_trace(None)

    def _abstain(self, text: str, reason: str, plan=None) -> Answer:
        return Answer(text, abstained=True, abstain_reason=reason, plan=asdict(plan) if plan else {})

    def _ask(self, principal, question, session_id, filters, tr):
        cfg = self.cfg
        self.limiter.check(principal, "query")
        question = normalize_query(question, cfg.max_question_chars)
        if not question:
            return self._abstain("Please enter a question.", "empty question")

        yield {"type": "status", "stage": "understanding"}

        # --- Conversation routing gate ---
        # Small-talk is classified deterministically (no LLM, no retrieval).
        from .query import detect_small_talk  # local import to avoid circular at module level
        st = detect_small_talk(question)
        if st in _SMALL_TALK_REPLIES:
            reply = _SMALL_TALK_REPLIES[st]
            ans = Answer(reply, abstained=False)
            yield {"type": "final", "answer": ans}
            return ans

        # --- Normal RAG path ---
        index = self.store.get(principal.tenant_id)
        system_index = self.store.get(SYSTEM_TENANT)
        has_user_docs = bool(self.list_documents(principal))

        self.memory.maybe_store_facts(principal, question)           # long-term memory write
        history = self.memory.recent_turns(principal, session_id)    # short-term memory

        # known_meta merges user docs + system docs for filter extraction
        from .types import Principal as _P
        _sys = _P(SYSTEM_TENANT, "system", ("system",))
        merged_known = index.known_meta(principal)
        if system_index:
            sys_known = system_index.known_meta(_sys)
            merged_known["filenames"] = merged_known["filenames"] + sys_known["filenames"]
        plan = self.query.plan(question, history, merged_known)
        if filters:
            plan.filters = {**plan.filters, **filters}

        # Help queries get a source_type hint so system docs are prioritised
        if plan.intent == "help" and not plan.filters.get("source_type"):
            plan.filters["source_type"] = ["system", "user"]

        yield {"type": "plan", "plan": asdict(plan)}
        mem = self.memory.retrieve(principal, session_id, plan.rewritten)   # semantic + long-term + summary

        #  semantic cache (scoped by tenant+roles+filters; versioned)
        qvec = self.embedder.embed_query(plan.rewritten)
        scope = f"{principal.tenant_id}|{sha(*sorted(principal.roles), n=8)}|{sha(stable_json(plan.filters), n=8)}"
        sys_ver = system_index.version if system_index else ""
        versions = sha(index.version, sys_ver, self.embedder.name, PROMPT_VERSION, self.llm.name, mem.facts_hash, n=16)
        if self.sem_cache:
            hit = self.sem_cache.get(scope, qvec, versions)
            if hit is not None:
                ans = Answer(**hit)
                ans.cached = True
                self.memory.record_turn(principal, session_id, question, ans.text)
                return ans

        yield {"type": "status", "stage": "retrieving"}
        # Retrieve from user index; for help/factual also search system index and merge hits
        res = self.retriever.retrieve(principal, plan)
        # Merge system index hits only for help queries — never for factual document queries
        if plan.intent == "help" and len(system_index) > 0:
            sys_res = self.retriever.retrieve_system(plan)
            if sys_res and sys_res.hits:
                from .types import Hit  # noqa: F401
                merged_hits = {h.chunk_id: h for h in res.hits}
                for h in sys_res.hits:
                    if h.chunk_id not in merged_hits:
                        merged_hits[h.chunk_id] = h
                res.hits = sorted(merged_hits.values(), key=lambda h: -h.score)
                if not res.hits or res.strength == 0.0:
                    # If user retrieval was empty, adopt system result strength
                    res.strength = sys_res.strength
                    res.kind = sys_res.kind

        # Guard: require user docs for non-help queries; help can fall through to system docs
        if not has_user_docs and plan.intent not in ("help",):
            ans = self._abstain("No documents are indexed for you yet. Upload some first.", "no user documents", plan)
            self.memory.record_turn(principal, session_id, question, ans.text)
            return ans

        if not self.retriever.is_answerable(res):
            ans = self._abstain("I couldn't find anything in your documents that answers this.",
                                f"retrieval evidence too weak ({res.kind} score {res.strength:.3f})", plan)
            self.memory.record_turn(principal, session_id, question, ans.text)
            return ans

        mem_text = mem.render(cfg.memory_budget_tokens)
        overhead = Generator.overhead_tokens(self.gen.system_prompt, history, question, mem_text)
        budget = max(300, cfg.context_window - cfg.answer_reserve_tokens - overhead)
        ctx = self.ctx.build(principal, plan, res.hits, budget)
        if not ctx.items:
            return self._abstain("I couldn't assemble usable evidence for this question.", "empty context", plan)

        #  generation (streamed) 
        yield {"type": "status", "stage": "generating"}
        msgs = self.gen.build_messages(question, ctx.items, mem_text, history)
        gate, parts = StreamGate(), []
        with tr.span("generation"):
            for tok in self.llm.stream(msgs, max_tokens=cfg.max_new_tokens, temperature=cfg.temperature):
                parts.append(tok)
                out = gate.feed(tok)
                if out:
                    yield {"type": "token", "text": out}
            tail = gate.flush()
            if tail:
                yield {"type": "token", "text": tail}
        items = ctx.items
        ans = self.gen.parse("".join(parts), items)

        #  verification + self-correction 
        if not ans.abstained:
            yield {"type": "status", "stage": "verifying"}
            report = self.verifier.verify(ans.text, items)
            retries = 0
            while report.groundedness < cfg.min_groundedness and retries < cfg.max_retries:
                retries += 1
                yield {"type": "status", "stage": "self-correcting"}
                wide = self.ctx.build(principal, plan, res.hits, budget, wide=True)  # retrieve additional evidence
                items = wide.items or items
                feedback = ("Your previous answer contained statements the documents do not support:\n"
                            + "\n".join(f"- {c}" for c in report.unsupported[:6])
                            + "\nRewrite the answer using ONLY information stated in the documents, keep [S#] citations, "
                              f"or reply with {NO_ANSWER_HINT} if the documents cannot answer.")
                raw = self.llm.complete(self.gen.build_messages(question, items, mem_text, history, feedback),
                                        max_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
                ans = self.gen.parse(raw, items)
                if ans.abstained:
                    break
                report = self.verifier.verify(ans.text, items)
                yield {"type": "replace", "text": ans.text}
            if not ans.abstained:
                ans.claims = [asdict(c) for c in report.claims]
                ans.groundedness, ans.citation_accuracy = report.groundedness, report.citation_accuracy
                strength = min(1.0, max(0.0, res.strength))
                ans.confidence = round(0.6 * report.groundedness + 0.4 * strength, 3)
                if report.groundedness < cfg.abstain_groundedness:
                    # keep the retrieved-and-generated answer; just say plainly that it could not be verified
                    ans.text = ans.text.rstrip() + "\n\n> ⚠️ **Could not verify** this answer against your documents. Treat it with caution."
                    ans.warnings.append("Not verified against the documents: " + "; ".join(report.unsupported[:3]))
                elif report.groundedness < cfg.min_groundedness:
                    ans.text = ans.text.rstrip() + "\n\n> ⚠️ Only partially verified against your documents."
                    ans.warnings.append("Only partially verified against the documents: " + "; ".join(report.unsupported[:3]))
        ans.plan = asdict(plan)
        ans.context = [{"label": i.label, "chunk_ids": i.meta.get("chunk_ids", [i.chunk_id]), "filename": i.meta["filename"],
                        "page": i.meta.get("page"), "section": i.meta["section_path"], "text": i.text} for i in items]
        self.memory.record_turn(principal, session_id, question, ans.text)
        if self.sem_cache and not ans.abstained and (ans.groundedness or 0) >= cfg.min_groundedness:
            self.sem_cache.set(scope, qvec, versions, {**ans.to_dict(), "cached": False})
        return ans


NO_ANSWER_HINT = "NO_ANSWER: <reason>"
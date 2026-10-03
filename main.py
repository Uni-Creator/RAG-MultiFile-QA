"""Streamlit UI for the RAG pipeline.   streamlit run main.py

Env: HUGGINGFACE_HUB_TOKEN (or st.secrets), RAG_* overrides (see rag/config.py),
     RAG_OFFLINE=1 to run with the stub embedder/LLM (no models, no key),
     RAG_AUTH_TOKENS='{"token": {"tenant":"acme","user":"alice","roles":["member"]}}' to require login.
"""
from __future__ import annotations

import html
import json
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag import Principal, RAGConfig, RAGPipeline

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
load_dotenv()

try:
    with open(os.path.join(os.path.dirname(__file__), "ui", "style.css"), encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except OSError:
    pass


st.markdown("<style>.source-chip{display:inline-block;padding:2px 10px;margin:0 4px 4px 0;border-radius:12px;"
            "background:rgba(120,120,160,.18);font-size:.8rem}</style>", unsafe_allow_html=True)


#  setup
def _secret(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


OFFLINE = os.getenv("RAG_OFFLINE") == "1"
token = _secret("HUGGINGFACE_HUB_TOKEN")
if not token and not OFFLINE:
    st.error("HUGGINGFACE_HUB_TOKEN not found (or set RAG_OFFLINE=1 for a model-free demo).")
    st.stop()
if token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token


@st.cache_resource(show_spinner="Loading models…")
def get_pipeline() -> RAGPipeline:
    cfg = RAGConfig.from_env()
    if OFFLINE:
        from rag.embeddings import HashEmbedder
        from rag.testing import FakeLLM
        cfg.use_reranker = cfg.use_nli = False
        cfg.min_dense_sim = -1.0
        return RAGPipeline(cfg, llm=FakeLLM(), embedder=HashEmbedder())
    return RAGPipeline(cfg)


pipe = get_pipeline()


#  authentication
def current_principal() -> Principal | None:
    """AuthN: who are you? (AuthZ - what may you read - is enforced in the store by tenant + role ACL.)"""
    raw = _secret("RAG_AUTH_TOKENS")
    if not raw:
        return Principal("default", "local", ("member",))
    try:
        table = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        st.error("RAG_AUTH_TOKENS is not valid JSON.")
        st.stop()
    if "principal" in st.session_state:
        return st.session_state.principal
    tok = st.text_input("Access token", type="password")
    if tok:
        import hmac
        for known, info in table.items():
            if hmac.compare_digest(tok.encode(), known.encode()):
                st.session_state.principal = Principal(info["tenant"], info["user"], tuple(info.get("roles", ["member"])))
                st.rerun()
        st.error("Invalid token.")
    return None


who = current_principal()
if who is None:
    st.stop()

if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:12]
    st.session_state.messages = []


#  helpers
def render_meta(ans: dict):
    """Source chips, groundedness / confidence, warnings."""
    if ans.get("abstained"):
        st.caption(f"Abstained · {ans.get('abstain_reason') or 'no supporting evidence'}")
        return
    chips = "".join(
        f"<span class='source-chip'>[{c['label']}] "
        f"{html.escape(c['filename'])}{' · p.' + str(c['page']) if c.get('page') else ''}</span> "
        for c in ans.get("citations", []))
    if chips:
        st.markdown(chips, unsafe_allow_html=True)
    bits = []
    if ans.get("groundedness") is not None:
        bits.append(f"groundedness {ans['groundedness']:.0%}")
    bits.append(f"confidence {ans.get('confidence', 0):.0%}")
    if ans.get("cached"):
        bits.append("cached")
    st.caption(" · ".join(bits))
    for w in ans.get("warnings", []):
        st.warning(w, icon="⚠️")
    if ans.get("citations"):
        with st.expander("Sources"):
            for c in ans["citations"]:
                st.markdown(f"**[{c['label']}] {c['filename']}**"
                            f"{' · p.' + str(c['page']) if c.get('page') else ''} — _{c.get('section', '')}_")
                st.text((c.get("snippet", "") or "").strip()[:240])
    if ans.get("timings_ms"):
        with st.expander("Trace"):
            st.json({"trace_id": ans.get("trace_id"), "timings_ms": ans["timings_ms"], "plan": ans.get("plan")})


#  sidebar
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True)
    if files and st.button("Ingest", use_container_width=True, type="primary"):
        with st.spinner("Parsing, chunking, embedding…"):
            reports = pipe.ingest(who, [(f.name, f.getvalue()) for f in files])
        for r in reports:
            icon = {"indexed": "✅", "skipped": "⏭️", "rejected": "⛔", "failed": "❌"}.get(r["status"], "•")
            st.write(f"{icon} **{r.get('filename', '')}** — {r['status']}"
                     + (f" ({r.get('chunks', 0)} chunks)" if r["status"] == "indexed" else f": {r.get('detail', '')}"))
            if r.get("injection_blocks"):
                st.warning(f"{r['injection_blocks']} instruction-like block(s) neutralized")

    docs = pipe.list_documents(who)
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{d.get('filename')} · v{d.get('version', 1)}")
        if c2.button("✕", key=f"del-{d['doc_id']}"):
            pipe.delete_document(who, d["doc_id"])
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        pipe.memory.clear_session(who, st.session_state.sid)
        st.session_state.messages = []
        st.session_state.sid = uuid.uuid4().hex[:12]
        st.rerun()

    with st.expander("Memory"):
        facts = pipe.memory.facts(who)
        if not facts:
            st.caption("Nothing remembered. Try “Remember that I prefer short answers.”")
        for f in facts:
            c1, c2 = st.columns([5, 1])
            c1.caption(f["text"])
            if c2.button("✕", key=f"fact-{f['id']}"):
                pipe.memory.forget_fact(who, f["id"])
                st.rerun()
        if facts and st.button("Forget everything"):
            pipe.memory.forget_all(who)
            st.rerun()

    with st.expander("Stats"):
        st.json(pipe.stats(who))


#  chat
st.title("📄 Document Q&A")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("meta"):
            render_meta(m["meta"])

if not pipe.list_documents(who):
    st.info("Upload documents in the sidebar to get started.")
    st.chat_input("Upload documents to start…", disabled=True)
    st.stop()

if q := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        box, status, buf, final = st.empty(), st.empty(), "", None
        for ev in pipe.ask_stream(who, q, session_id=st.session_state.sid):
            t = ev["type"]
            if t == "status":
                status.caption(f"{ev['stage']}…")
            elif t == "token":
                buf += ev["text"]
                box.markdown(buf + "▌")
            elif t == "replace":
                buf = ev["text"]
                box.markdown(buf)
            elif t == "final":
                final = ev["answer"]
        status.empty()
        box.markdown(final.text)
        meta = final.to_dict()
        render_meta(meta)
    st.session_state.messages.append({"role": "assistant", "content": final.text, "meta": meta})
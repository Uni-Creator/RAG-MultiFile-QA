"""Streamlit UI for the RAG pipeline.

Run via:
    streamlit run frontend/app.py
or
    RAG_OFFLINE=1 streamlit run frontend/app.py

Env: HUGGINGFACE_HUB_TOKEN (or st.secrets), RAG_* overrides (see backend/rag/config.py),
     RAG_OFFLINE=1 to run with the stub embedder/LLM (no models, no key),
     RAG_AUTH_TOKENS='{"token": {"tenant":"acme","user":"alice","roles":["member"]}}' to require login.
"""
from __future__ import annotations

import html
import json
import os
import sys
import uuid
from pathlib import Path

# Ensure backend directory is in sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import streamlit as st
from dotenv import load_dotenv

from rag import Principal, RAGConfig, RAGPipeline

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
load_dotenv()

# Load custom CSS
css_file = Path(__file__).resolve().parent / "ui" / "style.css"
if not css_file.exists():
    css_file = Path(__file__).resolve().parent / "style.css"
if css_file.exists():
    try:
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except OSError:
        pass

st.markdown(
    "<style>.source-chip{display:inline-block;padding:2px 10px;margin:0 4px 4px 0;border-radius:12px;"
    "background:rgba(120,120,160,.18);font-size:.8rem}</style>",
    unsafe_allow_html=True,
)


# Setup secrets and pipeline
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
    print(">>> INITIALIZING RAG PIPELINE <<<")
    cfg = RAGConfig.from_env()
    if OFFLINE:
        from rag.embeddings import HashEmbedder
        from rag.testing import FakeLLM
        cfg.use_reranker = cfg.use_nli = False
        cfg.min_dense_sim = -1.0
        return RAGPipeline(cfg, llm=FakeLLM(), embedder=HashEmbedder())
    return RAGPipeline(cfg)


pipe = get_pipeline()


# Authentication
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
                st.session_state.principal = Principal(
                    info["tenant"], info["user"], tuple(info.get("roles", ["member"]))
                )
                st.rerun()
        st.error("Invalid token.")
    return None


who = current_principal()
if who is None:
    st.stop()

if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:12]
    st.session_state.messages = []


# Helpers
def render_meta(ans: dict):
    """Source chips, groundedness / confidence, warnings."""
    if ans.get("abstained"):
        st.caption(f"Abstained · {ans.get('abstain_reason') or 'no supporting evidence'}")
        return
    chips = "".join(
        f"<span class='source-chip'>[{c['label']}] "
        f"{html.escape(c['filename'])}{' · p.' + str(c['page']) if c.get('page') else ''}</span> "
        for c in ans.get("citations", [])
    )
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
                st.markdown(
                    f"**[{c['label']}] {c['filename']}**"
                    f"{' · p.' + str(c['page']) if c.get('page') else ''} — _{c.get('section', '')}_"
                )
                st.text((c.get("snippet", "") or "").strip()[:240])
    if ans.get("timings_ms"):
        with st.expander("Trace"):
            st.json(
                {
                    "trace_id": ans.get("trace_id"),
                    "timings_ms": ans["timings_ms"],
                    "plan": ans.get("plan"),
                }
            )


# Sidebar
with st.sidebar:
    st.header("Documents")

    files = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )

    if files and st.button("Ingest", use_container_width=True, type="primary"):
        with st.spinner("Parsing, chunking, embedding…"):
            reports = pipe.ingest(
                who,
                [(f.name, f.getvalue()) for f in files],
            )

        for r in reports:
            status = r.get("status") if isinstance(r, dict) else getattr(r, "status", None)
            fname = r.get("filename", "file") if isinstance(r, dict) else getattr(r, "filename", "file")
            if status in ("ok", "indexed"):
                parents = r.get("parents", 0) if isinstance(r, dict) else getattr(r, "parents", 0)
                chunks = r.get("chunks", 0) if isinstance(r, dict) else getattr(r, "chunks", 0)
                chars = r.get("chars", 0) if isinstance(r, dict) else getattr(r, "chars", 0)
                st.success(f"**{fname}**: {parents} sections, {chunks} chunks, {chars:,} chars")
            elif status in ("unchanged", "skipped"):
                st.info(f"**{fname}**: unchanged (hash matches)")
            else:
                err = r.get("detail") or r.get("error") if isinstance(r, dict) else getattr(r, "error", status)
                st.error(f"**{fname}**: {err}")

    docs = pipe.list_documents(who)

    st.caption(f"Indexed documents ({len(docs)}) · Tenant: `{who.tenant_id}`")

    for d in docs:
        col1, col2 = st.columns([4, 1])
        chunks = d.get("n_chunks") or d.get("chunks", 0)
        chars = d.get("chars") or d.get("size_bytes", 0)
        version = d.get("version", 1)
        created_time = str(d.get("ingested_at") or d.get("time") or "")[:10]
        filename = d.get("filename", "Untitled")
        doc_id = d.get("doc_id", filename)

        with col1:
            st.markdown(
                f"**{filename}** · {chunks} chunks  \n"
                f"<span style='font-size:0.75rem;color:#8896aa'>"
                f"{chars:,} bytes · v{version}{' · ' + created_time if created_time else ''}</span>",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("🗑️", key=f"del_{doc_id}", help=f"Delete {filename}"):
                pipe.delete_document(who, doc_id)
                st.rerun()

    st.divider()

    st.subheader("Session")
    st.caption(f"User: `{who.user}` | Roles: `{','.join(who.roles)}` | Session: `{st.session_state.sid}`")

    if st.button("Clear conversation", use_container_width=True):
        pipe.clear_memory(who, st.session_state.sid)
        st.session_state.messages = []
        st.rerun()

    facts = pipe.get_facts(who)
    if facts:
        with st.expander(f"Long-term memory ({len(facts)} facts)"):
            for f in facts:
                st.caption(f"• {f}")

    if st.button("Reset facts", use_container_width=True):
        pipe.clear_facts(who)
        st.rerun()

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear caches", use_container_width=True):
            pipe.clear_caches()
            st.success("Caches cleared")
    with col_b:
        if st.button("Compact index", use_container_width=True):
            pipe.store.get(who.tenant).compact()
            st.success("Compacted")


# Main chat view
st.title("Document Q&A")
st.caption("Answers grounded in your documents with verified citations and abstention when uncertain.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            render_meta(msg["meta"])

query = st.chat_input("Ask a question about your documents…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        container = st.empty()
        full_text = ""
        final_answer = None

        with st.spinner("Searching and reasoning…"):
            for event in pipe.ask_stream(who, query, session_id=st.session_state.sid):
                etype = event.get("type")
                if etype == "token":
                    full_text += event["text"]
                    container.markdown(full_text + "▌")
                elif etype == "final":
                    final_answer = event["answer"]

        if final_answer:
            container.markdown(final_answer.text)
            render_meta(final_answer.to_dict())
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer.text,
                "meta": final_answer.to_dict(),
            })
        else:
            container.markdown(full_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text,
                "meta": {},
            })

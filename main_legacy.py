from httpx import HTTPStatusError
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
import tempfile
import os
import numpy as np
from dotenv import load_dotenv

# Page config
st.set_page_config(
    page_title="Ask RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

:root {
  --bg:       #0d0f12;
  --surface:  #13161b;
  --border:   #1f2530;
  --accent:   #c8f060;
  --accent2:  #60c8f0;
  --muted:    #4a5568;
  --text:     #e8edf4;
  --text-dim: #8896aa;
  --radius:   12px;
  --user-bg:  #1a2235;
  --bot-bg:   #111418;
}

html, body, [class*="css"] {
  font-family: 'DM Mono', monospace;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

#MainMenu, footer { visibility: hidden; }

/* Make header transparent so the sidebar toggle button stays clickable */
header[data-testid="stHeader"] {
  background: transparent !important;
  box-shadow: none !important;
}
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] svg { color: var(--text-dim) !important; fill: var(--text-dim) !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}

.sidebar-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: var(--accent);
  margin-bottom: 0.2rem;
}
.sidebar-sub {
  font-size: 0.7rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 1.5rem;
}

[data-testid="stFileUploader"] {
  border: 1px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  background: rgba(255,255,255,0.02) !important;
  padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.file-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(200,240,96,0.08);
  border: 1px solid rgba(200,240,96,0.2);
  border-radius: 999px;
  padding: 3px 10px 3px 8px;
  font-size: 0.68rem;
  color: var(--accent);
  margin: 3px 2px;
}

.stat-row { display: flex; gap: 8px; margin: 1rem 0; }
.stat-card {
  flex: 1;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  text-align: center;
}
.stat-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent);
}
.stat-lbl { font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; }

.mem-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(96,200,240,0.08);
  border: 1px solid rgba(96,200,240,0.2);
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 0.65rem;
  color: var(--accent2);
  margin-top: 0.5rem;
}

.chat-header {
  padding: 1.5rem 2rem 1rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.chat-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
}
.chat-status { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em; }
.chat-status.online { color: var(--accent); }

.msg-wrap { 
    padding: 0.8rem 1.5rem; /* Increased vertical, adjusted horizontal */
    display: flex; 
    flex-direction: column; 
    width: 100%; /* Ensure it spans the full width */
}
.msg-wrap.user { align-items: flex-end; }
.msg-wrap.bot  { align-items: flex-start; }

.msg-bubble {
  max-width: 72%;
  padding: 0.75rem 1rem;
  border-radius: var(--radius);
  font-size: 0.85rem;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg-bubble.user {
  background: var(--user-bg);
  border: 1px solid rgba(200,240,96,0.15);
  border-bottom-right-radius: 3px;
  color: var(--text);
}
.msg-bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 3px;
    color: var(--text);
    /* Prevent text from touching the absolute edges if container shrinks */
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.msg-meta { 
    font-size: 0.65rem; 
    color: var(--text-dim); 
    margin-top: 6px; /* Space between bubble and label */
    margin-left: 4px; /* Slight offset so it's not flush against the screen edge */
    letter-spacing: 0.05em; 
    text-transform: uppercase;
}
.msg-wrap.user .msg-meta {
    margin-right: 4px;
    margin-left: 0;
}
.msg-meta.mem-used { color: var(--accent2); }

.src-row { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.src-chip {
  background: rgba(96,200,240,0.07);
  border: 1px solid rgba(96,200,240,0.2);
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 0.62rem;
  color: var(--accent2);
}
.src-chip-label { font-size: 0.55rem; color: var(--text-dim); margin-right: 3px; }

.empty-state {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 12px; padding: 4rem 2rem;
  text-align: center; color: var(--text-dim);
}
.empty-icon { font-size: 2.5rem; opacity: 0.3; }
.empty-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: var(--text-dim); }
.empty-body { font-size: 0.75rem; max-width: 320px; line-height: 1.6; }

/* Enhanced Button Styling */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    width: 100%;
    height: auto;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Hover State */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: rgba(200, 240, 96, 0.05) !important;
}

/* Active/Click State */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:active {
    transform: scale(0.98);
    background-color: rgba(200, 240, 96, 0.1) !important;
}

/* Remove default Streamlit focus rings */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:focus:not(:active) {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
    color: var(--accent) !important;
}

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# Load API key
load_dotenv()
api_key = None
try:
    if "HUGGINGFACE_HUB_TOKEN" in st.secrets:
        api_key = st.secrets["HUGGINGFACE_HUB_TOKEN"]
    else:
        api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")
except (HTTPStatusError, KeyError):
    api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")
except Exception:
    api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not api_key:
    st.error("HUGGINGFACE_HUB_TOKEN not found.")
    st.stop()

# LLM and embeddings
@st.cache_resource
def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        huggingfacehub_api_token=api_key,
        task="conversational",
        max_new_tokens=512,
        temperature=0.5,
    )
    return ChatHuggingFace(llm=endpoint)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# File ingestion
@st.cache_resource(show_spinner=False)
def load_files(file_key: str, files):
    """
    Loads files and patches metadata['source'] with the real filename
    so source chips never show temp paths.
    Returns (vectorstore, temp_paths, chunk_count, file_names).
    """
    documents = []
    temp_paths = []
    file_names = []

    for f in files:
        suffix = os.path.splitext(f.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            path = tmp.name
        temp_paths.append(path)
        file_names.append(f.name)
        try:
            if f.name.endswith(".pdf"):
                docs = PyPDFLoader(path).load()
            elif f.name.endswith(".txt"):
                docs = TextLoader(path).load()
            elif f.name.endswith(".docx"):
                docs = Docx2txtLoader(path).load()
            elif f.name.endswith(".csv"):
                docs = CSVLoader(path).load()
            else:
                docs = []
            for doc in docs:
                doc.metadata["source"] = f.name
            documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not load {f.name}: {e}")

    if not documents:
        return None, temp_paths, 0, file_names

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vs = FAISS.from_documents(chunks, get_embeddings())
    return vs, temp_paths, len(chunks), file_names

# Semantic memory helpers
def embed_text(text: str) -> np.ndarray:
    return np.array(get_embeddings().embed_query(text), dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def retrieve_relevant_history(query: str, messages: list, top_k: int = 3, threshold: float = 0.35) -> str:
    if not messages:
        return ""
    q_vec = embed_text(query)
    pairs, i = [], 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    scored = []
    for u, a in pairs:
        turn = f"Q: {u}\nA: {a}"
        scored.append((cosine_sim(q_vec, embed_text(turn)), turn))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join(t for s, t in scored[:top_k] if s >= threshold)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []

# Vectorstore lives in session state so it persists when sidebar collapses.
# We only rebuild it when the file set changes (keyed by file_key).
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "last_file_key" not in st.session_state:
    st.session_state.last_file_key = ""

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">◈ Ask RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Document Intelligence · v2</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        file_key = "_".join(sorted(f.name + str(f.size) for f in uploaded_files))

        # Only re-index if the file set actually changed
        if file_key != st.session_state.last_file_key:
            with st.spinner("Indexing documents…"):
                result = load_files(file_key, uploaded_files)
            if result[0] is not None:
                vs, temp_paths, chunk_count, file_names = result
                st.session_state.vectorstore   = vs
                st.session_state.chunk_count   = chunk_count
                st.session_state.file_names    = file_names
                st.session_state.temp_files    = temp_paths
                st.session_state.last_file_key = file_key
            else:
                st.warning("No content could be extracted from the uploaded files.")

    # Show stats whenever a vectorstore is loaded, even if sidebar was just reopened
    if st.session_state.vectorstore is not None:
        st.markdown("**Loaded files**")
        for fn in st.session_state.file_names:
            ext = fn.split(".")[-1].upper()
            icon = {"PDF": "📄", "DOCX": "📝", "TXT": "📃", "CSV": "📊"}.get(ext, "📁")
            st.markdown(f'<span class="file-pill">{icon} {fn}</span>', unsafe_allow_html=True)

        n_turns = len(st.session_state.messages) // 2
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-card"><div class="stat-val">{len(st.session_state.file_names)}</div><div class="stat-lbl">files</div></div>'
            f'<div class="stat-card"><div class="stat-val">{st.session_state.chunk_count}</div><div class="stat-lbl">chunks</div></div>'
            f'<div class="stat-card"><div class="stat-val">{n_turns}</div><div class="stat-lbl">turns</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if n_turns > 0:
            label = f'{n_turns} turn{"s" if n_turns != 1 else ""} indexed'
            st.markdown(
                f'<div class="mem-badge">⬡ Semantic memory active · {label}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset all"):
            st.session_state.messages      = []
            st.session_state.vectorstore   = None
            st.session_state.chunk_count   = 0
            st.session_state.file_names    = []
            st.session_state.last_file_key = ""
            for p in st.session_state.temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass
            st.session_state.temp_files = []
            st.cache_resource.clear()
            st.rerun()

    st.markdown(
        '<div style="font-size:0.6rem;color:var(--text-dim);margin-top:1.5rem;line-height:1.8">'
        'Llama-3-8B · MiniLM-L6 · FAISS<br>'
        'Semantic memory via cosine retrieval'
        '</div>',
        unsafe_allow_html=True,
    )

# Read vectorstore from session state (works regardless of sidebar state)
vectorstore = st.session_state.vectorstore

# Chat header
status_cls = "online" if vectorstore else ""
status_txt = "● ready" if vectorstore else "○ waiting for documents"
st.markdown(
    f'<div class="chat-header">'
    f'<span class="chat-title">Chat</span>'
    f'<span class="chat-status {status_cls}">{status_txt}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Render helpers
def render_user(content: str):
    st.markdown(
        f'<div class="msg-wrap user">'
        f'<div class="msg-bubble user">{content}</div>'
        f'<div class="msg-meta">you</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_bot(content: str, sources: list, mem_used: bool):
    src_html = ""
    if sources:
        chips = "".join(
            f'<span class="src-chip"><span class="src-chip-label">src</span>{s}</span>'
            for s in sources[:4]
        )
        src_html = f'<div class="src-row">{chips}</div>'
    footer_html = (
        '<div class="msg-meta mem-used">⬡ memory retrieved</div>'
        if mem_used
        else '<div class="msg-meta">assistant</div>'
    )
    st.markdown(
        f'<div class="msg-wrap bot">'
        f'<div class="msg-bubble bot">{content}{src_html}</div>'
        f'{footer_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

# Message history
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">◈</div>'
        '<div class="empty-title">No messages yet</div>'
        '<div class="empty-body">Upload one or more documents in the sidebar, then ask anything about them.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user(msg["content"])
        else:
            render_bot(msg["content"], msg.get("sources", []), msg.get("mem_used", False))

# Chat input and RAG chain
if vectorstore:
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    user_input = st.chat_input("Ask about your documents…")

    if user_input:
        # Store and immediately render the user message before the spinner
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_user(user_input)

        # Retrieve relevant past turns via cosine similarity
        history_context = retrieve_relevant_history(
            user_input, st.session_state.messages[:-1]
        )
        mem_used = bool(history_context)

        # Build prompt based on whether memory was retrieved
        if history_context:
            prompt_template = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Use the document context and relevant "
                "conversation history below to answer the question. Be concise.\n\n"
                "Relevant past conversation:\n{history}\n\n"
                "Document context:\n{context}\n\n"
                "Question: {input}\n\nAnswer:"
            )
        else:
            prompt_template = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer using only the document context "
                "below. Be concise.\n\n"
                "Document context:\n{context}\n\n"
                "Question: {input}\n\nAnswer:"
            )

        retrieved_docs = retriever.invoke(user_input)
        doc_context = "\n\n".join(d.page_content for d in retrieved_docs)
        sources = list({d.metadata.get("source", "unknown") for d in retrieved_docs})

        chain_input = {"context": doc_context, "input": user_input}
        if history_context:
            chain_input["history"] = history_context

        with st.spinner("Thinking…"):
            response = (prompt_template | llm).invoke(chain_input)

        output = response.content if hasattr(response, "content") else str(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": output,
            "sources": sources,
            "mem_used": mem_used,
        })
        render_bot(output, sources, mem_used)

else:
    st.chat_input("Upload documents to start…", disabled=True)
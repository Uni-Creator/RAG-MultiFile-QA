"""Main entry point for the RAG MultiFile QA Application.

Usage:
    - Run UI via Streamlit:
        streamlit run frontend/app.py
        or
        streamlit run main.py
    - Run directly with Python:
        python main.py
    - Run backend CLI:
        python -m backend.rag.cli --help
        or
        PYTHONPATH=backend python -m rag.cli --help
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add backend and root paths to sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

for p in (BACKEND_DIR, ROOT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def run_app():
    app_file = FRONTEND_DIR / "app.py"
    # If run under streamlit process (e.g. `streamlit run main.py`)
    try:
        import streamlit.runtime.scriptrunner as sr
        is_streamlit = sr.get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False

    if is_streamlit:
        with open(app_file, encoding="utf-8") as f:
            code = compile(f.read(), str(app_file), "exec")
            exec(code, globals())
    else:
        import subprocess
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    run_app()
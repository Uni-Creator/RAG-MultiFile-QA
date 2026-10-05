import sys
from pathlib import Path

# Add backend and root paths to sys.path so 'rag' and 'backend' are directly importable
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

for path in (BACKEND_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

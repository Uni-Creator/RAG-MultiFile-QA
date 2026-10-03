from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import numpy as np

_ZW = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")
_TOKEN = re.compile(r"\w+(?:[.\-]\w+)*")


def sha(*parts, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _ZW.sub("", s)
    return _CTRL.sub("", s)


def clean_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def count_tokens(text: str) -> int:
    """Cheap, conservative estimate (no tokenizer download needed)."""
    return max(1, int(len(text) / 3.6)) if text else 0


def truncate_to_tokens(text: str, n: int) -> str:
    chars = int(n * 3.6)
    if len(text) <= chars:
        return text
    cut = text[:chars]
    idx = max(cut.rfind(". "), cut.rfind("\n"))
    if idx > chars * 0.5:
        cut = cut[: idx + 1]
    return cut.rstrip() + " …"


def l2norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))

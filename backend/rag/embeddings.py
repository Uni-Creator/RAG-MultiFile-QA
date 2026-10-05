"""Layer 2: embeddings with batching + cache. HashEmbedder gives a model-free fallback for tests/offline."""
from __future__ import annotations

import hashlib

import numpy as np

from .cache import KVCache, make_key
from .observability import current
from .utils import l2norm, tokenize


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.name = model_name
        self.batch_size = batch_size
        self._model = None
        self._dim = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
            getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self._dim = getter()
        return self._model

    @property
    def dim(self) -> int:
        self._load()
        return self._dim

    def embed_documents(self, texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._load().encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (uni+bigrams). Not semantic - for tests and offline demos."""
    name = "hash-384"
    dim = 384

    def embed_documents(self, texts: list) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for tok in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return l2norm(out)


class CachedEmbedder:
    """Batches texts, serves hits from the embedding cache, only sends misses to the model."""

    def __init__(self, base, cache: KVCache | None, batch_size: int = 64):
        self.base, self.cache, self.batch_size = base, cache, batch_size

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def dim(self) -> int:
        return self.base.dim

    def embed_documents(self, texts: list) -> np.ndarray:
        tr = current()
        tr.count("embed.texts", len(texts))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list = [None] * len(texts)
        miss_idx = []
        keys = [make_key("emb", self.name, t) for t in texts]
        for i, k in enumerate(keys):
            v = self.cache.get_vec("embedding", k) if self.cache else None
            if v is None:
                miss_idx.append(i)
            else:
                out[i] = v
        for s in range(0, len(miss_idx), self.batch_size):
            idx = miss_idx[s:s + self.batch_size]
            vecs = self.base.embed_documents([texts[i] for i in idx])
            tr.count("embed.model_calls")
            for i, v in zip(idx, vecs):
                out[i] = v
                if self.cache:
                    self.cache.set_vec("embedding", keys[i], v)
        return np.vstack(out).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]
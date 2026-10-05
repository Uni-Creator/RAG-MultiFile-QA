"""LLM adapters. Anything with name / complete() / stream() works (see rag/testing.py for an offline fake)."""
from __future__ import annotations

import os
from typing import Iterator

from .cache import KVCache, make_key
from .observability import current
from .utils import count_tokens


class HFChatLLM:
    """Hugging Face Inference chat-completion client (same model as the original app)."""

    def __init__(self, model: str, token: str | None = None, timeout: int = 120):
        self.name = model
        self.token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        self.timeout = timeout
        self._client = None

    def _c(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(model=self.name, token=self.token, timeout=self.timeout)
        return self._client

    def complete(self, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> str:
        r = self._c().chat_completion(messages=messages, max_tokens=max_tokens, temperature=max(temperature, 0.01))
        return r.choices[0].message.content or ""

    def stream(self, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> Iterator[str]:
        for chunk in self._c().chat_completion(messages=messages, max_tokens=max_tokens,
                                               temperature=max(temperature, 0.01), stream=True):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class CachedLLM:
    """LLM response cache (key = messages + model + sampling config) + token accounting for cost metrics."""

    def __init__(self, llm, cache: KVCache | None):
        self.llm, self.cache = llm, cache

    @property
    def name(self) -> str:
        return self.llm.name

    def _key(self, messages, max_tokens, temperature) -> str:
        return make_key("llm", self.llm.name, messages, max_tokens, temperature)

    def complete(self, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> str:
        tr = current()
        t_in = sum(count_tokens(m["content"]) for m in messages)
        key = self._key(messages, max_tokens, temperature)
        if self.cache:
            hit = self.cache.get_json("llm", key)
            if hit is not None:
                tr.count("llm.tokens_saved", t_in + count_tokens(hit["text"]))
                return hit["text"]
        text = self.llm.complete(messages, max_tokens=max_tokens, temperature=temperature)
        tr.count("llm.calls")
        tr.count("llm.tokens_in", t_in)
        tr.count("llm.tokens_out", count_tokens(text))
        if self.cache:
            self.cache.set_json("llm", key, {"text": text})
        return text

    def stream(self, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> Iterator[str]:
        tr = current()
        t_in = sum(count_tokens(m["content"]) for m in messages)
        key = self._key(messages, max_tokens, temperature)
        if self.cache:
            hit = self.cache.get_json("llm", key)
            if hit is not None:
                tr.count("llm.tokens_saved", t_in + count_tokens(hit["text"]))
                text = hit["text"]
                for i in range(0, len(text), 40):  # replay as a stream so the UI behaves the same
                    yield text[i:i + 40]
                return
        parts: list = []
        for tok in self.llm.stream(messages, max_tokens=max_tokens, temperature=temperature):
            parts.append(tok)
            yield tok
        text = "".join(parts)
        tr.count("llm.calls")
        tr.count("llm.tokens_in", t_in)
        tr.count("llm.tokens_out", count_tokens(text))
        if self.cache and text:
            self.cache.set_json("llm", key, {"text": text})

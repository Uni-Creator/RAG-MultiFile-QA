"""Offline stand-ins so the whole pipeline runs (tests, CI, demos) without any model download or API key."""
from __future__ import annotations

import json
import re

from .utils import split_sentences, tokenize


class FakeLLM:
    """Extractive 'LLM': answers with the best-overlapping document sentences and cites them.
    Understands the pipeline's own helper prompts (rewrite / decompose / claims / summary)."""
    name = "fake-extractive-llm"

    def _respond(self, messages: list) -> str:
        sys_, user = messages[0]["content"], messages[-1]["content"]
        if sys_.startswith("You rewrite a follow-up"):
            m = re.search(r"Follow-up: (.*?)\n", user)
            return m.group(1) if m else user
        if sys_.startswith("Split the user's question"):
            return "[]"
        if sys_.startswith("Split the answer into atomic"):
            return json.dumps(split_sentences(user))
        if sys_.startswith("Summarize the conversation"):
            return "Summary: " + re.sub(r"\s+", " ", user)[:200]
        q = re.search(r"<question>\n(.*?)\n</question>", user, re.S)
        qtok = {t for t in tokenize(q.group(1)) if len(t) > 3} if q else set()
        scored = []
        for lab, text in re.findall(r'<document id="(S\d+)"[^\n]*>\n(.*?)\n</document>', user, re.S):
            for s in split_sentences(text):
                ov = len(qtok & set(tokenize(s)))
                if ov:
                    scored.append((ov, lab, s))
        if not scored:
            return "NO_ANSWER: the documents do not mention this."
        scored.sort(key=lambda x: -x[0])
        return " ".join(f"{s} [{lab}]" for _, lab, s in scored[:2])

    def complete(self, messages: list, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return self._respond(messages)

    def stream(self, messages: list, max_tokens: int = 512, temperature: float = 0.0):
        text = self._respond(messages)
        for i in range(0, len(text), 6):
            yield text[i:i + 6]

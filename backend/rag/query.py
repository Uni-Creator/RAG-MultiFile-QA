"""Layer 3 (part 1): query understanding -> normalization, rewriting, decomposition, metadata filters."""
from __future__ import annotations

import json
import re
import unicodedata

from .observability import current
from .security import neutralize
from .types import QueryPlan
from .utils import normalize_text

_PRONOUN = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|the above|the former|the latter)\b", re.I)
_FOLLOWUP = re.compile(r"^(and|but|also|what about|how about|why|then|so)\b", re.I)
_SUMMARY = re.compile(r"\b(summari[sz]e|summary|overview|tl;?dr|main points|key points)\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference(?:s)? between|better than|pros and cons)\b", re.I)
_MULTI = re.compile(r"\b(and then|first.+then|as well as)\b|\?.+\?", re.I | re.S)
_HELP = re.compile(
    r"\b(how (do|can) i|how does|what file types|how to use|how do documents|what does groundedness|what does confidence|how (to|do i|can i) (upload|delete|ingest|use))\b",
    re.I,
)
_WH = r"(?:what|how|why|which|who|when|where)"

GREETING_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
)

CASUAL_PATTERNS = (
    "how are you",
    "how's it going",
    "what's up",
)

GRATITUDE_PATTERNS = (
    "thanks",
    "thank you",
    "thx",
)

FAREWELL_PATTERNS = (
    "bye",
    "goodbye",
    "see you",
)


def detect_small_talk(text: str) -> str | None:
    q = " ".join(text.lower().strip().split())

    if q in GREETING_PATTERNS:
        return "greeting"

    if any(q.startswith(x) for x in CASUAL_PATTERNS):
        return "casual"

    if q in GRATITUDE_PATTERNS:
        return "gratitude"

    if q in FAREWELL_PATTERNS:
        return "farewell"

    return None

REWRITE_SYS = ("You rewrite a follow-up question into ONE standalone search query. Use the conversation only to resolve "
               "references such as 'it', 'they', 'that'. Do not answer the question. Output only the query.")
DECOMP_SYS = ("Split the user's question into at most {n} self-contained sub-questions, each answerable by looking up one "
              "thing in a document collection. Output ONLY a JSON array of strings, nothing else.")


def normalize_query(q: str, max_chars: int = 2000) -> str:
    q = normalize_text(unicodedata.normalize("NFKC", q))
    q = re.sub(r"\s+", " ", q).strip()
    return q[:max_chars]


def classify_intent(q: str, has_history: bool) -> str:
    st = detect_small_talk(q)
    if st:
        return st
    if _HELP.search(q):
        return "help"
    if _SUMMARY.search(q):
        return "summarization"
    if _COMPARE.search(q):
        return "comparative"
    if _MULTI.search(q) or len(re.findall(rf"\b{_WH}\b", q, re.I)) >= 2:
        return "multi_step"
    if has_history and (_FOLLOWUP.match(q) or (_PRONOUN.search(q) and len(q.split()) <= 12)):
        return "conversational"
    return "factual"


def _first_line(s: str, limit: int = 300) -> str:
    s = (s or "").strip().splitlines()[0] if (s or "").strip() else ""
    return s.strip(" \"'`")[:limit]


def _parse_json_list(s: str) -> list:
    m = re.search(r"\[.*\]", s or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x.strip() for x in data if isinstance(x, str) and x.strip()]


class QueryUnderstanding:
    def __init__(self, llm, cfg):
        self.llm, self.cfg = llm, cfg

    #  rewrite
    def rewrite(self, q: str, history: list) -> str:
        if not history:
            return q
        convo = "\n".join(f"{'User' if t['role'] == 'user' else 'Assistant'}: {neutralize(t['content'])[:240]}"
                          for t in history[-4:])
        try:
            out = _first_line(self.llm.complete(
                [{"role": "system", "content": REWRITE_SYS},
                 {"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up: {q}\n\nStandalone query:"}],
                max_tokens=80, temperature=0.0))
            if 3 <= len(out) <= 300:
                return out
        except Exception as e:
            current().fail("rewrite", e)
        last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
        return f"{q} ({last_user[:120]})" if last_user else q  # degrade gracefully, never block the request

    #  decompose
    def decompose(self, q: str) -> list:
        n = self.cfg.max_subqueries
        subs: list = []
        try:
            subs = _parse_json_list(self.llm.complete(
                [{"role": "system", "content": DECOMP_SYS.format(n=n)}, {"role": "user", "content": q}],
                max_tokens=250, temperature=0.0))
        except Exception as e:
            current().fail("decompose", e)
        if not subs:
            parts = [p.strip() for p in re.split(rf"\?\s*|;\s*|\band\b(?=\s+{_WH}\b)", q, flags=re.I) if p and p.strip()]
            subs = [p if p.endswith("?") else p + "?" for p in parts] if len(parts) > 1 else []
        seen, out = set(), []
        for s in subs:
            if s.lower() not in seen and s.lower() != q.lower():
                seen.add(s.lower())
                out.append(s[:300])
        return out[:n]

    #  filters
    @staticmethod
    def extract_filters(q: str, known: dict) -> dict:
        """Only emit filters that can actually match something in the index (never invent values)."""
        flt: dict = {}
        low = q.lower()
        files = [f for f in known.get("filenames", []) if f.lower() in low or
                 (len(f.rsplit(".", 1)[0]) >= 5 and f.rsplit(".", 1)[0].lower() in low)]
        if files:
            flt["filename"] = files
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q) if int(y) in known.get("years", set())]
        if len(years) == 1:
            flt["year"] = years[0]
        types = {"pdf": r"\bpdf\b", "docx": r"\b(word doc|docx)\b", "csv": r"\b(csv|spreadsheet)\b"}
        for t, pat in types.items():
            if t in known.get("doc_types", set()) and re.search(pat, low):
                flt["doc_type"] = t
        return flt

    #  plan
    def plan(self, question: str, history: list, known: dict) -> QueryPlan:
        tr = current()
        with tr.span("query_understanding"):
            norm = normalize_query(question, self.cfg.max_question_chars)
            st = detect_small_talk(norm)
            if st:
                plan = QueryPlan(question, norm, norm, st, [], {})
                tr.log("query_plan", intent=st, rewritten=norm, subqueries=[], filters={})
                return plan
            intent = classify_intent(norm, bool(history))
            rewritten = norm
            if history and (intent == "conversational" or _PRONOUN.search(norm)):
                rewritten = self.rewrite(norm, history)
            subs = self.decompose(rewritten) if intent in ("comparative", "multi_step") else []
            flt = self.extract_filters(norm, known)
            plan = QueryPlan(question, norm, rewritten, intent, subs, flt)
            tr.log("query_plan", intent=intent, rewritten=rewritten[:200], subqueries=subs, filters=flt)
            return plan

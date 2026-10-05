from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Principal:
    """Caller representation (auth disabled)."""
    tenant_id: str = "default"
    user_id: str = "user"
    roles: tuple = ("member",)

    @property
    def tenant(self) -> str:
        return self.tenant_id

    @property
    def user(self) -> str:
        return self.user_id

    @property
    def key(self) -> str:
        return f"{self.tenant_id}/{self.user_id}"


DEFAULT_PRINCIPAL = Principal()


@dataclass
class Block:
    type: str  # heading | paragraph | table | code
    text: str
    level: int = 0
    page: Optional[int] = None


@dataclass
class ParsedDoc:
    filename: str
    blocks: list
    meta: dict


@dataclass
class Parent:
    parent_id: str
    doc_id: str
    text: str
    meta: dict


@dataclass
class Child:
    chunk_id: str
    parent_id: str
    doc_id: str
    text: str          # what the user/LLM reads
    embed_text: str    # "Section > Path\ntext" - what is embedded / BM25-indexed
    meta: dict


@dataclass
class Hit:
    chunk_id: str
    fused: float = 0.0
    dense: float = 0.0
    rerank: Optional[float] = None

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.fused


@dataclass
class QueryPlan:
    original: str
    normalized: str
    rewritten: str
    intent: str
    subqueries: list = field(default_factory=list)
    filters: dict = field(default_factory=dict)

    @property
    def all_queries(self) -> list:
        seen, out = set(), []
        for q in [self.rewritten, *self.subqueries]:
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out


@dataclass
class ContextItem:
    label: str
    chunk_id: str
    parent_id: str
    text: str
    score: float
    meta: dict
    tokens: int = 0
    suspicious: bool = False


@dataclass
class Claim:
    text: str
    citations: list = field(default_factory=list)
    status: str = "unverified"   # supported | partial | unsupported | contradicted
    score: float = 0.0
    evidence: list = field(default_factory=list)  # [{label, sentence, sim}]
    citation_ok: Optional[bool] = None
    note: str = ""


@dataclass
class VerificationReport:
    claims: list
    groundedness: float
    citation_accuracy: Optional[float]
    unsupported: list
    contradicted: list


@dataclass
class Answer:
    text: str
    abstained: bool = False
    abstain_reason: str = ""
    citations: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    groundedness: Optional[float] = None
    citation_accuracy: Optional[float] = None
    confidence: float = 0.0
    warnings: list = field(default_factory=list)
    cached: bool = False
    trace_id: str = ""
    plan: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    context: list = field(default_factory=list)   # [{label, chunk_ids, filename, page, text}] shown to the LLM

    def to_dict(self) -> dict:
        return asdict(self)

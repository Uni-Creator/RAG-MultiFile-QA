"""Layer 6: claim extraction -> evidence matching -> verification (NLI when available, else lexical+embedding)."""
from __future__ import annotations

import re

import numpy as np

from .generation import _CITE
from .observability import current
from .types import Claim, VerificationReport
from .utils import split_sentences, tokenize

CLAIM_SYS = ("Split the answer into atomic factual claims. One fact per claim. Keep any source labels like [S1] attached "
             "to the claim they support. Output ONLY a JSON array of strings.")
_STOP = set("this that with from have been were which their there about would could should into also than then they them "
            "what when where while will does using used uses such each other more most some many these those its".split())
_META = re.compile(r"^(here(?:'s| is| are)|below|in summary|overall|to summarize|i hope|let me know|sure|based on the (?:documents|context))", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)*%?")


class NLIModel:
    def __init__(self, name: str):
        self.name, self._m, self._failed, self._order = name, None, False, None

    @property
    def available(self) -> bool:
        if self._m is None and not self._failed:
            try:
                from sentence_transformers import CrossEncoder
                self._m = CrossEncoder(self.name)
                id2label = getattr(getattr(self._m.model, "config", None), "id2label", None) or {}
                self._order = [str(id2label.get(i, d)).lower() for i, d in enumerate(["contradiction", "entailment", "neutral"])]
            except Exception as e:
                self._failed = True
                current().fail("nli_load", e)
        return self._m is not None

    def predict(self, premise: str, hypothesis: str) -> dict | None:
        if not self.available:
            return None
        x = np.asarray(self._m.predict([(premise[:1800], hypothesis)])[0], dtype=np.float64)
        if x.min() < 0 or abs(x.sum() - 1) > 1e-3:
            e = np.exp(x - x.max())
            x = e / e.sum()
        return dict(zip(self._order, x.tolist()))


class Verifier:
    def __init__(self, llm, embedder, nli, cfg):
        self.llm, self.embedder, self.nli, self.cfg = llm, embedder, nli, cfg

    #  claim extraction
    def extract_claims(self, answer: str) -> list:
        raw = []
        if self.cfg.claim_extraction == "llm":
            try:
                out = self.llm.complete([{"role": "system", "content": CLAIM_SYS}, {"role": "user", "content": answer}],
                                        max_tokens=500, temperature=0.0)
                from .query import _parse_json_list
                raw = _parse_json_list(out)
            except Exception as e:
                current().fail("claim_extraction", e)
        if not raw:
            raw = split_sentences(answer)
        claims = []
        for r in raw:
            labels = [x for m in _CITE.finditer(r) for x in re.split(r"\s*[,;]\s*", m.group(1))]
            text = re.sub(r"\s+", " ", _CITE.sub("", r)).strip(" -•*")
            content = [t for t in tokenize(text) if len(t) > 3 and t not in _STOP]
            if len(text) >= 8 and len(content) >= 2 and not text.endswith("?") and not _META.match(text):
                claims.append(Claim(text, labels))
        # a claim without its own label inherits the previous label when the answer cites per-sentence
        return claims

    #  evidence matching
    def _evidence(self, items: list) -> list:
        """Evidence units: every sentence/line (tables, code and lists are line-based) and every adjacent pair, so a
        fact spread over two sentences can still be matched as a whole."""
        out = []
        for it in items:
            units = []
            for line in it.text.split("\n"):
                units += [u for u in (split_sentences(line) if len(line) > 160 else [line.strip()]) if len(u) >= 12]
            for i, u in enumerate(units):
                out.append((it.label, u[:600]))
                if i + 1 < len(units):
                    out.append((it.label, (u + " " + units[i + 1])[:900]))
        return out

    @staticmethod
    def _lexical_cov(claim: str, premise: str) -> float:
        ct = {t for t in tokenize(claim) if len(t) > 3 and t not in _STOP}
        return len(ct & set(tokenize(premise))) / len(ct) if ct else 0.0

    @staticmethod
    def _numbers_ok(claim: str, premise: str) -> bool:
        norm = lambda s: s.replace(",", "")
        prem = norm(premise)
        # only meaningful figures: ignore list markers and single digits ("1.", "2")
        nums = [n for n in _NUM.findall(claim) if len(norm(n).rstrip("%")) >= 2 or "." in n or n.endswith("%")]
        return all(norm(n) in prem for n in nums)

    #  verify
    def verify(self, answer: str, items: list) -> VerificationReport:
        tr = current()
        with tr.span("verification"):
            claims = self.extract_claims(answer)
            evid = self._evidence(items)
            if not claims or not evid:
                return VerificationReport(claims, 0.0 if claims else 1.0, None, [c.text for c in claims], [])
            vecs = self.embedder.embed_documents([e[1] for e in evid] + [c.text for c in claims])
            E, C = vecs[: len(evid)], vecs[len(evid):]
            for claim, cvec in zip(claims, C):
                self._verify_claim(claim, cvec, E, evid, items)
            n = len(claims)
            sup = sum(c.status == "supported" for c in claims)
            par = sum(c.status == "partial" for c in claims)
            cited = [c for c in claims if c.citations]
            report = VerificationReport(
                claims=claims,
                groundedness=round((sup + 0.5 * par) / n, 3),
                citation_accuracy=round(sum(bool(c.citation_ok) for c in cited) / len(cited), 3) if cited else None,
                unsupported=[c.text for c in claims if c.status in ("unsupported", "contradicted")],
                contradicted=[c.text for c in claims if c.status == "contradicted"])
            tr.log("verification", groundedness=report.groundedness, citation_accuracy=report.citation_accuracy,
                   claims=[{"claim": c.text[:120], "status": c.status, "score": round(c.score, 3)} for c in claims])
            return report

    def _verify_claim(self, claim: Claim, cvec: np.ndarray, E: np.ndarray, evid: list, items: list | None = None):
        sims = E @ cvec
        cited_idx = [i for i, (lab, _) in enumerate(evid) if lab in claim.citations]
        pool = cited_idx or list(range(len(evid)))
        top = sorted(pool, key=lambda i: -sims[i])[:3]
        best_all = int(np.argmax(sims))
        premise = " ".join(evid[i][1] for i in top)
        claim.evidence = [{"label": evid[i][0], "sentence": evid[i][1][:300], "sim": round(float(sims[i]), 3)} for i in top]

        # signal 1: lexical + embedding coverage (robust to paraphrase/summary of structured text)
        cited_text = " ".join(it.text for it in (items or []) if it.label in claim.citations) or \
            " ".join(it.text for it in (items or []))
        cov = max(self._lexical_cov(claim.text, premise), self._lexical_cov(claim.text, cited_text))
        lex_score = 0.6 * cov + 0.4 * max(float(sims[top[0]]), 0.0)
        lex = "supported" if (cov >= 0.7 and lex_score >= 0.55) else "partial" if (cov >= 0.5 or lex_score >= 0.45) else "unsupported"

        # signal 2: NLI entailment (strict; small models under-credit summaries, so it can only help, not veto)
        nli = self.nli.predict(premise, claim.text) if self.nli else None
        rank = {"unsupported": 0, "partial": 1, "supported": 2}
        status, claim.score = lex, lex_score
        if nli:
            p_ent, p_con = nli.get("entailment", 0.0), nli.get("contradiction", 0.0)
            n_status = "supported" if p_ent > 0.5 else "partial" if p_ent > 0.15 else "unsupported"
            if rank[n_status] > rank[status]:
                status, claim.score = n_status, p_ent
            if p_con > 0.85 and lex != "supported":      # contradiction only counts when the text does not back the claim
                status, claim.score = "contradicted", p_ent
        claim.status = status

        if claim.status in ("supported", "partial") and not self._numbers_ok(claim.text, premise):
            claim.status = "partial" if claim.status == "supported" else "unsupported"
            claim.note = "a number in the claim does not appear in the evidence"

        if claim.citations:
            wrong_source = (evid[best_all][0] not in claim.citations) and (sims[best_all] - sims[top[0]] > 0.10)
            claim.citation_ok = claim.status in ("supported", "partial") and not wrong_source
            if wrong_source:
                claim.note = (claim.note + "; " if claim.note else "") + f"stronger evidence is in {evid[best_all][0]}"
        else:
            claim.note = (claim.note + "; " if claim.note else "") + "uncited"
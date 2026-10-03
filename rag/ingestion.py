"""Layer 1 (part 1): validate -> parse -> clean -> structure extraction -> metadata."""
from __future__ import annotations

import io
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone

from .security import SecurityError, neutralize, redact_pii, safe_filename, scan_injection, validate_upload
from .types import Block, ParsedDoc, Principal
from .utils import clean_whitespace, normalize_text, sha

PAGE_NO = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.I)


#  shared helpers
def heading_level(s: str) -> int:
    """Heuristic heading detector for formats without real structure (PDF, plain text)."""
    if len(s) > 90 or len(s) < 3 or s[-1] in ".,;:":
        return 0
    m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+[A-Z]", s)
    if m:
        return min(m.group(1).count(".") + 1, 4)
    if re.match(r"^(chapter|section|appendix|part)\s+\w+", s, re.I):
        return 1
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and s.upper() == s and not re.search(r"\d{3,}", s):
        return 1
    return 0


def rows_to_markdown(rows: list) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _join_lines(lines: list) -> str:
    out = ""
    for ln in lines:
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return re.sub(r"\s+", " ", out).strip()


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


#  PDF
def _lines_to_blocks(lines: list, page: int) -> list:
    blocks, buf = [], []
    long_lines = [len(x) for x in lines if len(x.strip()) > 20]
    med = statistics.median(long_lines) if long_lines else 80

    def flush():
        if buf:
            text = _join_lines(buf)
            if text:
                blocks.append(Block("paragraph", text, 0, page))
            buf.clear()

    for raw in lines:
        s = raw.strip()
        if not s:
            flush()
            continue
        lvl = heading_level(s)
        if lvl:
            flush()
            blocks.append(Block("heading", re.sub(r"\s+", " ", s), lvl, page))
            continue
        buf.append(s)
        if s[-1] in ".!?:" and len(s) < 0.6 * med:  # short last line of a wrapped paragraph
            flush()
    flush()
    return blocks


def parse_pdf(name: str, data: bytes, cfg) -> list:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise SecurityError("encrypted PDF not supported")
    n = len(reader.pages)
    if n > cfg.max_pages:
        raise SecurityError(f"PDF has {n} pages (limit {cfg.max_pages})")
    pages = [[ln.rstrip() for ln in normalize_text(p.extract_text() or "").splitlines()] for p in reader.pages]

    def sig(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter = Counter()
    for lines in pages:
        counts.update({sig(x) for x in lines if x.strip() and len(x) < 100})
    boiler = {s for s, c in counts.items() if n >= 3 and c >= max(3, 0.5 * n)}  # repeated headers/footers

    blocks = []
    for pno, lines in enumerate(pages, start=1):
        lines = [x for x in lines if sig(x) not in boiler and not PAGE_NO.match(x)]
        blocks.extend(_lines_to_blocks(lines, pno))
    return blocks, n


#  DOCX
def parse_docx(name: str, data: bytes, cfg):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    blocks = []
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            text = normalize_text(p.text).strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            if m:
                blocks.append(Block("heading", text, int(m.group(1))))
            elif style == "Title":
                blocks.append(Block("heading", text, 1))
            elif "List" in style:
                blocks.append(Block("paragraph", "- " + text))
            else:
                blocks.append(Block("paragraph", text))
        elif tag == "tbl":
            t = Table(el, d)
            rows = [[normalize_text(c.text).strip().replace("\n", " ") for c in r.cells] for r in t.rows]
            blocks.append(Block("table", rows_to_markdown(rows)))
    return blocks, None


#  TXT / Markdown
_FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
_SKIP_LINE = re.compile(r"^\s*(?:-{3,}|\[BREAK[\w-]*\])\s*$")


def parse_text(name: str, data: bytes, cfg):
    text = normalize_text(_decode(data))
    text = _FRONT.sub("", text, count=1)
    blocks, para, table, code, in_code = [], [], [], [], False

    def flush_para():
        if para:
            joined = _join_lines(para)
            lvl = heading_level(joined) if len(para) == 1 else 0
            blocks.append(Block("heading", joined, lvl) if lvl else Block("paragraph", joined))
            para.clear()

    def flush_table():
        if table:
            blocks.append(Block("table", "\n".join(table)))
            table.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                blocks.append(Block("code", "\n".join(code)))
                code.clear()
            else:
                flush_para(); flush_table()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        line = re.sub(r"^\s*>\s?", "", line)  # blockquote / callout
        m = re.match(r"^(#{1,6})\s+(.*\S)", line)
        if m:
            flush_para(); flush_table()
            blocks.append(Block("heading", m.group(2).strip(), len(m.group(1))))
        elif _SKIP_LINE.match(line):
            flush_para(); flush_table()
        elif line.lstrip().startswith("|"):
            flush_para()
            table.append(line.strip())
        elif not line.strip():
            flush_para(); flush_table()
        elif re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            flush_para(); flush_table()
            blocks.append(Block("paragraph", line.strip()))
        else:
            flush_table()
            para.append(line.strip())
    if in_code and code:
        blocks.append(Block("code", "\n".join(code)))
    flush_para(); flush_table()
    return blocks, None


#  CSV
def parse_csv(name: str, data: bytes, cfg):
    import pandas as pd

    df = pd.read_csv(io.BytesIO(data), nrows=cfg.max_csv_rows, dtype=str, keep_default_na=False,
                     on_bad_lines="skip", encoding_errors="replace")
    cols = [normalize_text(str(c)).strip() for c in df.columns]
    stem = os.path.splitext(name)[0]
    blocks = [Block("heading", stem, 1)]
    step = 20
    for start in range(0, len(df), step):
        rows = [cols] + [[normalize_text(str(v)).strip() for v in r] for r in df.iloc[start:start + step].values.tolist()]
        blocks.append(Block("heading", f"Rows {start + 1}-{min(start + step, len(df))}", 2))
        blocks.append(Block("table", rows_to_markdown(rows)))
    return blocks, None


PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_text, ".md": parse_text, ".csv": parse_csv}


#  orchestration
def parse_document(name: str, data: bytes, principal: Principal, cfg, allowed_roles=None) -> tuple:
    """Returns (ParsedDoc, ingest_stats). Raises SecurityError for rejected files."""
    name = safe_filename(name)
    ext = validate_upload(name, data, cfg)
    blocks, n_pages = PARSERS[ext](name, data, cfg)

    # cleaning + safety flags
    cleaned, inj_blocks, pii_total = [], 0, Counter()
    for b in blocks:
        b.text = clean_whitespace(b.text) if b.type != "code" else b.text.rstrip()
        if not b.text:
            continue
        if scan_injection(b.text):
            inj_blocks += 1
            if cfg.injection_policy == "drop":
                continue
            if cfg.injection_policy == "neutralize":
                b.text = neutralize(b.text)
        if cfg.redact_pii_in_index:
            b.text, counts = redact_pii(b.text)
            pii_total.update(counts)
        cleaned.append(b)
    if not cleaned:
        raise SecurityError("no extractable text (scanned PDF? OCR is not enabled)")

    full = "\n".join(b.text for b in cleaned)
    # document title: first un-numbered level-1 heading, else the file name ("1. Introduction" is a section, not a title)
    title = next((b.text for b in cleaned if b.type == "heading" and b.level == 1
                  and not re.match(r"^\d+(?:\.\d+)*[.)]?\s", b.text)), os.path.splitext(name)[0])
    years = Counter(re.findall(r"\b(19\d{2}|20\d{2})\b", name + " " + full[:3000]))
    meta = {
        "doc_id": sha(data, n=16),
        "filename": name,
        "doc_type": ext.lstrip("."),
        "title": title[:150],
        "year": int(years.most_common(1)[0][0]) if years else None,
        "n_pages": n_pages,
        "tenant_id": principal.tenant_id,
        "owner": principal.user_id,
        "allowed_roles": list(allowed_roles) if allowed_roles else None,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": len(data),
        "injection_blocks": inj_blocks,
    }
    stats = {"blocks": len(cleaned), "injection_blocks": inj_blocks, "pii_redacted": dict(pii_total)}
    return ParsedDoc(name, cleaned, meta), stats
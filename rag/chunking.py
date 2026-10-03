r"""Layer 1 (part 2): structure extraction + structure-aware parent-child chunking.

  blocks -> section tree -> PARENTS (a section, split if huge)
                              \-> CHILDREN (small sentence-packed chunks; tables/code stay atomic)

Children are embedded (precision); parents are what the LLM can read (context).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .security import scan_injection
from .types import Block, Child, Parent, ParsedDoc
from .utils import split_sentences


@dataclass
class Section:
    path: list
    blocks: list = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def build_sections(blocks: list, title: str) -> list:
    """Structure extraction: turn the flat block stream into sections keyed by heading path."""
    sections, stack = [], []
    cur = Section([title])

    def path():
        if stack and stack[0][1] == title:
            return [h for _, h in stack]
        return [title] + [h for _, h in stack]

    for b in blocks:
        if b.type == "heading":
            if cur.blocks:
                sections.append(cur)
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            cur = Section(path())
        else:
            cur.blocks.append(b)
    if cur.blocks:
        sections.append(cur)
    return sections


def merge_small(sections: list, cfg) -> list:
    """Tiny sibling sections (a 2-line 'Summary') make useless parents - fold them into the next sibling."""
    out: list = []
    for s in sections:
        if (out and out[-1].chars < cfg.parent_min_chars and out[-1].path[:-1] == s.path[:-1]
                and out[-1].chars + s.chars <= cfg.parent_max_chars):
            out[-1].blocks.append(Block("paragraph", s.path[-1]))
            out[-1].blocks.extend(s.blocks)
        else:
            out.append(Section(list(s.path), list(s.blocks)))
    return out


def _hard_split(s: str, n: int) -> list:
    out = []
    while len(s) > n:
        cut = s.rfind(" ", 0, n)
        cut = cut if cut > n * 0.4 else n
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    return out + ([s] if s else [])


def split_big(b: Block, max_chars: int) -> list:
    """Split an oversized block without breaking its nature (tables keep the header row)."""
    if len(b.text) <= max_chars:
        return [b]
    if b.type == "table":
        lines = b.text.split("\n")
        head, rows = lines[:2], lines[2:]
        pieces, cur = [], list(head)
        for r in rows:
            if len("\n".join(cur + [r])) > max_chars and len(cur) > 2:
                pieces.append(Block("table", "\n".join(cur), 0, b.page))
                cur = list(head)
            cur.append(r)
        pieces.append(Block("table", "\n".join(cur), 0, b.page))
        return pieces
    if b.type == "code":
        pieces, cur = [], []
        for ln in b.text.split("\n"):
            if sum(len(x) + 1 for x in cur) + len(ln) > max_chars and cur:
                pieces.append(Block("code", "\n".join(cur), 0, b.page))
                cur = []
            cur.append(ln)
        pieces.append(Block("code", "\n".join(cur), 0, b.page))
        return pieces
    pieces, cur = [], ""
    for s in split_sentences(b.text):
        for part in _hard_split(s, max_chars):
            if cur and len(cur) + len(part) + 1 > max_chars:
                pieces.append(Block(b.type, cur, 0, b.page))
                cur = ""
            cur = f"{cur} {part}".strip()
    if cur:
        pieces.append(Block(b.type, cur, 0, b.page))
    return pieces


def split_section(blocks: list, max_chars: int) -> list:
    parts, cur, size = [], [], 0
    for b in blocks:
        for piece in split_big(b, max_chars):
            if cur and size + len(piece.text) > max_chars:
                parts.append(cur)
                cur, size = [], 0
            cur.append(piece)
            size += len(piece.text)
    if cur:
        parts.append(cur)
    return parts


def make_children(blocks: list, cfg) -> list:
    """-> [(text, page, block_type)]. Prose is sentence-packed with overlap; tables/code are atomic."""
    out, buf, buf_page = [], [], None

    def flush(carry: bool):
        nonlocal buf, buf_page
        if buf:
            out.append((" ".join(buf), buf_page, "text"))
        buf = buf[-cfg.child_overlap_sentences:] if (carry and cfg.child_overlap_sentences) else []
        buf_page = None

    for b in blocks:
        if b.type in ("table", "code"):
            flush(carry=False)
            for piece in split_big(b, cfg.child_chars * 3):
                out.append((piece.text, b.page, b.type))
            continue
        for sent in split_sentences(b.text):
            for s in _hard_split(sent, cfg.child_chars):
                if buf and len(" ".join(buf)) + len(s) + 1 > cfg.child_chars:
                    flush(carry=True)
                if buf_page is None:
                    buf_page = b.page
                buf.append(s)
    flush(carry=False)
    return out


def chunk_document(parsed: ParsedDoc, cfg) -> tuple:
    m = parsed.meta
    base = {k: m.get(k) for k in ("doc_id", "filename", "doc_type", "title", "year", "tenant_id", "owner", "allowed_roles")}
    sections = merge_small(build_sections(parsed.blocks, m["title"]), cfg)
    parents, children = [], []
    for sec in sections:
        header = " > ".join(sec.path)
        for part in split_section(sec.blocks, cfg.parent_max_chars):
            pid = f"{m['doc_id']}:p{len(parents)}"
            pages = [b.page for b in part if b.page is not None]
            pmeta = {**base, "section_path": header, "section_title": sec.path[-1],
                     "page": pages[0] if pages else None, "page_end": pages[-1] if pages else None}
            parents.append(Parent(pid, m["doc_id"], "\n\n".join(b.text for b in part), pmeta))
            for text, page, btype in make_children(part, cfg):
                cid = f"{m['doc_id']}:c{len(children)}"
                cmeta = {**base, "section_path": header, "section_title": sec.path[-1],
                         "page": page if page is not None else pmeta["page"], "block_type": btype,
                         "chunk_index": len(children), "suspicious": bool(scan_injection(text))}
                children.append(Child(cid, pid, m["doc_id"], text, f"{header}\n{text}", cmeta))
    return parents, children

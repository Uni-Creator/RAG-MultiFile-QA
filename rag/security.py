"""Layer 9: untrusted-input handling, file validation, ACLs, auth, PII, rate limits."""
from __future__ import annotations

import hmac
import io
import os
import re
import secrets
import threading
import time
import zipfile
from collections import defaultdict

from .types import Principal


class SecurityError(Exception):
    pass


class RateLimitError(SecurityError):
    pass


#  ids
_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def validate_id(value: str, what: str = "id") -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise SecurityError(f"invalid {what}: only letters, digits, '_' and '-' (max 64)")
    return value


def safe_filename(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    base = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return base[:150] or "unnamed"


#  file security
ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md", ".csv"}


def validate_upload(name: str, data: bytes, cfg) -> str:
    """Reject unsafe uploads *before* any parser touches them. Returns the extension."""
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        raise SecurityError(f"unsupported file type '{ext}'")
    if not data:
        raise SecurityError("empty file")
    if len(data) > cfg.max_file_mb * 1024 * 1024:
        raise SecurityError(f"file larger than {cfg.max_file_mb} MB")

    if ext == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise SecurityError("not a PDF (bad magic bytes)")
        for marker in (b"/JavaScript", b"/JS ", b"/Launch"):   # /OpenAction alone is benign (Word/LaTeX emit it for zoom)
            if marker in data:
                raise SecurityError(f"PDF contains active content ({marker.decode().strip()})")
    elif ext == ".docx":
        if not data.startswith(b"PK"):
            raise SecurityError("not a DOCX (bad magic bytes)")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                infos = z.infolist()
                if len(infos) > 5000:
                    raise SecurityError("too many archive entries")
                total = sum(i.file_size for i in infos)
                if total > 500 * 1024 * 1024 or total > cfg.max_zip_ratio * max(len(data), 1):
                    raise SecurityError("archive expands too much (possible zip bomb)")
                for i in infos:
                    n = i.filename
                    if n.startswith("/") or ".." in n.split("/"):
                        raise SecurityError("unsafe path in archive")
                    if "vbaproject" in n.lower() or n.lower().endswith((".exe", ".dll", ".bat", ".js")):
                        raise SecurityError("DOCX contains macros or executables")
        except zipfile.BadZipFile as e:
            raise SecurityError("corrupt DOCX archive") from e
    else:
        if b"\x00" in data[:4096]:
            raise SecurityError("binary content in text file")
    return ext


#  prompt injection
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
    r"disregard\s+.{0,40}(?:instructions?|rules?|prompt)",
    r"(?:reveal|print|show|repeat|leak)\s+.{0,30}(?:system\s+prompt|instructions|hidden\s+prompt)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"new\s+instructions?\s*:",
    r"(?:act|pretend)\s+(?:as|to\s+be)\s+(?:a|an|the)\b",
    r"</?\s*(?:system|assistant|documents?|memory|question|context)\s*>",
    r"\[/?INST\]",
    r"<\|[^|>]{1,30}\|>",
]
_INJ = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I | re.S)
_TAGS = re.compile(r"</?\s*(documents?|memory|question|system|assistant)\b", re.I)


def scan_injection(text: str) -> list:
    return [m.group(0)[:80] for m in _INJ.finditer(text)]


def neutralize(text: str) -> str:
    """Defang instruction-like spans and prompt-structure tags inside retrieved text."""
    text = _INJ.sub("[instruction-like text removed]", text)
    return _TAGS.sub(lambda m: m.group(0).replace("<", "‹"), text)


def sanitize_for_prompt(text: str) -> str:
    """Always-on escaping of structural tags (even when injection_policy == flag)."""
    return _TAGS.sub(lambda m: m.group(0).replace("<", "‹"), text)


def make_canary() -> str:
    return "CANARY-" + secrets.token_hex(6)


#  PII
def _luhn(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        s += d
    return s % 10 == 0


_PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "aadhaar": re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "phone": re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|\d{10})(?![\w.])"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
}


def redact_pii(text: str) -> tuple:
    counts: dict = defaultdict(int)

    def sub(kind):
        def _f(m):
            if kind == "card" and not _luhn(m.group(0)):
                return m.group(0)
            counts[kind] += 1
            return f"[{kind.upper()}]"
        return _f

    for kind in ("email", "ssn", "card", "aadhaar", "pan", "phone"):
        text = _PII[kind].sub(sub(kind), text)
    return text, dict(counts)


#  authn / authz
def can_read(principal: Principal, meta: dict) -> bool:
    """AuthZ: tenant must match; if the doc lists allowed_roles, caller needs one."""
    if meta.get("tenant_id") != principal.tenant_id:
        return False
    allowed = meta.get("allowed_roles")
    return not allowed or bool(set(allowed) & set(principal.roles))


class StaticTokenAuthenticator:
    """AuthN: RAG_AUTH_TOKENS="token:tenant:user:role1|role2;token2:tenant:user".
    Constant-time comparison. Swap for OIDC/JWT in a real deployment."""

    def __init__(self, spec: str | None = None):
        spec = spec if spec is not None else os.getenv("RAG_AUTH_TOKENS", "")
        self.entries = []
        for item in filter(None, (s.strip() for s in spec.split(";"))):
            parts = item.split(":")
            if len(parts) >= 3:
                roles = tuple(parts[3].split("|")) if len(parts) > 3 and parts[3] else ("member",)
                self.entries.append((parts[0], Principal(validate_id(parts[1]), validate_id(parts[2]), roles)))

    @property
    def enabled(self) -> bool:
        return bool(self.entries)

    def authenticate(self, token: str):
        found = None
        for tok, principal in self.entries:  # no early exit: constant-ish time
            if hmac.compare_digest(tok.encode(), (token or "").encode()):
                found = principal
        return found


class RateLimiter:
    """Token bucket per (principal, endpoint)."""

    def __init__(self, limits: dict):
        self.limits = limits  # endpoint -> per-minute rate
        self._state: dict = {}
        self._lock = threading.Lock()

    def check(self, principal: Principal, endpoint: str):
        rate = self.limits.get(endpoint)
        if not rate:
            return
        now = time.monotonic()
        with self._lock:
            tokens, last = self._state.get((principal.key, endpoint), (float(rate), now))
            tokens = min(float(rate), tokens + (now - last) * rate / 60.0)
            if tokens < 1:
                raise RateLimitError(f"rate limit exceeded for '{endpoint}' ({rate}/min)")
            self._state[(principal.key, endpoint)] = (tokens - 1, now)
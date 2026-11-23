"""Keep secrets out of the index (and therefore out of prompts sent to LLM providers)."""

from __future__ import annotations

import fnmatch
import re

# Files that are never indexed, whatever their extension.
SECRET_FILE_PATTERNS = [
    ".env", ".env.*", "*.env", "*.pem", "*.key", "*.pfx", "*.p12", "*.keystore", "*.jks", "id_rsa*", "id_ed25519*",
    "*secret*", "*credential*", ".npmrc", ".pypirc", ".netrc", "*.tfstate", "*.tfvars",
]

_PATTERNS = [
    # private key blocks (redact the body, keep line count)
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),  # GitHub tokens
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),  # Google API key
    re.compile(r"\bgsk_[A-Za-z0-9]{40,}\b"),  # Groq API key
    re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}\b"),  # generic "sk-" style keys
    re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"),  # Slack
]
# key = "value" style assignments; only the value is replaced
_ASSIGN = re.compile(
    r"""(?i)((?:api[_-]?key|secret|password|passwd|pwd|token|access[_-]?key|client[_-]?secret)\w*["']?\s*[:=]\s*["'])"""
    r"""([^"'\s]{8,})(["'])"""
)


def is_secret_file(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1].lower()
    return any(fnmatch.fnmatch(name, p) for p in SECRET_FILE_PATTERNS)


def redact(text: str) -> tuple[str, int]:
    """Replace likely secrets with [REDACTED]. Returns (text, number_of_redactions). Line count is preserved."""
    count = 0

    def keep_lines(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "[REDACTED]" + "\n" * m.group(0).count("\n")

    for p in _PATTERNS:
        text = p.sub(keep_lines, text)

    def assign(m: re.Match) -> str:
        nonlocal count
        value = m.group(2)
        # skip obvious placeholders / code references
        if value.startswith(("{", "$", "<", "%")) or value.lower() in ("password", "changeme", "example"):
            return m.group(0)
        count += 1
        return m.group(1) + "[REDACTED]" + m.group(3)

    text = _ASSIGN.sub(assign, text)
    return text, count

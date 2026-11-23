"""Load source files from a local directory or a git URL."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import config
from .models import SourceFile
from .safety import is_secret_file, redact

LANGUAGES = {
    ".py": "python",
    ".cs": "csharp",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rs": "rust",
    ".swift": "swift",
    ".php": "php",
    ".rb": "ruby",
    ".scala": "scala",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "text",
    ".txt": "text",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bin", "obj", "dist", "build", "out", "target",
    ".venv", "venv", "__pycache__", ".idea", ".vs", ".vscode", ".askcode", "packages",
}

MAX_FILE_BYTES = 1_000_000


@dataclass
class Repo:
    root: Path
    source: str  # what the user passed (path or URL)
    commit: str | None
    files: list[SourceFile]
    web_base: str | None = None  # e.g. https://github.com/o/r/blob/<sha>
    skipped_secret_files: int = 0
    redactions: int = 0


def _is_url(source: str) -> bool:
    return bool(re.match(r"^(https?://|git@)", source))


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def clone(url: str, ref: str | None = None) -> Path:
    """Shallow-clone `url` (optionally at a branch, tag or commit sha) into ASKCODE_HOME/repos.

    Without a ref this fetches the latest HEAD every time, which is what `askcode update` relies on.
    """
    name = re.sub(r"\.git$", "", url.rstrip("/").split("/")[-1])
    dest = config.repos_dir() / (f"{name}@{ref[:12]}" if ref else name)
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", cwd=dest)
        _git("remote", "add", "origin", url, cwd=dest)
    _git("fetch", "-q", "--depth", "1", "origin", ref or "HEAD", cwd=dest)
    _git("checkout", "-q", "--force", "FETCH_HEAD", cwd=dest)
    return dest


def web_base(remote: str | None, commit: str | None) -> str | None:
    """Turn a git remote + commit into a permalink prefix for GitHub / GitLab / Bitbucket."""
    if not remote or not commit:
        return None
    r = remote.strip()
    m = re.match(r"^git@([^:]+):(.+?)(?:\.git)?$", r)
    if m:
        r = f"https://{m.group(1)}/{m.group(2)}"
    r = re.sub(r"\.git$", "", r.rstrip("/"))
    r = re.sub(r"^https?://[^@/]+@", "https://", r)  # drop credentials from https remotes
    if "github.com" in r:
        return f"{r}/blob/{commit}"
    if "gitlab" in r:
        return f"{r}/-/blob/{commit}"
    if "bitbucket.org" in r:
        return f"{r}/src/{commit}"
    return None


def permalink(base: str | None, path: str, start: int, end: int) -> str | None:
    if not base:
        return None
    if "bitbucket.org" in base:
        return f"{base}/{path}#lines-{start}:{end}"
    return f"{base}/{path}#L{start}-L{end}" if "github.com" in base else f"{base}/{path}#L{start}-{end}"


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, p.rstrip("/") + "/*") for p in patterns)


def load(
    source: str,
    ref: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Repo:
    root = clone(source, ref) if _is_url(source) else Path(source).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    try:
        commit = _git("rev-parse", "HEAD", cwd=root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    remote = source if _is_url(source) else None
    if remote is None:
        try:
            remote = _git("remote", "get-url", "origin", cwd=root)
        except (subprocess.CalledProcessError, FileNotFoundError):
            remote = None

    files: list[SourceFile] = []
    skipped = redactions = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        rel = p.relative_to(root).as_posix()
        if is_secret_file(rel):
            skipped += 1
            continue
        lang = LANGUAGES.get(p.suffix.lower())
        if lang is None or p.stat().st_size > MAX_FILE_BYTES:
            continue
        if include and not _matches(rel, include):
            continue
        if exclude and _matches(rel, exclude):
            continue
        try:
            text = p.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if text.strip():
            text, n = redact(text.replace("\r\n", "\n"))
            redactions += n
            files.append(SourceFile(path=rel, text=text, language=lang))
    return Repo(root=root, source=source, commit=commit, files=files, web_base=web_base(remote, commit),
                skipped_secret_files=skipped, redactions=redactions)

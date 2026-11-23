from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines) so users don't need python-dotenv."""
    env = Path.cwd() / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def home() -> Path:
    p = Path(os.environ.get("ASKCODE_HOME") or Path.cwd() / ".askcode")
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_dir(name: str) -> Path:
    return home() / "indexes" / name


def repos_dir() -> Path:
    p = home() / "repos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v else default

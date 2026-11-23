from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class SourceFile:
    path: str  # repo-relative, forward slashes
    text: str
    language: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    @property
    def sha(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    kind: str  # file | function | property | type | gap | group | window | section
    language: str
    symbol: str | None = None
    id: str = field(default="")
    # small-to-big: the larger span this chunk belongs to (whole method, neighbouring windows...)
    parent_start: int | None = None
    parent_end: int | None = None
    # contextual retrieval: one LLM-written sentence saying what this chunk does (optional)
    context: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raw = f"{self.path}:{self.start_line}-{self.end_line}:{self.kind}"
            self.id = hashlib.sha1(raw.encode()).hexdigest()[:16]

    @property
    def citation(self) -> str:
        return f"{self.path}:L{self.start_line}-L{self.end_line}"

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1

    def embedding_text(self) -> str:
        """Text sent to the embedding model (and the keyword index).

        A short header naming the file and symbol ("contextual chunk header") helps a lot:
        a method body alone often never mentions the class or file it lives in.
        """
        header = f"File: {self.path}"
        if self.symbol:
            header += f"\nSymbol: {self.symbol}"
        if self.context:
            header += f"\nSummary: {self.context}"
        return f"{header}\n\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Hit:
    chunk: Chunk
    score: float
    # rank of this chunk in each retriever that found it, e.g. {"vector": 3, "keyword": 1}
    ranks: dict[str, int] = field(default_factory=dict)


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for code). Good enough for budgeting."""
    return max(1, len(text) // 4)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

"""Keyword search: BM25 with a code-aware tokenizer.

Code identifiers are glued together (``QueryFirstOrDefaultAsync``), so a plain word tokenizer would
never match a question that says "query first or default". We index the whole identifier *and* its
camelCase / snake_case parts.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by", "with", "as", "at",
    "it", "this", "that", "from", "how", "what", "which", "when", "where", "why", "does", "do", "can", "i",
    "if", "into", "its", "there", "their", "was", "were", "will", "would", "should", "use", "used", "using",
    "s", "var", "return", "public", "private", "static", "void", "new", "class", "get", "set",
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text):
        lower = word.lower()
        parts = [p.lower() for p in _PARTS.findall(word.replace("_", " "))]
        if lower not in STOPWORDS:
            out.append(lower)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, d in enumerate(docs):
            tf = Counter(tokenize(d))
            self.doc_len.append(sum(tf.values()))
            for term, n in tf.items():
                self.postings[term].append((i, n))
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for doc, tf in self.postings.get(term, ()):
                dl = self.doc_len[doc]
                scores[doc] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

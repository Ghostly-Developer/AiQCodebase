"""LLM providers for answer generation: Google Gemini and Groq. Streaming and a disk cache included."""

from __future__ import annotations

import time
from collections.abc import Iterator

from . import config
from .cache import KVCache


def _retryable(e: Exception) -> bool:
    s = str(e)
    return any(code in s for code in ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate"))


class LLM:
    name: str

    def stream(self, system: str, user: str) -> Iterator[str]:
        raise NotImplementedError

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return "".join(self.stream(system, user))


class GoogleLLM(LLM):
    def __init__(self, model: str | None = None):
        from google import genai

        key = config.env("GOOGLE_API_KEY") or config.env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self._client = genai.Client(api_key=key)
        self.model = model or config.env("ASKCODE_GOOGLE_MODEL", "gemini-2.5-flash")
        self.name = f"google:{self.model}"

    def _config(self, system: str, json_mode: bool):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=8192,
            response_mime_type="application/json" if json_mode else None,
        )

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        for attempt in range(6):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=user, config=self._config(system, json_mode)
                )
                return resp.text or ""
            except Exception as e:  # free-tier rate limits (429) and transient 503s
                if attempt == 5 or not _retryable(e):
                    raise
                time.sleep(min(60, 5 * 2**attempt))
        return ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=user, config=self._config(system, False)
        ):
            if chunk.text:
                yield chunk.text


class GroqLLM(LLM):
    def __init__(self, model: str | None = None):
        from groq import Groq

        key = config.env("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self._client = Groq(api_key=key, max_retries=6)  # SDK honours retry-after on 429
        self.model = model or config.env("ASKCODE_GROQ_MODEL", "llama-3.3-70b-versatile")
        self.name = f"groq:{self.model}"

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._client.chat.completions.create(
            model=self.model, messages=self._messages(system, user), temperature=0.0,
            max_completion_tokens=4096, **extra,
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        for chunk in self._client.chat.completions.create(
            model=self.model, messages=self._messages(system, user), temperature=0.0,
            max_completion_tokens=4096, stream=True,
        ):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class CachedLLM(LLM):
    """Wraps any LLM with a disk cache keyed on (model, system, user, json_mode)."""

    def __init__(self, inner: LLM):
        self.inner = inner
        self.name = inner.name
        self._cache = KVCache("llm")
        self.hits = self.misses = 0

    def _key(self, system: str, user: str, json_mode: bool) -> str:
        return KVCache.key(self.inner.name, system, user, str(json_mode))

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        k = self._key(system, user, json_mode)
        cached = self._cache.get(k)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        out = self.inner.complete(system, user, json_mode)
        if out:
            self._cache.set(k, out)
        return out

    def stream(self, system: str, user: str) -> Iterator[str]:
        k = self._key(system, user, False)
        cached = self._cache.get(k)
        if cached is not None:
            self.hits += 1
            yield cached
            return
        self.misses += 1
        parts = []
        for piece in self.inner.stream(system, user):
            parts.append(piece)
            yield piece
        if parts:
            self._cache.set(k, "".join(parts))


def get_llm(spec: str | None = None, cache: bool = True) -> LLM:
    """spec: 'google', 'groq', 'google:<model>', 'groq:<model>' or None (first provider with a key)."""
    if not spec:
        if config.env("GOOGLE_API_KEY") or config.env("GEMINI_API_KEY"):
            spec = "google"
        elif config.env("GROQ_API_KEY"):
            spec = "groq"
        else:
            raise RuntimeError("Set GOOGLE_API_KEY or GROQ_API_KEY (see .env.example) to use an LLM.")
    kind, _, model = spec.partition(":")
    if kind == "google":
        llm: LLM = GoogleLLM(model or None)
    elif kind == "groq":
        llm = GroqLLM(model or None)
    else:
        raise ValueError(f"Unknown LLM {spec!r} (use google or groq)")
    return CachedLLM(llm) if cache else llm

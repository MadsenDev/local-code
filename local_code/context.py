"""Transparent, dependency-free context-window accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ContextUsage:
    conversation: int = 0
    memory: int = 0
    repo: int = 0
    tools: int = 0
    other: int = 0
    limit: int = 16384

    @property
    def total(self):
        return self.conversation + self.memory + self.repo + self.tools + self.other

    @property
    def remaining(self):
        return max(0, self.limit - self.total)

    @property
    def percent(self):
        return round(self.total / self.limit * 100, 1) if self.limit else 0.0

    def to_dict(self):
        return {**asdict(self), "total": self.total, "remaining": self.remaining, "percent": self.percent}


def estimate_tokens(value):
    """Conservative token estimate suitable without provider tokenizers."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(estimate_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(estimate_tokens(key) + estimate_tokens(item) for key, item in value.items())
    text = str(value)
    return (len(text.encode("utf-8")) + 3) // 4


def build_context_usage(*, history=None, memory="", repo="", tools="", other="", limit=16384):
    return ContextUsage(
        conversation=estimate_tokens(history or []),
        memory=estimate_tokens(memory),
        repo=estimate_tokens(repo),
        tools=estimate_tokens(tools),
        other=estimate_tokens(other),
        limit=limit,
    )

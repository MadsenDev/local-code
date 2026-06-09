"""Storage policy for reviewable project knowledge and private local state."""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .records import IntelligenceRecord, RecordKind, RecordStatus


class StorageMode(StrEnum):
    """How repository intelligence participates in collaboration."""

    LOCAL_ONLY = "local-only"
    SHARED = "shared"
    HYBRID = "hybrid"


class StorageScope(StrEnum):
    """The explicit privacy boundary beneath a repository's ``.rist`` folder."""

    PROJECT = "project"
    LOCAL = "local"


DEFAULT_STORAGE_MODE = StorageMode.HYBRID

# Only reviewed, durable classifications belong in the commit-friendly scope.
SHAREABLE_RECORD_KINDS = frozenset(
    {
        RecordKind.PROJECT_IDENTITY,
        RecordKind.DECISION,
        RecordKind.COMPONENT,
        RecordKind.RELATIONSHIP,
        RecordKind.LIFECYCLE_STATUS,
        RecordKind.CONVENTION,
    }
)
SHAREABLE_RECORD_STATUSES = frozenset(
    {
        RecordStatus.ACTIVE,
        RecordStatus.DEPRECATED,
        RecordStatus.SUPERSEDED,
        RecordStatus.ARCHIVED,
    }
)

_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:api_?key|access_?token|auth|authorization|cookie|credential|env|environment|"
    r"home|password|personal|private|prompt|provider|secret|session|token)(?:$|_)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk|ghp|github_pat|xox[baprs])-[-A-Za-z0-9_]{8,}|"
    r"\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*\S+|\$\{?[A-Z][A-Z0-9_]*\}?)",
    re.IGNORECASE,
)
_PRIVATE_TEXT_RE = re.compile(r"\b(?:system|user|assistant)?\s*prompt\s*[:=]|\bprovider\s*[:=]|\benvironment\s*[:=]|\b(?:ollama|llama\.cpp|openrouter|openai)\b", re.IGNORECASE)
_ABSOLUTE_LOCAL_PATH_RE = re.compile(r"(?:^|[\s'\"(])(?:/(?:home|Users|tmp|var|etc|opt)/[^\s'\")]+|[A-Za-z]:\\Users\\[^\\\s]+)")
_HOME_PATH_RE = re.compile(r"(?:^|[\s'\"(])(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)(?:[/\\]|$)")


class ShareableContentError(ValueError):
    """Raised when content is not safe or reviewed enough for project storage."""


def coerce_storage_mode(value: StorageMode | str | None) -> StorageMode:
    if value is None:
        value = os.environ.get("RIST_STORAGE_MODE", DEFAULT_STORAGE_MODE.value)
    try:
        return StorageMode(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in StorageMode)
        raise ValueError(f"storage mode must be one of: {choices}") from exc


def scope_path(root: str | Path, scope: StorageScope | str) -> Path:
    return Path(root) / StorageScope(scope).value


def record_is_reviewable(record: IntelligenceRecord) -> bool:
    """Return whether a record category/lifecycle is eligible for team review."""
    return record.kind in SHAREABLE_RECORD_KINDS and record.status in SHAREABLE_RECORD_STATUSES


def _walk_values(value: Any, key: str = ""):
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            child_name = str(child_key)
            yield child_name, child
            yield from _walk_values(child, child_name)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk_values(child, key)


def validate_shareable_record(record: IntelligenceRecord) -> None:
    """Reject unreviewed classes and common private/secret material.

    This is deliberately conservative. A rejected record remains suitable for
    ``.rist/local/`` and can be rewritten into a minimal, reviewed project fact.
    """
    if not record_is_reviewable(record):
        raise ShareableContentError(
            f"{record.kind.value}/{record.status.value} records are local-only until reviewed"
        )
    fields = {
        "statement": record.statement,
        "source": record.source,
        "source_ref": record.source_ref,
        "details": record.details,
    }
    for key, value in _walk_values(fields):
        if _SENSITIVE_KEY_RE.search(key):
            raise ShareableContentError(f"shareable records cannot contain sensitive field {key!r}")
        if isinstance(value, str) and (_SECRET_TEXT_RE.search(value) or _PRIVATE_TEXT_RE.search(value) or _HOME_PATH_RE.search(value) or _ABSOLUTE_LOCAL_PATH_RE.search(value)):
            raise ShareableContentError("shareable records cannot contain secrets, environment values, or home paths")

"""Dependency-free schema validation for intelligence JSON."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .records import RecordKind, RecordStatus

SCHEMA_VERSION = 1


class IntelligenceValidationError(ValueError):
    """Raised when persisted intelligence does not match the supported schema."""


def _timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise IntelligenceValidationError(f"{field} must be a string")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntelligenceValidationError(f"{field} must be an ISO-8601 timestamp") from exc


def validate_document(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping):
        raise IntelligenceValidationError("intelligence document must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise IntelligenceValidationError(f"unsupported schema_version: {document.get('schema_version')!r}")
    records = document.get("records")
    if not isinstance(records, list):
        raise IntelligenceValidationError("records must be an array")
    seen: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"records[{index}]"
        if not isinstance(record, Mapping):
            raise IntelligenceValidationError(f"{prefix} must be an object")
        for field in ("id", "kind", "statement", "status", "source", "source_ref", "created_at", "updated_at"):
            if not isinstance(record.get(field), str):
                raise IntelligenceValidationError(f"{prefix}.{field} must be a string")
        if not record["id"] or record["id"] in seen:
            raise IntelligenceValidationError(f"{prefix}.id must be non-empty and unique")
        seen.add(record["id"])
        try:
            RecordKind(record["kind"])
            RecordStatus(record["status"])
        except ValueError as exc:
            raise IntelligenceValidationError(f"{prefix} has an unknown kind or status") from exc
        if not record["statement"].strip():
            raise IntelligenceValidationError(f"{prefix}.statement must not be empty")
        confidence = record.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise IntelligenceValidationError(f"{prefix}.confidence must be between 0 and 1")
        supersedes = record.get("supersedes")
        if not isinstance(supersedes, list) or not all(isinstance(item, str) for item in supersedes):
            raise IntelligenceValidationError(f"{prefix}.supersedes must be an array of IDs")
        if not isinstance(record.get("details"), Mapping):
            raise IntelligenceValidationError(f"{prefix}.details must be an object")
        _timestamp(record["created_at"], f"{prefix}.created_at")
        _timestamp(record["updated_at"], f"{prefix}.updated_at")
    hashes = document.get("view_hashes", {})
    if not isinstance(hashes, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in hashes.items()):
        raise IntelligenceValidationError("view_hashes must be an object of strings")

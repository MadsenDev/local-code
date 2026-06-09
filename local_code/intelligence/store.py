"""Versioned, atomic persistence for repository intelligence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .markdown import VIEW_SECTIONS, content_hash, parse_legacy_view, parse_view, render_view
from .records import IntelligenceRecord, RecordKind, RecordStatus, record_from_dict, stable_record_id, utc_now
from .schema import SCHEMA_VERSION, IntelligenceValidationError, validate_document
from .storage import StorageScope, validate_shareable_record

INTELLIGENCE_FILENAME = "intelligence.json"


def atomic_write_text(path: Path, content: str) -> None:
    """Durably replace a text file without exposing a partially-written value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(slots=True)
class IntelligenceStore:
    base_path: Path
    scope: StorageScope = StorageScope.LOCAL
    records: dict[str, IntelligenceRecord] = field(default_factory=dict)
    view_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.base_path / INTELLIGENCE_FILENAME

    @classmethod
    def load(
        cls,
        base_path: str | Path,
        *,
        sync_views: bool = True,
        scope: StorageScope | str = StorageScope.LOCAL,
    ) -> "IntelligenceStore":
        base = Path(base_path)
        storage_scope = StorageScope(scope)
        base.mkdir(parents=True, exist_ok=True)
        path = base / INTELLIGENCE_FILENAME
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise IntelligenceValidationError(f"invalid JSON in {path}: {exc}") from exc
            document = migrate_document(raw)
            validate_document(document)
            store = cls(
                base_path=base,
                scope=storage_scope,
                records={item["id"]: record_from_dict(item) for item in document["records"]},
                view_hashes=dict(document.get("view_hashes", {})),
            )
        else:
            store = cls(base_path=base, scope=storage_scope)
            store._import_legacy_views()
        if sync_views:
            store.sync_markdown_views()
        return store

    def values(self, *, kinds: Iterable[RecordKind] | None = None, include_archived: bool = False) -> list[IntelligenceRecord]:
        selected = set(kinds) if kinds is not None else None
        return [
            record
            for record in self.records.values()
            if (selected is None or record.kind in selected)
            and (include_archived or record.status != RecordStatus.ARCHIVED)
        ]

    def upsert(self, record: IntelligenceRecord) -> None:
        self.records[record.id] = record

    def _validate_scope(self) -> None:
        if self.scope == StorageScope.PROJECT:
            for record in self.records.values():
                validate_shareable_record(record)

    def save(self) -> None:
        self._validate_scope()
        document = self.to_document()
        validate_document(document)
        atomic_write_text(self.path, json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "records": [record.to_dict() for record in sorted(self.records.values(), key=lambda item: item.id)],
            "view_hashes": dict(sorted(self.view_hashes.items())),
        }

    def sync_markdown_views(self) -> None:
        """Import changed human views, then normalize all views from structured data."""
        for filename in VIEW_SECTIONS:
            path = self.base_path / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if self.view_hashes.get(filename) == content_hash(text):
                continue
            for record in parse_view(filename, text, self.records):
                self.upsert(record)
        self._validate_scope()
        for filename in VIEW_SECTIONS:
            rendered = render_view(filename, self.records.values())
            atomic_write_text(self.base_path / filename, rendered)
            self.view_hashes[filename] = content_hash(rendered)
        self.save()

    def _import_legacy_views(self) -> None:
        for filename in VIEW_SECTIONS:
            path = self.base_path / filename
            if not path.exists():
                continue
            for record in parse_legacy_view(filename, path.read_text(encoding="utf-8", errors="replace")):
                self.records.setdefault(record.id, record)


def migrate_document(raw: Any) -> dict[str, Any]:
    """Upgrade all previously-supported loose/version-0 shapes to version 1."""
    if isinstance(raw, list):
        raw = {"schema_version": 0, "records": raw}
    if not isinstance(raw, Mapping):
        raise IntelligenceValidationError("intelligence document must be an object")
    version = raw.get("schema_version", raw.get("version", 0))
    if version == SCHEMA_VERSION:
        return dict(raw)
    if version != 0:
        raise IntelligenceValidationError(f"unsupported schema_version: {version!r}")
    now = utc_now()
    aliases = {"identity": RecordKind.PROJECT_IDENTITY.value, "file_association": RecordKind.FILE_ASSOCIATION.value}
    migrated = []
    for item in raw.get("records", []):
        if not isinstance(item, Mapping):
            raise IntelligenceValidationError("version 0 records must be objects")
        statement = str(item.get("statement", item.get("text", ""))).strip()
        kind = aliases.get(str(item.get("kind", "fact")), str(item.get("kind", "fact")))
        source_ref = str(item.get("source_ref", ""))
        migrated.append({
            "id": str(item.get("id") or stable_record_id(kind, statement, source_ref)),
            "kind": kind,
            "statement": statement,
            "status": str(item.get("status", "active")),
            "confidence": item.get("confidence", 1.0),
            "source": str(item.get("source", "migration_v0")),
            "source_ref": source_ref,
            "created_at": str(item.get("created_at", now)),
            "updated_at": str(item.get("updated_at", item.get("created_at", now))),
            "supersedes": list(item.get("supersedes", [])),
            "details": dict(item.get("details", item.get("metadata", {}))),
        })
    return {"schema_version": SCHEMA_VERSION, "records": migrated, "view_hashes": dict(raw.get("view_hashes", {}))}

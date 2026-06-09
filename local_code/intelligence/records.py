"""Typed records for durable repository intelligence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, ClassVar, Mapping


class RecordKind(StrEnum):
    PROJECT_IDENTITY = "project_identity"
    PRINCIPLE = "principle"
    FACT = "fact"
    DECISION = "decision"
    COMPONENT = "component"
    RELATIONSHIP = "relationship"
    LIFECYCLE_STATUS = "lifecycle_status"
    CONVENTION = "convention"
    WORKFLOW = "workflow"
    FILE_ASSOCIATION = "file_association"


class RecordStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    ARCHIVED = "archived"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_record_id(kind: RecordKind | str, statement: str, source_ref: str = "") -> str:
    """Create a repeatable, readable ID for imported records."""
    kind_value = RecordKind(kind).value
    slug = re.sub(r"[^a-z0-9]+", "-", statement.lower()).strip("-")[:36] or "record"
    digest = hashlib.sha256(f"{kind_value}\0{statement.strip()}\0{source_ref}".encode()).hexdigest()[:10]
    return f"{kind_value}:{slug}:{digest}"


@dataclass(frozen=True, slots=True)
class IntelligenceRecord:
    """Common durable fields shared by every intelligence record."""

    KIND: ClassVar[RecordKind] = RecordKind.FACT

    id: str
    statement: str
    status: RecordStatus = RecordStatus.ACTIVE
    confidence: float = 1.0
    source: str = "user"
    source_ref: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    supersedes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> RecordKind:
        return self.KIND

    def with_updates(self, **changes: Any) -> "IntelligenceRecord":
        changes.setdefault("updated_at", utc_now())
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence,
            "source": self.source,
            "source_ref": self.source_ref,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "supersedes": list(self.supersedes),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ProjectIdentityRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.PROJECT_IDENTITY


@dataclass(frozen=True, slots=True)
class PrincipleRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.PRINCIPLE


@dataclass(frozen=True, slots=True)
class FactRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.FACT


@dataclass(frozen=True, slots=True)
class DecisionRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.DECISION


@dataclass(frozen=True, slots=True)
class ComponentRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.COMPONENT


@dataclass(frozen=True, slots=True)
class RelationshipRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.RELATIONSHIP


@dataclass(frozen=True, slots=True)
class LifecycleStatusRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.LIFECYCLE_STATUS


@dataclass(frozen=True, slots=True)
class ConventionRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.CONVENTION


@dataclass(frozen=True, slots=True)
class WorkflowRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.WORKFLOW


@dataclass(frozen=True, slots=True)
class FileAssociationRecord(IntelligenceRecord):
    KIND: ClassVar[RecordKind] = RecordKind.FILE_ASSOCIATION


RECORD_TYPES: dict[RecordKind, type[IntelligenceRecord]] = {
    cls.KIND: cls
    for cls in (
        ProjectIdentityRecord,
        PrincipleRecord,
        FactRecord,
        DecisionRecord,
        ComponentRecord,
        RelationshipRecord,
        LifecycleStatusRecord,
        ConventionRecord,
        WorkflowRecord,
        FileAssociationRecord,
    )
}


def record_from_dict(value: Mapping[str, Any]) -> IntelligenceRecord:
    kind = RecordKind(value["kind"])
    record_type = RECORD_TYPES[kind]
    return record_type(
        id=str(value["id"]),
        statement=str(value["statement"]),
        status=RecordStatus(value.get("status", RecordStatus.ACTIVE)),
        confidence=float(value.get("confidence", 1.0)),
        source=str(value.get("source", "unknown")),
        source_ref=str(value.get("source_ref", "")),
        created_at=str(value.get("created_at", utc_now())),
        updated_at=str(value.get("updated_at", value.get("created_at", utc_now()))),
        supersedes=tuple(str(item) for item in value.get("supersedes", ())),
        details=dict(value.get("details", {})),
    )

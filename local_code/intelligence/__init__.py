"""Structured, durable repository intelligence and editable Markdown views."""

from .records import (
    ComponentRecord,
    ConventionRecord,
    DecisionRecord,
    FactRecord,
    FileAssociationRecord,
    IntelligenceRecord,
    LifecycleStatusRecord,
    PrincipleRecord,
    ProjectIdentityRecord,
    RecordKind,
    RecordStatus,
    RelationshipRecord,
    WorkflowRecord,
    stable_record_id,
)
from .schema import SCHEMA_VERSION, IntelligenceValidationError, validate_document
from .store import IntelligenceStore, atomic_write_text

__all__ = [
    "ComponentRecord",
    "ConventionRecord",
    "DecisionRecord",
    "FactRecord",
    "FileAssociationRecord",
    "IntelligenceRecord",
    "IntelligenceStore",
    "IntelligenceValidationError",
    "LifecycleStatusRecord",
    "PrincipleRecord",
    "ProjectIdentityRecord",
    "RecordKind",
    "RecordStatus",
    "RelationshipRecord",
    "SCHEMA_VERSION",
    "WorkflowRecord",
    "atomic_write_text",
    "stable_record_id",
    "validate_document",
]

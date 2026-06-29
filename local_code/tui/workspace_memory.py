"""In-session Workspace Memory records for the Rist TUI Decision Browser."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable


class DecisionType(StrEnum):
    """Kinds of engineering memory shown by the Decision Browser."""

    PLAN = "plan"
    DECISION = "decision"
    ASSUMPTION = "assumption"
    QUESTION = "question"
    REJECTED = "rejected"
    OBSERVATION = "observation"
    WARNING = "warning"
    APPLIED_CHANGE = "applied_change"


class DecisionStatus(StrEnum):
    """Lifecycle state for a workspace-memory record."""

    ACTIVE = "active"
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DISMISSED = "dismissed"


@dataclass(frozen=True, slots=True)
class Decision:
    """A structured in-session engineering-memory entry.

    The public name remains Decision because the screen presents a Decision
    Browser, but this model is broad enough to carry plans, assumptions,
    questions, observations, warnings, and applied changes.
    """

    id: str
    timestamp: datetime
    type: DecisionType
    title: str
    summary: str = ""
    reason: str = ""
    confidence: float | None = None
    status: DecisionStatus = DecisionStatus.ACTIVE
    files: tuple[str, ...] = ()
    proposal_id: str | None = None
    conversation_anchor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches_text(self, query: str) -> bool:
        needle = query.strip().casefold()
        if not needle:
            return True
        haystack = "\n".join(
            [self.title, self.summary, self.reason, self.proposal_id or "", self.conversation_anchor or "", *self.files]
        ).casefold()
        return needle in haystack


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _items(values: Iterable[str] | None = None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in (values or ()) if str(item).strip()))


def _record_id(record_type: DecisionType, title: str, timestamp: datetime, seed: str = "") -> str:
    digest = hashlib.sha256(f"{record_type.value}\0{title}\0{timestamp.isoformat()}\0{seed}".encode()).hexdigest()[:10]
    return f"WM-{record_type.value.upper().replace('_', '-')}-{digest}"


@dataclass(slots=True)
class DecisionFilter:
    type: DecisionType | None = None
    status: DecisionStatus | None = None
    file: str | None = None
    search: str = ""
    min_confidence: float | None = None


class DecisionStore:
    """In-memory workspace memory for the current TUI session."""

    def __init__(self) -> None:
        self._records: list[Decision] = []

    def add(
        self,
        *,
        type: DecisionType | str,
        title: str,
        summary: str = "",
        reason: str = "",
        confidence: float | None = None,
        status: DecisionStatus | str = DecisionStatus.ACTIVE,
        files: Iterable[str] = (),
        proposal_id: str | None = None,
        conversation_anchor: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
        decision_id: str | None = None,
    ) -> Decision:
        record_type = DecisionType(type)
        ts = timestamp or _now()
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("decision title is required")
        record = Decision(
            id=decision_id or _record_id(record_type, clean_title, ts, proposal_id or conversation_anchor or ""),
            timestamp=ts,
            type=record_type,
            title=clean_title,
            summary=summary.strip(),
            reason=reason.strip(),
            confidence=confidence,
            status=DecisionStatus(status),
            files=_items(files),
            proposal_id=proposal_id,
            conversation_anchor=conversation_anchor,
            metadata=dict(metadata or {}),
        )
        self._records.append(record)
        return record

    def dismiss(self, decision_id: str) -> Decision:
        for index, record in enumerate(self._records):
            if record.id == decision_id:
                updated = replace(record, status=DecisionStatus.DISMISSED)
                self._records[index] = updated
                return updated
        raise KeyError(decision_id)

    def all(self) -> list[Decision]:
        return sorted(self._records, key=lambda item: item.timestamp, reverse=True)

    def filter(self, criteria: DecisionFilter | None = None) -> list[Decision]:
        criteria = criteria or DecisionFilter()
        records = self.all()
        if criteria.type is not None:
            records = [record for record in records if record.type == criteria.type]
        if criteria.status is not None:
            records = [record for record in records if record.status == criteria.status]
        if criteria.file:
            needle = criteria.file.casefold()
            records = [record for record in records if any(needle in path.casefold() for path in record.files)]
        if criteria.min_confidence is not None:
            records = [record for record in records if record.confidence is not None and record.confidence >= criteria.min_confidence]
        if criteria.search:
            records = [record for record in records if record.matches_text(criteria.search)]
        return records

    def ingest_activity_event(self, event: dict[str, Any]) -> Decision | None:
        """Convert existing ActivityTimeline events into workspace memory."""
        kind = event.get("kind")
        if kind == "milestone":
            return self.add(type=DecisionType.PLAN, title=str(event.get("text") or "Plan generated"), reason="Captured from activity timeline milestone.", status=DecisionStatus.PENDING)
        if kind == "apply":
            return self.add(type=DecisionType.DECISION, title="Proposal accepted", summary=str(event.get("text") or ""), reason="User accepted the pending proposal.", status=DecisionStatus.ACCEPTED, files=event.get("files") or ())
        if kind == "reject":
            return self.add(type=DecisionType.REJECTED, title="Proposal rejected", summary=str(event.get("text") or ""), reason="User rejected the pending proposal.", status=DecisionStatus.REJECTED, files=event.get("files") or ())
        if kind == "diff":
            title = f"Prepared diff for {event.get('files', 0)} file(s)"
            return self.add(type=DecisionType.APPLIED_CHANGE, title=title, summary=f"+{event.get('added', 0)} -{event.get('removed', 0)}", status=DecisionStatus.PENDING)
        if kind == "task_failed":
            return self.add(type=DecisionType.WARNING, title="Runtime task failed", summary=str(event.get("text") or ""), status=DecisionStatus.ACTIVE)
        if kind == "task_complete":
            return self.add(type=DecisionType.OBSERVATION, title="Runtime task completed", summary=str(event.get("text") or ""), status=DecisionStatus.ACTIVE)
        if kind == "question":
            return self.add(type=DecisionType.QUESTION, title=str(event.get("title") or "Waiting for user approval"), summary=str(event.get("text") or ""), status=DecisionStatus.PENDING)
        return None

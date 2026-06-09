"""Markdown views over structured repository intelligence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping

from .records import IntelligenceRecord, RecordKind, RecordStatus, RECORD_TYPES, stable_record_id

VIEW_SECTIONS: dict[str, tuple[tuple[str, RecordKind], ...]] = {
    "project.md": (
        ("Project Identity", RecordKind.PROJECT_IDENTITY),
        ("Principles", RecordKind.PRINCIPLE),
        ("Facts", RecordKind.FACT),
        ("Conventions", RecordKind.CONVENTION),
        ("Workflows", RecordKind.WORKFLOW),
        ("Learned File Associations", RecordKind.FILE_ASSOCIATION),
    ),
    "architecture.md": (
        ("Components", RecordKind.COMPONENT),
        ("Relationships", RecordKind.RELATIONSHIP),
        ("Lifecycle Status", RecordKind.LIFECYCLE_STATUS),
    ),
    "decisions.md": (("Decisions", RecordKind.DECISION),),
}

_TITLES = {"project.md": "Project Intelligence", "architecture.md": "Architecture", "decisions.md": "Decisions"}
_META_RE = re.compile(r"\s*<!--\s*rist:(\{.*\})\s*-->\s*$")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def render_view(filename: str, records: Iterable[IntelligenceRecord]) -> str:
    by_kind: dict[RecordKind, list[IntelligenceRecord]] = {}
    for record in records:
        if record.status != RecordStatus.ARCHIVED:
            by_kind.setdefault(record.kind, []).append(record)
    lines = [
        f"# {_TITLES[filename]}",
        "",
        "<!-- Rist view: edit bullet text/status/confidence; structured data remains authoritative. -->",
    ]
    for heading, kind in VIEW_SECTIONS[filename]:
        lines.extend(("", f"## {heading}", ""))
        items = sorted(by_kind.get(kind, ()), key=lambda item: (item.created_at, item.id))
        if not items:
            lines.append("_No records._")
            continue
        for record in items:
            metadata = json.dumps(
                {"id": record.id, "status": record.status.value, "confidence": record.confidence},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            statement = record.statement.replace("\n", " ").strip()
            lines.append(f"- {statement} <!-- rist:{metadata} -->")
    return "\n".join(lines).rstrip() + "\n"


def parse_view(filename: str, text: str, existing: Mapping[str, IntelligenceRecord]) -> list[IntelligenceRecord]:
    """Parse supported bullet edits, preserving hidden structured fields by stable ID."""
    heading_to_kind = {heading.casefold(): kind for heading, kind in VIEW_SECTIONS[filename]}
    current_kind: RecordKind | None = None
    parsed: list[IntelligenceRecord] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current_kind = heading_to_kind.get(line[3:].strip().casefold())
            continue
        if current_kind is None or not line.startswith(('- ', '* ')):
            continue
        statement = line[2:].strip()
        metadata: dict[str, object] = {}
        if match := _META_RE.search(statement):
            try:
                metadata = json.loads(match.group(1))
            except json.JSONDecodeError:
                metadata = {}
            statement = statement[: match.start()].strip()
        if not statement or statement.endswith(":"):
            continue
        record_id = str(metadata.get("id") or stable_record_id(current_kind, statement, filename))
        old = existing.get(record_id)
        status_value = metadata.get("status", old.status.value if old else RecordStatus.ACTIVE.value)
        confidence = metadata.get("confidence", old.confidence if old else 0.8)
        try:
            status = RecordStatus(str(status_value))
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        confidence_value = min(1.0, max(0.0, confidence_value))
        if old and old.kind == current_kind:
            parsed.append(old.with_updates(statement=statement, status=status, confidence=confidence_value))
        else:
            parsed.append(RECORD_TYPES[current_kind](
                id=record_id,
                statement=statement,
                status=status,
                confidence=confidence_value,
                source="markdown_edit",
                source_ref=filename,
            ))
    return parsed


def parse_legacy_view(filename: str, text: str) -> list[IntelligenceRecord]:
    """Best-effort import of the Markdown files created by ensure_memory_files."""
    default_kind = {
        "project.md": RecordKind.FACT,
        "architecture.md": RecordKind.COMPONENT,
        "decisions.md": RecordKind.DECISION,
    }[filename]
    records: list[IntelligenceRecord] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("<!--") or line == "_No records._":
            continue
        statement = line[2:].strip() if line.startswith(("- ", "* ")) else line
        if not statement or statement.endswith(":"):
            continue
        record_id = stable_record_id(default_kind, statement, filename)
        records.append(RECORD_TYPES[default_kind](
            id=record_id,
            statement=statement,
            confidence=0.7,
            source="legacy_markdown",
            source_ref=filename,
        ))
    return records

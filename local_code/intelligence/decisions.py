"""Reviewable architecture and implementation decision service.

Decision candidates produced by a model are deliberately kept separate from
accepted records until a person reviews them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .store import atomic_write_text

DECISIONS_FILENAME = "decisions.json"
DECISIONS_MARKDOWN = "decisions.md"


class DecisionStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class Decision:
    id: str
    title: str
    status: DecisionStatus
    date: str
    rationale: str
    alternatives: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()
    affected_components: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    superseding_decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        for key in ("alternatives", "consequences", "affected_components", "source_references"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True, slots=True)
class DecisionCandidate:
    id: str
    title: str
    rationale: str
    alternatives: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()
    affected_components: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    source_run: str = ""
    extraction_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("alternatives", "consequences", "affected_components", "source_references"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True, slots=True)
class DecisionConflict:
    decision_id: str
    candidate_id: str | None
    kind: str
    score: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _items(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:48] or "decision"


def stable_decision_id(title: str, *, seed: str = "") -> str:
    digest = hashlib.sha256(f"{title.strip()}\0{seed}".encode()).hexdigest()[:8]
    return f"DEC-{_slug(title)}-{digest}"


def _tokens(*values: str) -> set[str]:
    stop = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "use", "using", "with"}
    return {word for value in values for word in re.findall(r"[a-z0-9_.-]+", value.casefold()) if len(word) > 1 and word not in stop}


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _opposes(left: str, right: str) -> bool:
    negations = (("not ", ""), ("avoid ", "use "), ("disable ", "enable "), ("remove ", "add "), ("reject ", "adopt "))
    a, b = left.casefold(), right.casefold()
    return any((x in a and y in b) or (x in b and y in a) for x, y in negations if x and y) or ((" not " in f" {a} ") != (" not " in f" {b} ") and _similarity(_tokens(a), _tokens(b)) >= 0.45)


class DecisionService:
    """Persist decisions, pending candidates, conflicts, and the Markdown view."""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.path = self.base_path / DECISIONS_FILENAME
        self.markdown_path = self.base_path / DECISIONS_MARKDOWN
        self.decisions: dict[str, Decision] = {}
        self.pending: dict[str, DecisionCandidate] = {}

    @classmethod
    def load(cls, base_path: str | Path) -> "DecisionService":
        service = cls(base_path)
        service.base_path.mkdir(parents=True, exist_ok=True)
        if service.path.exists():
            raw = json.loads(service.path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("decision document must be an object")
            for item in raw.get("decisions", []):
                decision = service._decision_from_mapping(item)
                service.decisions[decision.id] = decision
            for item in raw.get("pending", []):
                candidate = service._candidate_from_mapping(item)
                service.pending[candidate.id] = candidate
        service.reconcile_markdown()
        return service

    @staticmethod
    def _decision_from_mapping(value: Mapping[str, Any]) -> Decision:
        return Decision(
            id=str(value["id"]), title=str(value["title"]).strip(), status=DecisionStatus(value.get("status", "proposed")),
            date=str(value.get("date") or date.today().isoformat()), rationale=str(value.get("rationale", "")).strip(),
            alternatives=_items(value.get("alternatives")), consequences=_items(value.get("consequences")),
            affected_components=_items(value.get("affected_components")), source_references=_items(value.get("source_references")),
            superseding_decision=str(value["superseding_decision"]) if value.get("superseding_decision") else None,
        )

    @staticmethod
    def _candidate_from_mapping(value: Mapping[str, Any]) -> DecisionCandidate:
        title = str(value.get("title", "")).strip()
        if not title:
            raise ValueError("decision candidate title is required")
        return DecisionCandidate(
            id=str(value.get("id") or stable_decision_id(title, seed=str(value.get("source_run", "candidate")))),
            title=title, rationale=str(value.get("rationale", "")).strip(), alternatives=_items(value.get("alternatives")),
            consequences=_items(value.get("consequences")), affected_components=_items(value.get("affected_components")),
            source_references=_items(value.get("source_references")), source_run=str(value.get("source_run", "")),
            extraction_error=str(value["extraction_error"]) if value.get("extraction_error") else None,
        )

    def save(self) -> None:
        document = {
            "schema_version": 1,
            "decisions": [self.decisions[key].to_dict() for key in sorted(self.decisions)],
            "pending": [self.pending[key].to_dict() for key in sorted(self.pending)],
        }
        atomic_write_text(self.path, json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        atomic_write_text(self.markdown_path, self.render_markdown())

    def add(self, *, title: str, rationale: str, status: DecisionStatus | str = DecisionStatus.PROPOSED,
            alternatives: Iterable[str] = (), consequences: Iterable[str] = (), affected_components: Iterable[str] = (),
            source_references: Iterable[str] = (), decision_id: str | None = None, decision_date: str | None = None) -> Decision:
        title = title.strip()
        if not title:
            raise ValueError("decision title is required")
        duplicate = self.find_duplicate(title, affected_components)
        if duplicate:
            raise ValueError(f"duplicate decision: {duplicate.id}")
        decision = Decision(
            id=decision_id or stable_decision_id(title), title=title, status=DecisionStatus(status),
            date=decision_date or date.today().isoformat(), rationale=rationale.strip(), alternatives=_items(tuple(alternatives)),
            consequences=_items(tuple(consequences)), affected_components=_items(tuple(affected_components)),
            source_references=_items(tuple(source_references)),
        )
        self.decisions[decision.id] = decision
        self.save()
        return decision

    def find_duplicate(self, title: str, components: Iterable[str] = ()) -> Decision | None:
        title_tokens = _tokens(title)
        component_set = {item.casefold() for item in components}
        for decision in self.decisions.values():
            if decision.title.casefold() == title.casefold():
                return decision
            overlap = component_set & {item.casefold() for item in decision.affected_components}
            if _similarity(title_tokens, _tokens(decision.title)) >= 0.9 and (not component_set or overlap):
                return decision
        return None

    def accept(self, decision_id: str) -> Decision:
        if decision_id in self.pending:
            candidate = self.pending.pop(decision_id)
            duplicate = self.find_duplicate(candidate.title, candidate.affected_components)
            if duplicate:
                raise ValueError(f"duplicate decision: {duplicate.id}")
            decision = Decision(candidate.id, candidate.title, DecisionStatus.ACCEPTED, date.today().isoformat(), candidate.rationale,
                                candidate.alternatives, candidate.consequences, candidate.affected_components, candidate.source_references)
            self.decisions[decision.id] = decision
        else:
            current = self.decisions[decision_id]
            decision = replace(current, status=DecisionStatus.ACCEPTED)
            self.decisions[decision_id] = decision
        self.save()
        return decision

    def reject(self, candidate_id: str, rationale: str = "") -> Decision:
        candidate = self.pending.pop(candidate_id)
        decision = Decision(candidate.id, candidate.title, DecisionStatus.REJECTED, date.today().isoformat(), rationale.strip() or candidate.rationale,
                            candidate.alternatives, candidate.consequences, candidate.affected_components, candidate.source_references)
        self.decisions[decision.id] = decision
        self.save()
        return decision

    def edit_candidate(self, candidate_id: str, **changes: Any) -> DecisionCandidate:
        candidate = self.pending[candidate_id]
        allowed = {key: value for key, value in changes.items() if value is not None and key in candidate.__dataclass_fields__ and key != "id"}
        for key in ("alternatives", "consequences", "affected_components", "source_references"):
            if key in allowed:
                allowed[key] = _items(allowed[key])
        updated = replace(candidate, **allowed)
        self.pending[candidate_id] = updated
        self.save()
        return updated

    def merge_candidates(self, candidate_ids: Iterable[str], *, title: str | None = None) -> DecisionCandidate:
        candidates = [self.pending[item] for item in candidate_ids]
        if len(candidates) < 2:
            raise ValueError("merge requires at least two candidates")
        merged_title = title or candidates[0].title
        merged = DecisionCandidate(
            id=stable_decision_id(merged_title, seed="merge:" + ",".join(sorted(c.id for c in candidates))), title=merged_title,
            rationale="\n\n".join(dict.fromkeys(c.rationale for c in candidates if c.rationale)),
            alternatives=_items([item for c in candidates for item in c.alternatives]), consequences=_items([item for c in candidates for item in c.consequences]),
            affected_components=_items([item for c in candidates for item in c.affected_components]),
            source_references=_items([item for c in candidates for item in c.source_references]), source_run=",".join(c.source_run for c in candidates if c.source_run),
        )
        for candidate in candidates:
            del self.pending[candidate.id]
        self.pending[merged.id] = merged
        self.save()
        return merged

    def supersede(self, old_id: str, new_id: str) -> tuple[Decision, Decision]:
        if old_id == new_id:
            raise ValueError("a decision cannot supersede itself")
        old, new = self.decisions[old_id], self.decisions[new_id]
        if new.status != DecisionStatus.ACCEPTED:
            new = replace(new, status=DecisionStatus.ACCEPTED)
            self.decisions[new_id] = new
        old = replace(old, status=DecisionStatus.SUPERSEDED, superseding_decision=new_id)
        self.decisions[old_id] = old
        self.save()
        return old, new

    def extract_candidates(self, model_output: Any, *, source_run: str = "") -> list[DecisionCandidate]:
        """Validate model extraction; malformed entries are ignored, never promoted."""
        if isinstance(model_output, str):
            try:
                model_output = json.loads(model_output)
            except json.JSONDecodeError:
                return []
        if isinstance(model_output, Mapping):
            model_output = model_output.get("decision_candidates", [])
        if not isinstance(model_output, list):
            return []
        added = []
        for raw in model_output:
            if not isinstance(raw, Mapping):
                continue
            try:
                value = dict(raw)
                value["source_run"] = source_run or value.get("source_run", "")
                candidate = self._candidate_from_mapping(value)
            except (KeyError, TypeError, ValueError):
                continue
            if self.find_duplicate(candidate.title, candidate.affected_components):
                continue
            pending_duplicate = next((item for item in self.pending.values() if item.title.casefold() == candidate.title.casefold()), None)
            if pending_duplicate:
                continue
            self.pending[candidate.id] = candidate
            added.append(candidate)
        if added:
            self.save()
        return added

    def relevant(self, task: str, *, limit: int = 8) -> list[Decision]:
        task_tokens = _tokens(task)
        ranked = []
        for decision in self.decisions.values():
            if decision.status != DecisionStatus.ACCEPTED:
                continue
            score = _similarity(task_tokens, _tokens(decision.title, decision.rationale, *decision.affected_components))
            if score > 0:
                ranked.append((score, decision.id, decision))
        return [item[2] for item in sorted(ranked, reverse=True)[:limit]]

    def conflicts_for(self, task: str, candidates: Iterable[DecisionCandidate | Mapping[str, Any]] = ()) -> list[DecisionConflict]:
        relevant = self.relevant(task, limit=50)
        conflicts: list[DecisionConflict] = []
        normalized_candidates = [candidate if isinstance(candidate, DecisionCandidate) else self._candidate_from_mapping(candidate) for candidate in candidates]
        for candidate in normalized_candidates:
            candidate_tokens = _tokens(candidate.title, candidate.rationale, *candidate.affected_components)
            for decision in relevant:
                component_overlap = set(map(str.casefold, candidate.affected_components)) & set(map(str.casefold, decision.affected_components))
                score = _similarity(candidate_tokens, _tokens(decision.title, decision.rationale, *decision.affected_components))
                if component_overlap and _opposes(candidate.title + " " + candidate.rationale, decision.title + " " + decision.rationale):
                    conflicts.append(DecisionConflict(decision.id, candidate.id, "deterministic", 1.0, "Opposing direction for an affected component."))
                elif score >= 0.55 and _opposes(candidate.title + " " + candidate.rationale, decision.title + " " + decision.rationale):
                    conflicts.append(DecisionConflict(decision.id, candidate.id, "semantic", round(score, 3), "Semantically similar decision appears contradictory."))
        # The task itself is checked so conflicts are available before a model has proposed candidates.
        for decision in relevant:
            score = _similarity(_tokens(task), _tokens(decision.title, decision.rationale, *decision.affected_components))
            if score >= 0.35 and _opposes(task, decision.title + " " + decision.rationale):
                conflicts.append(DecisionConflict(decision.id, None, "task", round(score, 3), "The requested task may deviate from an accepted decision."))
        unique = {(c.decision_id, c.candidate_id, c.kind): c for c in conflicts}
        return list(unique.values())

    def contract_context(self, task: str) -> dict[str, Any]:
        decisions = self.relevant(task)
        conflicts = self.conflicts_for(task)
        return {
            "relevant_decisions": [decision.to_dict() for decision in decisions],
            "decision_conflicts": [conflict.to_dict() for conflict in conflicts],
            "requires_deviation_explanation": bool(conflicts),
        }

    def render_markdown(self) -> str:
        lines = ["# Decisions", "", "<!-- Rist decisions: IDs are stable. Edit fields below; structured storage is reconciled by ID. -->", ""]
        accepted = [item for item in self.decisions.values() if item.status == DecisionStatus.ACCEPTED]
        if not accepted:
            lines.append("_No accepted decisions._")
        for decision in sorted(accepted, key=lambda item: (item.date, item.id)):
            lines.extend([
                f"## {decision.id}: {decision.title}", "", f"- **Status:** {decision.status.value}", f"- **Date:** {decision.date}",
                f"- **Affected components:** {', '.join(decision.affected_components) or 'None'}", f"- **Source references:** {', '.join(decision.source_references) or 'None'}", "",
                "### Rationale", "", decision.rationale or "_None provided._", "", "### Alternatives", "",
                *([f"- {item}" for item in decision.alternatives] or ["- None"]), "", "### Consequences", "",
                *([f"- {item}" for item in decision.consequences] or ["- None"]), "",
            ])
        return "\n".join(lines).rstrip() + "\n"

    def reconcile_markdown(self) -> None:
        if not self.markdown_path.exists() or not self.decisions:
            return
        text = self.markdown_path.read_text(encoding="utf-8", errors="replace")
        headings = list(re.finditer(r"(?m)^## (DEC-[^:]+):\s*(.+?)\s*$", text))
        changed = False
        for index, match in enumerate(headings):
            decision_id, title = match.group(1), match.group(2).strip()
            old = self.decisions.get(decision_id)
            if not old:
                continue
            body = text[match.end():headings[index + 1].start() if index + 1 < len(headings) else len(text)]
            def field(label: str) -> str | None:
                found = re.search(rf"(?m)^- \*\*{re.escape(label)}:\*\*\s*(.*?)\s*$", body)
                return found.group(1).strip() if found else None
            rationale_match = re.search(r"(?s)### Rationale\s*\n+(.*?)(?=\n### Alternatives)", body)
            alternatives_match = re.search(r"(?s)### Alternatives\s*\n+(.*?)(?=\n### Consequences)", body)
            consequences_match = re.search(r"(?s)### Consequences\s*\n+(.*)$", body)
            bullets = lambda matched: tuple(line[2:].strip() for line in matched.group(1).splitlines() if line.startswith("- ") and line[2:].strip() != "None") if matched else ()
            status_text = field("Status") or old.status.value
            try:
                status = DecisionStatus(status_text)
            except ValueError:
                status = old.status
            updated = replace(
                old, title=title, status=status, date=field("Date") or old.date,
                rationale=(rationale_match.group(1).strip() if rationale_match else old.rationale).replace("_None provided._", ""),
                alternatives=bullets(alternatives_match), consequences=bullets(consequences_match),
                affected_components=_items(field("Affected components") or old.affected_components),
                source_references=_items(field("Source references") or old.source_references),
            )
            if updated != old:
                self.decisions[decision_id] = updated
                changed = True
        if changed:
            self.save()


def execute_decision_command(service: DecisionService, command: str) -> str:
    """Execute the compact slash-command grammar used by both interactive UIs."""
    import shlex

    parts = shlex.split(command)
    if parts and parts[0] in {"decisions", "/decisions"}:
        parts = parts[1:]
    action = parts[0] if parts else "list"
    args = parts[1:]
    if action == "list":
        if not service.decisions:
            return "No decisions."
        return "\n".join(f"{item.id} [{item.status.value}] {item.title}" for item in sorted(service.decisions.values(), key=lambda value: value.id))
    if action == "add":
        if not args:
            raise ValueError("usage: /decisions add TITLE [| RATIONALE]")
        joined = " ".join(args)
        title, _, rationale = joined.partition("|")
        item = service.add(title=title.strip(), rationale=rationale.strip())
        return f"Added {item.id} [proposed] {item.title}"
    if action == "accept" and args:
        item = service.accept(args[0])
        return f"Accepted {item.id}: {item.title}"
    if action == "supersede" and len(args) == 2:
        old, new = service.supersede(args[0], args[1])
        return f"{old.id} superseded by {new.id}"
    if action == "review":
        if not args:
            if not service.pending:
                return "No pending decision candidates."
            return "\n".join(f"{item.id} [pending] {item.title}" for item in sorted(service.pending.values(), key=lambda value: value.id))
        verb = args[0]
        if verb == "accept" and len(args) == 2:
            item = service.accept(args[1])
            return f"Accepted {item.id}: {item.title}"
        if verb == "reject" and len(args) >= 2:
            item = service.reject(args[1], " ".join(args[2:]))
            return f"Rejected {item.id}: {item.title}"
        if verb == "edit" and len(args) >= 3:
            joined = " ".join(args[2:])
            title, _, rationale = joined.partition("|")
            item = service.edit_candidate(args[1], title=title.strip(), rationale=rationale.strip() or None)
            return f"Updated pending candidate {item.id}: {item.title}"
        if verb == "merge" and len(args) >= 3:
            item = service.merge_candidates(args[1:])
            return f"Merged into pending candidate {item.id}: {item.title}"
    raise ValueError("usage: /decisions list|add|accept|supersede|review")

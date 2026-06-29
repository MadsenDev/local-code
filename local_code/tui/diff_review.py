"""Structured diff review models for Rist TUI proposals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewFile:
    """One file in a pending proposal review."""

    filename: str
    added: int = 0
    removed: int = 0
    diff: str = ""


@dataclass(frozen=True)
class ReviewSummary:
    """Aggregate review information shown before applying changes."""

    files: int
    added: int
    removed: int
    impact: str


@dataclass(frozen=True)
class ReviewModel:
    """Complete structured data consumed by the review screen."""

    repository: str
    files: tuple[ReviewFile, ...]
    summary: ReviewSummary
    plan: tuple[str, ...] = ()


def build_review_model(partner) -> ReviewModel | None:
    """Build a review model from the partner's pending proposal/report data."""

    pending = getattr(partner, "pending_plan", None)
    report = (pending or {}).get("report") or getattr(partner, "last_report", None) or {}
    if not (pending or report.get("needs_approval")):
        return None
    diff_summary = str(report.get("diff_summary") or "")
    files = parse_unified_diff(diff_summary)
    if not files:
        files = tuple(ReviewFile(filename=str(path)) for path in (report.get("files_changed") or []))
    summary = summarize_review(files)
    repository = getattr(partner, "workdir", "") or "repository"
    return ReviewModel(repository=repository, files=files, summary=summary, plan=tuple(report.get("plan") or ()))


def summarize_review(files: tuple[ReviewFile, ...] | list[ReviewFile]) -> ReviewSummary:
    """Summarize review files without inventing unavailable details."""

    added = sum(file.added for file in files)
    removed = sum(file.removed for file in files)
    count = len(files)
    churn = added + removed
    if count == 0:
        impact = "No file details available"
    elif churn < 25:
        impact = "Low"
    elif churn < 150:
        impact = "Moderate"
    else:
        impact = "High"
    return ReviewSummary(files=count, added=added, removed=removed, impact=impact)


def parse_unified_diff(diff: str) -> tuple[ReviewFile, ...]:
    """Parse a unified diff into per-file review entries."""

    files: list[ReviewFile] = []
    current_name: str | None = None
    current_lines: list[str] = []
    added = removed = 0

    def finish() -> None:
        nonlocal current_name, current_lines, added, removed
        if current_name is not None:
            files.append(ReviewFile(current_name, added, removed, "\n".join(current_lines).rstrip()))
        current_name = None
        current_lines = []
        added = removed = 0

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            finish()
            parts = line.split()
            current_name = _clean_name(parts[-1] if len(parts) >= 4 else parts[-1])
            current_lines = [line]
            continue
        if line.startswith("--- ") and current_name is None:
            current_name = _clean_name(line[4:].strip())
            current_lines = [line]
            continue
        if current_name is None:
            continue
        if line.startswith("+++ "):
            name = _clean_name(line[4:].strip())
            if name != "/dev/null":
                current_name = name
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
        current_lines.append(line)
    finish()
    return tuple(files)


def _clean_name(name: str) -> str:
    name = name.split("\t", 1)[0].strip()
    if name.startswith("a/") or name.startswith("b/"):
        return name[2:]
    return name

"""Repository Explorer data models for Rist TUI session visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path


class RepositoryBadge(str, Enum):
    """Transient Rist session badge attached to a repository file."""

    READ = "READ"
    EDITED = "EDITED"
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"


@dataclass
class RepositoryNode:
    """One directory or file in the repository tree."""

    name: str
    path: str
    is_dir: bool
    children: list["RepositoryNode"] = field(default_factory=list)
    expanded: bool = False
    badges: set[RepositoryBadge] = field(default_factory=set)
    size: int | None = None
    language: str = ""
    modified: float | None = None

    def find(self, path: str) -> "RepositoryNode | None":
        if self.path == path:
            return self
        for child in self.children:
            found = child.find(path)
            if found is not None:
                return found
        return None


@dataclass
class RepositorySelection:
    """Selection state independent of the presentation widget."""

    index: int = 0
    query: str = ""


@dataclass
class SessionFileState:
    """AI session footprint for a repository file."""

    read_count: int = 0
    edited_count: int = 0
    proposed: bool = False
    active: bool = False

    def badges(self) -> set[RepositoryBadge]:
        badges: set[RepositoryBadge] = set()
        if self.read_count:
            badges.add(RepositoryBadge.READ)
        if self.edited_count:
            badges.add(RepositoryBadge.EDITED)
        if self.proposed:
            badges.add(RepositoryBadge.PROPOSED)
        if self.active:
            badges.add(RepositoryBadge.ACTIVE)
        return badges


class RepositoryTree:
    """Cached repository tree plus transient Rist session metadata."""

    def __init__(self, root: Path | str, *, max_depth: int = 4, max_entries: int = 2000):
        self.root_path = Path(root).resolve()
        self.max_depth = max_depth
        self.max_entries = max_entries
        self.session: dict[str, SessionFileState] = {}
        self.root = RepositoryNode(self.root_path.name or str(self.root_path), "", True, expanded=True)
        self._built = False

    def build(self, *, force: bool = False) -> RepositoryNode:
        if self._built and not force:
            return self.root
        count = 0

        def walk(directory: Path, rel: str, depth: int) -> list[RepositoryNode]:
            nonlocal count
            if depth > self.max_depth or count >= self.max_entries:
                return []
            nodes: list[RepositoryNode] = []
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return []
            for entry in entries:
                if entry.name in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                    continue
                count += 1
                child_rel = f"{rel}/{entry.name}" if rel else entry.name
                is_dir = entry.is_dir()
                node = RepositoryNode(entry.name, child_rel, is_dir)
                if is_dir:
                    node.children = walk(entry, child_rel, depth + 1)
                else:
                    try:
                        stat = entry.stat()
                        node.size = stat.st_size
                        node.modified = stat.st_mtime
                    except OSError:
                        pass
                    node.language = language_for(entry)
                    node.badges = self.session.get(child_rel, SessionFileState()).badges()
                nodes.append(node)
                if count >= self.max_entries:
                    break
            return nodes

        self.root.children = walk(self.root_path, "", 1)
        self._built = True
        self.assign_badges()
        return self.root

    def ingest_report(self, report: dict | None) -> None:
        if not report:
            return
        for path in report.get("files_read") or []:
            self.mark_read(str(path))
        for path in report.get("files_changed") or []:
            self.mark_edited(str(path))
        if report.get("needs_approval"):
            self.set_proposed(report.get("files_changed") or [])
        self.assign_badges()

    def set_proposed(self, paths) -> None:
        for state in self.session.values():
            state.proposed = False
        for path in paths or []:
            self._state(str(path)).proposed = True
        self.assign_badges()

    def apply_proposal(self) -> None:
        for state in self.session.values():
            if state.proposed:
                state.proposed = False
                state.edited_count += 1
        self.assign_badges()

    def clear_proposal(self) -> None:
        for state in self.session.values():
            state.proposed = False
        self.assign_badges()

    def mark_read(self, path: str) -> None:
        self._state(path).read_count += 1

    def mark_edited(self, path: str) -> None:
        self._state(path).edited_count += 1

    def assign_badges(self) -> None:
        self._assign(self.root)

    def visible_nodes(self, query: str = "", *, badge: RepositoryBadge | None = None) -> list[RepositoryNode]:
        self.build()
        nodes: list[RepositoryNode] = []
        query = query.strip().lower()

        def include(node: RepositoryNode) -> bool:
            if badge is not None and badge not in node.badges:
                return False
            if not query:
                return True
            hay = f"{node.path} {node.name}".lower()
            if query in hay:
                return True
            return _fuzzy(query, hay) > 0.45

        def visit(node: RepositoryNode) -> None:
            if node is not self.root and include(node):
                nodes.append(node)
            if query or node.expanded or node is self.root:
                for child in node.children:
                    visit(child)

        visit(self.root)
        return nodes

    def preview(self, path: str, *, limit: int = 80, diff: str = "") -> str:
        if diff:
            return "\n".join(diff.splitlines()[:limit])
        target = (self.root_path / path).resolve()
        try:
            target.relative_to(self.root_path)
        except ValueError:
            return "Outside repository."
        if not target.is_file():
            return "Directory. Press Enter to expand or collapse."
        try:
            data = target.read_text(errors="replace").splitlines()
        except OSError as exc:
            return f"Unable to load preview: {exc}"
        return "\n".join(data[:limit]) or "Empty file."

    def _state(self, path: str) -> SessionFileState:
        clean = path.replace("\\", "/").lstrip("./")
        return self.session.setdefault(clean, SessionFileState())

    def _assign(self, node: RepositoryNode) -> set[RepositoryBadge]:
        if not node.is_dir:
            node.badges = self.session.get(node.path, SessionFileState()).badges()
            return set(node.badges)
        badges: set[RepositoryBadge] = set()
        for child in node.children:
            badges |= self._assign(child)
        node.badges = badges
        return badges


def language_for(path: Path) -> str:
    mapping = {".py": "Python", ".md": "Markdown", ".toml": "TOML", ".json": "JSON", ".yml": "YAML", ".yaml": "YAML", ".css": "CSS"}
    return mapping.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "text")


def _fuzzy(query: str, text: str) -> float:
    if not query:
        return 1.0
    pos = -1
    hits = 0
    for char in query:
        pos = text.find(char, pos + 1)
        if pos < 0:
            return 0.0
        hits += 1
    return (hits / max(len(query), 1) + SequenceMatcher(None, query, text).ratio()) / 2

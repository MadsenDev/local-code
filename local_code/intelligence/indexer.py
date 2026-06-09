"""Deterministic, Git-aware repository indexing for Rist."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..tools import ENTRYPOINTS, NOISY_DIRS
from .graph import build_dependency_graph, render_architecture

SCHEMA_VERSION = 1
DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024
ARTIFACT_NAMES = ("repo-map.json", "manifests.json", "conventions.json", "dependency-graph.json")
ARCHITECTURE_VIEW = "architecture.md"
PUBLISHED_NAMES = (*ARTIFACT_NAMES, ARCHITECTURE_VIEW)
RIST_EXCLUDED_DIRS = frozenset({".rist"})
EXCLUDED_DIRS = frozenset(NOISY_DIRS) | RIST_EXCLUDED_DIRS

LANGUAGES = {
    ".py": "Python", ".pyi": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".hpp": "C++",
    ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".scala": "Scala",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".fish": "Shell",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
    ".vue": "Vue", ".svelte": "Svelte", ".sql": "SQL", ".proto": "Protocol Buffers",
    ".ex": "Elixir", ".exs": "Elixir", ".erl": "Erlang", ".hrl": "Erlang",
}

MANIFEST_NAMES = frozenset({
    "package.json", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "Pipfile", "poetry.lock", "uv.lock", "Cargo.toml", "go.mod", "pom.xml",
    "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
    "Gemfile", "composer.json", "mix.exs", "Package.swift", "*.csproj", "*.sln",
    "pnpm-workspace.yaml", "lerna.json", "nx.json", "turbo.json",
})
CONFIG_NAMES = frozenset({
    ".editorconfig", ".prettierrc", ".prettierrc.json", ".prettierrc.js", ".eslintrc",
    ".eslintrc.json", "eslint.config.js", "eslint.config.mjs", "tsconfig.json", "jsconfig.json",
    "vite.config.js", "vite.config.ts", "next.config.js", "next.config.mjs", "webpack.config.js",
    "pytest.ini", "tox.ini", "mypy.ini", "ruff.toml", ".ruff.toml", "Dockerfile",
    "docker-compose.yml", "docker-compose.yaml", "Makefile", "Justfile", "Taskfile.yml",
    ".github", ".gitlab-ci.yml", "Cargo.toml", "pyproject.toml", "package.json",
})
TEST_PARTS = frozenset({"test", "tests", "spec", "specs", "__tests__", "testdata", "fixtures"})
GENERATED_NAME_PATTERNS = (
    re.compile(r"(?:^|[._-])(generated|gen|min|bundle|vendor)(?:[._-]|$)", re.I),
    re.compile(r"\.(?:map|lock)$", re.I),
)
GENERATED_HEADER_MARKERS = ("generated file", "code generated", "do not edit", "automatically generated")


def _run_git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=check, timeout=30,
    )


def _git_root(workdir: Path) -> Path | None:
    result = _run_git(workdir, "rev-parse", "--show-toplevel")
    if result.returncode:
        return None
    return Path(result.stdout.decode("utf-8", "surrogateescape").strip()).resolve()


def _git_revision(root: Path) -> str | None:
    result = _run_git(root, "rev-parse", "HEAD")
    return result.stdout.decode().strip() if result.returncode == 0 else None


def _git_state(root: Path) -> dict[str, Any]:
    status = _run_git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=all",
        "--", ".", ":(exclude).rist/**",
    )
    raw = status.stdout if status.returncode == 0 else b""
    return {
        "revision": _git_revision(root),
        "status_sha256": hashlib.sha256(raw).hexdigest(),
        "dirty": bool(raw),
    }


def _fallback_files(root: Path) -> list[str]:
    files: list[str] = []
    for current, dirs, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirs[:] = sorted(
            name for name in dirs
            if name not in EXCLUDED_DIRS and not (current_path / name).is_symlink()
        )
        for name in sorted(names):
            files.append((current_path / name).relative_to(root).as_posix())
    return files


def _candidate_files(root: Path) -> tuple[list[str], str]:
    result = _run_git(root, "ls-files", "-co", "--exclude-standard", "-z")
    if result.returncode == 0:
        paths = result.stdout.decode("utf-8", "surrogateescape").split("\0")
        return sorted({path for path in paths if path}), "git"
    return _fallback_files(root), "filesystem"


def _excluded_reason(relative: Path) -> str | None:
    if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
        return "excluded_directory"
    return None


def _is_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return False
    if b"\0" in chunk:
        return True
    if not chunk:
        return False
    suspicious = sum(byte < 9 or 13 < byte < 32 for byte in chunk)
    return suspicious / len(chunk) > 0.30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _language(path: Path) -> str | None:
    if path.name == "Dockerfile":
        return "Dockerfile"
    return LANGUAGES.get(path.suffix.lower())


def _is_manifest(path: Path) -> bool:
    return path.name in MANIFEST_NAMES or path.suffix in {".csproj", ".sln"}


def _is_test(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return bool(lower_parts & TEST_PARTS) or bool(re.search(r"(?:^|[._-])(test|spec)(?:[._-]|$)", name))


def _is_entrypoint(path: Path) -> bool:
    posix = path.as_posix()
    name = path.name.lower()
    if posix in ENTRYPOINTS:
        return True
    return name in {"main.py", "__main__.py", "app.py", "manage.py", "main.rs", "main.go", "main.java", "index.js", "index.ts", "server.js", "server.ts"}


def _is_generated(path: Path, prefix: str) -> bool:
    if any(pattern.search(path.name) for pattern in GENERATED_NAME_PATTERNS):
        return True
    lowered = prefix.lower()
    return any(marker in lowered for marker in GENERATED_HEADER_MARKERS)


def _read_prefix(path: Path, limit: int = 16384) -> str:
    try:
        return path.read_bytes()[:limit].decode("utf-8", "replace")
    except OSError:
        return ""


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _manifest_details(root: Path, paths: list[str]) -> tuple[list[dict[str, Any]], list[str], dict[str, str], list[str]]:
    manifests: list[dict[str, Any]] = []
    workspaces: set[str] = {"."}
    commands: dict[str, str] = {}
    unsupported: list[str] = []
    for relative in sorted(path for path in paths if _is_manifest(Path(path))):
        path = root / relative
        kind = path.name
        item: dict[str, Any] = {"path": relative, "kind": kind, "workspace": Path(relative).parent.as_posix() or "."}
        workspaces.add(item["workspace"])
        if kind == "package.json":
            package = _json_file(path)
            item["name"] = package.get("name")
            scripts = package.get("scripts") if isinstance(package.get("scripts"), dict) else {}
            item["scripts"] = {str(k): str(v) for k, v in sorted(scripts.items())}
            bins = package.get("bin")
            if isinstance(bins, str) and item.get("name"):
                item["entry_points"] = {str(item["name"]): bins}
            elif isinstance(bins, dict):
                item["entry_points"] = {str(k): str(v) for k, v in sorted(bins.items())}
            prefix = "" if item["workspace"] == "." else f"--dir {item['workspace']} "
            for name in sorted(scripts):
                commands.setdefault(name, f"npm {prefix}run {name}".replace("npm --dir", "npm --prefix"))
            workspace_value = package.get("workspaces")
            patterns = workspace_value.get("packages", []) if isinstance(workspace_value, dict) else workspace_value
            if patterns and not isinstance(patterns, list):
                unsupported.append(f"Unsupported package.json workspaces value in {relative}")
            elif isinstance(patterns, list):
                item["workspace_patterns"] = sorted(str(value) for value in patterns)
        elif kind == "pyproject.toml":
            text = _read_prefix(path, 65536)
            try:
                pyproject = tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                pyproject = {}
            project = pyproject.get("project", {}) if isinstance(pyproject.get("project"), dict) else {}
            name = re.search(r"(?m)^name\s*=\s*['\"]([^'\"]+)", text)
            item["name"] = project.get("name") or (name.group(1) if name else None)
            scripts = project.get("scripts") if isinstance(project.get("scripts"), dict) else {}
            gui_scripts = project.get("gui-scripts") if isinstance(project.get("gui-scripts"), dict) else {}
            item["entry_points"] = {str(k): str(v) for k, v in sorted({**scripts, **gui_scripts}.items())}
            commands.setdefault("test", "pytest")
            if "[tool.ruff" in text:
                commands.setdefault("lint", "ruff check .")
            if "[tool.mypy" in text:
                commands.setdefault("typecheck", "mypy .")
        elif kind == "Cargo.toml":
            commands.setdefault("build", "cargo build")
            commands.setdefault("test", "cargo test")
        elif kind == "go.mod":
            commands.setdefault("build", "go build ./...")
            commands.setdefault("test", "go test ./...")
        elif kind in {"pom.xml"}:
            commands.setdefault("build", "mvn package")
            commands.setdefault("test", "mvn test")
        elif kind.startswith("build.gradle"):
            commands.setdefault("build", "./gradlew build")
            commands.setdefault("test", "./gradlew test")
        manifests.append(item)
    return manifests, sorted(workspaces), dict(sorted(commands.items())), sorted(unsupported)


def _build_tools(paths: list[str], manifests: list[dict[str, Any]]) -> list[str]:
    names = {Path(path).name for path in paths}
    kinds = {item["kind"] for item in manifests}
    tools = set()
    checks = {
        "npm": "package.json", "pnpm": "pnpm-lock.yaml", "Yarn": "yarn.lock", "Poetry": "poetry.lock",
        "uv": "uv.lock", "pip": "requirements.txt", "Cargo": "Cargo.toml", "Go": "go.mod",
        "Maven": "pom.xml", "Gradle": "build.gradle", "Make": "Makefile", "Docker": "Dockerfile",
    }
    for tool, marker in checks.items():
        if marker in names or marker in kinds:
            tools.add(tool)
    return sorted(tools)


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_previous(output_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((output_dir / "repo-map.json").read_text(encoding="utf-8"))
        return data if data.get("schema_version") == SCHEMA_VERSION else None
    except (OSError, ValueError):
        return None


def _write_atomic(output_dir: Path, artifacts: dict[str, dict[str, Any] | str]) -> None:
    """Publish a complete snapshot by atomically switching one symlink."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = output_dir / ".index-snapshots"
    snapshots.mkdir(exist_ok=True)
    snapshot_id = _fingerprint(artifacts)
    destination = snapshots / snapshot_id
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=snapshots))
        try:
            for name in PUBLISHED_NAMES:
                path = temporary / name
                with path.open("w", encoding="utf-8") as handle:
                    content = artifacts[name]
                    if isinstance(content, str):
                        handle.write(content)
                    else:
                        json.dump(content, handle, indent=2, sort_keys=True, ensure_ascii=False)
                        handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()

    current = output_dir / ".index-current"
    temporary_link = output_dir / f".index-current.{os.getpid()}.tmp"
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(Path(".index-snapshots") / snapshot_id, target_is_directory=True)

    # Public artifact links are stable after their first creation. Later indexes
    # switch the complete set together through the single atomic current-link replace.
    os.replace(temporary_link, current)
    for name in PUBLISHED_NAMES:
        public = output_dir / name
        expected = Path(".index-current") / name
        if public.is_symlink() and Path(os.readlink(public)) == expected:
            continue
        link = output_dir / f".{name}.{os.getpid()}.tmp"
        link.unlink(missing_ok=True)
        link.symlink_to(expected)
        os.replace(link, public)


def index_repository(
    workdir: str | Path,
    *,
    force: bool = False,
    preview: bool = False,
    status_only: bool = False,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Index a repository and optionally atomically publish project artifacts."""
    requested = Path(workdir).resolve()
    root = _git_root(requested) or requested
    output_dir = root / ".rist" / "project"
    previous = _load_previous(output_dir)
    try:
        previous_graph = json.loads((output_dir / "dependency-graph.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous_graph = None
    candidates, discovery = _candidate_files(root)
    git_state = _git_state(root) if discovery == "git" else {"revision": None, "status_sha256": None, "dirty": None}
    previous_files = {item["path"]: item for item in (previous or {}).get("files", [])}

    files: list[dict[str, Any]] = []
    skipped = Counter()
    unsupported: list[str] = []
    changed_paths: list[str] = []
    symlinks: list[str] = []
    for relative_string in candidates:
        relative = Path(relative_string)
        reason = _excluded_reason(relative)
        if reason:
            skipped[reason] += 1
            continue
        path = root / relative
        if path.is_symlink():
            symlinks.append(relative.as_posix())
            skipped["symlink"] += 1
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            skipped["unreadable"] += 1
            unsupported.append(f"Cannot stat {relative.as_posix()}: {exc.strerror or exc}")
            continue
        if not path.is_file():
            skipped["not_regular_file"] += 1
            continue
        if stat.st_size > max_file_size:
            skipped["too_large"] += 1
            continue
        if _is_binary(path):
            skipped["binary"] += 1
            continue
        digest = _sha256(path)
        old = previous_files.get(relative.as_posix())
        if not force and old and old.get("sha256") == digest:
            item = {key: old[key] for key in ("path", "size", "sha256", "language", "test", "entrypoint", "generated") if key in old}
        else:
            prefix = _read_prefix(path)
            item = {
                "path": relative.as_posix(), "size": stat.st_size, "sha256": digest,
                "language": _language(relative), "test": _is_test(relative),
                "entrypoint": _is_entrypoint(relative), "generated": _is_generated(relative, prefix),
            }
            changed_paths.append(relative.as_posix())
        files.append(item)

    current_paths = {item["path"] for item in files}
    removed_paths = sorted(set(previous_files) - current_paths)
    changed_paths.extend(removed_paths)
    files.sort(key=lambda item: item["path"])
    indexed_paths = [item["path"] for item in files]
    manifests, workspaces, commands, manifest_unsupported = _manifest_details(root, indexed_paths)
    unsupported.extend(manifest_unsupported)
    language_counts = Counter(item["language"] for item in files if item.get("language"))
    generated = sorted(item["path"] for item in files if item.get("generated"))
    tests = sorted({str(Path(item["path"]).parent.as_posix()) for item in files if item.get("test")})
    entrypoints = sorted(item["path"] for item in files if item.get("entrypoint"))
    configs = sorted(path for path in indexed_paths if Path(path).name in CONFIG_NAMES or ".github/workflows/" in path)
    inputs = {
        "discovery": discovery,
        "excluded_directories": sorted(EXCLUDED_DIRS),
        "max_file_size": max_file_size,
        "include": "tracked and untracked files",
        "gitignore": discovery == "git",
        "follow_symlinks": False,
    }
    content_fingerprint = _fingerprint([{"path": item["path"], "sha256": item["sha256"]} for item in files])
    stale = (
        previous is None
        or previous.get("content_fingerprint") != content_fingerprint
        or previous.get("git_state") != git_state
        or previous.get("inputs") != inputs
    )
    generated_timestamp = (now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    timestamp = previous.get("index_timestamp", generated_timestamp) if not stale and not force else generated_timestamp
    common = {
        "schema_version": SCHEMA_VERSION,
        "repository_revision": git_state["revision"],
        "index_timestamp": timestamp,
        "inputs": inputs,
        "content_fingerprint": content_fingerprint,
    }
    repo_map = {
        **common,
        "repository_root": root.as_posix(),
        "git_state": git_state,
        "languages": dict(sorted(language_counts.items())),
        "files": files,
        "entry_points": entrypoints,
        "test_locations": tests,
        "generated_files": generated,
        "symlinks_skipped": sorted(symlinks),
    }
    manifest_payload = {
        **common,
        "manifests": manifests,
        "workspace_boundaries": workspaces,
        "build_tools": _build_tools(indexed_paths, manifests),
        "common_commands": commands,
    }
    conventions_payload = {
        **common,
        "configuration_files": configs,
        "source_roots": sorted({item["path"].split("/", 1)[0] for item in files if "/" in item["path"] and item.get("language")}),
        "test_locations": tests,
        "generated_files": generated,
        "unsupported_constructs": sorted(set(unsupported)),
    }
    graph = build_dependency_graph(root, files, manifests, commands, previous=previous_graph)
    graph.metadata.update(common)
    graph_payload = {**common, **graph.to_dict()}
    artifacts: dict[str, dict[str, Any] | str] = {
        "repo-map.json": repo_map,
        "manifests.json": manifest_payload,
        "conventions.json": conventions_payload,
        "dependency-graph.json": graph_payload,
        ARCHITECTURE_VIEW: render_architecture(graph),
    }
    artifact_fingerprints = {name: _fingerprint(payload) for name, payload in artifacts.items()}
    report = {
        "schema_version": SCHEMA_VERSION,
        "repository_root": root.as_posix(),
        "repository_revision": git_state["revision"],
        "index_timestamp": timestamp,
        "status": "missing" if previous is None else ("stale" if stale else "current"),
        "mode": "status" if status_only else ("preview" if preview else ("full" if force or previous is None else "incremental")),
        "files_scanned": len(candidates),
        "files_indexed": len(files),
        "files_skipped": sum(skipped.values()),
        "skip_reasons": dict(sorted(skipped.items())),
        "files_changed": len(set(changed_paths)),
        "changed_paths": sorted(set(changed_paths)),
        "unsupported_constructs": sorted(set(unsupported)),
        "content_fingerprint": content_fingerprint,
        "artifact_fingerprints": artifact_fingerprints,
        "artifacts": {name: str(output_dir / name) for name in PUBLISHED_NAMES},
        "stale_graph_nodes": graph.metadata["incremental"]["stale_node_ids"],
        "stale_graph_edges": graph.metadata["incremental"]["stale_edge_ids"],
        "written": False,
        "preview_artifacts": artifacts if preview else None,
    }
    if not status_only and not preview and (stale or force):
        _write_atomic(output_dir, artifacts)
        report["written"] = True
        report["status"] = "current"
    return report


def format_index_report(report: dict[str, Any]) -> str:
    action = {"status": "Index status", "preview": "Index preview", "full": "Full index", "incremental": "Incremental index"}[report["mode"]]
    lines = [
        f"{action}: {report['status']}",
        f"Repository: {report['repository_root']}",
        f"Revision: {report['repository_revision'] or '(unborn or unavailable)'}",
        f"Files scanned: {report['files_scanned']}",
        f"Files skipped: {report['files_skipped']}",
        f"Files changed: {report['files_changed']}",
    ]
    if report["skip_reasons"]:
        lines.append("Skipped: " + ", ".join(f"{key}={value}" for key, value in report["skip_reasons"].items()))
    if report["unsupported_constructs"]:
        lines.append("Unsupported constructs:")
        lines.extend(f"- {item}" for item in report["unsupported_constructs"])
    if report["written"]:
        lines.append("Artifacts:")
        lines.extend(f"- {path}" for path in report["artifacts"].values())
    elif report["mode"] == "preview":
        lines.append("No files written (preview).")
    return "\n".join(lines)

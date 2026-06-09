"""Per-repository storage with an explicit shareable/private boundary."""

import json
import shutil
import subprocess
from pathlib import Path

from .config import LEGACY_MEMORY_DIR_NAME, MAX_HISTORY_MESSAGES, MEMORY_DIR_NAME
from .intelligence import (
    IntelligenceStore,
    StorageMode,
    StorageScope,
    atomic_write_text,
    coerce_storage_mode,
)
from .ui import clip

_PRIVATE_FILENAMES = ("runs.jsonl", "chat_history.jsonl", "preferences.json")
_LEGACY_INTELLIGENCE_FILENAMES = ("intelligence.json", "project.md", "decisions.md", "architecture.md")


def memory_paths(workdir, storage_mode=None):
    root = Path(workdir)
    base = root / MEMORY_DIR_NAME
    legacy = root / LEGACY_MEMORY_DIR_NAME
    if not base.exists() and legacy.is_dir():
        shutil.copytree(legacy, base)
    mode = coerce_storage_mode(storage_mode)
    project_scope = base / StorageScope.PROJECT.value
    local_scope = base / StorageScope.LOCAL.value
    local_intelligence = local_scope / "intelligence"
    active_intelligence = project_scope if mode == StorageMode.SHARED else local_intelligence
    return {
        "base": base,
        "mode": mode,
        "project_scope": project_scope,
        "local": local_scope,
        "local_intelligence": local_intelligence,
        "active_intelligence": active_intelligence,
        "intelligence": active_intelligence / "intelligence.json",
        "project": active_intelligence / "project.md",
        "decisions": active_intelligence / "decisions.md",
        "architecture": active_intelligence / "architecture.md",
        "runs": local_scope / "runs.jsonl",
        "chat_history": local_scope / "chat_history.jsonl",
        "preferences": local_scope / "preferences.json",
        "gitignore": base / ".gitignore",
    }


def _move_if_needed(source: Path, destination: Path, conflict_root: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        if source.is_file() and destination.is_file() and source.read_bytes() == destination.read_bytes():
            source.unlink()
            return
        conflict = conflict_root / destination.name
        index = 1
        while conflict.exists():
            conflict = conflict_root / f"{destination.stem}-{index}{destination.suffix}"
            index += 1
        conflict.parent.mkdir(parents=True, exist_ok=True)
        source.replace(conflict)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def _migrate_fully_ignored_layout(paths) -> None:
    """Move legacy root data into private storage without touching Git's index."""
    base = paths["base"]
    local_intelligence = paths["local_intelligence"]
    conflicts = paths["local"] / "migration-conflicts"
    for filename in _LEGACY_INTELLIGENCE_FILENAMES:
        _move_if_needed(base / filename, local_intelligence / filename, conflicts / "intelligence")
    for filename in _PRIVATE_FILENAMES:
        _move_if_needed(base / filename, paths["local"] / filename, conflicts / "state")
    old_private = base / "private"
    if old_private.is_dir():
        for filename in _PRIVATE_FILENAMES:
            _move_if_needed(old_private / filename, paths["local"] / filename, conflicts / "private")
        try:
            old_private.rmdir()
        except OSError:
            pass


def _gitignore_content(mode: StorageMode) -> str:
    lines = [
        "# Rist private state: never commit chat, runs, caches, paths, providers, or preferences.",
        "/local/",
    ]
    if mode == StorageMode.LOCAL_ONLY:
        lines.extend(("# local-only mode also keeps reviewable project knowledge untracked.", "/project/"))
    return "\n".join(lines) + "\n"


def ensure_memory_files(workdir, storage_mode=None):
    """Create scoped knowledge stores and migrate the old fully ignored layout.

    Migration only moves files beneath ``.rist`` and updates ignore metadata. It
    never invokes ``git add`` or otherwise stages/commits project knowledge.
    """
    paths = memory_paths(workdir, storage_mode)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["local"].mkdir(parents=True, exist_ok=True)
    _migrate_fully_ignored_layout(paths)
    ensure_git_exclude(workdir, paths["mode"])
    atomic_write_text(paths["gitignore"], _gitignore_content(paths["mode"]))
    for key in ("runs", "chat_history"):
        if not paths[key].exists():
            paths[key].touch()
    if not paths["preferences"].exists():
        atomic_write_text(paths["preferences"], "{}\n")

    if paths["mode"] != StorageMode.SHARED:
        IntelligenceStore.load(paths["local_intelligence"], sync_views=True, scope=StorageScope.LOCAL)
    if paths["mode"] != StorageMode.LOCAL_ONLY:
        IntelligenceStore.load(paths["project_scope"], sync_views=True, scope=StorageScope.PROJECT)
    return paths


def ensure_git_exclude(workdir, storage_mode=None):
    """Ignore only private state and remove obsolete whole-directory excludes."""
    mode = coerce_storage_mode(storage_mode)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=workdir,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return
    if completed.returncode != 0:
        return
    exclude_path = Path(completed.stdout.strip())
    if not exclude_path.is_absolute():
        exclude_path = (Path(workdir) / exclude_path).resolve()
    try:
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text(encoding="utf-8", errors="replace") if exclude_path.exists() else ""
        obsolete = {f"{MEMORY_DIR_NAME}/", f"/{MEMORY_DIR_NAME}/", f"{MEMORY_DIR_NAME}/**", f"/{MEMORY_DIR_NAME}/**"}
        if mode != StorageMode.LOCAL_ONLY:
            obsolete.add(f"/{MEMORY_DIR_NAME}/project/")
            obsolete.add(f"{MEMORY_DIR_NAME}/project/")
        lines = [line for line in existing.splitlines() if line.strip() not in obsolete]
        required = [f"/{MEMORY_DIR_NAME}/local/"]
        if mode == StorageMode.LOCAL_ONLY:
            required.append(f"/{MEMORY_DIR_NAME}/project/")
        for entry in required:
            if entry not in lines:
                lines.append(entry)
        content = "\n".join(lines).rstrip()
        atomic_write_text(exclude_path, content + ("\n" if content else ""))
    except Exception:  # noqa: BLE001
        return


def load_repo_memory(workdir, storage_mode=None):
    paths = ensure_memory_files(workdir, storage_mode)
    stores = []
    if paths["mode"] in {StorageMode.SHARED, StorageMode.HYBRID}:
        stores.append(paths["project_scope"])
    if paths["mode"] in {StorageMode.LOCAL_ONLY, StorageMode.HYBRID}:
        stores.append(paths["local_intelligence"])
    chunks = []
    for store in stores:
        label = store.relative_to(paths["base"])
        for filename in ("project.md", "decisions.md", "architecture.md"):
            path = store / filename
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                chunks.append(f"{label}/{filename}:\n{clip(text, 2000)}")
    return "\n\n".join(chunks)


def load_recent_runs(path, limit=5):
    if not path.exists():
        return ""
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    summaries = []
    for line in lines[-limit:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = clip(entry.get("user_prompt", ""), 240)
        status = entry.get("status", "unknown")
        result = clip(entry.get("result", ""), 500)
        if prompt or result:
            summaries.append(f"- {entry.get('timestamp', 'unknown')} [{status}] user: {prompt}\n  assistant: {result}")
    return "\n".join(summaries)


def append_run_log(workdir, entry, storage_mode=None):
    paths = ensure_memory_files(workdir, storage_mode)
    with paths["runs"].open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _trim_run_log(paths["runs"])


def _trim_run_log(path, limit=500):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        if len(lines) > limit:
            path.write_text("".join(lines[-limit:]), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def load_chat_history(workdir, storage_mode=None):
    path = memory_paths(workdir, storage_mode)["chat_history"]
    if not path.exists():
        return []
    history = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if isinstance(msg, dict) and msg.get("role") in {"user", "assistant"} and isinstance(msg.get("content"), str):
                history.append(msg)
        except json.JSONDecodeError:
            continue
    return history[-MAX_HISTORY_MESSAGES:]


def save_chat_history(workdir, history, storage_mode=None):
    path = memory_paths(workdir, storage_mode)["chat_history"]
    try:
        content = "".join(json.dumps(msg, ensure_ascii=False) + "\n" for msg in history)
        atomic_write_text(path, content)
    except Exception:  # noqa: BLE001
        pass


def clear_chat_history(workdir, storage_mode=None):
    path = memory_paths(workdir, storage_mode)["chat_history"]
    try:
        atomic_write_text(path, "")
    except Exception:  # noqa: BLE001
        pass

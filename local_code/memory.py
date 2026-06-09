import json
import shutil
import subprocess
from pathlib import Path

from .config import LEGACY_MEMORY_DIR_NAME, MAX_HISTORY_MESSAGES, MEMORY_DIR_NAME
from .ui import clip


def memory_paths(workdir):
    root = Path(workdir)
    base = root / MEMORY_DIR_NAME
    legacy = root / LEGACY_MEMORY_DIR_NAME
    if not base.exists() and legacy.is_dir():
        shutil.copytree(legacy, base)
    return {
        "base": base,
        "project": base / "project.md",
        "decisions": base / "decisions.md",
        "architecture": base / "architecture.md",
        "runs": base / "runs.jsonl",
        "chat_history": base / "chat_history.jsonl",
        "gitignore": base / ".gitignore",
    }


def ensure_memory_files(workdir):
    paths = memory_paths(workdir)
    paths["base"].mkdir(parents=True, exist_ok=True)
    ensure_git_exclude(workdir)
    defaults = {
        "project": "# Project Notes\n\n- Purpose:\n- Stack:\n- Common commands:\n",
        "decisions": "# Decisions\n\n- Date: \n- Decision: \n- Reason: \n",
        "architecture": "# Architecture\n\n- Entry points:\n- Key modules:\n- Constraints:\n",
    }
    for key, content in defaults.items():
        if not paths[key].exists():
            paths[key].write_text(content, encoding="utf-8")
    if not paths["runs"].exists():
        paths["runs"].touch()
    if not paths["gitignore"].exists():
        paths["gitignore"].write_text("*\n", encoding="utf-8")
    return paths


def ensure_git_exclude(workdir):
    try:
        completed = subprocess.run(
            "git rev-parse --git-path info/exclude",
            cwd=workdir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return
    if completed.returncode != 0:
        return
    exclude_path = Path(workdir) / completed.stdout.strip()
    if not exclude_path.is_absolute():
        exclude_path = exclude_path.resolve()
    try:
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text(encoding="utf-8", errors="replace") if exclude_path.exists() else ""
        if f"{MEMORY_DIR_NAME}/" not in existing:
            suffix = "" if existing.endswith("\n") or not existing else "\n"
            exclude_path.write_text(existing + suffix + f"{MEMORY_DIR_NAME}/\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        return


def load_repo_memory(workdir):
    paths = ensure_memory_files(workdir)
    chunks = []
    for key in ("project", "decisions", "architecture"):
        text = paths[key].read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append(f"{key}.md:\n{clip(text, 2000)}")
    recent_runs = load_recent_runs(paths["runs"])
    if recent_runs:
        chunks.append("recent runs:\n" + recent_runs)
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


def append_run_log(workdir, entry):
    paths = ensure_memory_files(workdir)
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


def load_chat_history(workdir):
    path = memory_paths(workdir)["chat_history"]
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


def save_chat_history(workdir, history):
    path = memory_paths(workdir)["chat_history"]
    try:
        with path.open("w", encoding="utf-8") as fh:
            for msg in history:
                fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def clear_chat_history(workdir):
    path = memory_paths(workdir)["chat_history"]
    try:
        path.write_text("", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

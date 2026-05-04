import json
import re
from pathlib import Path

from .config import EDIT_INTENT_RE


def infer_file_hints(user_prompt):
    hints = []
    for match in re.findall(r"(?<!\w)([A-Za-z0-9_./-]+\.[A-Za-z0-9_+-]+)", user_prompt):
        if "/" not in match:
            name, _, ext = match.rpartition(".")
            # Skip single-char basenames and extensions with no letters (e.g. "e.g", "1.0")
            if len(name) < 2 or not re.search(r"[A-Za-z]", ext):
                continue
        if match not in hints:
            hints.append(match)
    return hints


def infer_command_hints(user_prompt):
    commands = []
    pattern = re.compile(r"\b(pnpm|npm|yarn|bun|cargo|python|pytest|vitest|jest|go test|make)\b[^\n`.;,]*")
    for line in user_prompt.splitlines():
        line = line.strip().strip('`')
        if not line:
            continue
        for match in pattern.finditer(line):
            cmd = re.sub(r"\b(fails?|failing|failed|error|broken)\b.*$", "", match.group(0), flags=re.I).strip()
            if cmd and cmd not in commands:
                commands.append(cmd)
    return commands


def resolve_repo_file_hints(workdir, hints):
    resolved = []
    root = Path(workdir)
    for hint in hints:
        candidate = root / hint
        if candidate.exists():
            value = str(candidate.relative_to(root))
            if value not in resolved:
                resolved.append(value)
            continue
        name = Path(hint).name
        matches = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob(name)
            if p.is_file() and ".git" not in p.parts and "node_modules" not in p.parts
        )
        for match in matches[:3]:
            if match not in resolved:
                resolved.append(match)
    return resolved


def normalize_contract(contract, user_prompt, mode, planning=False):
    contract = dict(contract or {})
    inferred_files = infer_file_hints(user_prompt)
    edit_policy = contract.get("edit_policy")
    if not edit_policy:
        if planning:
            edit_policy = "plan"
        elif mode == "agent" and EDIT_INTENT_RE.search(user_prompt):
            edit_policy = "execute"
        elif EDIT_INTENT_RE.search(user_prompt):
            edit_policy = "propose"
        else:
            edit_policy = "inspect"
    goal = contract.get("goal") or user_prompt.strip()
    expected_result = contract.get("expected_result") or user_prompt.strip()
    normalized = {
        "goal": goal,
        "scope": contract.get("scope") or inferred_files or ["."],
        "constraints": contract.get("constraints") or [],
        "commands_allowed": contract.get("commands_allowed") or infer_command_hints(user_prompt),
        "edit_policy": edit_policy,
        "expected_result": expected_result,
        "files_of_interest": contract.get("files_of_interest") or inferred_files,
    }
    if has_pasted_context(user_prompt):
        normalized["constraints"].extend(
            [
                "The user supplied pasted context/output. Analyze that evidence directly before suggesting commands.",
                "Do not ask the user to rerun or provide the same output unless the pasted context is incomplete.",
            ]
        )
    if has_database_context(user_prompt):
        normalized["scope"] = contract.get("scope") or ["electron", "src", "package.json"]
        normalized["constraints"].extend(
            [
                "This is a database/foreign-key investigation. Prioritize schema, migrations, database initialization, account/user/mailbox inserts, and connection-test persistence paths.",
                "Avoid broad repo_overview loops after initial orientation; use targeted searches for sqlite, foreign key, schema, migration, account, connection, onboarding, and testConnection.",
                "Do not inspect built output such as dist-electron unless source files are unavailable.",
            ]
        )
    return normalized


def has_pasted_context(text):
    return "--- PASTED CONTENT" in text or "[Pasted Content" in text


def has_database_context(text):
    return bool(re.search(r"\b(database|sqlite|foreign key|constraint|schema|migration|account|imap|smtp|onboarding)\b", text, re.I))


def normalize_backend_report(report, fallback_message=""):
    if isinstance(report, str):
        text = report.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                report = parsed
            else:
                report = {"summary": text}
        except json.JSONDecodeError:
            report = {"summary": text or fallback_message}
    elif not isinstance(report, dict):
        report = {"summary": fallback_message}
    return {
        "summary": report.get("summary", fallback_message),
        "findings": report.get("findings") or [],
        "commands_run": report.get("commands_run") or [],
        "files_read": report.get("files_read") or [],
        "files_changed": report.get("files_changed") or [],
        "diff_summary": report.get("diff_summary", ""),
        "tests_run": report.get("tests_run") or [],
        "risks": report.get("risks") or [],
        "needs_approval": bool(report.get("needs_approval", False)),
        "plan": report.get("plan") or [],
    }


def load_json_layers(text, max_depth=3):
    value = text
    for _ in range(max_depth):
        if not isinstance(value, str):
            return value
        value = value.strip()
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def unwrap_frontend_reply_text(text):
    text = (text or "").strip()
    if not text:
        return text
    payload = load_json_layers(text)
    if not isinstance(payload, dict):
        return text
    if isinstance(payload, dict) and payload.get("mode") == "reply" and isinstance(payload.get("message"), str):
        return payload["message"].strip()
    return text

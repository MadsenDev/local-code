import json
import re
from pathlib import Path

from .config import EDIT_INTENT_RE

VALID_EDIT_POLICIES = {"inspect", "plan", "propose", "execute"}
VALID_TASK_KINDS = {"conversation", "inspection", "edit_existing", "bootstrap_new"}
VALID_EXECUTION_STRATEGIES = {"direct", "inspect_then_execute", "plan_only", "apply_approved_plan"}
PROJECT_MARKERS = (
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "src",
)
BOOTSTRAP_REGISTRY = {
    "vite-react-ts": {
        "commands_allowed": [
            "npm create vite@latest . -- --template react-ts",
            "npm install",
        ],
        "approval_prefixes": [
            "npm create vite@latest",
            "npm install",
        ],
        "verification_checks": [
            "package.json exists",
            "src/main.tsx exists",
            "src/App.tsx exists",
        ],
        "target_paths": ["package.json", "src/main.tsx", "src/App.tsx"],
    },
    "vite-react-js": {
        "commands_allowed": [
            "npm create vite@latest . -- --template react",
            "npm install",
        ],
        "approval_prefixes": [
            "npm create vite@latest",
            "npm install",
        ],
        "verification_checks": [
            "package.json exists",
            "src/main.jsx exists",
            "src/App.jsx exists",
        ],
        "target_paths": ["package.json", "src/main.jsx", "src/App.jsx"],
    },
}
BOOTSTRAP_RE = re.compile(r"\b(set up|setup|scaffold|bootstrap|initialize|init|create|start)\b", re.I)
VITE_REACT_RE = re.compile(r"\b(vite|react)\b", re.I)
INSPECTION_RE = re.compile(
    r"\b(check|inspect|search|read|explain|tell me what .* does|look in|review|analy[sz]e|why|where|exists?)\b",
    re.I,
)
BOOTSTRAP_RISKY_RE = re.compile(r"\b(monorepo|workspace|docker|deploy|production|architecture|migrate|restructure)\b", re.I)


def inspect_workdir_state(workdir):
    root = Path(workdir)
    entries = [p for p in root.iterdir() if p.name not in {".git", ".rist", ".local-code"}]
    files = [p for p in entries if p.is_file()]
    dirs = [p for p in entries if p.is_dir()]
    markers = []
    for marker in PROJECT_MARKERS:
        if (root / marker).exists():
            markers.append(marker)
    return {
        "workdir_name": root.name,
        "exists": root.exists(),
        "is_git_repo": (root / ".git").exists(),
        "is_empty": not entries,
        "entry_names": sorted(p.name for p in entries),
        "file_names": sorted(p.name for p in files),
        "dir_names": sorted(p.name for p in dirs),
        "project_markers": markers,
        "has_project_markers": bool(markers),
    }


def infer_edit_policy(user_prompt, mode, planning=False):
    if planning:
        return "plan"
    if mode == "agent" and EDIT_INTENT_RE.search(user_prompt):
        return "execute"
    if EDIT_INTENT_RE.search(user_prompt):
        return "propose"
    return "inspect"


def infer_file_hints(user_prompt):
    hints = []
    for match in re.findall(r"(?<!\w)([A-Za-z0-9_./-]+\.[A-Za-z0-9_+-]+)", user_prompt):
        if "/" not in match:
            name, _, ext = match.rpartition(".")
            if len(name) < 2 or not re.search(r"[A-Za-z]", ext):
                continue
        if match not in hints:
            hints.append(match)
    return hints


def infer_command_hints(user_prompt):
    commands = []
    pattern = re.compile(r"\b(pnpm|npm|yarn|bun|cargo|python|pytest|vitest|jest|go test|make)\b[^\n`.;,]*")
    for line in user_prompt.splitlines():
        line = line.strip().strip("`")
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


def infer_bootstrap_variant(user_prompt):
    prompt = user_prompt.lower()
    if VITE_REACT_RE.search(prompt):
        if re.search(r"\bjavascript\b|\bjsx\b|\breact js\b", prompt):
            return "vite-react-js"
        return "vite-react-ts"
    return None


def classify_task(user_prompt, workdir_state):
    prompt = user_prompt.strip()
    if has_pasted_context(prompt) and not EDIT_INTENT_RE.search(prompt):
        return "inspection"
    if INSPECTION_RE.search(prompt) and not EDIT_INTENT_RE.search(prompt):
        return "inspection"
    bootstrap_variant = infer_bootstrap_variant(prompt)
    if BOOTSTRAP_RE.search(prompt) and bootstrap_variant:
        return "bootstrap_new"
    if workdir_state["is_empty"] and bootstrap_variant:
        return "bootstrap_new"
    if EDIT_INTENT_RE.search(prompt):
        return "edit_existing" if workdir_state["has_project_markers"] else "bootstrap_new" if bootstrap_variant else "edit_existing"
    return "conversation"


def infer_execution_strategy(task_kind, user_prompt, mode, workdir_state, planning=False):
    if planning:
        return "plan_only"
    if task_kind == "conversation":
        return "direct"
    if task_kind == "inspection":
        return "direct"
    if task_kind == "bootstrap_new":
        return "inspect_then_execute"
    if task_kind == "edit_existing":
        if mode == "agent":
            return "inspect_then_execute"
        if BOOTSTRAP_RISKY_RE.search(user_prompt) or len(user_prompt.split()) > 40:
            return "plan_only"
        return "inspect_then_execute"
    return "direct"


def default_target_paths(task_kind, workdir_state, bootstrap_variant):
    if task_kind == "bootstrap_new" and bootstrap_variant in BOOTSTRAP_REGISTRY:
        return BOOTSTRAP_REGISTRY[bootstrap_variant]["target_paths"]
    if task_kind == "inspection":
        return []
    if workdir_state["has_project_markers"]:
        return workdir_state["project_markers"][:4]
    return []


def default_verification_checks(task_kind, bootstrap_variant):
    if task_kind == "bootstrap_new" and bootstrap_variant in BOOTSTRAP_REGISTRY:
        return BOOTSTRAP_REGISTRY[bootstrap_variant]["verification_checks"]
    return []


def default_commands_allowed(task_kind, bootstrap_variant, user_prompt):
    if task_kind == "bootstrap_new" and bootstrap_variant in BOOTSTRAP_REGISTRY:
        return BOOTSTRAP_REGISTRY[bootstrap_variant]["commands_allowed"]
    return infer_command_hints(user_prompt)


def default_approval_prefixes(task_kind, bootstrap_variant):
    if task_kind == "bootstrap_new" and bootstrap_variant in BOOTSTRAP_REGISTRY:
        return BOOTSTRAP_REGISTRY[bootstrap_variant]["approval_prefixes"]
    return []


def normalize_contract(contract, user_prompt, mode, planning=False, workdir_state=None):
    contract = dict(contract or {})
    workdir_state = workdir_state or inspect_workdir_state(Path.cwd())
    inferred_files = infer_file_hints(user_prompt)
    task_kind = str(contract.get("task_kind") or "").strip().lower()
    if task_kind not in VALID_TASK_KINDS:
        task_kind = classify_task(user_prompt, workdir_state)
    bootstrap_variant = infer_bootstrap_variant(user_prompt) if task_kind == "bootstrap_new" else None
    execution_strategy = str(contract.get("execution_strategy") or "").strip().lower()
    if execution_strategy not in VALID_EXECUTION_STRATEGIES:
        execution_strategy = infer_execution_strategy(task_kind, user_prompt, mode, workdir_state, planning=planning)
    edit_policy = str(contract.get("edit_policy") or "").strip().lower()
    if edit_policy not in VALID_EDIT_POLICIES:
        if planning or execution_strategy == "plan_only":
            edit_policy = "plan"
        elif task_kind in {"bootstrap_new", "edit_existing"} and execution_strategy == "inspect_then_execute":
            edit_policy = "execute"
        elif task_kind == "inspection":
            edit_policy = "inspect"
        else:
            edit_policy = infer_edit_policy(user_prompt, mode, planning=planning)
    goal = contract.get("goal") or user_prompt.strip()
    expected_result = contract.get("expected_result") or user_prompt.strip()
    constraints = list(contract.get("constraints") or [])
    target_paths = contract.get("target_paths") or default_target_paths(task_kind, workdir_state, bootstrap_variant)
    verification_checks = contract.get("verification_checks") or default_verification_checks(task_kind, bootstrap_variant)
    commands_allowed = contract.get("commands_allowed") or default_commands_allowed(task_kind, bootstrap_variant, user_prompt)
    approval_prefixes = contract.get("approval_prefixes") or default_approval_prefixes(task_kind, bootstrap_variant)
    normalized = {
        "goal": goal,
        "scope": contract.get("scope") or inferred_files or ["."],
        "constraints": constraints,
        "commands_allowed": commands_allowed,
        "approval_prefixes": approval_prefixes,
        "edit_policy": edit_policy,
        "expected_result": expected_result,
        "files_of_interest": contract.get("files_of_interest") or inferred_files,
        "task_kind": task_kind,
        "execution_strategy": execution_strategy,
        "target_paths": target_paths,
        "verification_checks": verification_checks,
    }
    if edit_policy == "plan" and workdir_state["has_project_markers"]:
        extra = [m for m in workdir_state["project_markers"] if m not in (normalized.get("files_of_interest") or [])]
        normalized["files_of_interest"] = (normalized.get("files_of_interest") or []) + extra[:6]
    if bootstrap_variant:
        normalized["bootstrap_template"] = bootstrap_variant
    if task_kind == "bootstrap_new":
        normalized["target_directory"] = contract.get("target_directory") or workdir_state["workdir_name"]
        if workdir_state["is_empty"]:
            normalized["constraints"].append("The current directory is empty. Create the project in place.")
        elif workdir_state["has_project_markers"]:
            normalized["constraints"].append("This directory already looks like a project. Modify it in place only if the request clearly asks for that.")
        else:
            normalized["constraints"].append(
                "This directory is non-empty and not yet a project. Refuse in-place scaffolding if existing files would conflict, unless the user names a target subdirectory."
            )
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
        "decision_candidates": report.get("decision_candidates") or [],
        "decision_conflicts": report.get("decision_conflicts") or [],
        "requires_deviation_explanation": bool(report.get("requires_deviation_explanation", False)),
        "execution_status": str(report.get("execution_status") or ""),
    }


# Deliberately broad: "add"/"generate" count as creation verbs so new-file steps
# pass validation; command detection below is limited to infer_command_hints's
# known runners (npm, pytest, cargo, ...).
PLAN_CREATE_RE = re.compile(r"\b(create|add|new file|scaffold|generate)\b", re.I)
DIFF_MARKERS = ("--- ", "+++ ", "@@")


def validate_plan_report(report, workdir):
    """Check that a plan/propose report is concrete enough to apply mechanically.

    Returns a list of human-readable problems; an empty list means the plan can
    be applied mechanically. A step is concrete when it references an existing
    repo file, a new file it explicitly creates, or a runnable command. If any
    step is a file edit, diff_summary must contain real unified-diff markers.
    """
    problems = []
    steps = report.get("plan") or []
    if not steps:
        return ["The plan is empty. Provide one step per concrete change."]
    needs_diff = False
    for index, step in enumerate(steps, start=1):
        text = str(step)
        file_hints = infer_file_hints(text)
        existing = resolve_repo_file_hints(workdir, file_hints)
        is_creation = bool(file_hints and PLAN_CREATE_RE.search(text))
        is_command = bool(infer_command_hints(text))
        if existing or is_creation:
            needs_diff = True
            continue
        if is_command:
            continue
        problems.append(
            f"Step {index} names no existing repo file and no runnable command: "
            f"{text[:120]!r}. Name the exact file and change, or the exact command."
        )
    if needs_diff:
        diff = str(report.get("diff_summary") or "")
        if not all(marker in diff for marker in DIFF_MARKERS):
            problems.append(
                "diff_summary must contain a real unified diff "
                "(--- a/path, +++ b/path, @@ hunks) for the file edits in the plan."
            )
    return problems


def compute_execution_status(plan_steps, files_changed, commands_run):
    """Harness-derived status of an approved plan after an execute run.

    Returns (status, missed_steps) where status is one of "applied",
    "partially_applied", or "plan_not_applied". A step is covered when a file
    it names appears in files_changed (exact or basename match) or a command
    it names matches a commands_run entry by prefix. Steps naming no file or
    command cannot be verified mechanically and are excluded.
    """
    changed = [str(p) for p in files_changed or []]
    changed_names = {Path(p).name for p in changed}
    run = [str(c) for c in commands_run or []]
    missed = []
    verifiable = 0
    for step in plan_steps or []:
        text = str(step)
        file_refs = infer_file_hints(text)
        command_refs = infer_command_hints(text)
        if not file_refs and not command_refs:
            continue
        verifiable += 1
        file_hit = any(ref in changed or Path(ref).name in changed_names for ref in file_refs)
        command_hit = any(
            r == cmd or r.startswith(cmd + " ") or cmd.startswith(r + " ")
            for cmd in command_refs
            for r in run
        )
        if not (file_hit or command_hit):
            missed.append(text)
    if not changed and not run:
        return "plan_not_applied", missed
    if not verifiable:
        return ("applied" if changed else "plan_not_applied"), missed
    if not missed:
        return "applied", missed
    return "partially_applied", missed


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

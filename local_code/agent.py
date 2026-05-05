import json
import difflib
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

from .config import (
    CODE_ACTION_RE,
    DEFAULT_MODE,
    DEFAULT_VERBOSITY,
    EDIT_INTENT_RE,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_STEPS,
    PROMPT_YES_RE,
    READ_FILE_DEFAULT_END,
    TEST_COMMAND_PATTERNS,
)
from .contracts import (
    has_database_context,
    has_pasted_context,
    infer_command_hints,
    inspect_workdir_state,
    load_json_layers,
    normalize_backend_report,
    normalize_contract,
    resolve_repo_file_hints,
    unwrap_frontend_reply_text,
)
from .memory import append_run_log, ensure_memory_files, load_chat_history, load_recent_runs, load_repo_memory, memory_paths, save_chat_history
from .models import ollama_chat, ollama_stream
from .permissions import command_is_blocked, confirm_action
from .tools import (
    fetch_url,
    git_context,
    insert_after,
    list_files,
    read_file,
    replace_in_file,
    replace_lines,
    repo_overview,
    resolve_path,
    run_subprocess,
    search_files,
    write_file,
)
from .ui import Spinner, UI, summarize_action, summarize_text


class LocalCodeAgent:
    def __init__(
        self,
        model,
        ollama,
        workdir,
        command_permission="ask",
        edit_permission="ask",
        max_steps=MAX_TOOL_STEPS,
        verbosity=DEFAULT_VERBOSITY,
        show_raw_actions=False,
    ):
        self.model = model
        self.ollama = ollama.rstrip("/")
        self.workdir = str(Path(workdir).resolve())
        self.command_permission = command_permission
        self.edit_permission = edit_permission
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.show_raw_actions = show_raw_actions
        self.ui = UI()
        self.last_tool_display = None
        self.last_backend_action = None
        self.last_backend_action_signature = None
        self.repeated_backend_action_count = 0
        self.command_approval_cache = {}
        self.command_prefix_approval_cache = {}

    def chat(self, messages):
        return ollama_chat(self.ollama, self.model, messages)

    def trace_print(self, message):
        if self.verbosity == "debug":
            prefix = self.ui.style("backend", UI.BOLD, UI.BLUE)
            print(f"{prefix} {message}", file=sys.stderr)

    def milestone(self, message):
        if self.verbosity in {"normal", "debug"}:
            print(self.ui.tool_line(message, color=UI.BLUE), file=sys.stderr)

    def action_print(self, title, summary, detail=None, footer=None, color=UI.CYAN):
        if self.verbosity in {"normal", "debug"}:
            print(self.ui.action_card(title, summary, detail=detail, footer=footer, color=color), file=sys.stderr)

    def transcript_print(self, title, lines=None, color=UI.CYAN):
        if self.verbosity in {"normal", "debug"}:
            print(self.ui.transcript_item(title, lines or [], color=color), file=sys.stderr)

    def allow_command(self, command, allowed_commands, approval_prefixes=None):
        if allowed_commands:
            for allowed in allowed_commands:
                if command == allowed or command.startswith(allowed + " "):
                    break
            else:
                return False, "Command is outside the allowed command list for this task."
        if command_is_blocked(command):
            return False, "Blocked dangerous command."
        if self.command_permission == "deny":
            return False, "Command execution is disabled by permission mode."
        if self.command_permission == "ask":
            for prefix, ok in self.command_prefix_approval_cache.items():
                if command == prefix or command.startswith(prefix + " "):
                    return ok, "Command not approved by user." if not ok else ""
            if command in self.command_approval_cache:
                ok = self.command_approval_cache[command]
                return ok, "Command not approved by user." if not ok else ""
            ok = confirm_action("Command", command, "Verify or inspect the current task before continuing.", self.ui)
            self.command_approval_cache[command] = ok
            for prefix in approval_prefixes or []:
                if command == prefix or command.startswith(prefix + " "):
                    self.command_prefix_approval_cache[prefix] = ok
            return ok, "Command not approved by user." if not ok else ""
        return True, ""

    def allow_edit(self, label):
        if self.edit_permission == "deny":
            return False, "File edits are disabled by permission mode."
        if self.edit_permission == "ask":
            ok = confirm_action("Edit", label, "Modify the file as part of the approved task.", self.ui)
            return ok, "Edit not approved by user." if not ok else ""
        return True, ""

    def system_prompt(self, contract, memory_text):
        return textwrap.dedent(
            f"""\
            You are the backend half of a local coding partner.
            You do repo inspection, command execution, and code edits. You are not user-facing.

            Working directory: {self.workdir}
            Repo context:
            {git_context(self.workdir)}

            Repo memory:
            {memory_text or 'No repo memory yet.'}

            Task contract:
            {json.dumps(contract, ensure_ascii=False, indent=2)}

            Rules:
            - Follow the contract strictly.
            - If the contract contains pasted context/output, analyze that pasted evidence first.
            - Do not respond with generic reproduction commands when the user already pasted command output.
            - For database/foreign-key tasks, prioritize targeted searches for schema, migrations, SQLite/database initialization, account insertion, onboarding, and connection-test persistence.
            - Avoid repeated repo_overview calls. After one orientation pass, search/read specific source files or finish with findings.
            - Avoid dist/build output unless no source file exists.
            - Use tools instead of guessing.
            - Read files before editing them.
            - Respect edit_policy:
              - inspect: do not edit files
              - plan: do not edit files; produce a concrete plan
              - propose: do not edit files; inspect and propose changes that would be made
              - execute: inspect, then apply the requested changes if justified
            - Respect commands_allowed. Do not try commands outside it when that list is non-empty.
            - Final output must be a JSON report in the final tool with this shape:
              {{
                "summary": "brief factual summary",
                "findings": ["..."],
                "commands_run": ["..."],
                "files_read": ["..."],
                "files_changed": ["..."],
                "diff_summary": "unified diff of proposed changes (--- a/path, +++ b/path, @@ lines) for plan/propose mode; git diff --stat output for execute mode",
                "tests_run": ["..."],
                "risks": ["..."],
                "needs_approval": true_or_false,
                "plan": ["concrete step 1", "concrete step 2"]
              }}
            - In plan or propose mode: populate diff_summary with a real unified diff showing exactly what lines would change. Read each file first, then produce --- / +++ / @@ hunks. Populate plan with one entry per file change.
            - Return exactly one JSON object using the tool schema and no prose outside it.

            Tool schema:
            {{"tool":"TOOL_NAME","args":{{...}}}}

            Allowed tools:
            - fetch_url: {{"url":"https://..."}}
            - repo_overview: {{}}
            - list_files: {{"path":"optional path"}}
            - search_files: {{"query":"text or regex","path":"optional path"}}
            - read_file: {{"path":"file path","start":1,"end":500}}
            - run_command: {{"command":"shell command","timeout":30}}
            - write_file: {{"path":"file path","content":"full file content"}}
            - replace_lines: {{"path":"file path","start":10,"end":15,"content":"replacement lines"}}  ← preferred for surgical edits; read the file first to get line numbers
            - replace_in_file: {{"path":"file path","old":"exact old text","new":"replacement text","count":1}}
            - insert_after: {{"path":"file path","anchor":"exact anchor text","content":"text to insert","occurrence":1}}
            - final: {{"summary":"...","findings":[...],"files_read":[...],"files_changed":[...],"diff_summary":"...","commands_run":[...],"tests_run":[...],"risks":[...],"needs_approval":false,"plan":[...]}}
            """
        ).strip()

    def direct_report(self, contract, memory_text):
        resolved_hints = resolve_repo_file_hints(self.workdir, contract.get("files_of_interest") or [])
        if resolved_hints:
            contract = dict(contract)
            contract["files_of_interest"] = resolved_hints
        file_sections = []
        for path in (contract.get("files_of_interest") or [])[:3]:
            file_sections.append(f"File: {path}\n{read_file(self.workdir, path, 1, 220)}")
        command_section = ""
        if contract.get("commands_allowed"):
            command_section = "\n\nSuggested commands to inspect or reproduce:\n" + "\n".join(contract["commands_allowed"][:5])
        messages = [
            {
                "role": "system",
                "content": textwrap.dedent(
                    f"""\
                    You are the backend half of a local coding partner.
                    Produce exactly one JSON object with this schema:
                    {{
                      "summary": "brief factual summary",
                      "findings": ["..."],
                      "commands_run": ["..."],
                      "files_read": ["..."],
                      "files_changed": ["..."],
                      "diff_summary": "what changed or would change",
                      "tests_run": ["..."],
                      "risks": ["..."],
                      "needs_approval": true_or_false,
                      "plan": ["step 1", "step 2"]
                    }}

                    This is a direct analysis pass. Do not propose tool actions. Return JSON only.
                    If the contract contains pasted context/output, analyze that pasted evidence first.
                    Do not tell the user to run commands whose output they already pasted.
                    Repo memory:
                    {memory_text or 'No repo memory yet.'}
                    """
                ).strip(),
            },
            {
                "role": "user",
                "content": "Contract:\n"
                + json.dumps(contract, ensure_ascii=False, indent=2)
                + "\n\nRepo overview:\n"
                + repo_overview(self.workdir)
                + ("\n\n" + "\n\n".join(file_sections) if file_sections else "")
                + command_section,
            },
        ]
        with Spinner(self.ui, self.ui.style("analyzing", UI.DIM)):
            raw = self.chat(messages)
        report = normalize_backend_report(raw, fallback_message="Backend direct analysis failed.")
        if not report["files_read"]:
            report["files_read"] = [str(resolve_path(self.workdir, p)) for p in (contract.get("files_of_interest") or [])[:3]]
        if contract.get("edit_policy") in {"plan", "propose"}:
            report["needs_approval"] = True
        self.transcript_print(
            f"Inspected repo ({len(report.get('files_read') or [])} file(s) read)",
            [summarize_text(report.get("summary", ""), 220)] if report.get("summary") else [],
            color=UI.BLUE,
        )
        return report

    def parse_action(self, text):
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        if fence:
            data = load_json_layers(fence.group(1))
            if isinstance(data, dict) and "tool" in data and "args" in data:
                return data
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return obj
                except json.JSONDecodeError:
                    continue
        data = load_json_layers(text)
        if isinstance(data, dict) and "tool" in data and "args" in data:
            return data
        raise ValueError(f"Backend did not return a valid tool JSON object:\n{text}")

    def tool_result(self, tool, args, contract, tracker):
        self.last_tool_display = None
        try:
            if tool == "fetch_url":
                return fetch_url(args["url"])
            if tool == "repo_overview":
                return repo_overview(self.workdir)
            if tool == "list_files":
                return list_files(self.workdir, args.get("path", "."))
            if tool == "search_files":
                return search_files(self.workdir, args["query"], args.get("path", "."))
            if tool == "read_file":
                path = args["path"]
                tracker["files_read"].add(str(resolve_path(self.workdir, path)))
                result = read_file(self.workdir, path, args.get("start", 1), args.get("end", READ_FILE_DEFAULT_END))
                self.last_tool_display = {
                    "kind": "read",
                    "path": path,
                    "range": f"{args.get('start', 1)}-{args.get('end', 200)}",
                }
                return result
            if tool == "run_command":
                command = args["command"]
                allowed, reason = self.allow_command(
                    command,
                    contract.get("commands_allowed") or [],
                    contract.get("approval_prefixes") or [],
                )
                if not allowed:
                    return reason
                tracker["commands_run"].append(command)
                requested_timeout = int(args.get("timeout", 30))
                if tool == "run_command" and contract.get("task_kind") == "bootstrap_new":
                    timeout = min(requested_timeout, 300)
                else:
                    timeout = min(requested_timeout, 120)
                code, output = run_subprocess(command, cwd=self.workdir, timeout=timeout)
                self.last_tool_display = {
                    "kind": "command",
                    "command": command,
                    "exit_code": code,
                    "output": output,
                }
                return f"exit_code={code}\n{output}"
            if tool in {"write_file", "replace_in_file", "replace_lines", "insert_after"} and contract.get("edit_policy") != "execute":
                return f"Edit blocked by edit_policy={contract.get('edit_policy')}"
            if tool == "write_file":
                target = str(resolve_path(self.workdir, args["path"]))
                allowed, reason = self.allow_edit(target)
                if not allowed:
                    return reason
                before = self._read_text_for_diff(target)
                tracker["files_changed"].add(target)
                result = write_file(self.workdir, args["path"], args["content"])
                self.last_tool_display = self.edit_display(target, before, args["content"])
                return result
            if tool == "replace_in_file":
                target = str(resolve_path(self.workdir, args["path"]))
                allowed, reason = self.allow_edit(target)
                if not allowed:
                    return reason
                before = self._read_text_for_diff(target)
                tracker["files_changed"].add(target)
                result = replace_in_file(self.workdir, args["path"], args["old"], args["new"], int(args.get("count", 1)))
                after = self._read_text_for_diff(target)
                self.last_tool_display = self.edit_display(target, before, after)
                return result
            if tool == "replace_lines":
                target = str(resolve_path(self.workdir, args["path"]))
                allowed, reason = self.allow_edit(target)
                if not allowed:
                    return reason
                before = self._read_text_for_diff(target)
                tracker["files_changed"].add(target)
                result = replace_lines(self.workdir, args["path"], args["start"], args["end"], args.get("content", ""))
                after = self._read_text_for_diff(target)
                self.last_tool_display = self.edit_display(target, before, after)
                return result
            if tool == "insert_after":
                target = str(resolve_path(self.workdir, args["path"]))
                allowed, reason = self.allow_edit(target)
                if not allowed:
                    return reason
                before = self._read_text_for_diff(target)
                tracker["files_changed"].add(target)
                result = insert_after(self.workdir, args["path"], args["anchor"], args["content"], int(args.get("occurrence", 1)))
                after = self._read_text_for_diff(target)
                self.last_tool_display = self.edit_display(target, before, after)
                return result
            return f"Unknown tool: {tool}"
        except subprocess.TimeoutExpired:
            command = args.get("command", tool)
            return f"Command timed out: {command}"
        except KeyError as exc:
            return f"Missing required argument: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Tool error: {exc}"

    def _read_text_for_diff(self, path):
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def edit_display(self, path, before, after):
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        added = sum(1 for line in difflib.ndiff(before_lines, after_lines) if line.startswith("+ "))
        removed = sum(1 for line in difflib.ndiff(before_lines, after_lines) if line.startswith("- "))
        diff = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=path,
                tofile=path,
                lineterm="",
                n=2,
            )
        )
        preview = [line for line in diff if not line.startswith(("---", "+++"))][:16]
        return {
            "kind": "edit",
            "path": path,
            "added": added,
            "removed": removed,
            "preview": preview,
        }

    def print_tool_transcript(self, tool, args, result, elapsed):
        if self.verbosity not in {"normal", "debug"}:
            return
        display = self.last_tool_display or {}
        if display.get("kind") == "command":
            cmd = display["command"]
            status = self.ui.style(f"  exit {display['exit_code']} · {elapsed:.1f}s", UI.DIM)
            print(self.ui.tool_line(f"Run  {cmd}{status}", color=UI.BLUE), file=sys.stderr)
            if self.verbosity == "debug":
                for line in self.ui.truncate_lines(display.get("output", ""), limit=4):
                    print(f"       {line}", file=sys.stderr)
            return
        if display.get("kind") == "edit":
            path = display["path"]
            added, removed = display["added"], display["removed"]
            print(self.ui.tool_line(f"Edit  {path}  +{added} −{removed}", color=UI.GREEN), file=sys.stderr)
            if self.verbosity == "debug":
                for line in (self.ui.render_diff_lines(display.get("preview") or []) or [])[:8]:
                    print(f"       {line}", file=sys.stderr)
            return
        if display.get("kind") == "read":
            print(self.ui.tool_line(f"Read  {display['path']}:{display['range']}", color=UI.CYAN), file=sys.stderr)
            return
        if tool == "search_files":
            matches = 0 if result == "(no matches)" else len([line for line in result.splitlines() if ":" in line])
            q = args.get("query", "")[:40]
            print(self.ui.tool_line(f"Search  {q!r}  in {args.get('path', '.')}  ({matches} matches)", color=UI.CYAN), file=sys.stderr)
            return
        if tool == "list_files":
            count = len([line for line in result.splitlines() if line.strip()])
            print(self.ui.tool_line(f"List  {args.get('path', '.')}  ({count} files)", color=UI.CYAN), file=sys.stderr)
            return
        if tool == "repo_overview":
            print(self.ui.tool_line("Overview  repo structure", color=UI.CYAN), file=sys.stderr)

    def run_contract(self, contract, memory_text):
        self.command_approval_cache = {}
        self.command_prefix_approval_cache = {}
        if contract.get("edit_policy") in {"inspect", "plan", "propose"} and contract.get("files_of_interest"):
            return self.direct_report(contract, memory_text)
        messages = [{"role": "system", "content": self.system_prompt(contract, memory_text)}]
        if contract.get("edit_policy") in {"inspect", "plan", "propose"}:
            messages.append({"role": "user", "content": "This is a non-editing backend pass unless edit_policy is execute. Inspect first and avoid speculative claims."})
        tracker = {
            "commands_run": [],
            "files_read": set(),
            "files_changed": set(),
        }
        self.last_backend_action = None
        self.last_backend_action_signature = None
        self.repeated_backend_action_count = 0
        invalid_action_count = 0

        for step in range(1, self.max_steps + 1):
            intent = self.print_step_intent(step, contract, messages)
            spinner_label = f"step {step}/{self.max_steps}"
            if intent and step == 1:
                spinner_label += f"  ·  {intent}"
            with Spinner(self.ui, self.ui.style(spinner_label, UI.DIM)):
                chat_started = time.monotonic()
                try:
                    raw = self.chat(messages)
                except TimeoutError:
                    if step == 1:
                        try:
                            raw = self.chat(messages)
                        except TimeoutError:
                            return normalize_backend_report(
                                {
                                    "summary": f"Backend model timed out while {spinner_label}.",
                                    "commands_run": tracker["commands_run"],
                                    "files_read": sorted(tracker["files_read"]),
                                    "files_changed": sorted(tracker["files_changed"]),
                                    "risks": ["Backend model timeout; task context was preserved for retry."],
                                }
                            )
                    else:
                        return normalize_backend_report(
                            {
                                "summary": f"Backend model timed out while {spinner_label}.",
                                "commands_run": tracker["commands_run"],
                                "files_read": sorted(tracker["files_read"]),
                                "files_changed": sorted(tracker["files_changed"]),
                                "risks": ["Backend model timeout; task context was preserved for retry."],
                            }
                        )
                chat_elapsed = time.monotonic() - chat_started
            try:
                action = self.parse_action(raw)
            except Exception as exc:  # noqa: BLE001
                invalid_action_count += 1
                self.trace_print(f"invalid backend action after {chat_elapsed:.1f}s; asking it to retry")
                messages.append(
                    {
                        "role": "user",
                        "content": self.invalid_action_hint(contract, step, exc, invalid_action_count, tracker),
                    }
                )
                if invalid_action_count >= 4:
                    self.transcript_print(
                        "Stopped backend loop",
                        ["Backend did not produce a valid tool call after repeated retries."],
                        color=UI.YELLOW,
                    )
                    return normalize_backend_report(
                        {
                            "summary": "Stopped because the backend failed to produce a valid tool call.",
                            "commands_run": tracker["commands_run"],
                            "files_read": sorted(tracker["files_read"]),
                            "files_changed": sorted(tracker["files_changed"]),
                            "needs_approval": False,
                            "risks": ["Backend produced invalid actions repeatedly before making progress."],
                        }
                    )
                continue

            tool = action["tool"]
            args = action.get("args") or {}
            self.last_backend_action = action
            signature = self.action_signature(tool, args)
            if signature == self.last_backend_action_signature:
                self.repeated_backend_action_count += 1
            else:
                self.repeated_backend_action_count = 0
            self.last_backend_action_signature = signature
            if self.show_raw_actions and self.verbosity == "debug":
                self.trace_print("raw action: " + json.dumps(action, ensure_ascii=False))
            messages.append({"role": "assistant", "content": json.dumps(action)})

            if tool == "final":
                message = args.get("message", "")
                if isinstance(message, dict) or (isinstance(message, str) and message.strip().startswith("{")):
                    report = normalize_backend_report(message, fallback_message="Backend finished without a detailed report.")
                else:
                    report = normalize_backend_report(args, fallback_message="Backend finished without a detailed report.")
                report["commands_run"] = report["commands_run"] or tracker["commands_run"]
                report["files_read"] = report["files_read"] or sorted(tracker["files_read"])
                report["files_changed"] = report["files_changed"] or sorted(tracker["files_changed"])
                if contract.get("edit_policy") in {"plan", "propose"}:
                    report["needs_approval"] = True
                if contract.get("edit_policy") == "execute" and report["files_changed"]:
                    _, diff_stat = run_subprocess("git diff HEAD --stat", cwd=self.workdir, timeout=10)
                    if diff_stat.strip():
                        report["diff_summary"] = report["diff_summary"] or diff_stat.strip()
                        self.transcript_print("diff --stat", [diff_stat.strip()[:220]], color=UI.CYAN)
                    test_cmds = [c for c in (contract.get("commands_of_interest") or []) if any(t in c for t in TEST_COMMAND_PATTERNS)]
                    if test_cmds and not report["tests_run"]:
                        cmd = test_cmds[0]
                        self.transcript_print("Running tests", [cmd], color=UI.CYAN)
                        code, test_out = run_subprocess(cmd, cwd=self.workdir, timeout=60)
                        report["tests_run"] = [f"{cmd}:\n{test_out[:1000]}"]
                        self.transcript_print("Test results", [test_out[:220]], color=UI.GREEN if code == 0 else UI.RED)
                missing = self.verify_contract_outputs(contract)
                if missing:
                    report["summary"] = "Task ran but verification did not fully pass."
                    report["risks"] = report.get("risks") or []
                    report["risks"].append("Missing expected artifacts: " + ", ".join(missing))
                elif contract.get("verification_checks") and contract.get("task_kind") == "bootstrap_new" and "verified" not in report["summary"].lower():
                    report["summary"] = report["summary"].rstrip(".") + ". Verified expected project files."
                self.transcript_print("Finished backend report", [summarize_text(report.get("summary", "Backend finished."), 220)], color=UI.GREEN)
                return report

            tool_started = time.monotonic()
            result = self.tool_result(tool, args, contract, tracker)
            tool_elapsed = time.monotonic() - tool_started
            self.print_tool_transcript(tool, args, result, tool_elapsed)
            messages.append({"role": "user", "content": f"Tool result for {tool}:\n{result}"})
            if self.repeated_backend_action_count >= 2 and tool != "final":
                self.transcript_print(
                    "Stopped backend loop",
                    [f"Repeated {summarize_action(tool, args)} without new information."],
                    color=UI.YELLOW,
                )
                return normalize_backend_report(
                    {
                        "summary": "Stopped because the backend repeated the same action without new information.",
                        "commands_run": tracker["commands_run"],
                        "files_read": sorted(tracker["files_read"]),
                        "files_changed": sorted(tracker["files_changed"]),
                        "needs_approval": False,
                        "risks": [
                            "Backend repeated the same action multiple times without converging on a finding.",
                        ],
                    }
                )

        return normalize_backend_report(
            {
                "summary": "Stopped after reaching the maximum backend tool steps.",
                "commands_run": tracker["commands_run"],
                "files_read": sorted(tracker["files_read"]),
                "files_changed": sorted(tracker["files_changed"]),
                "needs_approval": False,
                "risks": ["Backend step limit reached before completion."],
            }
        )

    def print_step_intent(self, step, contract, messages):
        return self.step_intent(step, contract, messages)

    def step_intent(self, step, contract, messages):
        if step == 1:
            if has_database_context(json.dumps(contract)):
                return "target database/schema and connection-test paths"
            return "inspect the repo and find the relevant path"
        action = self.last_backend_action or {}
        tool = action.get("tool")
        args = action.get("args") or {}
        if tool:
            if self.repeated_backend_action_count > 0:
                return "repeating " + summarize_action(tool, args)
            return "after " + summarize_action(tool, args)
        return "continue the investigation"

    @staticmethod
    def action_signature(tool, args):
        try:
            normalized_args = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
        except TypeError:
            normalized_args = str(args)
        return f"{tool}:{normalized_args}"

    def invalid_action_hint(self, contract, step, error, invalid_action_count, tracker):
        database_context = has_database_context(json.dumps(contract))
        last_result = []
        if tracker["commands_run"]:
            last_result.append("commands: " + ", ".join(tracker["commands_run"][-2:]))
        if tracker["files_read"]:
            last_result.append("files read: " + ", ".join(sorted(tracker["files_read"])[-2:]))
        progress = " Previous progress: " + "; ".join(last_result) + "." if last_result else ""
        if database_context:
            if step <= 1 or invalid_action_count == 1:
                tool_hint = 'Use {"tool":"repo_overview","args":{}}.'
            elif invalid_action_count == 2:
                tool_hint = (
                    'Use {"tool":"search_files","args":{"query":"sqlite|foreign key|schema|migration|account|connection|onboarding|testConnection","path":"src"}}.'
                )
            else:
                tool_hint = 'Use a targeted read/search tool, or {"tool":"final","args":{"summary":"...","findings":[...]}} if you already have the finding.'
        else:
            tool_hint = 'Use a single JSON tool call, then wait for the result.'
        return (
            "Your previous response was invalid. Return exactly one JSON object only. "
            f"Error: {error}.{progress} {tool_hint}"
        )

    def verify_contract_outputs(self, contract):
        missing = []
        for path in contract.get("target_paths") or []:
            if not Path(self.workdir, path).exists():
                missing.append(path)
        return missing


class LocalPartner:
    def __init__(
        self,
        frontend_model,
        backend_model,
        ollama,
        workdir,
        command_permission="ask",
        edit_permission="ask",
        max_steps=MAX_TOOL_STEPS,
        verbosity=DEFAULT_VERBOSITY,
        show_raw_actions=False,
        mode=DEFAULT_MODE,
    ):
        self.frontend_model = frontend_model
        self.backend_model = backend_model
        self.ollama = ollama.rstrip("/")
        self.workdir = str(Path(workdir).resolve())
        self.command_permission = command_permission
        self.edit_permission = edit_permission
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.show_raw_actions = show_raw_actions
        self.mode = mode
        self.ui = UI()
        self.pending_plan = None
        self.latest_plan = None
        self.last_report = None
        self.last_status = None
        self.on_token = None
        self.last_streamed = False
        self.backend_runs = 0
        ensure_memory_files(self.workdir)
        self.history = load_chat_history(self.workdir)
        self.executor = LocalCodeAgent(
            model=self.backend_model,
            ollama=self.ollama,
            workdir=self.workdir,
            command_permission=self.command_permission,
            edit_permission=self.edit_permission,
            max_steps=self.max_steps,
            verbosity=self.verbosity,
            show_raw_actions=self.show_raw_actions,
        )

    def _prune_history(self):
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]
        save_chat_history(self.workdir, self.history)

    def sync_executor(self):
        self.executor.model = self.backend_model
        self.executor.command_permission = self.command_permission
        self.executor.edit_permission = self.edit_permission
        self.executor.max_steps = self.max_steps
        self.executor.verbosity = self.verbosity
        self.executor.show_raw_actions = self.show_raw_actions

    def trace_print(self, message):
        if self.verbosity == "debug":
            prefix = self.ui.style("frontend", UI.BOLD, UI.YELLOW)
            print(f"{prefix} {message}", file=sys.stderr)

    def milestone(self, message):
        if self.verbosity in {"normal", "debug"}:
            print(self.ui.tool_line(message, color=UI.BLUE), file=sys.stderr)

    def chat(self, model, messages):
        return ollama_chat(self.ollama, model, messages)

    def _chat_streaming(self, model, messages):
        """Stream a response via on_token callbacks; return full accumulated text."""
        chunks = []
        first = True
        for chunk in ollama_stream(self.ollama, model, messages):
            if first:
                sys.stdout.write("\n")
                sys.stdout.flush()
                first = False
            self.on_token(chunk)
            chunks.append(chunk)
        full = "".join(chunks)
        if full and not full.endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.last_streamed = True
        return full.strip()

    def _stream_frontend_turn(self, model, messages):
        """Stream frontend turn: text replies go to on_token, delegation JSON is buffered silently."""
        chunks = []
        decided = False
        is_text = False
        for chunk in ollama_stream(self.ollama, model, messages):
            chunks.append(chunk)
            if not decided:
                so_far = "".join(chunks).lstrip()
                if so_far:
                    is_text = not so_far.startswith("{")
                    decided = True
                    if is_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        for c in chunks:
                            self.on_token(c)
            elif is_text:
                self.on_token(chunk)
        full = "".join(chunks)
        if is_text:
            if full and not full.endswith("\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
            self.last_streamed = True
        return full.strip()

    def frontend_system_prompt(self):
        return textwrap.dedent(
            f"""\
            You are the frontend half of a local coding partner.
            You are the only user-facing model. The backend never speaks directly to the user.

            Working directory: {self.workdir}
            Repo context:
            {git_context(self.workdir)}

            Repo memory:
            {load_repo_memory(self.workdir)}

            Current mode: {self.mode}

            Behavior:
            - In chat mode, do not delegate to the backend.
            - In hybrid mode, execute clear coding tasks by delegating immediately. Propose only when the task is broad, risky, or genuinely ambiguous.
            - In agent mode, delegation is allowed immediately for code/repo tasks.
            - If the user pasted logs/output or asks "does this tell you anything?", analyze the pasted evidence directly.
            - Do not ask the user to share or rerun output that is already in their message.
            - When delegation is useful, produce a strict contract for the backend.
            - For clear bootstrap tasks, use edit_policy=execute.
            - When you emit a contract, set edit_policy to exactly one literal value: inspect, plan, propose, or execute.
            - Include task_kind, execution_strategy, target_paths, and verification_checks in delegated contracts.

            For direct answers: respond with plain natural-language text only.
            To delegate to the backend: respond with ONLY this JSON object (no other text before or after):
            {{
              "mode":"delegate",
              "message":"brief explanation shown to user while backend works",
              "contract": {{
                "goal":"...",
                "scope":["src","electron"],
                "constraints":["Do not install packages"],
                "commands_allowed":["npm test"],
                "edit_policy":"propose",
                "task_kind":"inspection",
                "execution_strategy":"direct",
                "expected_result":"...",
                "files_of_interest":["src/main.tsx"],
                "target_paths":["src/main.tsx"],
                "verification_checks":["src/main.tsx exists"]
              }}
            }}
            """
        ).strip()

    def parse_frontend_action(self, text):
        text = text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        if fence:
            data = load_json_layers(fence.group(1))
            if isinstance(data, dict) and data.get("mode") in {"reply", "delegate"}:
                if data.get("mode") == "reply" and isinstance(data.get("message"), str):
                    data["message"] = unwrap_frontend_reply_text(data["message"])
                return data
        decoder = json.JSONDecoder()
        if text.startswith("{"):
            try:
                obj, end = decoder.raw_decode(text)
                if isinstance(obj, dict) and obj.get("mode") in {"reply", "delegate"} and text[end:].strip() == "":
                    if obj.get("mode") == "reply" and isinstance(obj.get("message"), str):
                        obj["message"] = unwrap_frontend_reply_text(obj["message"])
                    return obj
            except json.JSONDecodeError:
                pass
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fence:
            data = load_json_layers(fence.group(1))
            surrounding = (text[:fence.start()] + text[fence.end():]).strip()
            if (
                isinstance(data, dict)
                and data.get("mode") == "delegate"
                and "?" not in surrounding
            ):
                return data
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            surrounding = (text[:i] + text[end:]).strip()
            if (
                isinstance(obj, dict)
                and obj.get("mode") == "delegate"
                and "?" not in surrounding
            ):
                return obj
        if text:
            return {"mode": "reply", "message": unwrap_frontend_reply_text(text)}
        raise ValueError(f"Frontend did not return a valid JSON action:\n{text}")

    def wants_code_action(self, prompt):
        return bool(CODE_ACTION_RE.search(prompt))

    def wants_edit(self, prompt):
        return bool(EDIT_INTENT_RE.search(prompt))

    def classify_contract(self, prompt, planning=False):
        state = inspect_workdir_state(self.workdir)
        return normalize_contract({}, prompt, self.mode, planning=planning, workdir_state=state)

    def must_delegate(self, prompt, planning=False):
        contract = self.classify_contract(prompt, planning=planning)
        if planning:
            return True
        if contract["task_kind"] in {"inspection", "edit_existing", "bootstrap_new"}:
            return True
        if has_pasted_context(prompt):
            return True
        if infer_command_hints(prompt):
            return True
        return bool(re.search(r"\b(inspect|check|search|read|tell me what .* does|look in|explain this repo|analy[sz]e|fails?|failing|error|broken|build)\b", prompt, re.I))

    def progress_label(self, contract):
        if contract.get("execution_strategy") == "plan_only" or contract.get("edit_policy") in {"plan", "propose"}:
            return "planning task"
        if contract.get("task_kind") == "bootstrap_new":
            target = contract.get("target_directory") or Path(self.workdir).name
            return f"scaffolding project ({target})"
        if contract.get("task_kind") == "inspection":
            return "investigating repo"
        if contract.get("verification_checks"):
            return "verifying files"
        return "executing task"

    def frontend_turn(self, user_prompt, planning=False):
        messages = [{"role": "system", "content": self.frontend_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_prompt})
        if planning:
            messages.append({"role": "user", "content": "Create a concrete backend plan. Do not execute edits."})
        elif self.mode == "agent" and self.wants_code_action(user_prompt):
            messages.append({"role": "user", "content": "Agent mode is active. Prefer delegation to the backend for code or repo tasks."})
        elif self.mode == "hybrid" and self.wants_code_action(user_prompt):
            messages.append({"role": "user", "content": "Hybrid mode is active. Delegate if repo inspection or implementation detail is needed. Prefer propose before execute for edit requests."})
        elif self.mode == "chat":
            messages.append({"role": "user", "content": "Chat mode is active. Do not delegate; answer conversationally."})

        started = time.monotonic()
        try:
            if self.on_token:
                raw = self._stream_frontend_turn(self.frontend_model, messages)
            else:
                with Spinner(self.ui, self.ui.style("thinking", UI.DIM)):
                    raw = self.chat(self.frontend_model, messages)
        except TimeoutError:
            if self.on_token:
                raw = self._stream_frontend_turn(self.frontend_model, messages)
            else:
                with Spinner(self.ui, self.ui.style("retrying frontend", UI.DIM)):
                    raw = self.chat(self.frontend_model, messages)
        elapsed = time.monotonic() - started
        action = self.parse_frontend_action(raw)
        if self.show_raw_actions and self.verbosity == "debug":
            self.trace_print("raw action: " + json.dumps(action, ensure_ascii=False))
        else:
            self.trace_print(
                self.ui.style("Action", UI.MAGENTA, UI.BOLD)
                + " "
                + ("delegate to backend" if action["mode"] == "delegate" else "reply directly")
                + " "
                + self.ui.style(f"({elapsed:.1f}s)", UI.DIM)
            )
        return action

    def frontend_finalize(self, original_prompt, backend_report, frontend_message=""):
        messages = [{"role": "system", "content": textwrap.dedent(
            """\
            You are the frontend half of a local coding partner.
            A backend code worker has completed repo work and returned a factual report.
            Your job is to explain the result clearly and naturally to the user.
            Do not mention hidden orchestration or say 'the backend said' unless the user explicitly asks.
            If the user supplied pasted output, address what that output indicates directly.
            Do not ask for the same output again unless the report says it is missing or truncated.
            Preserve uncertainty where the report is uncertain.
            Be concise but useful.
            Return plain text only.
            """
        ).strip()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": original_prompt})
        messages.append({"role": "user", "content": (("Context to preserve:\n" + frontend_message + "\n\n") if frontend_message else "") + "Backend report:\n" + json.dumps(backend_report, ensure_ascii=False, indent=2)})
        started = time.monotonic()
        if self.on_token:
            reply = self._chat_streaming(self.frontend_model, messages)
        else:
            with Spinner(self.ui, self.ui.style("composing", UI.DIM)):
                reply = self.chat(self.frontend_model, messages).strip()
        elapsed = time.monotonic() - started
        self.trace_print(self.ui.style("Compose", UI.MAGENTA, UI.BOLD) + " frontend answer " + self.ui.style(f"({elapsed:.1f}s)", UI.DIM))
        return reply

    def frontend_summarize_proposal(self, original_prompt, contract, backend_report, frontend_message=""):
        messages = [{"role": "system", "content": textwrap.dedent(
            """\
            You are the frontend half of a local coding partner.
            The backend inspected the repo and prepared a proposal.
            Summarize what would change, mention key files and risks, and end by asking the user to approve before edits happen.
            Return plain text only.
            """
        ).strip()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": original_prompt})
        messages.append({"role": "user", "content": (("Context to preserve:\n" + frontend_message + "\n\n") if frontend_message else "") + "Contract:\n" + json.dumps(contract, ensure_ascii=False, indent=2) + "\n\nBackend report:\n" + json.dumps(backend_report, ensure_ascii=False, indent=2)})
        if self.on_token:
            return self._chat_streaming(self.frontend_model, messages)
        with Spinner(self.ui, self.ui.style("preparing proposal", UI.DIM)):
            reply = self.chat(self.frontend_model, messages).strip()
        return reply

    def log_run(self, user_prompt, result, contract=None, report=None, status="completed"):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "user_prompt": user_prompt,
            "status": status,
            "frontend_model": self.frontend_model,
            "backend_model": self.backend_model,
            "command_permission": self.command_permission,
            "edit_permission": self.edit_permission,
            "contract": contract,
            "backend_report": report,
            "result": result,
        }
        append_run_log(self.workdir, entry)

    def _run_backend(self, contract):
        self.backend_runs += 1
        return self.executor.run_contract(contract, load_repo_memory(self.workdir))

    def update_memory(self):
        paths = memory_paths(self.workdir)
        current = paths["project"].read_text(encoding="utf-8", errors="replace").strip()
        recent = load_recent_runs(paths["runs"], limit=8)
        if not recent:
            return
        messages = [{
            "role": "user",
            "content": (
                "Update this project memory file with any new facts learned from recent runs.\n\n"
                f"Current project.md:\n{current}\n\n"
                f"Recent runs:\n{recent}\n\n"
                "Rules: only add new reusable facts (commands, key files, stack, patterns). "
                "Keep each addition to one line. Do not remove existing entries. "
                "Return ONLY the full updated file content, no explanation."
            ),
        }]
        with Spinner(self.ui, self.ui.style("updating memory", UI.DIM)):
            updated = ollama_chat(self.ollama, self.backend_model, messages)
        updated = updated.strip()
        if updated and updated != current:
            paths["project"].write_text(updated + "\n", encoding="utf-8")
            self.milestone("project memory updated")

    def apply_pending_plan(self):
        if not self.pending_plan:
            return None
        contract = dict(self.pending_plan["contract"])
        contract["edit_policy"] = "execute"
        contract["execution_strategy"] = "inspect_then_execute"
        self.milestone("applying approved plan")
        report = self._run_backend(contract)
        reply = self.frontend_finalize(self.pending_plan["original_prompt"], report, self.pending_plan.get("frontend_message", ""))
        self.last_report = report
        self.last_status = "completed"
        self.latest_plan = {"contract": contract, "report": report, "original_prompt": self.pending_plan["original_prompt"]}
        self.pending_plan = None
        self.history.append({"role": "user", "content": "Apply the approved plan."})
        self.history.append({"role": "assistant", "content": reply})
        save_chat_history(self.workdir, self.history)
        self.log_run("/apply", reply, contract=contract, report=report)
        return reply

    def run_readonly_turn(self, user_prompt):
        self.sync_executor()
        self._prune_history()
        contract = normalize_contract({}, user_prompt, "hybrid", workdir_state=inspect_workdir_state(self.workdir))
        contract["edit_policy"] = "inspect"
        contract["files_of_interest"] = resolve_repo_file_hints(self.workdir, contract.get("files_of_interest") or [])
        self.milestone("read-only inspection")
        report = self._run_backend(contract)
        reply = self.frontend_finalize(user_prompt, report)
        self.last_report = report
        self.last_status = "completed"
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": reply})
        self.log_run(user_prompt, reply, contract=contract, report=report)
        return reply

    def run_turn(self, user_prompt, planning=False):
        self.sync_executor()
        self._prune_history()
        if self.pending_plan and PROMPT_YES_RE.search(user_prompt.strip()):
            applied = self.apply_pending_plan()
            if applied is not None:
                return applied

        if self.mode == "agent":
            contract = self.classify_contract(user_prompt)
            contract["files_of_interest"] = resolve_repo_file_hints(self.workdir, contract.get("files_of_interest") or [])
            self.milestone(self.progress_label(contract))
            report = self._run_backend(contract)
            reply = self.frontend_finalize(user_prompt, report)
            self.last_report = report
            self.last_status = "completed"
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": reply})
            self.log_run(user_prompt, reply, contract=contract, report=report)
            return reply

        if self.mode == "hybrid" and self.must_delegate(user_prompt, planning=planning):
            contract = self.classify_contract(user_prompt, planning=planning)
            contract["files_of_interest"] = resolve_repo_file_hints(self.workdir, contract.get("files_of_interest") or [])
            self.milestone(self.progress_label(contract))
            report = self._run_backend(contract)
            if contract["edit_policy"] in {"plan", "propose"} or report.get("needs_approval"):
                self.pending_plan = {
                    "original_prompt": user_prompt,
                    "frontend_message": "",
                    "contract": contract,
                    "report": report,
                }
                self.latest_plan = dict(self.pending_plan)
                reply = self.frontend_summarize_proposal(user_prompt, contract, report, "")
                self.last_report = report
                self.last_status = "proposal"
                self.history.append({"role": "user", "content": user_prompt})
                self.history.append({"role": "assistant", "content": reply})
                self.log_run(user_prompt, reply, contract=contract, report=report, status="proposal")
                return reply
            reply = self.frontend_finalize(user_prompt, report, "")
            self.last_report = report
            self.last_status = "completed"
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": reply})
            self.log_run(user_prompt, reply, contract=contract, report=report)
            return reply

        action = self.frontend_turn(user_prompt, planning=planning)
        if action["mode"] == "reply":
            message = action.get("message", "").strip()
            self.last_report = None
            self.last_status = "reply"
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": message})
            self.log_run(user_prompt, message, status="reply")
            return message

        frontend_message = action.get("message", "").strip()
        contract = normalize_contract(
            action.get("contract") or {},
            user_prompt,
            self.mode,
            planning=planning,
            workdir_state=inspect_workdir_state(self.workdir),
        )
        contract["files_of_interest"] = resolve_repo_file_hints(self.workdir, contract.get("files_of_interest") or [])
        if planning:
            contract["edit_policy"] = "plan"
            contract["execution_strategy"] = "plan_only"

        self.milestone(self.progress_label(contract))
        report = self._run_backend(contract)
        if contract["edit_policy"] in {"plan", "propose"} or report.get("needs_approval"):
            self.pending_plan = {
                "original_prompt": user_prompt,
                "frontend_message": frontend_message,
                "contract": contract,
                "report": report,
            }
            self.latest_plan = dict(self.pending_plan)
            reply = self.frontend_summarize_proposal(user_prompt, contract, report, frontend_message)
            self.last_report = report
            self.last_status = "proposal"
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": reply})
            self.log_run(user_prompt, reply, contract=contract, report=report, status="proposal")
            return reply

        reply = self.frontend_finalize(user_prompt, report, frontend_message)
        self.last_report = report
        self.last_status = "completed"
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": reply})
        self.log_run(user_prompt, reply, contract=contract, report=report)
        return reply

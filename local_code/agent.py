import json
import difflib
import re
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
from datetime import datetime
from pathlib import Path

from .config import (
    CODE_ACTION_RE,
    DEFAULT_MODE,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA,
    DEFAULT_TOOL_CALLING,
    DEFAULT_VERBOSITY,
    EDIT_INTENT_RE,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_STEPS,
    PROMPT_YES_RE,
    READ_FILE_DEFAULT_END,
    STRUCTURED_TEMPERATURE,
    TEST_COMMAND_PATTERNS,
)
from .model_profiles import classify_model
from .context import build_context_usage
from .providers import OllamaProvider
from .contracts import (
    compute_execution_status,
    has_database_context,
    has_pasted_context,
    infer_command_hints,
    inspect_workdir_state,
    load_json_layers,
    normalize_backend_report,
    normalize_contract,
    resolve_repo_file_hints,
    unwrap_frontend_reply_text,
    validate_plan_report,
)
from .intelligence import DecisionService, IntelligenceStore, atomic_write_text, execute_decision_command
from .memory import append_run_log, ensure_memory_files, load_chat_history, load_recent_runs, load_repo_memory, memory_paths, save_chat_history
from .permissions import command_is_blocked, confirm_action
from .tool_specs import native_tool_definitions, tool_prompt_lines, validate_tool_call
from .tools import (
    fetch_url,
    build_project_profile,
    format_project_profile,
    format_repo_map,
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
    search_web,
    write_file,
)
from .ui import Spinner, UI, summarize_action, summarize_text


class LocalCodeAgent:
    def __init__(
        self,
        model,
        ollama=DEFAULT_OLLAMA,
        workdir=".",
        command_permission="ask",
        edit_permission="ask",
        max_steps=MAX_TOOL_STEPS,
        verbosity=DEFAULT_VERBOSITY,
        show_raw_actions=False,
        tool_calling=DEFAULT_TOOL_CALLING,
        provider=None,
        observer=None,
        confirm_hook=None,
    ):
        self.model = model
        self.provider = provider or OllamaProvider(ollama)
        self.ollama = getattr(self.provider, "base_url", ollama)
        # Optional UI bridges. `observer(event_dict)` receives progress events
        # (milestones, tool activity); `confirm_hook(kind, label, content)`
        # replaces the blocking stdin prompt. Both stay None for the plain REPL.
        self.observer = observer
        self.confirm_hook = confirm_hook
        self.workdir = str(Path(workdir).resolve())
        self.command_permission = command_permission
        self.edit_permission = edit_permission
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.show_raw_actions = show_raw_actions
        self.tool_calling = tool_calling
        self.ui = UI()
        self.last_tool_display = None
        self.last_backend_action = None
        self.last_backend_action_signature = None
        self.repeated_backend_action_count = 0
        self._action_seen_counts: dict = {}
        self.cancel_event = threading.Event()
        self.command_approval_cache = {}
        self.command_prefix_approval_cache = {}

    def chat(self, messages):
        # Every backend self.chat caller wants a single JSON object (tool action
        # or report), so constrain output to JSON and decode deterministically.
        # This is the dominant reliability lever for weak local models.
        return self.provider.chat(self.model, messages, fmt="json", temperature=STRUCTURED_TEMPERATURE)

    def chat_tools(self, messages, tools):
        return self.provider.chat_tools(self.model, messages, tools)

    def model_profile(self):
        return classify_model(self.model)

    def emit(self, kind, **data):
        if self.observer is not None:
            self.observer({"kind": kind, "source": "backend", **data})

    def trace_print(self, message):
        if self.verbosity != "debug":
            return
        if self.observer is not None:
            self.emit("trace", text=message)
            return
        prefix = self.ui.style("backend", UI.BOLD, UI.BLUE)
        print(f"{prefix} {message}", file=sys.stderr)

    def milestone(self, message):
        if self.verbosity not in {"normal", "debug"}:
            return
        if self.observer is not None:
            self.emit("milestone", text=message)
            return
        print(self.ui.tool_line(message, color=UI.BLUE), file=sys.stderr)

    def action_print(self, title, summary, detail=None, footer=None, color=UI.CYAN):
        if self.verbosity not in {"normal", "debug"}:
            return
        if self.observer is not None:
            self.emit("action", title=title, summary=summary, detail=detail, footer=footer)
            return
        print(self.ui.action_card(title, summary, detail=detail, footer=footer, color=color), file=sys.stderr)

    def transcript_print(self, title, lines=None, color=UI.CYAN):
        if self.verbosity not in {"normal", "debug"}:
            return
        if self.observer is not None:
            self.emit("transcript", title=title, lines=list(lines or []))
            return
        print(self.ui.transcript_item(title, lines or [], color=color), file=sys.stderr)

    def allowed_tool_names(self, contract):
        if contract.get("read_only") or contract.get("edit_policy") == "inspect":
            return {
                "repo_map",
                "repo_overview",
                "list_files",
                "search_files",
                "read_file",
                "final",
            }
        return {
            "search_web",
            "fetch_url",
            "repo_map",
            "repo_overview",
            "list_files",
            "search_files",
            "read_file",
            "run_command",
            "write_file",
            "replace_lines",
            "replace_in_file",
            "insert_after",
            "final",
        }

    def allowed_tools_text(self, contract):
        return tool_prompt_lines(self.allowed_tool_names(contract))

    def native_tools(self, contract):
        return native_tool_definitions(self.allowed_tool_names(contract))

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
            ok = self._confirm("Command", command, "Verify or inspect the current task before continuing.")
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
            ok = self._confirm("Edit", label, "Modify the file as part of the approved task.")
            return ok, "Edit not approved by user." if not ok else ""
        return True, ""

    def _confirm(self, kind, label, content):
        if self.confirm_hook is not None:
            return bool(self.confirm_hook(kind, label, content))
        return confirm_action(kind, label, content, self.ui)

    def _approved_plan_rules_block(self, contract):
        """Rules injected when the contract carries a user-approved plan.

        Rendered inside system_prompt's Rules list; the 12-space indent matches
        the template body so the bullets align with the surrounding rules.
        """
        if not contract.get("approved_plan"):
            return ""
        indent = " " * 12
        lines = [
            "- An approved plan exists in the contract's approved_plan field. "
            "Your job is to apply each step, not to re-investigate.",
            "- The plan is already approved; do not re-evaluate whether the changes are justified.",
            "- Read a file only immediately before editing it; do not do exploratory reads — "
            "the plan already defines what needs changing.",
            "- Finish only when every step is applied, or report exactly "
            "which steps you could not apply and why.",
        ]
        return ("\n" + indent).join(lines) + "\n" + indent

    def _discovery_rules_block(self, contract):
        if contract.get("task_kind") != "repo_discovery":
            return ""
        indent = " " * 12
        lines = [
            "- This is AI-led repository discovery. Use repo_map/list_files/search_files/read_file yourself; do not merely restate discovery_seed.",
            "- First understand the user's requested level of detail. Then inspect documentation, package/config metadata, runtime entrypoints, and representative feature implementation files.",
            "- Treat README claims as documented intent and source files as implementation evidence. Distinguish the two in the final report.",
            "- Continue reading until the user's question is answered with file-grounded evidence, not merely until a framework is identified.",
            "- Cite repo-relative file paths in findings. Never claim documentation is absent without checking for it.",
            "- Final confidence must be high, medium, or low. If intent is ambiguous, set needs_clarification=true and ask one concrete clarifying_question.",
            "- If the answer is useful but meaningful areas remain unexplored, use medium confidence, list open_questions, and suggest specific deeper scopes the user may approve.",
        ]
        return ("\n" + indent).join(lines) + "\n" + indent

    def system_prompt(self, contract, memory_text):
        approved_plan_rules = self._approved_plan_rules_block(contract)
        discovery_rules = self._discovery_rules_block(contract)
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

            Active tool mode:
            {"read-only inspection; only repo_map, repo_overview, list_files, search_files, read_file, and final are allowed." if contract.get("read_only") or contract.get("edit_policy") == "inspect" else "full tool mode subject to permissions and edit_policy."}

            Rules:
            {approved_plan_rules}{discovery_rules}- Follow the contract strictly.
            - If the contract includes intent_analysis, use it as the first-pass interpretation of the user's actual goal, non-goals, needed context, risks, and success criteria.
            - Before editing, explicitly satisfy the intent_analysis needed_context where applicable and avoid the not_the_goal items.
            - If the contract contains pasted context/output, analyze that pasted evidence first.
            - Do not respond with generic reproduction commands when the user already pasted command output.
            - For database/foreign-key tasks, prioritize targeted searches for schema, migrations, SQLite/database initialization, account insertion, onboarding, and connection-test persistence.
            - Avoid repeated repo_overview calls. After one orientation pass, search/read specific source files or finish with findings.
            - Avoid dist/build output unless no source file exists.
            - Use tools instead of guessing.
            - Read files before editing them.
            - For improvement, review, or analysis tasks: run repo_overview first to discover all source files, then read the key source files directly (agent.py, cli.py, tools.py, config.py, etc.). Do NOT search for abstract concept strings like "CLI interface" or "configuration system" — those will not match source code. Read the actual files instead.
            - Respect edit_policy:
              - inspect: do not edit files
              - plan: do not edit files. First run repo_overview to see all files, then read all relevant source files. After reading, produce a concrete plan. Every plan step must name a real file that exists in the repo, a specific function or section to change, and exactly what the change would be. Generic suggestions without file references are not acceptable.
              - propose: do not edit files. Read the relevant source files first, then propose the exact changes that would be made, including file paths and code snippets showing before/after.
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
                "confidence": "high|medium|low",
                "open_questions": ["..."],
                "suggested_deeper_scope": ["..."],
                "needs_clarification": true_or_false,
                "clarifying_question": "one question when clarification is needed",
                "needs_approval": true_or_false,
                "plan": ["concrete step 1", "concrete step 2"],
                "decision_candidates": [{{"title":"...","rationale":"...","alternatives":[],"consequences":[],"affected_components":[],"source_references":[]}}]
              }}
            - In plan mode: the "plan" array must contain one entry per concrete change, formatted as "Edit <filename>:<function_or_section> — <what changes and why>". The "diff_summary" field must contain a real unified diff (--- a/path, +++ b/path, @@ ... @@) showing the exact lines that would change. Read files before writing the diff.
            - In propose mode: populate diff_summary with a real unified diff showing exactly what lines would change. Read each file first, then produce --- / +++ / @@ hunks. Populate plan with one entry per file change.
            - Return exactly one JSON object using the tool schema and no prose outside it.

            Tool schema:
            {{"tool":"TOOL_NAME","args":{{...}}}}

            Allowed tools:
            {self.allowed_tools_text(contract)}
            {self.few_shot_block(contract)}"""
        ).strip()

    def few_shot_block(self, contract):
        """Worked tool-call examples, shown only to weaker models that need them.

        Stronger models follow the schema from the description alone; for
        best-effort / medium-tier models a couple of concrete examples sharply
        cut malformed or chatty responses.
        """
        if not self.provider.is_local or not self.model_profile().use_few_shot:
            return ""
        read_only = contract.get("read_only") or contract.get("edit_policy") == "inspect"
        examples = [
            'To read a file:\n{"tool":"read_file","args":{"path":"src/app.py","start":1,"end":120}}',
            'To search the repo:\n{"tool":"search_files","args":{"query":"def main","path":"."}}',
        ]
        if read_only:
            examples.append(
                'To finish:\n{"tool":"final","args":{"summary":"app.py defines the entrypoint","findings":["main() in src/app.py:10"],"files_read":["src/app.py"]}}'
            )
        else:
            examples.append(
                'To edit exact text:\n{"tool":"replace_in_file","args":{"path":"src/app.py","old":"DEBUG = True","new":"DEBUG = False"}}'
            )
            examples.append(
                'To finish:\n{"tool":"final","args":{"summary":"disabled debug flag","files_changed":["src/app.py"],"diff_summary":"git diff --stat output"}}'
            )
        return (
            "\nExamples (respond with exactly one such object, nothing else):\n"
            + "\n".join(examples)
            + "\n"
        )

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
                      "plan": ["step 1", "step 2"],
                      "decision_candidates": [{{"title":"...","rationale":"...","alternatives":[],"consequences":[],"affected_components":[],"source_references":[]}}]
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
        files_read = report.get("files_read") or []
        direct_lines = [summarize_text(report.get("summary", ""), 220)] if report.get("summary") else []
        if files_read:
            direct_lines.append("read: " + ", ".join(Path(f).name for f in files_read))
        self.transcript_print(
            f"Inspected repo ({len(files_read)} file(s) read)",
            direct_lines,
            color=UI.BLUE,
        )
        return report

    def project_profile_report(self, contract):
        profile = build_project_profile(self.workdir, contract.get("exploration_budget") or None)
        summary = format_project_profile(profile)
        findings = list(profile.get("confirmed") or [])
        findings.extend(profile.get("likely") or [])
        findings.extend("Unclear: " + item for item in (profile.get("uncertainties") or []))
        return normalize_backend_report(
            {
                "summary": summary,
                "findings": findings,
                "commands_run": [],
                "files_read": profile.get("files_read") or [],
                "files_changed": [],
                "risks": profile.get("uncertainties") or [],
                "needs_approval": False,
                "plan": [],
            }
        )

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

    def request_action(self, messages, contract):
        if self.tool_calling in {"native", "auto"}:
            try:
                result = self.chat_tools(messages, self.native_tools(contract))
                action = self.action_from_native_tool_calls(result.tool_calls)
                if action:
                    return action, result.content
                if self.tool_calling == "native":
                    raise ValueError("Native tool calling did not return a tool call.")
                if result.content:
                    self.trace_print("native tool call absent; falling back to JSON content")
                    return self.parse_action(result.content), result.content
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001
                if self.tool_calling == "native":
                    raise
                self.trace_print(f"native tool calling unavailable; falling back to JSON protocol: {exc}")
        raw = self.chat(messages)
        return self.parse_action(raw), raw

    def action_from_native_tool_calls(self, tool_calls):
        if not tool_calls:
            return None
        call = tool_calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        if function is None and isinstance(call, dict):
            function = call
        if not isinstance(function, dict):
            raise ValueError("Native tool call did not include a function object.")
        name = function.get("name")
        args = function.get("arguments") or {}
        if isinstance(args, str):
            parsed = load_json_layers(args)
            args = parsed if isinstance(parsed, dict) else {}
        if not isinstance(name, str) or not name:
            raise ValueError("Native tool call did not include a tool name.")
        if not isinstance(args, dict):
            raise ValueError("Native tool call arguments must be an object.")
        return {"tool": name, "args": args}

    def tool_result(self, tool, args, contract, tracker):
        self.last_tool_display = None
        try:
            if tool == "search_web":
                query = args.get("query", "")
                max_results = int(args.get("max_results", 5))
                result = search_web(query, max_results=max_results)
                self.last_tool_display = {"kind": "search_web", "query": query}
                return result
            if tool == "fetch_url":
                return fetch_url(args["url"])
            if tool == "repo_map":
                return format_repo_map(self.workdir)
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
                if contract.get("read_only") or contract.get("edit_policy") == "inspect":
                    return "Command blocked by read-only inspection mode."
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

    def tool_transcript_line(self, tool, args, result, elapsed):
        """Plain-text one-line summary of a tool result (no ANSI), for observers."""
        display = self.last_tool_display or {}
        kind = display.get("kind")
        if kind == "command":
            return f"Run  {display['command']}  (exit {display['exit_code']}, {elapsed:.1f}s)"
        if kind == "edit":
            return f"Edit  {display['path']}  +{display['added']} -{display['removed']}"
        if kind == "read":
            return f"Read  {display['path']}:{display['range']}"
        if kind == "search_web":
            return f"Search web  {display['query']!r}"
        if tool == "search_files":
            matches = 0 if result == "(no matches)" else len([ln for ln in result.splitlines() if ":" in ln])
            return f"Search  {args.get('query', '')[:40]!r} in {args.get('path', '.')}  ({matches} matches)"
        if tool == "list_files":
            count = len([ln for ln in result.splitlines() if ln.strip()])
            return f"List  {args.get('path', '.')}  ({count} files)"
        if tool == "repo_overview":
            return "Overview  repo structure"
        if tool == "repo_map":
            return "Map  repo structure"
        return f"{tool}"

    def print_tool_transcript(self, tool, args, result, elapsed):
        if self.verbosity not in {"normal", "debug"}:
            return
        display = self.last_tool_display or {}
        if self.observer is not None:
            self.emit(
                "tool",
                tool=tool,
                args=args,
                display=display,
                elapsed=elapsed,
                line=self.tool_transcript_line(tool, args, result, elapsed),
            )
            return
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
        if display.get("kind") == "search_web":
            print(self.ui.tool_line(f"Search web  {display['query']!r}", color=UI.CYAN), file=sys.stderr)
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
        if (
            contract.get("edit_policy") == "inspect"
            and contract.get("files_of_interest")
            and contract.get("task_kind") != "repo_discovery"
        ):
            return self.direct_report(contract, memory_text)
        messages = [{"role": "system", "content": self.system_prompt(contract, memory_text)}]
        if contract.get("edit_policy") in {"inspect", "plan", "propose"}:
            messages.append({"role": "user", "content": "This is a non-editing backend pass unless edit_policy is execute. Inspect first and avoid speculative claims."})
        tracker = {
            "commands_run": [],
            "files_read": set(),
            "files_changed": set(),
            "evidence": [],
        }
        self.last_backend_action = None
        self.last_backend_action_signature = None
        self.repeated_backend_action_count = 0
        self._action_seen_counts: dict = {}
        self.cancel_event.clear()
        invalid_action_count = 0
        plan_pushback_count = 0
        noop_final_count = 0

        step_limit = min(24, max(1, int(contract.get("max_tool_steps") or self.max_steps)))
        for step in range(1, step_limit + 1):
            if self.cancel_event.is_set():
                self.transcript_print("Cancelled", ["User cancelled the backend loop."], color=UI.YELLOW)
                return normalize_backend_report(
                    {
                        "summary": "Cancelled by user.",
                        "commands_run": tracker["commands_run"],
                        "files_read": sorted(tracker["files_read"]),
                        "files_changed": sorted(tracker["files_changed"]),
                        "needs_approval": False,
                        "risks": [],
                    }
                )
            intent = self.print_step_intent(step, contract, messages)
            spinner_label = f"step {step}"
            if intent and step == 1:
                spinner_label += f"  ·  {intent}"
            action_error = None
            with Spinner(self.ui, self.ui.style(spinner_label, UI.DIM)):
                chat_started = time.monotonic()
                try:
                    action, raw = self.request_action(messages, contract)
                except TimeoutError:
                    if step == 1:
                        try:
                            action, raw = self.request_action(messages, contract)
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
                        except Exception as exc:  # noqa: BLE001
                            action_error = exc
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
                except Exception as exc:  # noqa: BLE001
                    action_error = exc
                chat_elapsed = time.monotonic() - chat_started
            if action_error is not None:
                exc = action_error
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
            if tool not in self.allowed_tool_names(contract):
                invalid_action_count += 1
                self.trace_print(f"invalid backend tool {tool!r}; asking it to retry")
                messages.append(
                    {
                        "role": "user",
                        "content": self.invalid_action_hint(
                            contract,
                            step,
                            f"Tool {tool!r} is not available in the current mode.",
                            invalid_action_count,
                            tracker,
                        ),
                    }
                )
                if invalid_action_count >= 4:
                    self.transcript_print(
                        "Stopped backend loop",
                        ["Backend did not produce a valid allowed tool call after repeated retries."],
                        color=UI.YELLOW,
                    )
                    return normalize_backend_report(
                        {
                            "summary": "Stopped because the backend failed to produce a valid allowed tool call.",
                            "commands_run": tracker["commands_run"],
                            "files_read": sorted(tracker["files_read"]),
                            "files_changed": sorted(tracker["files_changed"]),
                            "needs_approval": False,
                            "risks": ["Backend produced invalid actions repeatedly before making progress."],
                        }
                    )
                continue
            try:
                args = validate_tool_call(tool, args, self.allowed_tool_names(contract))
                action["args"] = args
            except Exception as exc:  # noqa: BLE001
                invalid_action_count += 1
                self.trace_print(f"invalid backend tool arguments for {tool!r}; asking it to retry")
                messages.append(
                    {
                        "role": "user",
                        "content": self.invalid_action_hint(contract, step, exc, invalid_action_count, tracker),
                    }
                )
                if invalid_action_count >= 4:
                    self.transcript_print(
                        "Stopped backend loop",
                        ["Backend did not produce valid tool arguments after repeated retries."],
                        color=UI.YELLOW,
                    )
                    return normalize_backend_report(
                        {
                            "summary": "Stopped because the backend failed to produce valid tool arguments.",
                            "commands_run": tracker["commands_run"],
                            "files_read": sorted(tracker["files_read"]),
                            "files_changed": sorted(tracker["files_changed"]),
                            "needs_approval": False,
                            "risks": ["Backend produced invalid tool arguments repeatedly before making progress."],
                        }
                    )
                continue
            self.last_backend_action = action
            signature = self.action_signature(tool, args)
            if signature == self.last_backend_action_signature:
                self.repeated_backend_action_count += 1
            else:
                self.repeated_backend_action_count = 0
            self.last_backend_action_signature = signature
            self._action_seen_counts[signature] = self._action_seen_counts.get(signature, 0) + 1
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
                    problems = validate_plan_report(report, self.workdir)
                    if problems and plan_pushback_count < 2:
                        plan_pushback_count += 1
                        self.trace_print("plan final too vague; asking for a concrete plan")
                        messages.append({"role": "user", "content": self.vague_plan_hint(problems)})
                        continue
                    if problems:
                        report["risks"] = list(report["risks"]) + [
                            "Plan may be too vague to apply mechanically: " + "; ".join(problems)[:300]
                        ]
                approved_plan = contract.get("approved_plan") or {}
                if contract.get("edit_policy") == "execute" and approved_plan:
                    # Status is derived from the loop's own tracker, never from
                    # model-claimed report fields — a fabricated files_changed
                    # must not bypass the no-op rejection.
                    status, missed = compute_execution_status(
                        approved_plan.get("steps") or [],
                        sorted(tracker["files_changed"]),
                        tracker["commands_run"],
                    )
                    # One retry per backend run by design: first no-op final gets
                    # pushed back with the unapplied steps, the second becomes an
                    # honest failure report.
                    if status == "plan_not_applied" and noop_final_count == 0:
                        noop_final_count += 1
                        self.trace_print("no-op final in execute mode; asking backend to apply the approved plan")
                        messages.append({"role": "user", "content": self.unapplied_plan_hint(approved_plan)})
                        continue
                    report["execution_status"] = status
                    if status == "plan_not_applied":
                        report["summary"] = "The plan was approved but not applied."
                        report["risks"] = list(report["risks"]) + [
                            "Approved plan steps were not applied: "
                            + "; ".join(str(s) for s in (approved_plan.get("steps") or []))[:300]
                        ]
                    elif status == "partially_applied":
                        report["risks"] = list(report["risks"]) + [
                            "Plan steps not verifiably applied: " + "; ".join(missed)[:300]
                        ]
                    if status != "applied":
                        self.transcript_print(
                            "no edits were made" if status == "plan_not_applied" else "plan only partially applied",
                            [str(s)[:220] for s in (missed or approved_plan.get("steps") or [])][:5],
                            color=UI.YELLOW,
                        )
                elif contract.get("edit_policy") == "execute" and tracker["files_changed"]:
                    report["execution_status"] = "applied"
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
                files_read = report.get("files_read") or []
                read_label = f"Finished backend report ({len(files_read)} file(s) read)"
                detail_lines = [summarize_text(report.get("summary", "Backend finished."), 220)]
                if files_read:
                    detail_lines.append("read: " + ", ".join(Path(f).name for f in files_read))
                self.transcript_print(read_label, detail_lines, color=UI.GREEN)
                return report

            tool_started = time.monotonic()
            result = self.tool_result(tool, args, contract, tracker)
            tool_elapsed = time.monotonic() - tool_started
            self.print_tool_transcript(tool, args, result, tool_elapsed)
            if contract.get("task_kind") == "repo_discovery" and tool in {"repo_map", "repo_overview", "list_files", "search_files", "read_file"}:
                tracker["evidence"].append({
                    "tool": tool,
                    "args": args,
                    "result": str(result)[:1800],
                })
            messages.append({"role": "user", "content": f"Tool result for {tool}:\n{result}"})
            cycle_detected = (
                self._action_seen_counts.get(signature, 0) >= 3
                or self.repeated_backend_action_count >= 2
            )
            if cycle_detected and tool != "final":
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

        if contract.get("task_kind") == "repo_discovery":
            # Reserve one synthesis-only call after exploration. Small local
            # models often spend the entire tool budget reading useful files
            # and otherwise never emit their final report.
            messages.append({
                "role": "user",
                "content": (
                    "The repository exploration budget is exhausted. Do not call another read/search/list tool. "
                    "Call final now and synthesize the evidence already collected. Include confidence, open_questions, "
                    "suggested_deeper_scope, and a yes/no clarifying_question only if needed."
                ),
            })
            try:
                action, _raw = self.request_action(messages, contract)
                if action.get("tool") == "final":
                    args = validate_tool_call("final", action.get("args") or {}, self.allowed_tool_names(contract))
                    message = args.get("message", "")
                    report = normalize_backend_report(
                        message if isinstance(message, dict) or (isinstance(message, str) and message.strip().startswith("{")) else args,
                        fallback_message="Repository exploration completed without a detailed report.",
                    )
                    report["commands_run"] = report["commands_run"] or tracker["commands_run"]
                    report["files_read"] = report["files_read"] or sorted(tracker["files_read"])
                    report["files_changed"] = report["files_changed"] or sorted(tracker["files_changed"])
                    self.transcript_print(
                        f"Finished backend report ({len(report['files_read'])} file(s) read)",
                        [summarize_text(report.get("summary", "Repository discovery finished."), 220)],
                        color=UI.GREEN,
                    )
                    return report
            except Exception as exc:  # noqa: BLE001
                self.trace_print(f"final discovery synthesis failed: {exc}")
            synthesized = self.synthesize_discovery_report(contract, tracker)
            if synthesized is not None:
                self.transcript_print(
                    f"Synthesized discovery report ({len(synthesized['files_read'])} file(s) read)",
                    [summarize_text(synthesized.get("summary", "Repository discovery finished."), 220)],
                    color=UI.GREEN,
                )
                return synthesized
        return normalize_backend_report({
            "summary": "Stopped after reaching the maximum backend tool steps.",
            "commands_run": tracker["commands_run"],
            "files_read": sorted(tracker["files_read"]),
            "files_changed": sorted(tracker["files_changed"]),
            "needs_approval": False,
            "risks": ["Backend step limit reached before completion."],
        })

    def synthesize_discovery_report(self, contract, tracker):
        """Give a small model one bounded, tool-free evidence synthesis job."""
        evidence = tracker.get("evidence") or []
        if not evidence:
            return None
        messages = [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """\
                    You synthesize repository evidence already collected by another pass.
                    Do not request tools. Do not guess beyond the evidence. Distinguish README claims
                    from implementation confirmed in source. Cite repo-relative paths in findings.
                    Open questions are uncertainties, not defects or improvement recommendations.

                    Return exactly one JSON object:
                    {
                      "summary": "direct answer to the user's question",
                      "findings": ["file-grounded finding"],
                      "risks": ["actual evidence limitation"],
                      "confidence": "high|medium|low",
                      "open_questions": ["unresolved factual question"],
                      "suggested_deeper_scope": ["specific area worth more inspection"],
                      "needs_clarification": false,
                      "clarifying_question": ""
                    }
                    """
                ).strip(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": contract.get("goal"),
                        "intent_analysis": contract.get("intent_analysis") or {},
                        "files_read": sorted(tracker.get("files_read") or []),
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            raw = self.provider.assess(self.model, messages)
            report = normalize_backend_report(raw, fallback_message="")
        except Exception as exc:  # noqa: BLE001
            self.trace_print(f"dedicated discovery synthesis failed: {exc}")
            return None
        if not report.get("summary"):
            return None
        report["commands_run"] = tracker["commands_run"]
        report["files_read"] = sorted(tracker["files_read"])
        report["files_changed"] = sorted(tracker["files_changed"])
        return report

    def run_contract_with_model(self, contract, memory_text, model):
        original = self.model
        self.model = model
        try:
            return self.run_contract(contract, memory_text)
        finally:
            self.model = original

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
            if contract.get("read_only") or contract.get("edit_policy") == "inspect":
                tool_hint = 'Use {"tool":"repo_map","args":{}} or a read/search tool.'
            else:
                tool_hint = 'Use a single JSON tool call, then wait for the result.'
        return "\n".join(
            [
                "Your last response was not a valid tool call.",
                "",
                "Available tools:",
                self.allowed_tools_text(contract),
                "",
                "You must respond with exactly one valid tool call as JSON.",
                "Do not explain.",
                "Do not invent tools.",
                f"Error: {error}.{progress} {tool_hint}",
            ]
        )

    def vague_plan_hint(self, problems):
        return "\n".join(
            [
                "Your plan is not concrete enough to execute. Problems:",
                *[f"- {p}" for p in problems],
                "",
                "Revise the plan and call final again.",
                "Every plan step must name a real repo file and the exact change, or an exact command.",
                "diff_summary must contain a real unified diff (--- a/path, +++ b/path, @@ hunks).",
            ]
        )

    def unapplied_plan_hint(self, approved_plan):
        steps = approved_plan.get("steps") or []
        if not steps:
            return (
                "You finished without applying the approved plan. No files were changed "
                "and no plan commands were run. Re-read the contract's approved_plan and task, "
                "apply the requested changes, then call final."
            )
        return "\n".join(
            [
                "You finished without applying the approved plan. No files were changed and no plan commands were run.",
                "Unapplied steps:",
                *[f"{index}. {step}" for index, step in enumerate(steps, start=1)],
                "",
                "Apply them now using replace_in_file, write_file, replace_lines, insert_after, or run_command.",
                "If a step truly cannot be applied, apply the others and state per step why it cannot be applied.",
            ]
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
        ollama=DEFAULT_OLLAMA,
        workdir=".",
        command_permission="ask",
        edit_permission="ask",
        max_steps=MAX_TOOL_STEPS,
        verbosity=DEFAULT_VERBOSITY,
        show_raw_actions=False,
        mode=DEFAULT_MODE,
        tool_calling=DEFAULT_TOOL_CALLING,
        provider=None,
        observer=None,
        confirm_hook=None,
        model_routing="adaptive",
        routing_decision=None,
        context_limit=DEFAULT_NUM_CTX,
        preferred_frontend_model=None,
        preferred_backend_model=None,
        storage_mode=None,
    ):
        self.frontend_model = frontend_model
        self.backend_model = backend_model
        self.preferred_frontend_model = preferred_frontend_model or frontend_model
        self.preferred_backend_model = preferred_backend_model or backend_model
        self.provider = provider or OllamaProvider(ollama)
        self.ollama = getattr(self.provider, "base_url", ollama)
        self.workdir = str(Path(workdir).resolve())
        self.command_permission = command_permission
        self.edit_permission = edit_permission
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.show_raw_actions = show_raw_actions
        self.mode = mode
        self.tool_calling = tool_calling
        self.model_routing = model_routing
        self.routing_decision = routing_decision or {"mode": "single" if frontend_model == backend_model else "dual", "reason": "Models configured directly."}
        self.context_limit = context_limit
        self.storage_mode = storage_mode
        self.active_context_extra = ""
        self.ui = UI()
        self.observer = observer
        self.confirm_hook = confirm_hook
        # When a TUI drives streaming via on_token, the partner must not write
        # to stdout itself (it would corrupt the full-screen app).
        self.stream_to_stdout = True
        self.pending_plan = None
        self.pending_discovery = None
        self.latest_plan = None
        self.last_report = None
        self.last_status = None
        self.on_token = None
        self.last_streamed = False
        self.backend_runs = 0
        decision_paths = ensure_memory_files(self.workdir, self.storage_mode)
        self.decisions = DecisionService.load(decision_paths["active_intelligence"])
        self.history = load_chat_history(self.workdir, self.storage_mode)
        self.executor = LocalCodeAgent(
            model=self.backend_model,
            provider=self.provider,
            workdir=self.workdir,
            command_permission=self.command_permission,
            edit_permission=self.edit_permission,
            max_steps=self.max_steps,
            verbosity=self.verbosity,
            show_raw_actions=self.show_raw_actions,
            tool_calling=self.tool_calling,
            observer=observer,
            confirm_hook=confirm_hook,
        )

    def run_decision_command(self, command):
        return execute_decision_command(self.decisions, command)

    def context_usage(self, extra=None):
        """Estimate the active prompt budget by source for transparent local tuning."""
        tool_text = tool_prompt_lines(self.executor.allowed_tool_names({"edit_policy": "execute"}))
        return build_context_usage(
            history=self.history,
            memory=load_repo_memory(self.workdir, self.storage_mode),
            repo=git_context(self.workdir),
            tools=tool_text,
            other=self.active_context_extra if extra is None else extra,
            limit=self.context_limit,
        )

    def _prune_history(self):
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]
        save_chat_history(self.workdir, self.history, self.storage_mode)

    def sync_executor(self):
        self.executor.model = self.backend_model
        self.executor.provider = self.provider
        self.executor.command_permission = self.command_permission
        self.executor.edit_permission = self.edit_permission
        self.executor.max_steps = self.max_steps
        self.executor.verbosity = self.verbosity
        self.executor.show_raw_actions = self.show_raw_actions
        self.executor.tool_calling = self.tool_calling
        self.executor.observer = self.observer
        self.executor.confirm_hook = self.confirm_hook

    def emit(self, kind, **data):
        if self.observer is not None:
            self.observer({"kind": kind, "source": "frontend", **data})

    def trace_print(self, message):
        if self.verbosity != "debug":
            return
        if self.observer is not None:
            self.emit("trace", text=message)
            return
        prefix = self.ui.style("frontend", UI.BOLD, UI.YELLOW)
        print(f"{prefix} {message}", file=sys.stderr)

    def milestone(self, message):
        if self.verbosity not in {"normal", "debug"}:
            return
        if self.observer is not None:
            self.emit("milestone", text=message)
            return
        print(self.ui.tool_line(message, color=UI.BLUE), file=sys.stderr)

    def chat(self, model, messages):
        return self.provider.chat(model, messages)

    def _emit_stdout(self, text):
        if self.stream_to_stdout:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _chat_streaming(self, model, messages):
        """Stream a response via on_token callbacks; return full accumulated text."""
        chunks = []
        first = True
        for chunk in self.provider.stream(model, messages):
            if first:
                self._emit_stdout("\n")
                first = False
            self.on_token(chunk)
            chunks.append(chunk)
        full = "".join(chunks)
        if full and not full.endswith("\n"):
            self._emit_stdout("\n")
        self.last_streamed = True
        return full.strip()

    def _stream_frontend_turn(self, model, messages):
        """Stream frontend turn: text replies go to on_token, delegation JSON is buffered silently."""
        chunks = []
        decided = False
        is_text = False
        for chunk in self.provider.stream(model, messages):
            chunks.append(chunk)
            if not decided:
                so_far = "".join(chunks).lstrip()
                if so_far:
                    is_text = not so_far.startswith("{")
                    decided = True
                    if is_text:
                        self._emit_stdout("\n")
                        for c in chunks:
                            self.on_token(c)
            elif is_text:
                self.on_token(chunk)
        full = "".join(chunks)
        if is_text:
            if full and not full.endswith("\n"):
                self._emit_stdout("\n")
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
            {load_repo_memory(self.workdir, self.storage_mode)}

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

    def wants_read_only_inspection(self, prompt):
        explicit_read_only = bool(re.search(r"\b(without (?:making )?edits?|do not modify|don't modify|do not edit|don't edit|no edits?)\b", prompt, re.I))
        if explicit_read_only:
            return True
        return bool(
            re.search(
                r"\b(inspect|review|summarize|tell me about|what can you tell me|read key files)\b",
                prompt,
                re.I,
            )
        ) and not self.wants_edit(prompt)

    def wants_project_discovery(self, prompt):
        discovery_re = re.compile(
            r"\b(what is this project|what can you tell me about this project|tell me about (?:this )?(?:repo|repository|project|codebase)|analy[sz]e (?:this )?(?:repo|repository|project|codebase)|figure out what this (?:does|is)|review the structure|understand this codebase|read key files until (?:you )?know what this is)\b",
            re.I,
        )
        if discovery_re.search(prompt):
            return True

        # Treat questions naming the current project as repo questions rather
        # than general-knowledge chat. This keeps "What is Skald?" grounded in
        # the checked-out repository even when the project name is not followed
        # by the word "project".
        names = {Path(self.workdir).name.lower()}
        package_path = Path(self.workdir) / "package.json"
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            package = {}
        for value in (package.get("name"), (package.get("build") or {}).get("productName")):
            if isinstance(value, str) and value.strip():
                names.add(value.strip().lower())
        for name in names:
            if re.search(
                rf"\b(?:what is|what does|tell me about|explain)\s+(?:the\s+)?{re.escape(name)}\b",
                prompt,
                re.I,
            ):
                return True

        # Resolve short follow-ups such as "what does it do?" against the last
        # user turn when that turn was explicitly about discovering the repo.
        follow_up = re.search(
            r"^\s*(?:ok(?:ay)?[, ]*)?(?:but\s+)?(?:what is (?:it|this)|what does (?:it|this) do|what is the (?:project|app)|tell me more)\??\s*$",
            prompt,
            re.I,
        )
        if follow_up:
            previous_user = next(
                (item.get("content", "") for item in reversed(self.history) if item.get("role") == "user"),
                "",
            )
            return bool(discovery_re.search(previous_user))
        return False

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
        if contract.get("task_kind") == "repo_discovery":
            return "read-only repo discovery"
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
            If requires_deviation_explanation is true, explicitly explain whether and why the work intentionally deviated from an accepted decision.
            Mention any decision candidates as pending human review, never as established fact.
            Be concise but useful.
            Return plain text only.
            """
        ).strip()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": original_prompt})
        messages.append({"role": "user", "content": (("Context to preserve:\n" + frontend_message + "\n\n") if frontend_message else "") + "Backend report:\n" + json.dumps(backend_report, ensure_ascii=False, indent=2)})
        if (backend_report or {}).get("grounding_evidence"):
            messages.append({
                "role": "user",
                "content": (
                    "This is a repository-discovery answer. The backend report came from AI-led tool exploration; "
                    "grounding_evidence is a deterministic verification layer. Synthesize a direct answer to the "
                    "user's actual question, cite the key repo-relative files that support it, preserve confidence "
                    "and open questions, and do not invent claims that conflict with grounding_evidence. Do not "
                    "emit a generic project-overview template. If grounding_corrections is present, explicitly fix "
                    "every listed issue and do not repeat the contradicted claim. Do not recommend reading files "
                    "that already appear in files_read. Treat open questions as uncertainty, not evidence of defects; "
                    "do not suggest improvements unless a finding identifies a concrete problem."
                ),
            })
        status = str((backend_report or {}).get("execution_status") or "")
        if status and status != "applied":
            if status == "partially_applied":
                directive = (
                    "IMPORTANT: Only some planned changes were applied; "
                    "the remaining steps were not written to disk. "
                    "State this plainly as the first sentence of your reply. "
                    "Do not imply the full task was completed."
                )
            else:
                directive = (
                    "IMPORTANT: No files were changed. The approved plan was not applied. "
                    "State this plainly as the first sentence of your reply. "
                    "Do not imply the task was completed."
                )
            messages.append({"role": "user", "content": directive})
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
        append_run_log(self.workdir, entry, self.storage_mode)

    def _compact_string_list(self, value, limit=6):
        if not isinstance(value, list):
            return []
        items = []
        for item in value:
            text = str(item).strip()
            if text and text not in items:
                items.append(text[:300])
            if len(items) >= limit:
                break
        return items

    def _intent_analysis(self, contract):
        """Run a small, bounded intent pass before tool work.

        This is deliberately separate from implementation so weaker local models get
        one narrow job: decide what the user is actually asking for, what is out of
        scope, and what context should be inspected before edits. Failures are
        non-fatal because the normal contract is still sufficient to proceed.
        """
        if contract.get("intent_analysis"):
            return contract.get("intent_analysis")
        messages = [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """\
                    You are the Intent Analyst for a local coding agent.
                    Your only job is to clarify the user's goal before any repo edits happen.
                    Do not solve the task. Do not propose code. Be compact and concrete.

                    Return exactly one JSON object with this schema and no prose:
                    {
                      "user_goal": "plain-language goal",
                      "not_the_goal": ["things that would be overreach"],
                      "needed_context": ["files, symbols, commands, or evidence to inspect before editing"],
                      "likely_files": ["repo-relative file paths or globs if inferable"],
                      "risks": ["ambiguities, assumptions, or things that could break"],
                      "success_criteria": ["observable outcomes that mean the task is done"]
                    }
                    """
                ).strip(),
            },
            {"role": "user", "content": "Task contract:\n" + json.dumps(contract, ensure_ascii=False, indent=2)},
        ]
        try:
            raw = self.provider.assess(self.frontend_model, messages)
            data = load_json_layers(raw)
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(data, dict):
            return {}
        user_goal = str(data.get("user_goal") or contract.get("goal") or "").strip()
        analysis = {
            "user_goal": user_goal[:500],
            "not_the_goal": self._compact_string_list(data.get("not_the_goal"), limit=6),
            "needed_context": self._compact_string_list(data.get("needed_context"), limit=8),
            "likely_files": self._compact_string_list(data.get("likely_files"), limit=8),
            "risks": self._compact_string_list(data.get("risks"), limit=6),
            "success_criteria": self._compact_string_list(data.get("success_criteria"), limit=6),
        }
        return {key: value for key, value in analysis.items() if value}

    def _collect_source_files(self):
        """Return repo-relative paths for the primary source files, excluding cache/test dirs."""
        root = Path(self.workdir)
        skip = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", "dist", "build", "*.egg-info"}
        results = []
        for p in sorted(root.rglob("*.py")):
            if any(part in skip or part.endswith(".egg-info") for part in p.parts):
                continue
            results.append(p.relative_to(root).as_posix())
        for ext in ("*.ts", "*.tsx", "*.js", "*.jsx"):
            for p in sorted(root.rglob(ext)):
                if any(part in skip for part in p.parts):
                    continue
                results.append(p.relative_to(root).as_posix())
        return results[:20]

    _BROAD_REVIEW_RE = re.compile(
        r"\b(read\s+(through|all|everything|the\s+whole|every\s+file)|"
        r"review\s+(everything|all|the\s+whole|every\s+file)|"
        r"look\s+(through|at)\s+(everything|all\s+(the\s+)?files)|"
        r"go\s+through\s+(everything|all)|"
        r"read\s+(?:it|them)\s+all\b)",
        re.IGNORECASE,
    )

    def _contract_with_intent_scaffold(self, contract):
        if contract.get("approved_plan"):
            return contract
        if contract.get("intent_analysis"):
            return contract
        if contract.get("task_kind") == "conversation":
            return contract
        analysis = self._intent_analysis(contract)
        if not analysis:
            return contract
        enriched = dict(contract)
        enriched["intent_analysis"] = analysis
        likely_files = analysis.get("likely_files") or []
        if likely_files:
            existing = list(enriched.get("files_of_interest") or [])
            for path in resolve_repo_file_hints(self.workdir, likely_files):
                if path not in existing:
                    existing.append(path)
            enriched["files_of_interest"] = existing
        guardrails = list(enriched.get("constraints") or [])
        if analysis.get("not_the_goal"):
            guardrails.append("Do not overreach beyond the intent analysis not_the_goal list.")
        if analysis.get("needed_context"):
            guardrails.append("Inspect the intent analysis needed_context before editing when applicable.")
        if analysis.get("success_criteria"):
            guardrails.append("Use the intent analysis success_criteria to decide when to stop.")
        # For broad review/read-everything tasks, pre-populate with all source files so the weak
        # backend model gets an explicit list to work through rather than stopping after 3 files.
        task_text = str(contract.get("task") or "")
        is_broad_review = (
            self._BROAD_REVIEW_RE.search(task_text)
            or (contract.get("scope") == "broad" and contract.get("edit_policy") in {"inspect", "plan", "propose"})
        )
        if is_broad_review and len(enriched.get("files_of_interest") or []) < 4:
            source_files = self._collect_source_files()
            if source_files:
                enriched["files_of_interest"] = source_files
                guardrails.append(
                    "Read each file in files_of_interest at most once, then call final with specific, "
                    "file-grounded findings. Do not re-read a file you have already read in this session."
                )
        enriched["constraints"] = guardrails
        return enriched

    def _assess_complexity(self, contract):
        edit_policy = contract.get("edit_policy")
        task_kind = contract.get("task_kind")
        if edit_policy in {"plan", "propose", "execute"}:
            return "escalate"
        if edit_policy == "inspect" or task_kind == "inspection":
            return "self"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task complexity assessor for a local coding assistant.\n\n"
                    "A SMALL model can handle:\n"
                    "- Reading or explaining files\n"
                    "- Answering questions about what the repo does\n"
                    "- Listing files or structure\n"
                    "- Simple single-file inspections\n"
                    "- Chat or conversational questions\n\n"
                    "A LARGE model is needed for:\n"
                    "- Editing or refactoring multiple files\n"
                    "- Bootstrapping or scaffolding new projects\n"
                    "- Debugging complex failures\n"
                    "- Tasks explicitly involving architecture changes\n\n"
                    "The contract's edit_policy is the strongest signal:\n"
                    '- inspect or plan → almost always "self"\n'
                    '- execute with multiple files_of_interest → usually "escalate"\n'
                    '- execute with one file → could be "self"\n\n'
                    "Reply with exactly one JSON object, no other text:\n"
                    '{"handle": "self"} or {"handle": "escalate", "reason": "..."}'
                ),
            },
            {"role": "user", "content": json.dumps(contract, ensure_ascii=False)},
        ]
        try:
            raw = self.provider.assess(self.frontend_model, messages)
            data = load_json_layers(raw)
            if isinstance(data, dict) and data.get("handle") == "self":
                return "self"
        except Exception:  # noqa: BLE001
            pass
        return "escalate"

    def _run_backend(self, contract):
        self.backend_runs += 1
        contract = dict(contract)
        contract.update(self.decisions.contract_context(contract.get("goal", "")))
        contract = self._contract_with_intent_scaffold(contract)
        memory = load_repo_memory(self.workdir, self.storage_mode)
        if self.frontend_model != self.backend_model:
            decision = self._assess_complexity(contract)
            if decision == "escalate":
                self.milestone(f"complex task → escalating to {self.backend_model}")
                report = self.executor.run_contract_with_model(contract, memory, self.backend_model)
            else:
                self.milestone(f"simple task → handling with {self.frontend_model}")
                try:
                    report = self.executor.run_contract_with_model(contract, memory, self.frontend_model)
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    self.milestone(f"{self.frontend_model} not available, falling back to {self.backend_model}")
                    report = self.executor.run_contract_with_model(contract, memory, self.backend_model)
        else:
            report = self.executor.run_contract(contract, memory)
        report = dict(report)
        report["decision_conflicts"] = contract.get("decision_conflicts", [])
        report["requires_deviation_explanation"] = bool(contract.get("requires_deviation_explanation"))
        candidates = self.decisions.extract_candidates(report, source_run=datetime.now().isoformat(timespec="seconds"))
        candidate_conflicts = self.decisions.conflicts_for(contract.get("goal", ""), candidates)
        report["decision_conflicts"] = list(report["decision_conflicts"]) + [conflict.to_dict() for conflict in candidate_conflicts]
        report["requires_deviation_explanation"] = bool(report["decision_conflicts"])
        report["pending_decision_candidates"] = [candidate.to_dict() for candidate in candidates]
        if candidates:
            self.milestone(f"{len(candidates)} decision candidate(s) pending review")
        return report

    def update_memory(self):
        paths = memory_paths(self.workdir, self.storage_mode)
        if paths["mode"].value == "shared":
            return
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
                "Keep each addition to one bullet line under the existing sections. "
                "Preserve rist metadata comments and headings. Do not remove existing entries. "
                "Return ONLY the full updated file content, no explanation."
            ),
        }]
        with Spinner(self.ui, self.ui.style("updating memory", UI.DIM)):
            updated = self.provider.chat(self.backend_model, messages)
        updated = updated.strip()
        if updated and updated != current:
            atomic_write_text(paths["project"], updated + "\n")
            IntelligenceStore.load(paths["active_intelligence"], sync_views=True)
            self.milestone("project memory updated")

    def apply_pending_plan(self):
        if not self.pending_plan:
            return None
        contract = dict(self.pending_plan["contract"])
        contract["edit_policy"] = "execute"
        contract["execution_strategy"] = "apply_approved_plan"
        plan_report = self.pending_plan.get("report") or {}
        contract["approved_plan"] = {
            "steps": list(plan_report.get("plan") or []),
            "diff_summary": str(plan_report.get("diff_summary") or ""),
            "files_read": list(plan_report.get("files_read") or []),
        }
        self.milestone("applying approved plan")
        report = self._run_backend(contract)
        reply = self.frontend_finalize(self.pending_plan["original_prompt"], report, self.pending_plan.get("frontend_message", ""))
        self.last_report = report
        self.last_status = "completed"
        self.latest_plan = {"contract": contract, "report": report, "original_prompt": self.pending_plan["original_prompt"]}
        self.pending_plan = None
        self.history.append({"role": "user", "content": "Apply the approved plan."})
        self.history.append({"role": "assistant", "content": reply})
        save_chat_history(self.workdir, self.history, self.storage_mode)
        self.log_run("/apply", reply, contract=contract, report=report)
        return reply

    def run_readonly_turn(self, user_prompt):
        self.sync_executor()
        self._prune_history()
        contract = normalize_contract({}, user_prompt, "hybrid", workdir_state=inspect_workdir_state(self.workdir))
        contract["edit_policy"] = "inspect"
        contract["read_only"] = True
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

    def _project_grounding_evidence(self, budget):
        """Build compact deterministic evidence used only to verify AI output."""
        profile = build_project_profile(self.workdir, budget)
        entrypoint_roles = {}
        role_candidates = {
            "src/main.tsx": "React renderer entrypoint",
            "src/main.jsx": "React renderer entrypoint",
            "src-main/main.ts": "Electron main-process entrypoint",
            "src-main/main.js": "Electron main-process entrypoint",
            "src-main/preload.ts": "Electron preload bridge",
            "src-main/preload.js": "Electron preload bridge",
            "electron/main.ts": "Electron main-process entrypoint",
            "electron/preload.ts": "Electron preload bridge",
        }
        for path in profile.get("entrypoints") or []:
            if path in role_candidates:
                entrypoint_roles[path] = role_candidates[path]
        return {
            "project_name": profile.get("display_name") or profile.get("project_name"),
            "documented_purpose": profile.get("documented_purpose"),
            "documentation_found": profile.get("documentation_found") or [],
            "documented_features": profile.get("documented_features") or [],
            "detected_stack": profile.get("detected_stack") or [],
            "entrypoints": profile.get("entrypoints") or [],
            "entrypoint_roles": entrypoint_roles,
        }

    @staticmethod
    def _discovery_grounding_issues(reply, evidence):
        issues = []
        if (
            evidence.get("entrypoint_roles", {}).get("src/main.tsx") == "React renderer entrypoint"
            and re.search(
                r"src/main\.tsx.{0,120}(?:Electron main(?:-| )process|main entry point for (?:the )?Electron)",
                reply,
                re.I | re.S,
            )
        ):
            issues.append("src/main.tsx is the React renderer entrypoint, not the Electron main process; src-main/main.ts is the Electron main-process entrypoint.")
        if evidence.get("documentation_found") and re.search(r"\b(no|without)\s+(?:public\s+)?documentation\b|\bno\s+README\b", reply, re.I):
            issues.append("Documentation exists: " + ", ".join(evidence["documentation_found"]))
        return issues

    @staticmethod
    def _discovery_followup_question(report, *, depth):
        confidence = str(report.get("confidence") or "").lower()
        files_read = report.get("files_read") or []
        if confidence not in {"high", "medium", "low"}:
            confidence = "low" if not files_read else "medium" if report.get("risks") else "high"
            report["confidence"] = confidence
        if report.get("needs_clarification"):
            return report.get("clarifying_question") or "Should I inspect the repository more deeply before answering?"
        scopes = [str(item).strip() for item in (report.get("suggested_deeper_scope") or []) if str(item).strip()]
        if depth != "deep" and confidence in {"medium", "low"}:
            target = ", ".join(scopes[:3]) or "the implementation and remaining unclear areas"
            return f"Should I dig deeper into {target}?"
        return ""

    @staticmethod
    def _normalize_discovery_uncertainty(report, evidence):
        risks = [str(item).strip() for item in (report.get("risks") or []) if str(item).strip()]
        if evidence.get("documented_purpose"):
            risks = [
                item for item in risks
                if not re.search(r"ambiguity in (?:the )?(?:exact )?definition|exact definition and scope", item, re.I)
            ]
        report["risks"] = risks
        if report.get("confidence") == "high" and (report.get("open_questions") or risks):
            report["confidence"] = "medium"
        return report

    def run_project_discovery_turn(self, user_prompt, *, depth="standard", prior_report=None, history_prompt=None):
        self.sync_executor()
        self._prune_history()
        is_deep = depth == "deep" or bool(re.search(r"\b(deep|thorough|detailed)\b", user_prompt, re.I))
        budget = {
            "max_root_files": 50,
            "max_reads": 20 if is_deep else 12,
            "max_searches": 12 if is_deep else 8,
            "max_lines_per_file": 500,
            "stop_when_confidence": "high" if is_deep else "medium",
        }
        goal = user_prompt.strip()
        if prior_report:
            goal = (
                f"Deepen the previous repository answer to: {user_prompt.strip()}\n"
                "Resolve its open questions and inspect implementation areas that were not yet read."
            )
        contract = {
            "goal": goal,
            "scope": ["."],
            "constraints": [
                "Read-only repo discovery.",
                "The model must retrieve evidence itself with repository tools.",
                "Inspect package/config, documentation, runtime entrypoints, and representative feature source.",
                "Do not stop after identifying the framework; answer what the product does and how the inspected code supports that conclusion.",
                "Ask a yes/no clarifying question instead of guessing when the user's desired scope is ambiguous.",
            ],
            "commands_allowed": [],
            "approval_prefixes": [],
            "edit_policy": "inspect",
            "read_only": True,
            "expected_result": "Structured project overview with confirmed facts, likely conclusions, uncertainties, and confidence levels.",
            "files_of_interest": [],
            "task_kind": "repo_discovery",
            "execution_strategy": "direct",
            "target_paths": [],
            "verification_checks": [],
            "exploration_budget": budget,
            "exploration_depth": "deep" if is_deep else "standard",
            "max_tool_steps": 18 if is_deep else 12,
            "discovery_seed": format_repo_map(self.workdir),
        }
        if prior_report:
            contract["previous_discovery"] = {
                "summary": prior_report.get("summary", ""),
                "files_read": prior_report.get("files_read") or [],
                "open_questions": prior_report.get("open_questions") or [],
                "suggested_deeper_scope": prior_report.get("suggested_deeper_scope") or [],
            }
            contract["constraints"].append("Do not merely repeat the previous answer; inspect additional relevant files.")
        self.milestone("mode: AI-led repository discovery")
        self.milestone("allowed tools: repo_map, repo_overview, list_files, search_files, read_file")
        report = self._run_backend(contract)
        relative_reads = []
        root = Path(self.workdir)
        for value in report.get("files_read") or []:
            path = Path(value)
            try:
                value = path.resolve().relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
            except ValueError:
                value = path.as_posix()
            if value not in relative_reads:
                relative_reads.append(value)
        report["files_read"] = relative_reads
        report["grounding_evidence"] = self._project_grounding_evidence(budget)
        self._normalize_discovery_uncertainty(report, report["grounding_evidence"])
        reply = self.frontend_finalize(user_prompt, report)
        grounding_issues = self._discovery_grounding_issues(reply, report["grounding_evidence"])
        if grounding_issues:
            report["grounding_corrections"] = grounding_issues
            reply = self.frontend_finalize(user_prompt, report)
        question = self._discovery_followup_question(report, depth="deep" if is_deep else "standard")
        if question:
            question = question.rstrip().rstrip("?") + "? Reply yes or no."
            reply = reply.rstrip() + "\n\n" + question
            self.pending_discovery = {
                "original_prompt": user_prompt,
                "contract": contract,
                "report": report,
                "question": question,
            }
        else:
            self.pending_discovery = None
        self.last_report = report
        self.last_status = "completed"
        turn_prompt = history_prompt or user_prompt
        self.history.append({"role": "user", "content": turn_prompt})
        self.history.append({"role": "assistant", "content": reply})
        self.log_run(turn_prompt, reply, contract=contract, report=report)
        return reply

    def continue_project_discovery(self):
        pending = self.pending_discovery
        if not pending:
            return None
        self.pending_discovery = None
        return self.run_project_discovery_turn(
            pending["original_prompt"],
            depth="deep",
            prior_report=pending["report"],
            history_prompt="yes",
        )

    def decline_project_discovery(self):
        self.pending_discovery = None
        reply = "Understood. I’ll keep the current answer at this level."
        self.last_status = "reply"
        self.history.append({"role": "user", "content": "no"})
        self.history.append({"role": "assistant", "content": reply})
        self.log_run("no", reply, status="reply")
        return reply

    def _direct_turn(self, user_prompt):
        """Single model call for chat mode — no JSON delegate logic."""
        messages = [{"role": "system", "content": self.frontend_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "user", "content": "Chat mode is active. Do not delegate; answer conversationally."})
        if self.on_token:
            reply = self._chat_streaming(self.frontend_model, messages)
        else:
            with Spinner(self.ui, self.ui.style("thinking", UI.DIM)):
                reply = self.chat(self.frontend_model, messages).strip()
        self.last_report = None
        self.last_status = "reply"
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": reply})
        self.log_run(user_prompt, reply, status="reply")
        return reply

    def run_turn(self, user_prompt, planning=False):
        self.active_context_extra = user_prompt
        self.sync_executor()
        self._prune_history()
        if self.pending_plan and PROMPT_YES_RE.search(user_prompt.strip()):
            applied = self.apply_pending_plan()
            if applied is not None:
                return applied
        if self.pending_discovery:
            if PROMPT_YES_RE.fullmatch(user_prompt.strip()):
                continued = self.continue_project_discovery()
                if continued is not None:
                    return continued
            if re.fullmatch(r"(?i)\s*(?:n|no|nope|stop)\s*[.!]?\s*", user_prompt):
                return self.decline_project_discovery()
            # A substantive new prompt supersedes the optional follow-up.
            self.pending_discovery = None

        if not planning and self.wants_project_discovery(user_prompt):
            return self.run_project_discovery_turn(user_prompt)

        if self.mode == "chat":
            return self._direct_turn(user_prompt)

        if self.mode == "agent":
            contract = self.classify_contract(user_prompt)
            if self.wants_read_only_inspection(user_prompt):
                contract["edit_policy"] = "inspect"
                contract["read_only"] = True
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
            if self.wants_read_only_inspection(user_prompt):
                contract["edit_policy"] = "inspect"
                contract["read_only"] = True
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
        if self.wants_read_only_inspection(user_prompt):
            contract["edit_policy"] = "inspect"
            contract["read_only"] = True
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

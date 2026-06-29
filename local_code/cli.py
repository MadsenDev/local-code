import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from .agent import LocalPartner
from .config import (
    DEFAULT_BACKEND_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_FRONTEND_MODEL,
    DEFAULT_MODEL,
    DEFAULT_MODE,
    DEFAULT_MODEL_ROUTING,
    DEFAULT_NUM_CTX,
    DEFAULT_TOOL_CALLING,
    DEFAULT_OLLAMA,
    DEFAULT_VERBOSITY,
    load_runtime_config,
    save_runtime_config,
    HELP_TEXT,
    MAX_TOOL_STEPS,
)
from .memory import clear_chat_history, save_chat_history
from .intelligence import DecisionService
from .intelligence.indexer import format_index_report, index_repository
from .paths import rist_home
from .llamacpp import DEFAULT_LLAMACPP_BASE_URL, LLAMACPP_GPU_PROFILES, format_llama_server_command, generate_llama_server_command, get_llamacpp_profile, recommend_gpu_profile, recommend_model_profiles
from .llama_runtime import (
    find_llama_server,
    install_llama_server,
    install_model,
    list_managed_models,
    log_path as managed_log_path,
    prepare_server_start,
    recent_log_lines,
    register_model,
    remove_model,
    server_status,
    start_server,
    stop_server,
)
from .diagnostics import benchmark_model, doctor_report, format_benchmark, format_doctor, to_json
from .hardware import detect_hardware
from .routing import resolve_model_routing
from .model_profiles import (
    RECOMMENDED_CEILING,
    RECOMMENDED_STANDARD,
    advisory_lines,
    classify_model,
)
from .providers import build_provider
from .tools import git_summary, list_files, read_file, resolve_path
from .ui import UI

os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout.processors import Processor, Transformation
    from prompt_toolkit.styles import Style
except Exception:
    PromptSession = None
    InMemoryHistory = None
    KeyBindings = None
    Keys = None
    Processor = object
    Transformation = None
    Style = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rist - local-first AI coding agent for real developer hardware."
    )
    parser.add_argument("command", nargs="?", choices=["chat", "setup", "doctor", "benchmark", "bench", "llama", "model", "index", "decisions"], help="Run diagnostics, benchmark, or llama.cpp helpers")
    parser.add_argument(
        "llama_action",
        nargs="?",
        choices=["doctor", "command", "install", "tune", "start", "stop", "status", "restart", "list", "register", "logs", "remove", "add", "accept", "supersede", "review", "reject", "merge"],
        help="llama.cpp or managed-model action",
    )
    parser.add_argument("model_target", nargs="?", help="Managed model profile, e.g. qwen36")
    parser.add_argument("-m", "--model", default=None, help=f"Model name override for both roles (default: role-specific)")
    parser.add_argument("--frontend-model", dest="frontend_model", default=None, help="Frontend/talker model name (falls back to --model)")
    parser.add_argument("--backend-model", dest="backend_model", default=None, help="Backend/coder model name (falls back to --model)")
    parser.add_argument("--planner-model", dest="frontend_model_alias", help="Compatibility alias for --frontend-model")
    parser.add_argument("--coder-model", dest="backend_model_alias", help="Compatibility alias for --backend-model")
    parser.add_argument("--ollama", default=DEFAULT_OLLAMA, help=f"Ollama base URL ({DEFAULT_OLLAMA})")
    parser.add_argument("--workdir", default=os.getcwd(), help="Working directory for repo-aware operations")
    parser.add_argument("--auto-approve", action="store_true", help="Compatibility flag. Allow shell commands without prompting")
    parser.add_argument("--prompt", help="Run a single prompt non-interactively and exit")
    parser.add_argument("--max-steps", type=int, default=MAX_TOOL_STEPS, help=f"Maximum tool steps per backend turn ({MAX_TOOL_STEPS})")
    parser.add_argument("--verbosity", choices=["quiet", "normal", "debug"], default=DEFAULT_VERBOSITY, help=f"Output level ({DEFAULT_VERBOSITY})")
    parser.add_argument("--show-raw-actions", action="store_true", help="Show raw JSON actions/contracts in debug mode")
    parser.add_argument("--tool-calling", choices=["json", "native", "auto"], default=DEFAULT_TOOL_CALLING, help=f"Backend tool protocol ({DEFAULT_TOOL_CALLING})")
    parser.add_argument("--mode", choices=["chat", "hybrid", "agent"], default=DEFAULT_MODE, help=f"Interaction mode ({DEFAULT_MODE})")
    parser.add_argument(
        "--storage-mode",
        choices=["local-only", "shared", "hybrid"],
        default=None,
        help="Repository memory policy (default: RIST_STORAGE_MODE or hybrid)",
    )
    parser.add_argument("--model-routing", choices=["single", "adaptive", "dual"], default=DEFAULT_MODEL_ROUTING, help=f"Model residency/routing strategy ({DEFAULT_MODEL_ROUTING})")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of measured benchmark runs per prompt size (3)")
    parser.add_argument("--start", action="store_true", help="During setup, start the managed llama.cpp runtime after registration")
    parser.add_argument("--benchmark", action="store_true", help="During setup, run a benchmark after a successful managed start")
    parser.add_argument("--long-context", action="store_true", help="Include an optional long-context benchmark prompt")
    parser.add_argument("--profile", default="qwen2.5-coder-7b", help="llama.cpp command profile")
    parser.add_argument("--gpu", default=None, help="llama.cpp hardware profile")
    parser.add_argument("--model-path", "--path", dest="model_path", default=None, help="GGUF path to print in the suggested llama-server command")
    parser.add_argument("--url", default=None, help="Explicit URL for a model GGUF or prebuilt llama-server binary")
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 for a downloaded artifact")
    parser.add_argument("--filename", default=None, help="Destination filename for a downloaded GGUF")
    parser.add_argument("--destination", default=None, help="Destination path for a downloaded llama-server binary")
    parser.add_argument("--title", default=None, help="Decision title or edited candidate title")
    parser.add_argument("--rationale", default="", help="Decision rationale")
    parser.add_argument("--alternatives", action="append", default=[], help="Rejected alternative (repeatable)")
    parser.add_argument("--consequences", action="append", default=[], help="Decision consequence (repeatable)")
    parser.add_argument("--component", action="append", default=[], help="Affected component (repeatable)")
    parser.add_argument("--source", action="append", default=[], help="Source reference (repeatable)")
    parser.add_argument("--with", dest="superseding_id", default=None, help="Superseding decision ID")
    parser.add_argument("--accept", dest="review_accept", default=None, help="Accept a pending candidate ID")
    parser.add_argument("--reject", dest="review_reject", default=None, help="Reject a pending candidate ID")
    parser.add_argument("--edit", dest="review_edit", default=None, help="Edit a pending candidate ID")
    parser.add_argument("--merge", dest="review_merge", nargs="+", default=None, help="Merge pending candidate IDs")
    parser.add_argument("--force", action="store_true", help="Force a full index rebuild, replace an artifact, or stop a running model before removal")
    parser.add_argument("--status", action="store_true", help="Report repository index freshness without writing artifacts")
    parser.add_argument("--preview", action="store_true", help="Preview repository index artifacts without writing them")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print model start details without launching a process")
    parser.add_argument("--delete-file", action="store_true", help="Delete a registered GGUF when removing a model")
    parser.add_argument("--yes", action="store_true", help="Run setup non-interactively and confirm destructive model file deletion")
    parser.add_argument("--tail", type=int, default=100, help="Number of managed llama.cpp log lines to show (100)")
    parser.add_argument("--follow", action="store_true", help="Continue streaming appended managed llama.cpp log output")
    parser.add_argument("--llama-server", dest="llama_server", default=None, help="Path to the external llama-server executable")
    parser.add_argument("--host", default="127.0.0.1", help="Host for a managed llama-server (127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port for a managed llama-server (8080)")
    parser.add_argument("--wait-timeout", type=float, default=120, help="Seconds to wait for a managed server to become ready (120)")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit doctor/benchmark output as JSON")
    parser.add_argument("--provider", choices=["auto", "ollama", "llamacpp", "openrouter", "openai"], default=None, help="Model provider (default: auto)")
    parser.add_argument("--base-url", dest="base_url", default=None, help="Override the provider base URL (e.g. an OpenAI-compatible server)")
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key for cloud providers (else read from env)")
    parser.add_argument("--no-preflight", dest="no_preflight", action="store_true", help="Skip the startup model/provider check")
    parser.add_argument("--no-tui", dest="no_tui", action="store_true", help="Use the plain line-based REPL instead of the full-screen TUI")
    return parser.parse_args()


def run_preflight(agent, verbose=False):
    """Check the provider + configured models and advise.

    Always non-fatal. For local Ollama it advises on the 12 GB standard and
    stays silent when everything is pulled and meets it (unless `verbose`). For
    cloud providers it checks the key/connectivity instead of VRAM tiers.
    """
    if not agent.provider.is_local:
        _run_preflight_cloud(agent, verbose)
        return

    ui = agent.ui
    if not agent.provider.available():
        if getattr(agent.provider, "name", "") == "llamacpp":
            title = "llama.cpp not reachable"
            details = agent.provider.failure_message().splitlines()
        else:
            title = "Ollama not reachable"
            details = [
                f"Could not reach {agent.provider.describe()}.",
                "Start it with `ollama serve`, or point at it with --ollama / --base-url.",
            ]
        print(ui.box(title, details, color=UI.YELLOW), file=sys.stderr)
        return

    if getattr(agent.provider, "name", "") == "llamacpp":
        if verbose:
            models = sorted(agent.provider.list_models())
            body = [
                f"Provider: {agent.provider.describe()}",
                "Reported models: " + (", ".join(models) or "none"),
                "External heavy backend; run `rist llama doctor` for a chat probe and server metadata.",
            ]
            print(ui.box("llama.cpp", body, color=UI.CYAN), file=sys.stderr)
        return

    models = list(dict.fromkeys([agent.frontend_model, agent.backend_model]))
    profiles = {m: classify_model(m) for m in models}
    missing = [m for m in models if agent.provider.model_available(m) is False]
    below = [m for m, p in profiles.items() if not p.meets_standard]
    advisory = advisory_lines(agent.frontend_model, agent.backend_model)
    dual_warn = any("won't stay resident" in line for line in advisory)

    if not verbose and not missing and not below and not dual_warn:
        return

    body = list(advisory)
    if missing:
        body.append("")
        body.extend(f"⚠ not pulled — run: ollama pull {m}" for m in missing)
    body.append("")
    body.append(f"Standard for 12 GB GPUs: {RECOMMENDED_STANDARD} (ceiling {RECOMMENDED_CEILING}).")
    color = UI.YELLOW if (missing or below or dual_warn) else UI.CYAN
    print(ui.box("Models", body, color=color), file=sys.stderr)


def _run_preflight_cloud(agent, verbose):
    ui = agent.ui
    provider = agent.provider
    models = list(dict.fromkeys([agent.frontend_model, agent.backend_model]))
    if not provider.available():
        if getattr(provider, "api_key", None):
            body = [f"Provider: {provider.describe()}", "Could not reach the provider (check --base-url / network)."]
        else:
            env = "OPENROUTER_API_KEY" if provider.name == "openrouter" else "OPENAI_API_KEY"
            body = [f"Provider: {provider.describe()}", f"No API key. Set {env}, or pass --api-key."]
        print(ui.box("Provider not ready", body, color=UI.YELLOW), file=sys.stderr)
        return
    if not verbose:
        return
    body = [
        f"Provider: {provider.describe()}",
        f"Models: {', '.join(models)}",
        "Cloud models — local VRAM tiers don't apply; structured output is still enforced.",
    ]
    print(ui.box("Models", body, color=UI.CYAN), file=sys.stderr)


def render_header(agent):
    repo, branch, changes = git_summary(agent.workdir)
    return agent.ui.compact_status(
        agent.mode,
        agent.frontend_model,
        agent.backend_model,
        repo,
        branch,
        changes,
        agent.edit_permission,
        agent.command_permission,
    )


def render_status(agent):
    repo, branch, changes = git_summary(agent.workdir)
    rows = [
        ("Mode", agent.ui.mode_label(agent.mode)),
        ("Safety", f"edits {agent.edit_permission}, commands {agent.command_permission}"),
        ("Workdir", agent.workdir),
        ("Git", f"{repo}/{branch}, {changes}"),
        ("Provider", agent.provider.describe()),
        ("Frontend", agent.frontend_model),
        ("Backend", agent.backend_model),
        ("Model routing", f"{agent.routing_decision.get('mode', agent.model_routing)} (requested: {agent.model_routing})"),
        ("Trace", "on" if agent.verbosity == "debug" else "off"),
        ("Raw JSON", "on" if agent.show_raw_actions else "off"),
        ("Tool calling", agent.tool_calling),
        ("Pending plan", "yes" if agent.pending_plan else "no"),
        ("Context", f"{agent.context_usage().total:,} / {agent.context_limit:,} estimated tokens ({agent.context_usage().percent}%)"),
    ]
    return agent.ui.kv_box("Rist status", rows, color=UI.CYAN)


def render_final_card(agent):
    report = getattr(agent, "last_report", None)
    if not report:
        return ""
    parts = []
    changed = len(report.get("files_changed") or [])
    read_count = len(report.get("files_read") or [])
    cmds = len(report.get("commands_run") or [])
    if changed:
        parts.append(f"{changed} file{'s' if changed != 1 else ''} edited")
    if read_count:
        parts.append(f"{read_count} read")
    if cmds:
        parts.append(f"{cmds} cmd{'s' if cmds != 1 else ''} run")
    if report.get("risks"):
        parts.append(agent.ui.style("⚠ risks", UI.YELLOW))
    if report.get("needs_approval"):
        parts.append(agent.ui.style("→ type 'yes' to apply", UI.YELLOW))
    if not parts:
        return ""
    return agent.ui.style("  · " + " · ".join(parts), UI.DIM)


def is_ui_chrome(prompt):
    stripped = strip_paste_markers(prompt).strip()
    if not stripped:
        return False
    prefixes = (
        "╭─",
        "│ ",
        "╰─",
        "─",
        "== Session ==",
        "== Rist ==",
        "== Action ==",
        "== Result ==",
        "== Done ==",
        "You > ",
        "rist ",
        "backend ",
        "frontend ",
        "⠋ ",
        "⠙ ",
        "⠹ ",
        "⠸ ",
        "⠼ ",
        "⠴ ",
        "⠦ ",
        "⠧ ",
        "⠇ ",
        "⠏ ",
    )
    return stripped.startswith(prefixes)


def starts_transcript_paste(prompt):
    stripped = strip_paste_markers(prompt).strip()
    return stripped.startswith(("You > ", "────────────────", "╭─", "== Session =="))


def should_capture_paste(prompt):
    stripped = strip_paste_markers(prompt).strip()
    return (
        has_literal_paste_start(prompt)
        or has_literal_paste_end(prompt)
        or "You >" in stripped
        or stripped.startswith(("────────────────", "╭─", "│ ", "╰─", "== Session =="))
    )


def has_literal_paste_start(prompt):
    return "[200~" in prompt or "\x1b[200~" in prompt


def has_literal_paste_end(prompt):
    return "[201~" in prompt or "\x1b[201~" in prompt


def strip_paste_markers(prompt):
    return (
        prompt.replace("\x1b[200~", "")
        .replace("\x1b[201~", "")
        .replace("[200~", "")
        .replace("[201~", "")
    )


def read_paste_block(agent):
    print(agent.ui.box("Paste", ["Paste context below.", "Finish with /end on its own line."], color=UI.CYAN))
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if line.strip() == "/end":
            return "\n".join(lines).strip()
        lines.append(line)


class PastedContentProcessor(Processor):
    TOKEN_RE = re.compile(r"(\[Pasted Content \d+ chars(?: #\d+)?\])")

    def apply_transformation(self, transformation_input):
        fragments = []
        for style, text, *rest in transformation_input.fragments:
            parts = self.TOKEN_RE.split(text)
            for part in parts:
                if not part:
                    continue
                if self.TOKEN_RE.fullmatch(part):
                    fragments.append(("class:pasted", part))
                else:
                    fragments.append((style, part, *rest))
        return Transformation(fragments)


def make_prompt_session(pastes):
    if PromptSession is None:
        return None
    bindings = KeyBindings()

    @bindings.add(Keys.BracketedPaste)
    def _(event):
        text = event.data
        index = len(pastes) + 1
        token = f"[Pasted Content {len(text)} chars]"
        if token in pastes:
            token = f"[Pasted Content {len(text)} chars #{index}]"
        pastes[token] = text
        event.current_buffer.insert_text(token)

    @bindings.add("backspace")
    def _(event):
        buffer = event.current_buffer
        before = buffer.document.text_before_cursor
        for token in sorted(pastes, key=len, reverse=True):
            if before.endswith(token):
                buffer.delete_before_cursor(len(token))
                return
        buffer.delete_before_cursor(1)

    return PromptSession(
        key_bindings=bindings,
        input_processors=[PastedContentProcessor()],
        style=Style.from_dict({"pasted": "ansigreen bold"}),
        reserve_space_for_menu=0,
        history=InMemoryHistory(),
    )


def read_prompt(session, pastes):
    pastes.clear()
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return input("").strip()
    if session is not None:
        message = [("bold ansicyan", "> ")]
        text = session.prompt(message)
        for token, content in list(pastes.items()):
            labelled = (
                f"\n\n--- PASTED CONTENT ({len(content)} chars) ---\n"
                f"{content}\n"
                "--- END PASTED CONTENT ---\n"
            )
            text = text.replace(token, labelled)
        return strip_paste_markers(text).strip()
    return input("> ").strip()


def _looks_like_diff(text):
    return bool(text and ("--- " in text or "+++ " in text or text.lstrip().startswith("@@")))


def print_reply(agent, reply):
    if not agent.last_streamed:
        print()
        agent.ui.print_markdown(reply)
    report = getattr(agent, "last_report", None)
    if report and report.get("needs_approval"):
        plan = [s for s in (report.get("plan") or []) if s]
        diff = (report.get("diff_summary") or "").strip()
        if plan:
            print()
            print(agent.ui.style("  Planned changes:", agent.ui.BOLD))
            for i, step in enumerate(plan, 1):
                print(f"    {i}. {step}")
        if diff and _looks_like_diff(diff):
            print()
            rendered = agent.ui.render_diff_lines(diff.splitlines())
            for line in rendered:
                print(line)
    card = render_final_card(agent)
    if card:
        print(card)
    print()
    agent.last_streamed = False


def _handle_cancelled(agent):
    sys.stderr.write("\r\033[2K")
    sys.stderr.flush()
    print("\nCancelled.", flush=True)
    agent.last_streamed = False


def run_with_temporary_mode(agent, mode, prompt, planning=False):
    previous = agent.mode
    agent.mode = mode
    try:
        return agent.run_turn(prompt, planning=planning)
    finally:
        agent.mode = previous


def pick_file(workdir):
    if shutil.which("fzf") is None:
        return None, "fzf is not installed."
    if shutil.which("rg") is None:
        return None, "ripgrep (rg) is not installed."
    try:
        files = subprocess.run(
            ["rg", "--files", "--hidden", "-g", "!.git"],
            cwd=workdir,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not list files: {exc}"
    if files.returncode != 0 or not files.stdout.strip():
        return None, "No files found."
    try:
        selected = subprocess.run(
            ["fzf", "--height=40%", "--reverse", "--prompt=files> "],
            input=files.stdout,
            text=True,
            capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"fzf failed: {exc}"
    if selected.returncode != 0:
        return None, "No file selected."
    return selected.stdout.strip(), ""


AT_REF_RE = re.compile(r"(?<!\S)@([^\s]+)")


def expand_at_references(prompt, workdir, ui):
    refs = []
    seen = set()
    for match in AT_REF_RE.finditer(prompt):
        raw = match.group(1).rstrip(".,;:)")
        if raw and raw not in seen:
            refs.append(raw)
            seen.add(raw)
    if not refs:
        return prompt

    sections = []
    for ref in refs[:8]:
        target = resolve_path(workdir, ref)
        try:
            if target.is_file():
                sections.append(
                    f"--- @FILE {ref} ---\n"
                    + read_file(workdir, ref, 1, 240)
                    + "\n--- END @FILE ---"
                )
            elif target.is_dir():
                sections.append(
                    f"--- @FOLDER {ref} ---\n"
                    + list_files(workdir, ref)
                    + "\n--- END @FOLDER ---"
                )
            else:
                sections.append(f"--- @MISSING {ref} ---\nPath not found: {target}\n--- END @MISSING ---")
        except Exception as exc:  # noqa: BLE001
            sections.append(f"--- @ERROR {ref} ---\n{exc}\n--- END @ERROR ---")
    if len(refs) > 8:
        sections.append(f"--- @REFERENCES OMITTED ---\n{len(refs) - 8} additional reference(s) omitted.\n--- END @REFERENCES OMITTED ---")
    print(ui.transcript_item("Expanded @ references", refs[:8], color=UI.CYAN))
    return prompt + "\n\n" + "\n\n".join(sections)


def _connection_error_message(agent, exc):
    if getattr(agent.provider, "name", "") == "llamacpp":
        return agent.provider.failure_message()
    return f"Connection error (is Ollama running?): {exc}"


def interactive_loop(agent):
    print(render_header(agent))
    print(agent.ui.style("  /help for commands · /status for details", UI.DIM))
    print()
    if agent.ui.enabled:
        agent.on_token = lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush())
    pastes = {}
    session = make_prompt_session(pastes)
    while True:
        try:
            prompt = read_prompt(session, pastes)
        except (EOFError, KeyboardInterrupt):
            save_chat_history(agent.workdir, agent.history, agent.storage_mode)
            print()
            return 0

        if not prompt:
            continue
        prompt = strip_paste_markers(prompt)
        if not prompt:
            continue
        if is_ui_chrome(prompt):
            continue
        if prompt == "/quit":
            return 0

        if prompt == "/help":
            print(HELP_TEXT, end="")
            continue
        if prompt == "/clear":
            agent.history.clear()
            agent.pending_plan = None
            clear_chat_history(agent.workdir, agent.storage_mode)
            print("History cleared.")
            continue
        if prompt.startswith("/mode "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value in {"chat", "hybrid", "agent"}:
                agent.mode = value
                print(render_header(agent))
            else:
                print("Use /mode chat, /mode hybrid, or /mode agent")
            continue
        if prompt.startswith("/model "):
            value = prompt.split(None, 1)[1].strip()
            agent.frontend_model = value
            agent.backend_model = value
            agent.preferred_frontend_model = value
            agent.preferred_backend_model = value
            print(render_header(agent))
            continue
        if prompt.startswith("/frontend "):
            agent.frontend_model = prompt.split(None, 1)[1].strip()
            agent.preferred_frontend_model = agent.frontend_model
            print(render_header(agent))
            continue
        if prompt.startswith("/backend "):
            agent.backend_model = prompt.split(None, 1)[1].strip()
            agent.preferred_backend_model = agent.backend_model
            print(render_header(agent))
            continue
        if prompt.startswith("/planner "):
            agent.frontend_model = prompt.split(None, 1)[1].strip()
            agent.preferred_frontend_model = agent.frontend_model
            print(render_header(agent))
            continue
        if prompt.startswith("/coder "):
            agent.backend_model = prompt.split(None, 1)[1].strip()
            agent.preferred_backend_model = agent.backend_model
            print(render_header(agent))
            continue
        if prompt.startswith("/permission "):
            parts = prompt.split()
            if len(parts) == 3:
                _, scope, mode = parts
            elif len(parts) == 2:
                _, mode = parts
                scope = "all"
            else:
                print("Use /permission [all|command|edit] ask|allow|deny")
                continue
            if scope not in {"all", "command", "edit"} or mode not in {"ask", "allow", "deny"}:
                print("Use /permission [all|command|edit] ask|allow|deny")
                continue
            if scope in {"all", "command"}:
                agent.command_permission = mode
            if scope in {"all", "edit"}:
                agent.edit_permission = mode
            print(render_header(agent))
            continue
        if prompt.startswith("/approve "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value == "on":
                agent.command_permission = "allow"
                print(render_header(agent))
            elif value == "off":
                agent.command_permission = "ask"
                print(render_header(agent))
            else:
                print("Use /approve on or /approve off")
            continue
        if prompt.startswith("/verbosity "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value in {"quiet", "normal", "debug"}:
                agent.verbosity = value
                print(f"Verbosity: {agent.verbosity}")
            else:
                print("Use /verbosity quiet, /verbosity normal, or /verbosity debug")
            continue
        if prompt.startswith("/trace "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value == "on":
                agent.verbosity = "debug"
                print("Verbosity: debug")
            elif value == "off":
                agent.verbosity = "normal"
                print("Verbosity: normal")
            else:
                print("Use /trace on or /trace off")
            continue
        if prompt.startswith("/tools "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value in {"json", "native", "auto"}:
                agent.tool_calling = value
                print(f"Tool calling: {agent.tool_calling}")
            else:
                print("Use /tools json, /tools native, or /tools auto")
            continue
        if prompt == "/context":
            usage = agent.context_usage()
            print(agent.ui.kv_box("Context usage", [
                ("Conversation", f"{usage.conversation:,}"),
                ("Memory", f"{usage.memory:,}"),
                ("Repo", f"{usage.repo:,}"),
                ("Tools", f"{usage.tools:,}"),
                ("Other", f"{usage.other:,}"),
                ("Total", f"{usage.total:,} / {usage.limit:,} ({usage.percent}%)"),
                ("Remaining", f"{usage.remaining:,}"),
            ], color=UI.CYAN))
            continue
        if prompt.startswith("/routing "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value not in {"single", "adaptive", "dual"}:
                print("Use /routing single, /routing adaptive, or /routing dual")
                continue
            hardware = detect_hardware(getattr(agent.provider, "base_url", None) if agent.provider.is_local else None)
            front, back, decision = resolve_model_routing(value, agent.provider.is_local, hardware, agent.preferred_frontend_model, agent.preferred_backend_model, agent.context_limit, provider_is_heavy=getattr(agent.provider, "is_heavy_backend", False))
            agent.model_routing, agent.frontend_model, agent.backend_model, agent.routing_decision = value, front, back, decision
            agent.sync_executor()
            print(f"Model routing: {decision['mode']} — {decision['reason']}")
            continue
        if prompt.startswith("/raw "):
            value = prompt.split(None, 1)[1].strip().lower()
            if value in {"on", "off"}:
                agent.show_raw_actions = value == "on"
                print(f"Raw actions: {'on' if agent.show_raw_actions else 'off'}")
            else:
                print("Use /raw on or /raw off")
            continue
        if prompt == "/paste":
            text = read_paste_block(agent)
            if not text:
                print("Paste cancelled.")
                continue
            try:
                reply = agent.run_turn(text)
            except KeyboardInterrupt:
                _handle_cancelled(agent)
                continue
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(f"HTTP error: {exc}\n{detail}")
                continue
            except urllib.error.URLError as exc:
                print(_connection_error_message(agent, exc))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            print_reply(agent, reply)
            continue
        if prompt.startswith("/ask "):
            text = prompt.split(None, 1)[1].strip()
            try:
                reply = agent.run_readonly_turn(text)
            except KeyboardInterrupt:
                _handle_cancelled(agent)
                continue
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(f"HTTP error: {exc}\n{detail}")
                continue
            except urllib.error.URLError as exc:
                print(_connection_error_message(agent, exc))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            print_reply(agent, reply)
            continue
        if prompt.startswith("/plan "):
            text = prompt.split(None, 1)[1].strip()
            try:
                reply = agent.run_turn(text, planning=True)
            except KeyboardInterrupt:
                _handle_cancelled(agent)
                continue
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(f"HTTP error: {exc}\n{detail}")
                continue
            except urllib.error.URLError as exc:
                print(_connection_error_message(agent, exc))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            print_reply(agent, reply)
            continue
        if prompt.startswith("/code "):
            text = prompt.split(None, 1)[1].strip()
            try:
                reply = run_with_temporary_mode(agent, "agent", text)
            except KeyboardInterrupt:
                _handle_cancelled(agent)
                continue
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(f"HTTP error: {exc}\n{detail}")
                continue
            except urllib.error.URLError as exc:
                print(_connection_error_message(agent, exc))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            print_reply(agent, reply)
            continue
        if prompt.startswith("/agent "):
            text = prompt.split(None, 1)[1].strip()
            try:
                reply = run_with_temporary_mode(agent, "agent", text)
            except KeyboardInterrupt:
                _handle_cancelled(agent)
                continue
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(f"HTTP error: {exc}\n{detail}")
                continue
            except urllib.error.URLError as exc:
                print(_connection_error_message(agent, exc))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            print_reply(agent, reply)
            continue
        if prompt == "/files":
            selected, error = pick_file(agent.workdir)
            if error:
                print(error)
            else:
                print(agent.ui.transcript_item("Selected file", [selected], color=UI.CYAN))
            continue
        if prompt == "/apply":
            if not agent.pending_plan:
                print("No active pending proposal to apply.")
                continue
            try:
                reply = agent.apply_pending_plan()
            except KeyboardInterrupt:
                _handle_cancelled(agent)
                continue
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                print(f"HTTP error: {exc}\n{detail}")
                continue
            except urllib.error.URLError as exc:
                print(_connection_error_message(agent, exc))
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            if reply is None:
                print("No pending plan to apply.")
            else:
                print_reply(agent, reply)
            continue
        if prompt == "/decisions" or prompt.startswith("/decisions "):
            try:
                print(agent.run_decision_command(prompt[1:]))
            except (KeyError, ValueError) as exc:
                print(f"Error: {exc}")
            continue
        if prompt == "/status":
            print(render_status(agent))
            continue
        if prompt == "/models":
            run_preflight(agent, verbose=True)
            continue
        if prompt.startswith("/"):
            print("Unknown command. Type /help.")
            continue

        prompt = expand_at_references(prompt, agent.workdir, agent.ui)
        try:
            reply = agent.run_turn(prompt)
        except KeyboardInterrupt:
            _handle_cancelled(agent)
            continue
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            print(f"HTTP error: {exc}\n{detail}")
            continue
        except urllib.error.URLError as exc:
            print(_connection_error_message(agent, exc))
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            continue

        print_reply(agent, reply)


def _default_model_for(provider_kind):
    # Local servers have a sensible alias; cloud providers must be told a model.
    if provider_kind == "ollama":
        return DEFAULT_MODEL
    if provider_kind == "llamacpp":
        return "local"
    return None


def _print_runtime_report(report, json_output=False):
    if json_output:
        print(to_json(report))
        return
    if not report.get("managed") and report.get("state") == "stopped":
        print("No managed llama.cpp server is currently running.")
        print(f"Log: {report.get('log_path', managed_log_path())}")
        return
    print("Managed llama.cpp runtime:")
    print(f"Status: {report.get('state', 'unknown')}")
    for key, label in (
        ("profile", "Model"), ("pid", "PID"), ("port", "Port"), ("base_url", "Base URL"),
        ("log_path", "Log"), ("started_at", "Started"), ("model_path", "Model path"),
        ("executable", "Executable"), ("context", "Context"), ("gpu", "GPU profile"), ("message", "Message"),
    ):
        if report.get(key) is not None:
            print(f"{label}: {report[key]}")
    health = report.get("health")
    health_state = "reachable" if health and health.get("ready") else "unreachable" if health else "not checked"
    print(f"Health: {health_state}")
    if health and health.get("models"):
        print("Reported models: " + ", ".join(health["models"]))


def _show_managed_logs(tail=100, follow=False):
    if tail < 0:
        raise ValueError("--tail must be zero or greater.")
    path = managed_log_path()
    try:
        if not path.exists():
            print("No managed llama.cpp server log has been created yet.")
            print("Rist only knows about logs for servers started with `rist model start`.")
            return 0
        lines = recent_log_lines(tail, path)
        for line in lines:
            print(line, end="" if line.endswith("\n") else "\n")
        if not follow:
            if not lines and path.stat().st_size == 0:
                print("The managed llama.cpp server log is empty.")
            return 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, 2)
            while True:
                line = handle.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(f"Unable to read the managed llama.cpp server log at {path}: {exc}")
        print("If llama-server was started manually, Rist does not know where its logs are stored.")
        return 0


def _print_dry_run(report, json_output=False):
    if json_output:
        print(to_json({"dry_run": True, **report}))
        return
    print("Dry run: llama.cpp managed server")
    for label, key in (
        ("Executable", "executable"), ("Model", "model_path"), ("Port", "port"),
        ("Base URL", "base_url"), ("Command", "command"), ("Log", "log_path"), ("State", "state_path"),
    ):
        print()
        print(f"{label}:")
        print(report[key])
    print()
    print("No process started.")


def _handle_runtime_command(args):
    action = args.llama_action
    if args.command == "llama" and action == "install":
        if not args.url:
            raise ValueError("`rist llama install` requires --url for a prebuilt llama-server binary.")
        report = install_llama_server(args.url, destination=args.destination, sha256=args.sha256, force=args.force)
        if args.json_output:
            print(to_json(report))
        else:
            print(f"Installed llama-server: {report['executable']}")
            print(f"SHA-256: {report['sha256']}")
        return 0
    if args.command == "llama" and action == "logs":
        return _show_managed_logs(args.tail, args.follow)
    if args.command == "llama" and action == "tune":
        return _run_llama_tune(args)
    if args.command != "model":
        return None
    if action == "list":
        models = list_managed_models()
        if args.json_output:
            print(to_json(models))
        elif not models:
            print("No managed llama.cpp models. Run: rist model install qwen36 --url URL")
        else:
            for model in models:
                state = "installed" if model["exists"] else "missing"
                print(f"{model['id']}: {state} — {model['path']}")
        return 0
    if action == "status":
        _print_runtime_report(server_status(), args.json_output)
        return 0
    if action == "logs":
        return _show_managed_logs(args.tail, args.follow)
    if action == "stop":
        _print_runtime_report(stop_server(), args.json_output)
        return 0
    if action == "restart":
        stop_server()
        action = "start"
    target = _runtime_profile(args)
    if not target:
        raise ValueError(f"`rist model {action or 'ACTION'}` requires a profile, e.g. qwen2.5-coder-7b.")
    if action == "install":
        if not args.url:
            raise ValueError("Model installation requires an explicit --url to a GGUF file; Hugging Face authentication is not managed yet.")
        if not args.json_output:
            profile = get_llamacpp_profile(target)
            hardware = profile["hardware"]
            print(f"Installing {profile['name']}...")
            print(f"Source: {args.url}")
            print(f"Recommended RAM: {hardware['recommended_ram_gb']} GB (ideal {hardware['ideal_ram_gb']} GB)")
            print(f"Recommended VRAM: {hardware['recommended_vram_gb']} GB")
        report = install_model(target, args.url, filename=args.filename, sha256=args.sha256, force=args.force)
        if args.json_output:
            print(to_json(report))
        else:
            print(f"Installed model: {report['profile']}")
            print(f"Path: {report['path']}")
            print(f"Size: {report['bytes'] / 1024**3:.2f} GiB")
            print(f"SHA-256: {report['sha256']}")
        return 0
    if action == "register":
        if not args.model_path:
            raise ValueError("Model registration requires --model-path /path/to/model.gguf.")
        report = register_model(target, args.model_path)
        print(to_json(report) if args.json_output else f"Registered {report['profile']}: {report['path']}")
        return 0
    if action == "remove":
        registered = next((item for item in list_managed_models() if item["id"] == get_llamacpp_profile(target)["id"]), None)
        if not registered:
            raise ValueError(f"Model {target!r} is not registered.")
        confirmed = args.yes
        if args.delete_file and not confirmed:
            answer = input(f"Delete registered GGUF {registered['path']}? [y/N] ").strip().lower()
            confirmed = answer in {"y", "yes"}
            if not confirmed:
                print("Removal cancelled; the registry and model file were not changed.")
                return 1
        report = remove_model(target, delete_file=args.delete_file, confirmed=confirmed, force=args.force)
        if args.json_output:
            print(to_json(report))
        else:
            print(f"Unregistered {report['id']}.")
            if report["file_deleted"]:
                print(f"Deleted model file: {report['path']}")
            elif report["file_missing"]:
                print(f"The registered model file was already missing: {report['path']}")
            else:
                print(f"Model file was kept: {report['path']}")
        return 0
    if action == "start":
        if not args.json_output:
            profile = get_llamacpp_profile(target)
            detected = detect_hardware()
            gpu_name = detected.gpus[0].name if detected.gpus else args.gpu
            print(f"Starting {profile['name']}...")
            print("Provider: llama.cpp")
            print(f"Port: {args.port}")
            print(f"GPU: {gpu_name}")
            requirements = profile["hardware"]
            system_ram = getattr(detected, "system_ram_gb", None)
            max_vram = getattr(detected, "max_vram_gb", None)
            if system_ram is not None and system_ram < requirements["recommended_ram_gb"]:
                print(f"Warning: detected RAM ({system_ram} GB) is below the profile recommendation ({requirements['recommended_ram_gb']} GB).")
            if max_vram is not None and max_vram < requirements["recommended_vram_gb"]:
                print(f"Warning: detected VRAM ({max_vram} GB) is below the profile recommendation ({requirements['recommended_vram_gb']} GB).")
            gpu_profile = _effective_gpu(args, detected)
            settings = LLAMACPP_GPU_PROFILES.get(gpu_profile, {})
            if settings.get("context", 0) > 32768 and (max_vram or 0) <= 12:
                print("Warning: context above 32768 on a 12 GB-or-smaller GPU is experimental.")
            if profile.get("role") == "heavy_backend":
                print("Warning: large MoE/heavy profiles are experimental; avoid frequent dual-model switching.")
        if args.dry_run:
            report = prepare_server_start(
                target, _effective_gpu(args), model_path=args.model_path, executable=args.llama_server,
                host=args.host, port=args.port,
            )
            _print_dry_run(report, args.json_output)
            return 0
        report = start_server(
            target,
            _effective_gpu(args),
            model_path=args.model_path,
            executable=args.llama_server,
            host=args.host,
            port=args.port,
            wait_timeout=args.wait_timeout,
        )
        _print_runtime_report(report, args.json_output)
        return 0 if report["state"] in {"running", "ready"} else 1
    raise ValueError("Model action must be one of: install, register, start, restart, stop, status, logs, list, remove.")


def _invoked_as_legacy_command() -> bool:
    return Path(sys.argv[0]).stem == "local-code"


def _decision_service(args):
    from .memory import ensure_memory_files
    paths = ensure_memory_files(args.workdir, args.storage_mode)
    return DecisionService.load(paths["active_intelligence"])


def _format_decisions(service, *, pending=False):
    items = service.pending.values() if pending else service.decisions.values()
    if not items:
        return "No pending decision candidates." if pending else "No decisions."
    lines = []
    for item in sorted(items, key=lambda value: value.id):
        status = "pending" if pending else item.status.value
        lines.append(f"{item.id}  [{status}]  {item.title}")
        if item.rationale:
            lines.append(f"  {item.rationale}")
    return "\n".join(lines)


def _run_decisions(args):
    service = _decision_service(args)
    action = args.llama_action or "list"
    target = args.model_target
    if action == "list":
        print(_format_decisions(service))
    elif action == "add":
        title = args.title or target
        if not title:
            raise SystemExit("rist decisions add requires a title")
        decision = service.add(title=title, rationale=args.rationale, alternatives=args.alternatives, consequences=args.consequences,
                               affected_components=args.component, source_references=args.source)
        print(f"Added {decision.id} [proposed] {decision.title}")
    elif action == "accept":
        if not target:
            raise SystemExit("rist decisions accept requires a decision ID")
        decision = service.accept(target)
        print(f"Accepted {decision.id}: {decision.title}")
    elif action == "supersede":
        if not target or not args.superseding_id:
            raise SystemExit("rist decisions supersede OLD_ID --with NEW_ID")
        old, new = service.supersede(target, args.superseding_id)
        print(f"{old.id} superseded by {new.id}")
    elif action == "reject":
        decision = service.reject(target, args.rationale)
        print(f"Rejected {decision.id}: {decision.title}")
    elif action == "merge":
        ids = ([target] if target else []) + (args.review_merge or [])
        candidate = service.merge_candidates(ids, title=args.title)
        print(f"Merged into pending candidate {candidate.id}: {candidate.title}")
    elif action == "review":
        if args.review_accept:
            decision = service.accept(args.review_accept)
            print(f"Accepted {decision.id}: {decision.title}")
        elif args.review_reject:
            decision = service.reject(args.review_reject, args.rationale)
            print(f"Rejected {decision.id}: {decision.title}")
        elif args.review_edit:
            candidate = service.edit_candidate(args.review_edit, title=args.title, rationale=args.rationale or None,
                                               alternatives=args.alternatives or None, consequences=args.consequences or None,
                                               affected_components=args.component or None, source_references=args.source or None)
            print(f"Updated pending candidate {candidate.id}: {candidate.title}")
        elif args.review_merge:
            candidate = service.merge_candidates(args.review_merge, title=args.title)
            print(f"Merged into pending candidate {candidate.id}: {candidate.title}")
        else:
            print(_format_decisions(service, pending=True))
    return 0



def _effective_gpu(args, hardware=None):
    if args.gpu:
        return args.gpu
    config = load_runtime_config()
    configured = (config.get("llamacpp") or {}).get("gpu_profile")
    if configured:
        return configured
    return recommend_gpu_profile(hardware or detect_hardware())



def _prompt_choice(prompt, choices, default):
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if not answer and default:
            return default
        if answer in choices:
            return answer
        print("Please choose one of: " + ", ".join(choices))


def _prompt_yes_no(prompt, default=True):
    label = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{label}] ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _format_hardware(hardware):
    lines = [f"CPU: {hardware.cpu_count or 'unknown'} cores", f"RAM: {hardware.system_ram_gb or 'unknown'} GB"]
    if hardware.gpus:
        for gpu in hardware.gpus:
            lines.append(f"GPU: {gpu.name}")
            lines.append(f"VRAM: {gpu.vram_total_gb or 'unknown'} GB")
    else:
        lines.extend(["GPU: not detected", "VRAM: none detected"])
    return lines


def _concise_doctor(report):
    ok = report.get("provider_available") and report.get("models_endpoint") and report.get("chat_completions")
    if ok:
        return "\n".join(["✓ Runtime reachable", "✓ Model loaded", "✓ Chat endpoint healthy"])
    return format_doctor(report)

def _runtime_profile(args):
    config = load_runtime_config()
    return args.model_target or (config.get("llamacpp") or {}).get("profile") or "qwen2.5-coder-7b"


def _managed_model_ready(profile_id):
    try:
        wanted = get_llamacpp_profile(profile_id)["id"]
    except ValueError:
        return False
    return any(model["id"] == wanted and model.get("exists") for model in list_managed_models())


def _try_autostart_managed_llamacpp(args, config):
    llama_cfg = config.get("llamacpp") or {}
    profile = llama_cfg.get("profile") or "qwen2.5-coder-7b"
    if config.get("default_runtime") != "llamacpp" or not _managed_model_ready(profile):
        return None
    executable = find_llama_server(args.llama_server or llama_cfg.get("llama_server"))
    if not executable:
        return None
    gpu = args.gpu or llama_cfg.get("gpu_profile") or recommend_gpu_profile(detect_hardware())
    host = args.host or "127.0.0.1"
    port = args.port or 8080
    try:
        return start_server(profile, gpu, executable=executable, host=host, port=port, wait_timeout=args.wait_timeout)
    except Exception as exc:  # noqa: BLE001 - surface actionable startup failure to caller
        args._auto_start_error = {
            "error": str(exc),
            "status": server_status(),
            "log_path": str(managed_log_path()),
            "profile": profile,
        }
        return None


def _select_auto_provider(args):
    config = load_runtime_config()
    managed = server_status()
    if managed.get("managed") and managed.get("state") in {"running", "ready"} and (managed.get("health") or {}).get("ready"):
        return "llamacpp", managed.get("base_url") or (config.get("llamacpp") or {}).get("base_url")
    started = _try_autostart_managed_llamacpp(args, config)
    if started and (started.get("health") or {}).get("ready"):
        return "llamacpp", started.get("base_url") or (config.get("llamacpp") or {}).get("base_url")
    llama_base = args.base_url or (config.get("llamacpp") or {}).get("base_url") or DEFAULT_LLAMACPP_BASE_URL
    llama = build_provider("llamacpp", base_url=llama_base, api_key=args.api_key)
    if not hasattr(llama, "available") or llama.available():
        return "llamacpp", llama_base
    ollama_base = args.ollama
    ollama = build_provider("ollama", base_url=ollama_base)
    if hasattr(ollama, "available") and ollama.available():
        return "ollama", ollama_base
    return "auto", None

def _run_setup(args):
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.yes
    if interactive:
        return _run_setup_interactive(args)
    return _run_setup_noninteractive(args)


def _base_setup(args, *, profile_id=None, gpu_profile=None, llama_server=None):
    hardware = detect_hardware()
    gpu_profile = gpu_profile or args.gpu or recommend_gpu_profile(hardware)
    recs = recommend_model_profiles(hardware)
    recommended_profile = recs[0]["id"].replace("-llamacpp", "") if recs else "qwen2.5-coder-7b"
    profile_id = profile_id or args.profile or recommended_profile
    settings = LLAMACPP_GPU_PROFILES[gpu_profile]
    llama_server = llama_server if llama_server is not None else find_llama_server(args.llama_server)
    config = load_runtime_config()
    config["provider"] = "auto"
    config["default_runtime"] = "llamacpp"
    config["model"] = "local"
    config["llamacpp"].update({
        "profile": profile_id, "gpu_profile": gpu_profile, "context": settings["context"],
        "batch": settings["batch"], "ubatch": settings["ubatch"],
        "threads": min(hardware.cpu_count or settings["threads"], settings["threads"]),
        "base_url": f"http://{'127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host}:{args.port}/v1",
    })
    if llama_server:
        config["llamacpp"]["llama_server"] = llama_server
    return hardware, recs, recommended_profile, settings, llama_server, config


def _run_setup_noninteractive(args):
    hardware, recs, recommended_profile, settings, llama_server, config = _base_setup(args)
    profile_id = config["llamacpp"]["profile"]
    registered = None
    if args.url and not args.sha256:
        raise ValueError("Installing a GGUF from URL requires --sha256.")
    if args.url:
        registered = install_model(profile_id, args.url, filename=args.filename, sha256=args.sha256, force=args.force)
    elif args.model_path:
        registered = register_model(profile_id, args.model_path)
    path = save_runtime_config(config)
    print("Rist setup")
    for line in _format_hardware(hardware):
        print(line)
    print(f"Recommended llama.cpp profile: {profile_id}")
    print(f"Recommended GPU profile: {config['llamacpp']['gpu_profile']}")
    print("llama-server:", llama_server or "not found; install llama.cpp, pass --llama-server, or set LLAMA_SERVER")
    if registered:
        print(f"Registered model: {registered['profile']} -> {registered['path']}")
    print(f"Saved config: {path}")
    started = None
    if args.start:
        if not _managed_model_ready(profile_id):
            print("Start skipped: no registered GGUF for the selected profile. Pass --model-path or run `rist model register`.")
        elif not llama_server:
            print("Start skipped: llama-server was not found. Pass --llama-server or set LLAMA_SERVER.")
        else:
            started = start_server(profile_id, config['llamacpp']['gpu_profile'], executable=llama_server, host=args.host, port=args.port, wait_timeout=args.wait_timeout)
            _print_runtime_report(started, args.json_output)
            provider = build_provider("llamacpp", base_url=started.get("base_url") or config["llamacpp"]["base_url"])
            report = doctor_report(provider, "local", "local", settings["context"])
            print(to_json(report) if args.json_output else _concise_doctor(report))
            if args.benchmark:
                report = benchmark_model(provider, "local", runs=args.benchmark_runs, num_ctx=settings["context"], long_context=args.long_context)
                print(to_json(report) if args.json_output else format_benchmark(report))
    print("Setup complete.")
    print("Run:\n\n    rist\n\nto begin.")
    return 0


def _run_setup_interactive(args):
    print("Welcome to Rist.\n")
    print("We'll configure your local AI runtime.\n")
    hardware, recs, recommended_profile, settings, llama_server, config = _base_setup(args)
    print("Detected hardware:")
    for line in _format_hardware(hardware):
        print(f"  {line}")
    print(f"\nDetected profile: {config['llamacpp']['gpu_profile']}")
    print("\nRecommended model:")
    options = []
    for i, profile in enumerate(recs[:3] or [{"id": recommended_profile+'-llamacpp', "name": recommended_profile}], 1):
        pid = profile["id"].replace("-llamacpp", "")
        options.append(pid)
        print(f"  {'✓' if pid == recommended_profile else '○'} {i}. {profile['name']}")
    choice = _prompt_choice("Choose model profile", {str(i) for i in range(1, len(options)+1)}, "1")
    profile_id = options[int(choice)-1]
    config["llamacpp"]["profile"] = profile_id
    llama_server = find_llama_server(args.llama_server)
    print("\nLocate llama-server.")
    if llama_server:
        print(f"✓ Found llama-server\n{llama_server}")
        config["llamacpp"]["llama_server"] = llama_server
    else:
        print("llama-server was not found.")
        print("- install upstream")
        print("- use LLAMA_SERVER")
        print("- pass --llama-server")
    print("\nModel setup:")
    print("1. Register existing GGUF")
    print("2. Install GGUF from explicit URL + SHA256")
    print("3. Skip for now")
    action = _prompt_choice("Choose", {"1", "2", "3"}, "1")
    registered = None
    if action == "1":
        path = input("Path to GGUF: ").strip()
        registered = register_model(profile_id, path)
        print(f"Registered model: {registered['path']}")
    elif action == "2":
        url = input("GGUF URL: ").strip()
        sha = input("SHA256: ").strip()
        if not sha:
            raise ValueError("Installing a GGUF from URL requires SHA256.")
        registered = install_model(profile_id, url, filename=args.filename, sha256=sha, force=args.force)
        print(f"Installed model: {registered['path']}")
    path = save_runtime_config(config)
    print(f"Saved config: {path}")
    started = None
    if llama_server and registered and _prompt_yes_no("Start managed runtime now?", True):
        try:
            started = start_server(profile_id, config['llamacpp']['gpu_profile'], executable=llama_server, host=args.host, port=args.port, wait_timeout=args.wait_timeout)
            provider = build_provider("llamacpp", base_url=started.get("base_url") or config["llamacpp"]["base_url"])
            report = doctor_report(provider, "local", "local", settings["context"])
            print(_concise_doctor(report))
            if _prompt_yes_no("Run a quick benchmark?", False):
                bench = benchmark_model(provider, "local", runs=args.benchmark_runs, num_ctx=settings["context"], long_context=args.long_context)
                print(format_benchmark(bench))
        except Exception as exc:  # noqa: BLE001
            print(f"Startup failed: {exc}")
            provider = build_provider("llamacpp", base_url=config["llamacpp"]["base_url"])
            print(format_doctor(doctor_report(provider, "local", "local", settings["context"])))
    print("\nSetup complete.\n")
    print("Run:\n\n    rist\n\nto begin.")
    return 0

def _print_setup_guidance():
    class _Args:
        profile = None; gpu = None; model_path = None; llama_server = None; host = "127.0.0.1"; port = 8080
        start = False; benchmark = False; benchmark_runs = 3; long_context = False; wait_timeout = 120; json_output = False
        url = None; sha256 = None; filename = None; force = False; yes = True
    return _run_setup(_Args())

def _run_llama_tune(args):
    config = load_runtime_config()
    gpu = _effective_gpu(args)
    settings = dict(LLAMACPP_GPU_PROFILES[gpu])
    config["provider"] = "auto"
    config["default_runtime"] = "llamacpp"
    config["llamacpp"].update({"gpu_profile": gpu, "context": settings["context"], "batch": settings["batch"], "ubatch": settings["ubatch"], "threads": settings["threads"]})
    path = save_runtime_config(config)
    print("llama.cpp conservative configuration")
    print("No measured tuning was run; saved conservative profile defaults only.")
    print(f"Profile: {gpu}; context={settings['context']} batch={settings['batch']} ubatch={settings['ubatch']} threads={settings['threads']}")
    print(f"Config: {path}")
    print("Run `rist benchmark` with the server running to measure TTFT/tokens/sec before calling these settings tuned.")
    return 0

def main():
    if _invoked_as_legacy_command():
        sys.stderr.write("local-code is deprecated and will be removed in a future release. Use `rist` instead.\n")
    rist_home()
    args = parse_args()
    if args.command == "setup":
        return _run_setup(args)
    if args.command == "decisions":
        return _run_decisions(args)
    if args.command == "chat":
        args.command = None
        args.mode = "chat"
    if args.command == "index":
        if args.status and (args.force or args.preview or args.dry_run):
            sys.stderr.write("--status cannot be combined with --force, --preview, or --dry-run.\n")
            return 2
        report = index_repository(
            args.workdir,
            force=args.force,
            preview=args.preview or args.dry_run,
            status_only=args.status,
        )
        print(to_json(report) if args.json_output else format_index_report(report))
        return 0
    if args.command == "model" or (args.command == "llama" and args.llama_action in {"install", "logs", "tune"}):
        try:
            return _handle_runtime_command(args)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        except (FileNotFoundError, FileExistsError, PermissionError, RuntimeError, KeyError, urllib.error.URLError) as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
    if args.command == "llama" and args.llama_action == "command":
        try:
            report = generate_llama_server_command(args.profile, _effective_gpu(args), args.model_path)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        print(to_json(report) if args.json_output else format_llama_server_command(report))
        return 0
    if args.command == "llama":
        args.provider = "llamacpp"
        args.command = args.llama_action or "doctor"
    runtime_config = load_runtime_config()
    requested_provider = args.provider or runtime_config.get("provider") or DEFAULT_PROVIDER
    if requested_provider == "auto":
        resolved_provider, resolved_base = _select_auto_provider(args)
        if resolved_provider == "auto":
            auto_error = getattr(args, "_auto_start_error", None)
            if auto_error:
                status = auto_error.get("status") or {}
                sys.stderr.write("Managed llama.cpp auto-start failed.\n")
                sys.stderr.write(f"Status: {status.get('state', 'unknown')}\n")
                sys.stderr.write(f"Log path: {auto_error.get('log_path')}\n")
                sys.stderr.write(f"Error: {auto_error.get('error')}\n")
                sys.stderr.write("Next commands:\n")
                sys.stderr.write("  rist llama logs --tail 50\n")
                sys.stderr.write("  rist model status\n")
                sys.stderr.write(f"  rist model start {auto_error.get('profile', '')}\n")
            else:
                sys.stderr.write("No local model runtime is ready. Run `rist setup --model-path /path/to/model.gguf --start`, or start Ollama.\n")
            return 1
        args.provider = resolved_provider
        if resolved_provider == "llamacpp" and not args.base_url:
            args.base_url = resolved_base
        elif resolved_provider == "ollama" and resolved_base:
            args.ollama = resolved_base
    else:
        args.provider = requested_provider
    explicit_model = args.model or args.frontend_model or args.frontend_model_alias or args.backend_model or args.backend_model_alias
    if args.provider not in {"ollama", "llamacpp"} and not explicit_model:
        sys.stderr.write(
            f"--provider {args.provider} needs a model. Pass --model "
            "(e.g. --provider openrouter --model qwen/qwen-2.5-coder-32b-instruct).\n"
        )
        return 2

    default_model = _default_model_for(args.provider)
    frontend_model = args.frontend_model_alias or args.frontend_model or args.model or default_model or DEFAULT_FRONTEND_MODEL
    backend_model = args.backend_model_alias or args.backend_model or args.model or default_model or DEFAULT_BACKEND_MODEL
    command_permission = "allow" if args.auto_approve else "ask"

    ollama_base = (args.base_url or args.ollama) if args.provider == "ollama" else args.base_url
    provider = build_provider(args.provider, base_url=ollama_base, api_key=args.api_key)

    if args.command == "doctor":
        report = doctor_report(provider, frontend_model, backend_model, DEFAULT_NUM_CTX)
        print(to_json(report) if args.json_output else format_doctor(report))
        return 0 if report.get("readiness", "ready") == "ready" and report["provider_available"] else 1
    if args.command in {"benchmark", "bench"}:
        if args.benchmark_runs < 1:
            sys.stderr.write("--benchmark-runs must be at least 1.\n")
            return 2
        if not provider.available():
            message = provider.failure_message() if hasattr(provider, "failure_message") else f"Provider unavailable: {provider.describe()}"
            sys.stderr.write(message + "\n")
            return 1
        try:
            report = benchmark_model(provider, backend_model, args.benchmark_runs, DEFAULT_NUM_CTX, long_context=args.long_context)
        except urllib.error.URLError as exc:
            message = provider.failure_message() if hasattr(provider, "failure_message") else f"Connection error: {exc}"
            sys.stderr.write(message + "\n")
            return 1
        print(to_json(report) if args.json_output else format_benchmark(report))
        return 0

    preferred_frontend_model, preferred_backend_model = frontend_model, backend_model
    hardware = detect_hardware(ollama_base if provider.is_local else None)
    frontend_model, backend_model, routing_decision = resolve_model_routing(
        args.model_routing, provider.is_local, hardware, frontend_model, backend_model, DEFAULT_NUM_CTX,
        provider_is_heavy=getattr(provider, "is_heavy_backend", False),
    )

    agent = LocalPartner(
        frontend_model=frontend_model,
        backend_model=backend_model,
        provider=provider,
        workdir=args.workdir,
        command_permission=command_permission,
        edit_permission="ask",
        max_steps=args.max_steps,
        verbosity=args.verbosity,
        show_raw_actions=args.show_raw_actions,
        mode=args.mode,
        tool_calling=args.tool_calling,
        model_routing=args.model_routing,
        routing_decision=routing_decision,
        context_limit=DEFAULT_NUM_CTX,
        preferred_frontend_model=preferred_frontend_model,
        preferred_backend_model=preferred_backend_model,
        storage_mode=args.storage_mode,
    )
    if not args.no_preflight:
        run_preflight(agent)
    if args.prompt:
        try:
            reply = agent.run_turn(args.prompt)
        except urllib.error.URLError as exc:
            sys.stderr.write(_connection_error_message(agent, exc) + "\n")
            return 1
        agent.ui.print_markdown(reply)
        card = render_final_card(agent)
        if card:
            print(card)
        return 0
    if not args.no_tui and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from .tui import run_tui
        except Exception:  # noqa: BLE001  (textual not installed)
            run_tui = None
        if run_tui is not None:
            return run_tui(agent)
    return interactive_loop(agent)

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.error

from .agent import LocalPartner
from .config import (
    DEFAULT_BACKEND_MODEL,
    DEFAULT_FRONTEND_MODEL,
    DEFAULT_MODE,
    DEFAULT_TOOL_CALLING,
    DEFAULT_OLLAMA,
    DEFAULT_VERBOSITY,
    HELP_TEXT,
    MAX_TOOL_STEPS,
)
from .memory import clear_chat_history, save_chat_history
from .model_profiles import (
    RECOMMENDED_CEILING,
    RECOMMENDED_STANDARD,
    advisory_lines,
    classify_model,
)
from .models import model_is_available, server_available
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
    parser = argparse.ArgumentParser(description="Local coding partner CLI backed by Ollama.")
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
    parser.add_argument("--no-preflight", dest="no_preflight", action="store_true", help="Skip the startup model/Ollama check")
    return parser.parse_args()


def run_preflight(agent, verbose=False):
    """Check Ollama + the configured models and advise on the 12 GB standard.

    Always non-fatal. Stays silent when everything is pulled and meets the
    recommended standard, unless `verbose` (the /models command) is set.
    """
    ui = agent.ui
    if not server_available(agent.ollama):
        print(
            ui.box(
                "Ollama not reachable",
                [
                    f"Could not reach Ollama at {agent.ollama}.",
                    "Start it with `ollama serve`, or point at it with --ollama.",
                ],
                color=UI.YELLOW,
            ),
            file=sys.stderr,
        )
        return

    models = list(dict.fromkeys([agent.frontend_model, agent.backend_model]))
    profiles = {m: classify_model(m) for m in models}
    missing = [m for m in models if model_is_available(agent.ollama, m) is False]
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
        ("Frontend", agent.frontend_model),
        ("Backend", agent.backend_model),
        ("Trace", "on" if agent.verbosity == "debug" else "off"),
        ("Raw JSON", "on" if agent.show_raw_actions else "off"),
        ("Tool calling", agent.tool_calling),
        ("Pending plan", "yes" if agent.pending_plan else "no"),
    ]
    return agent.ui.kv_box("local-code status", rows, color=UI.CYAN)


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
        "== Local Code ==",
        "== Action ==",
        "== Result ==",
        "== Done ==",
        "You > ",
        "local-code ",
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
            save_chat_history(agent.workdir, agent.history)
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
            clear_chat_history(agent.workdir)
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
            print(render_header(agent))
            continue
        if prompt.startswith("/frontend "):
            agent.frontend_model = prompt.split(None, 1)[1].strip()
            print(render_header(agent))
            continue
        if prompt.startswith("/backend "):
            agent.backend_model = prompt.split(None, 1)[1].strip()
            print(render_header(agent))
            continue
        if prompt.startswith("/planner "):
            agent.frontend_model = prompt.split(None, 1)[1].strip()
            print(render_header(agent))
            continue
        if prompt.startswith("/coder "):
            agent.backend_model = prompt.split(None, 1)[1].strip()
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
                print(f"Connection error (is Ollama running?): {exc}")
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
                print(f"Connection error (is Ollama running?): {exc}")
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
                print(f"Connection error (is Ollama running?): {exc}")
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
                print(f"Connection error (is Ollama running?): {exc}")
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
                print(f"Connection error (is Ollama running?): {exc}")
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
                print(f"Connection error (is Ollama running?): {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"Error: {exc}")
                continue
            if reply is None:
                print("No pending plan to apply.")
            else:
                print_reply(agent, reply)
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
            print(f"Connection error (is Ollama running?): {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            continue

        print_reply(agent, reply)


def main():
    args = parse_args()
    frontend_model = args.frontend_model_alias or args.frontend_model or args.model or DEFAULT_FRONTEND_MODEL
    backend_model = args.backend_model_alias or args.backend_model or args.model or DEFAULT_BACKEND_MODEL
    command_permission = "allow" if args.auto_approve else "ask"
    agent = LocalPartner(
        frontend_model=frontend_model,
        backend_model=backend_model,
        ollama=args.ollama,
        workdir=args.workdir,
        command_permission=command_permission,
        edit_permission="ask",
        max_steps=args.max_steps,
        verbosity=args.verbosity,
        show_raw_actions=args.show_raw_actions,
        mode=args.mode,
        tool_calling=args.tool_calling,
    )
    if not args.no_preflight:
        run_preflight(agent)
    if args.prompt:
        reply = agent.run_turn(args.prompt)
        agent.ui.print_markdown(reply)
        card = render_final_card(agent)
        if card:
            print(card)
        return 0
    return interactive_loop(agent)

"""Rist TUI v2 Textual application."""

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, Label, Static

from local_code.hardware import detect_hardware
from local_code.routing import resolve_model_routing

from .commands import build_commands
from .diff_review import build_review_model
from .screens.command_palette import CommandPaletteScreen
from .screens.confirm import ConfirmScreen
from .screens.diff_review import DiffReviewScreen
from .widgets.activity import ActivityTimeline
from .widgets.conversation import ConversationView
from .widgets.input import InputBar
from .widgets.runtime import RuntimePanel
from .widgets.status import StatusBar, render_status_text

HELP = """\
**Commands**

- `/help` — this help    · `/status` — settings    · `/models` — provider & tiers
- `/mode chat|hybrid|agent` · `/model NAME` · `/frontend NAME` · `/backend NAME`
- `/ask TEXT` — inspect (no edits) · `/plan TEXT` — propose · `/apply` — apply a proposal
- `/agent TEXT` — run a task in agent mode · `/decisions list|add|accept|supersede|review`
- `/copy` — copy last response · `/clear` — clear history · `/quit` — exit

**Keys**: Enter send · Ctrl+X cancel · Ctrl+L clear · Ctrl+C quit · in a proposal, type `yes` to apply.
"""


class LocalCodeApp(App):
    """Purpose-built Rist cockpit preserving the existing agent behavior."""

    CSS_PATH = "styles/rist.tcss"
    HELP_TEXT = HELP

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+x", "cancel_turn", "Cancel", show=True),
        Binding("ctrl+k", "open_palette", "Commands", show=True, priority=True),
        Binding("ctrl+shift+p", "open_palette", "Commands", show=False, priority=True),
    ]

    def __init__(self, partner):
        super().__init__()
        self.partner = partner
        self._live_text = ""
        self._busy = False
        self.commands = build_commands()

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Label("Rist", id="brand")
            yield StatusBar(id="status")
        with Horizontal(id="workspace"):
            yield ConversationView(id="conversation")
            yield ActivityTimeline(id="activity")
        with Vertical(id="confidence-strip"):
            yield RuntimePanel(id="runtime-panel")
        yield InputBar()
        yield Footer()

    @property
    def conversation(self) -> ConversationView:
        return self.query_one(ConversationView)

    @property
    def activity(self) -> ActivityTimeline:
        return self.query_one(ActivityTimeline)

    @property
    def status(self) -> StatusBar:
        return self.query_one(StatusBar)

    @property
    def runtime_panel(self) -> RuntimePanel:
        return self.query_one(RuntimePanel)

    @property
    def input(self) -> Input:
        return self.query_one("#input", Input)

    def on_mount(self) -> None:
        self.title = "Rist"
        self.conversation.write_welcome(self.partner.mode)
        self._refresh_status()
        self.input.focus()

    def _status_text(self) -> str:
        return render_status_text(self.partner, self._busy)

    def _refresh_status(self) -> None:
        self.status.refresh_from(self.partner, self._busy)
        self.runtime_panel.refresh_from(self.partner)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.input.value = ""
        if not text or self._busy:
            return
        if text.startswith("/"):
            if self._handle_command(text):
                return
        self.conversation.write_user(text)
        self._run_turn(text, kind="chat")

    def _handle_command(self, text) -> bool:
        """Handle a slash command. Returns True if fully handled (no turn)."""
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in {"/quit", "/exit"}:
            self.exit()
            return True
        if cmd == "/help":
            self.conversation.log.write(Panel(Markdown(HELP), title="help", border_style="cyan"))
            return True
        if cmd == "/clear":
            self.partner.history.clear()
            self.partner.pending_plan = None
            self.conversation.clear()
            self._refresh_status()
            return True
        if cmd == "/status":
            self.conversation.log.write(Panel(Markdown(self._status_text()), title="status", border_style="cyan"))
            return True
        if cmd == "/models":
            self._write_models()
            return True
        if cmd == "/decisions":
            try:
                result = self.partner.run_decision_command("decisions " + arg)
                self.conversation.log.write(Panel(Text(result), title="decisions", border_style="cyan"))
            except (KeyError, ValueError) as exc:
                self.conversation.log.write(Text(str(exc), style="yellow"))
            return True
        if cmd == "/context":
            self._show_context_usage()
            return True
        if cmd == "/routing" and arg in {"single", "adaptive", "dual"}:
            p = self.partner
            hardware = detect_hardware(getattr(p.provider, "base_url", None) if p.provider.is_local else None)
            front, back, decision = resolve_model_routing(
                arg, p.provider.is_local, hardware, p.preferred_frontend_model, p.preferred_backend_model, p.context_limit
            )
            p.model_routing, p.frontend_model, p.backend_model, p.routing_decision = arg, front, back, decision
            p.sync_executor()
            self.conversation.log.write(Text(f"Model routing: {decision['mode']} — {decision['reason']}", style="cyan"))
            self._refresh_status()
            return True
        if cmd == "/mode" and arg in {"chat", "hybrid", "agent"}:
            self.partner.mode = arg
            self._refresh_status()
            return True
        if cmd in {"/model", "/frontend", "/backend"} and arg:
            if cmd in {"/model", "/frontend"}:
                self.partner.frontend_model = arg
                self.partner.preferred_frontend_model = arg
            if cmd in {"/model", "/backend"}:
                self.partner.backend_model = arg
                self.partner.preferred_backend_model = arg
            self._refresh_status()
            return True
        if cmd == "/apply":
            if not self.partner.pending_plan:
                self.conversation.log.write(Text("No pending proposal to apply.", style="yellow"))
                return True
            self._open_review_screen()
            return True
        if cmd == "/ask" and arg:
            self.conversation.write_user(arg)
            self._run_turn(arg, kind="ask")
            return True
        if cmd == "/plan" and arg:
            self.conversation.write_user(arg)
            self._run_turn(arg, kind="plan")
            return True
        if cmd in {"/agent", "/code"} and arg:
            self.conversation.write_user(arg)
            self._run_turn(arg, kind="agent")
            return True
        if cmd == "/copy":
            self._copy_last_response()
            return True
        self.conversation.log.write(Text(f"Unknown or incomplete command: {text}", style="yellow"))
        return True

    def _show_context_usage(self) -> None:
        usage = self.partner.context_usage()
        body = "\n".join([
            f"Conversation: {usage.conversation:,}",
            f"Memory: {usage.memory:,}",
            f"Repo: {usage.repo:,}",
            f"Tools: {usage.tools:,}",
            f"Other: {usage.other:,}",
            f"Total: {usage.total:,} / {usage.limit:,} ({usage.percent}%)",
        ])
        self.conversation.log.write(Panel(Text(body), title="context", border_style="cyan"))

    def _write_command_event(self, title: str) -> None:
        self.activity.write_event({"kind": "command", "title": title})

    def execute_palette_command(self, command) -> None:
        self._write_command_event(command.title)
        try:
            command.callback(self)
        except Exception as exc:  # noqa: BLE001
            self.conversation.write_tool_summary(Text(str(exc), style="red"), title="command error", style="red")
        finally:
            self._refresh_status()
            self.input.focus()

    def _copy_last_response(self) -> None:
        import subprocess as _sp
        log = self.conversation.log
        history = self.partner.history
        last = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
        if not last:
            log.write(Text("No response to copy yet.", style="yellow"))
            return
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            try:
                _sp.run(cmd, input=last.encode(), check=True, timeout=5)
                log.write(Text(f"Copied ({len(last)} chars).", style="green"))
                return
            except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired):
                continue
        log.write(Text("Copy failed: install wl-copy (Wayland) or xclip/xsel (X11).", style="red"))

    def _write_models(self) -> None:
        p = self.partner
        if p.provider.is_local:
            from local_code.model_profiles import advisory_lines, RECOMMENDED_CEILING, RECOMMENDED_STANDARD

            lines = advisory_lines(p.frontend_model, p.backend_model)
            lines.append("")
            lines.append(f"Standard for 12 GB GPUs: {RECOMMENDED_STANDARD} (ceiling {RECOMMENDED_CEILING}).")
            body = "\n".join(lines)
        else:
            body = f"Provider: {p.provider.describe()}\nModels: {p.frontend_model}, {p.backend_model}\nCloud models — local VRAM tiers don't apply."
        self.conversation.log.write(Panel(Text(body), title="models", border_style="cyan"))

    # -- transcript helpers (main thread only) --------------------------
    def _write_user(self, text) -> None:
        self.conversation.write_user(text)

    def _append_live(self, chunk) -> None:
        self._live_text += chunk
        self.conversation.update_live(self._live_text)

    def _write_event(self, event) -> None:
        self.activity.write_event(event)

    def _confirm_result(self, kind, label, content) -> bool:
        # Called via call_from_thread; runs the modal and waits for dismissal.
        return bool(self.call_from_thread(self.push_screen_wait, ConfirmScreen(kind, label, content)))

    # -- the worker -----------------------------------------------------
    def _run_turn(self, text, kind) -> None:
        self._busy = True
        self._live_text = ""
        inp = self.input
        inp.disabled = True
        self.status.update("[b]working…[/]  " + self._status_text())

        partner = self.partner
        partner.on_token = lambda chunk: self.call_from_thread(self._append_live, chunk)
        partner.observer = lambda event: self.call_from_thread(self._write_event, event)
        partner.confirm_hook = lambda k, label, content: self._confirm_result(k, label, content)
        partner.stream_to_stdout = False
        self._process(text, kind)

    @work(thread=True, exclusive=True, group="turn")
    def _process(self, text, kind) -> None:
        partner = self.partner
        try:
            if kind == "ask":
                reply = partner.run_readonly_turn(text)
            elif kind == "plan":
                reply = partner.run_turn(text, planning=True)
            elif kind == "apply":
                reply = partner.apply_pending_plan() or "Nothing to apply."
            elif kind == "agent":
                previous = partner.mode
                partner.mode = "agent"
                try:
                    reply = partner.run_turn(text)
                finally:
                    partner.mode = previous
            else:
                reply = partner.run_turn(text)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._finish_turn, None, str(exc))
            return
        self.call_from_thread(self._finish_turn, reply, None)

    def _finish_turn(self, reply, error) -> None:
        # Clear the live streaming pane and commit the final reply to the log.
        live = self.conversation.query_one("#live", Static)
        live.update("")
        live.add_class("empty")
        self._live_text = ""
        log = self.conversation.log
        if error is not None:
            self.conversation.write_tool_summary(Text(error, style="red"), title="error", style="red")
        elif reply:
            self.conversation.write_assistant(reply)
        self._write_report_extras()
        log.scroll_end(animate=False)
        self.partner.on_token = None
        self.partner.observer = None
        self.partner.confirm_hook = None
        self._busy = False
        self.commands = build_commands()
        inp = self.input
        inp.disabled = False
        inp.focus()
        self._refresh_status()

    def _write_report_extras(self) -> None:
        report = getattr(self.partner, "last_report", None)
        if not report:
            return
        if report.get("needs_approval"):
            model = build_review_model(self.partner)
            if model:
                self.conversation.write_assistant("Proposal ready. Opened review screen.")
                self.activity.write_event({"kind": "milestone", "text": "Prepared code changes"})
                self.activity.write_event({"kind": "diff", "files": model.summary.files, "added": model.summary.added, "removed": model.summary.removed})
                self.call_later(self._open_review_screen)
        bits = []
        if report.get("files_changed"):
            bits.append(f"{len(report['files_changed'])} edited")
        if report.get("files_read"):
            bits.append(f"{len(report['files_read'])} read")
        if report.get("commands_run"):
            bits.append(f"{len(report['commands_run'])} cmds")
        if bits:
            self.activity.write_note(" · ".join(bits), title="REPORT", style="bright_black")

    def _open_review_screen(self) -> None:
        model = build_review_model(self.partner)
        if not model:
            self.conversation.log.write(Text("No pending proposal to review.", style="yellow"))
            return
        self.push_screen(DiffReviewScreen(model), self._review_dismissed)

    def _review_dismissed(self, action) -> None:
        if action == "apply":
            self.activity.write_event({"kind": "apply", "text": "Proposal accepted"})
            self.conversation.write_user("/apply")
            self._run_turn("", kind="apply")
        elif action == "reject":
            self.partner.pending_plan = None
            self.activity.write_event({"kind": "reject", "text": "Proposal rejected"})
            self.conversation.write_assistant("Proposal rejected.")
            self._refresh_status()
        else:
            self.input.focus()

    # -- actions --------------------------------------------------------
    def action_open_palette(self) -> None:
        if self._busy:
            return
        self.push_screen(CommandPaletteScreen(self.commands, self), self._palette_dismissed)

    def _palette_dismissed(self, command) -> None:
        if command is not None:
            self.execute_palette_command(command)

    def action_clear_log(self) -> None:
        self.conversation.clear()
        self.activity.clear()

    def action_cancel_turn(self) -> None:
        if self._busy:
            self.partner.executor.cancel_event.set()
            self.conversation.log.write(Text("Cancelling…", style="yellow"))


def run_tui(partner) -> int:
    """Launch the TUI for an already-constructed LocalPartner."""
    LocalCodeApp(partner).run()
    return 0

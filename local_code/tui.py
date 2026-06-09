"""Full-screen Textual TUI for Rist.

This is the default interactive experience. It drives the same `LocalPartner`
as the plain REPL, but renders the conversation, live token streaming, tool
activity, and edit/command approvals in a full-screen app.

The model work runs in a background thread (`@work(thread=True)`) so the UI
stays responsive; the partner pushes progress back through three bridges set up
in `_run_turn`: `on_token` (streaming text), `observer` (tool/milestone events),
and `confirm_hook` (a modal approval prompt). All cross-thread UI updates go
through `call_from_thread`.

Imported lazily by the CLI, so Textual is only required when the TUI runs.
"""

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, RichLog, Static

from .hardware import detect_hardware
from .routing import resolve_model_routing


class ConfirmScreen(ModalScreen[bool]):
    """Modal approval prompt used for edit/command confirmations."""

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, kind, label, content):
        super().__init__()
        self._kind = kind
        self._label = label
        self._content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(f"Approve {self._kind.lower()}?", id="confirm-title")
            yield Static(Text(self._label, style="bold"), id="confirm-label")
            if self._content:
                yield Static(Text(self._content, style="dim"), id="confirm-content")
            with Horizontal(id="confirm-buttons"):
                yield Button("Approve (y)", variant="success", id="yes")
                yield Button("Deny (n)", variant="error", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


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
    CSS = """
    Screen { layout: vertical; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #log { height: 1fr; padding: 0 1; }
    #live { padding: 0 1; color: $text; }
    #live.empty { display: none; }
    #input { border: tall $accent; }
    ConfirmScreen { align: center middle; }
    #confirm-box { width: 80; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
    #confirm-title { text-style: bold; color: $warning; }
    #confirm-label { margin: 1 0; }
    #confirm-buttons { height: auto; align: center middle; }
    #confirm-buttons Button { margin: 1 2; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+x", "cancel_turn", "Cancel", show=True),
    ]

    def __init__(self, partner):
        super().__init__()
        self.partner = partner
        self._live_text = ""
        self._busy = False

    # -- layout ---------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(self._status_text(), id="status")
        yield RichLog(id="log", wrap=True, markup=True, highlight=False)
        yield Static("", id="live", classes="empty")
        yield Input(placeholder="Ask, or /help …", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Rist"
        log = self.query_one("#log", RichLog)
        log.write(Panel(Markdown(f"Welcome to **Rist**. Mode: `{self.partner.mode}`. Type `/help` for commands."), border_style="cyan"))
        self.query_one("#input", Input).focus()

    # -- status bar -----------------------------------------------------
    def _status_text(self) -> str:
        p = self.partner
        models = p.frontend_model if p.frontend_model == p.backend_model else f"{p.frontend_model} → {p.backend_model}"
        return (
            f"[b]{p.provider.name}[/b]  ·  {models}  ·  mode {p.mode}  ·  "
            f"routing {p.routing_decision.get('mode', p.model_routing)}  ·  edits {p.edit_permission}/cmds {p.command_permission}"
            + ("  ·  [yellow]proposal pending — type 'yes'[/]" if p.pending_plan else "")
        )

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    # -- input handling -------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#input", Input).value = ""
        if not text or self._busy:
            return
        if text.startswith("/"):
            if self._handle_command(text):
                return
        self._write_user(text)
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
            self.query_one("#log", RichLog).write(Panel(Markdown(HELP), title="help", border_style="cyan"))
            return True
        if cmd == "/clear":
            self.partner.history.clear()
            self.partner.pending_plan = None
            self.query_one("#log", RichLog).clear()
            self._refresh_status()
            return True
        if cmd == "/status":
            self.query_one("#log", RichLog).write(Panel(Markdown(self._status_text()), title="status", border_style="cyan"))
            return True
        if cmd == "/models":
            self._write_models()
            return True
        if cmd == "/decisions":
            try:
                result = self.partner.run_decision_command("decisions " + arg)
                self.query_one("#log", RichLog).write(Panel(Text(result), title="decisions", border_style="cyan"))
            except (KeyError, ValueError) as exc:
                self.query_one("#log", RichLog).write(Text(str(exc), style="yellow"))
            return True
        if cmd == "/context":
            usage = self.partner.context_usage()
            body = "\n".join([
                f"Conversation: {usage.conversation:,}",
                f"Memory: {usage.memory:,}",
                f"Repo: {usage.repo:,}",
                f"Tools: {usage.tools:,}",
                f"Other: {usage.other:,}",
                f"Total: {usage.total:,} / {usage.limit:,} ({usage.percent}%)",
            ])
            self.query_one("#log", RichLog).write(Panel(Text(body), title="context", border_style="cyan"))
            return True
        if cmd == "/routing" and arg in {"single", "adaptive", "dual"}:
            p = self.partner
            hardware = detect_hardware(getattr(p.provider, "base_url", None) if p.provider.is_local else None)
            front, back, decision = resolve_model_routing(
                arg, p.provider.is_local, hardware, p.preferred_frontend_model, p.preferred_backend_model, p.context_limit
            )
            p.model_routing, p.frontend_model, p.backend_model, p.routing_decision = arg, front, back, decision
            p.sync_executor()
            self.query_one("#log", RichLog).write(Text(f"Model routing: {decision['mode']} — {decision['reason']}", style="cyan"))
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
                self.query_one("#log", RichLog).write(Text("No pending proposal to apply.", style="yellow"))
                return True
            self._write_user("/apply")
            self._run_turn("", kind="apply")
            return True
        if cmd == "/ask" and arg:
            self._write_user(arg)
            self._run_turn(arg, kind="ask")
            return True
        if cmd == "/plan" and arg:
            self._write_user(arg)
            self._run_turn(arg, kind="plan")
            return True
        if cmd in {"/agent", "/code"} and arg:
            self._write_user(arg)
            self._run_turn(arg, kind="agent")
            return True
        if cmd == "/copy":
            self._copy_last_response()
            return True
        self.query_one("#log", RichLog).write(Text(f"Unknown or incomplete command: {text}", style="yellow"))
        return True

    def _copy_last_response(self) -> None:
        import subprocess as _sp
        log = self.query_one("#log", RichLog)
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
            from .model_profiles import advisory_lines, RECOMMENDED_CEILING, RECOMMENDED_STANDARD

            lines = advisory_lines(p.frontend_model, p.backend_model)
            lines.append("")
            lines.append(f"Standard for 12 GB GPUs: {RECOMMENDED_STANDARD} (ceiling {RECOMMENDED_CEILING}).")
            body = "\n".join(lines)
        else:
            body = f"Provider: {p.provider.describe()}\nModels: {p.frontend_model}, {p.backend_model}\nCloud models — local VRAM tiers don't apply."
        self.query_one("#log", RichLog).write(Panel(Text(body), title="models", border_style="cyan"))

    # -- transcript helpers (main thread only) --------------------------
    def _write_user(self, text) -> None:
        self.query_one("#log", RichLog).write(Panel(Text(text), title="you", title_align="left", border_style="cyan"))

    def _append_live(self, chunk) -> None:
        self._live_text += chunk
        live = self.query_one("#live", Static)
        live.remove_class("empty")
        live.update(Text(self._live_text))
        self.query_one("#log", RichLog).scroll_end(animate=False)

    def _write_event(self, event) -> None:
        kind = event.get("kind")
        log = self.query_one("#log", RichLog)
        if kind == "tool":
            log.write(Text(f"  ⚙ {event.get('line', event.get('tool', ''))}", style="blue"))
        elif kind == "milestone":
            log.write(Text(f"  • {event.get('text', '')}", style="dim"))
        elif kind == "transcript":
            title = event.get("title", "")
            log.write(Text(f"  ✓ {title}", style="green"))
            for line in event.get("lines", []):
                log.write(Text(f"      {line}", style="dim"))
        elif kind == "trace":
            log.write(Text(f"  · {event.get('text', '')}", style="dim"))
        log.scroll_end(animate=False)

    def _confirm_result(self, kind, label, content) -> bool:
        # Called via call_from_thread; runs the modal and waits for dismissal.
        return bool(self.call_from_thread(self.push_screen_wait, ConfirmScreen(kind, label, content)))

    # -- the worker -----------------------------------------------------
    def _run_turn(self, text, kind) -> None:
        self._busy = True
        self._live_text = ""
        inp = self.query_one("#input", Input)
        inp.disabled = True
        self.query_one("#status", Static).update("[b]working…[/]  " + self._status_text())

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
        live = self.query_one("#live", Static)
        live.update("")
        live.add_class("empty")
        self._live_text = ""
        log = self.query_one("#log", RichLog)
        if error is not None:
            log.write(Panel(Text(error, style="red"), title="error", border_style="red"))
        elif reply:
            log.write(Panel(Markdown(reply), title="Rist", title_align="left", border_style="green"))
        self._write_report_extras()
        log.scroll_end(animate=False)
        self.partner.on_token = None
        self.partner.observer = None
        self.partner.confirm_hook = None
        self._busy = False
        inp = self.query_one("#input", Input)
        inp.disabled = False
        inp.focus()
        self._refresh_status()

    def _write_report_extras(self) -> None:
        report = getattr(self.partner, "last_report", None)
        if not report:
            return
        log = self.query_one("#log", RichLog)
        if report.get("needs_approval"):
            plan = [s for s in (report.get("plan") or []) if s]
            if plan:
                body = "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1))
                log.write(Panel(Text(body), title="planned changes", border_style="yellow"))
            diff = (report.get("diff_summary") or "").strip()
            if diff and ("---" in diff or "@@" in diff or "+++" in diff):
                log.write(Panel(Text(diff), title="diff", border_style="yellow"))
        bits = []
        if report.get("files_changed"):
            bits.append(f"{len(report['files_changed'])} edited")
        if report.get("files_read"):
            bits.append(f"{len(report['files_read'])} read")
        if report.get("commands_run"):
            bits.append(f"{len(report['commands_run'])} cmds")
        if bits:
            log.write(Text("  " + " · ".join(bits), style="dim"))

    # -- actions --------------------------------------------------------
    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def action_cancel_turn(self) -> None:
        if self._busy:
            self.partner.executor.cancel_event.set()
            self.query_one("#log", RichLog).write(Text("Cancelling…", style="yellow"))


def run_tui(partner) -> int:
    """Launch the TUI for an already-constructed LocalPartner."""
    LocalCodeApp(partner).run()
    return 0

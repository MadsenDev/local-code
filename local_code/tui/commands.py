"""Reusable command palette command registry for the Rist TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from local_code.config import load_runtime_config
from local_code.diagnostics import benchmark_model, doctor_report, format_benchmark, format_doctor
from local_code.hardware import detect_hardware
from local_code.llama_runtime import server_status, start_server, stop_server
from local_code.routing import resolve_model_routing


Callback = Callable[[object], None]
Enabled = Callable[[object], bool]


@dataclass(frozen=True)
class Command:
    """A single keyboard-first action exposed by the command palette."""

    id: str
    title: str
    subtitle: str
    category: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    callback: Callback = lambda app: None
    enabled: Enabled = lambda app: True

    def searchable_text(self) -> str:
        return " ".join((self.title, self.subtitle, self.category, *self.keywords)).lower()


def fuzzy_score(command: Command, query: str) -> float:
    """Score a command against a fuzzy query; zero means no match."""

    needle = query.strip().lower()
    if not needle:
        return 1.0
    haystack = command.searchable_text()
    fields = (command.title, command.subtitle, command.category, *command.keywords)
    lowered_fields = [field.lower() for field in fields]
    if any(needle == field for field in lowered_fields):
        return 100.0
    if any(needle in field for field in lowered_fields):
        return 50.0 + len(needle) / max(len(haystack), 1)
    pos = -1
    score = 0.0
    for char in needle:
        found = haystack.find(char, pos + 1)
        if found < 0:
            return 0.0
        score += 1.0 if found == pos + 1 else 0.6
        pos = found
    return score + SequenceMatcher(None, needle, haystack).ratio()


def filter_commands(commands: list[Command], query: str, app=None) -> list[Command]:
    """Return enabled commands matching query, best matches first within categories."""

    scored = []
    for index, command in enumerate(commands):
        if app is not None and not command.enabled(app):
            continue
        score = fuzzy_score(command, query)
        if score > 0:
            scored.append((-score, command.category, index, command))
    return [item[3] for item in sorted(scored)]


def build_commands() -> list[Command]:
    """Create the default command registry for the TUI."""

    commands: list[Command] = []

    def add(**kwargs):
        commands.append(Command(**kwargs))

    def write_panel(app, body, title):
        app.conversation.log.write(Panel(Text(str(body)), title=title, border_style="cyan"))

    def set_mode(mode: str):
        def callback(app):
            app.partner.mode = mode
            app._refresh_status()
        return callback

    def set_routing(mode: str):
        def callback(app):
            p = app.partner
            hardware = detect_hardware(getattr(p.provider, "base_url", None) if p.provider.is_local else None)
            front, back, decision = resolve_model_routing(
                mode, p.provider.is_local, hardware, p.preferred_frontend_model, p.preferred_backend_model, p.context_limit,
                provider_is_heavy=getattr(p.provider, "is_heavy_backend", False),
            )
            p.model_routing, p.frontend_model, p.backend_model, p.routing_decision = mode, front, back, decision
            p.sync_executor()
            app.conversation.log.write(Text(f"Model routing: {decision['mode']} — {decision['reason']}", style="cyan"))
            app._refresh_status()
        return callback

    def set_permission(scope: str, mode: str):
        def callback(app):
            if scope == "edit":
                app.partner.edit_permission = mode
            else:
                app.partner.command_permission = mode
            app.partner.sync_executor()
            app._refresh_status()
        return callback

    def start_runtime(app):
        cfg = load_runtime_config().get("llamacpp", {})
        report = start_server(cfg.get("profile", "qwen2.5-coder-7b"), cfg.get("gpu_profile", "cpu"), port=int(cfg.get("port", 8080)))
        write_panel(app, f"Managed runtime started: {report.get('base_url', '')}", "runtime")

    def stop_runtime(app):
        write_panel(app, stop_server().get("message", "Managed runtime stopped."), "runtime")

    def restart_runtime(app):
        stop_server()
        start_runtime(app)

    def runtime_status(app):
        write_panel(app, server_status(), "runtime status")

    def run_doctor(app):
        write_panel(app, format_doctor(doctor_report(app.partner.provider, app.partner.frontend_model, app.partner.backend_model, app.partner.context_limit)), "doctor")

    def run_benchmark(app):
        write_panel(app, format_benchmark(benchmark_model(app.partner.provider, app.partner.backend_model, num_ctx=app.partner.context_limit)), "benchmark")

    add(id="conversation.new", title="New conversation", subtitle="Start a fresh conversation", category="Conversation", keywords=("reset",), callback=lambda app: (app.partner.history.clear(), setattr(app.partner, "pending_plan", None), app.conversation.clear(), app.conversation.write_welcome(app.partner.mode), app._refresh_status()))
    add(id="conversation.clear", title="Clear conversation", subtitle="Clear the visible transcript", category="Conversation", keywords=("history",), callback=lambda app: (app.partner.history.clear(), setattr(app.partner, "pending_plan", None), app.conversation.clear(), app._refresh_status()))
    add(id="conversation.copy_last", title="Copy last response", subtitle="Copy the latest assistant response", category="Conversation", keywords=("clipboard",), callback=lambda app: app._copy_last_response())
    add(id="conversation.help", title="Show help", subtitle="Show available slash commands and keys", category="Conversation", keywords=("docs",), callback=lambda app: app.conversation.log.write(Panel(Markdown(app.HELP_TEXT), title="help", border_style="cyan")))

    for title, cb, kws in [("Start managed runtime", start_runtime, ("llama",)), ("Stop managed runtime", stop_runtime, ("llama",)), ("Restart managed runtime", restart_runtime, ("llama",)), ("Runtime status", runtime_status, ("health",)), ("Run doctor", run_doctor, ("diagnostics",)), ("Run benchmark", run_benchmark, ("bench", "performance"))]:
        add(id="runtime." + title.lower().replace(" ", "_"), title=title, subtitle="Manage or inspect the local runtime", category="Runtime", keywords=kws, callback=cb)

    add(id="models.show", title="Show current models", subtitle="Display provider and model details", category="Models", keywords=("model",), callback=lambda app: app._write_models())
    for mode in ("chat", "hybrid", "agent"):
        add(id=f"mode.{mode}", title=f"Switch mode → {mode}", subtitle="Change conversation operating mode", category="Models", keywords=(mode,), callback=set_mode(mode))
    for mode in ("single", "adaptive", "dual"):
        add(id=f"routing.{mode}", title=f"Routing → {mode}", subtitle="Change model routing strategy", category="Routing", keywords=(mode,), callback=set_routing(mode))
    for scope in ("edit", "command"):
        for mode in ("ask", "allow", "deny"):
            add(id=f"permission.{scope}.{mode}", title=f"{scope.title()} permission → {mode.title()}", subtitle=f"Set {scope} permission to {mode}", category="Permissions", keywords=(scope, mode, "permission"), callback=set_permission(scope, mode))
    add(id="context.usage", title="Show context usage", subtitle="Display token budget by source", category="Context", keywords=("tokens",), callback=lambda app: app._show_context_usage())
    add(id="session.quit", title="Quit", subtitle="Exit Rist", category="Session", keywords=("exit",), callback=lambda app: app.exit())
    return commands

"""Reusable command palette command registry for the Rist TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from local_code.hardware import detect_hardware
from local_code.routing import resolve_model_routing
from local_code.tui.repository import RepositoryBadge
from local_code.tui.workspace_memory import DecisionFilter, DecisionStatus, DecisionType


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

    def schedule_runtime(task: str):
        return lambda app: app.schedule_runtime_task(task)

    def has_pending(app):
        return bool(getattr(app.partner, "pending_plan", None))

    def apply_pending(app):
        app._open_review_screen()

    def reject_pending(app):
        app._review_dismissed("reject")

    add(id="conversation.new", title="New conversation", subtitle="Start a fresh conversation", category="Conversation", keywords=("reset",), callback=lambda app: (app.partner.history.clear(), setattr(app.partner, "pending_plan", None), setattr(app.partner, "pending_discovery", None), app.conversation.clear(), app.conversation.write_welcome(app.partner.mode), app._refresh_status()))
    add(id="conversation.clear", title="Clear conversation", subtitle="Clear the visible transcript", category="Conversation", keywords=("history",), callback=lambda app: (app.partner.history.clear(), setattr(app.partner, "pending_plan", None), setattr(app.partner, "pending_discovery", None), app.conversation.clear(), app._refresh_status()))
    add(id="conversation.copy_last", title="Copy last response", subtitle="Copy the latest assistant response", category="Conversation", keywords=("clipboard",), callback=lambda app: app._copy_last_response())
    add(id="conversation.help", title="Show help", subtitle="Show available slash commands and keys", category="Conversation", keywords=("docs",), callback=lambda app: app.conversation.log.write(Panel(Markdown(app.HELP_TEXT), title="help", border_style="cyan")))

    for title, task, kws in [("Start managed runtime", "start", ("llama",)), ("Stop managed runtime", "stop", ("llama",)), ("Restart managed runtime", "restart", ("llama",)), ("Runtime status", "status", ("health",)), ("Run doctor", "doctor", ("diagnostics",)), ("Run benchmark", "benchmark", ("bench", "performance"))]:
        add(
            id="runtime." + title.lower().replace(" ", "_"),
            title=title,
            subtitle="Manage or inspect the local runtime",
            category="Runtime",
            keywords=kws,
            callback=schedule_runtime(task),
            enabled=lambda app: not getattr(app, "_runtime_task_busy", False),
        )

    add(id="models.show", title="Show current models", subtitle="Display provider and model details", category="Models", keywords=("model",), callback=lambda app: app._write_models())
    for mode in ("chat", "hybrid", "agent"):
        add(id=f"mode.{mode}", title=f"Switch mode → {mode}", subtitle="Change conversation operating mode", category="Models", keywords=(mode,), callback=set_mode(mode))
    for mode in ("single", "adaptive", "dual"):
        add(id=f"routing.{mode}", title=f"Routing → {mode}", subtitle="Change model routing strategy", category="Routing", keywords=(mode,), callback=set_routing(mode))
    for scope in ("edit", "command"):
        for mode in ("ask", "allow", "deny"):
            add(id=f"permission.{scope}.{mode}", title=f"{scope.title()} permission → {mode.title()}", subtitle=f"Set {scope} permission to {mode}", category="Permissions", keywords=(scope, mode, "permission"), callback=set_permission(scope, mode))
    add(id="memory.open", title="Open Decision Browser", subtitle="Browse structured workspace memory", category="Workspace Memory", keywords=("decision", "memory", "why"), callback=lambda app: app._open_decision_browser())
    add(id="memory.pending", title="Show pending decisions", subtitle="Open Decision Browser filtered to pending records", category="Workspace Memory", keywords=("pending", "questions"), callback=lambda app: app._open_decision_browser(DecisionFilter(status=DecisionStatus.PENDING)))
    add(id="memory.rejected", title="Show rejected proposals", subtitle="Open Decision Browser filtered to rejections", category="Workspace Memory", keywords=("reject", "proposal"), callback=lambda app: app._open_decision_browser(DecisionFilter(type=DecisionType.REJECTED)))
    add(id="memory.assumptions", title="Show assumptions", subtitle="Open Decision Browser filtered to assumptions", category="Workspace Memory", keywords=("assumption",), callback=lambda app: app._open_decision_browser(DecisionFilter(type=DecisionType.ASSUMPTION)))
    add(id="repository.open", title="Open Repository Explorer", subtitle="Browse project structure and AI session footprint", category="Repository", keywords=("files", "tree", "explorer"), callback=lambda app: app._open_repository_explorer())
    add(id="repository.search", title="Focus search", subtitle="Open Repository Explorer with search focused", category="Repository", keywords=("find", "files"), callback=lambda app: app._focus_repository_search())
    add(id="repository.changed", title="Show changed files", subtitle="Filter Repository Explorer to AI-modified files", category="Repository", keywords=("edited", "modified"), callback=lambda app: app._open_repository_badge_filter(RepositoryBadge.EDITED))
    add(id="repository.read", title="Show read files", subtitle="Filter Repository Explorer to AI-inspected files", category="Repository", keywords=("inspected", "read"), callback=lambda app: app._open_repository_badge_filter(RepositoryBadge.READ))
    add(id="repository.proposed", title="Show pending proposal files", subtitle="Filter Repository Explorer to pending proposal files", category="Repository", keywords=("pending", "proposal"), callback=lambda app: app._open_repository_badge_filter(RepositoryBadge.PROPOSED), enabled=has_pending)
    add(id="proposal.review", title="Review pending proposal", subtitle="Open the dedicated diff review screen", category="Proposal", keywords=("diff", "review"), callback=lambda app: app._open_review_screen(), enabled=has_pending)
    add(id="proposal.apply", title="Apply pending proposal", subtitle="Review and apply all pending changes", category="Proposal", keywords=("approve", "apply"), callback=apply_pending, enabled=has_pending)
    add(id="proposal.reject", title="Reject pending proposal", subtitle="Discard the pending proposal", category="Proposal", keywords=("deny", "reject"), callback=reject_pending, enabled=has_pending)
    add(id="context.usage", title="Show context usage", subtitle="Display token budget by source", category="Context", keywords=("tokens",), callback=lambda app: app._show_context_usage())
    add(id="session.quit", title="Quit", subtitle="Exit Rist", category="Session", keywords=("exit",), callback=lambda app: app.exit())
    return commands

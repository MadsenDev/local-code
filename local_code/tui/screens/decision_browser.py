"""Decision Browser screen for in-session Workspace Memory."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

from local_code.tui.workspace_memory import Decision, DecisionFilter, DecisionStatus, DecisionStore, DecisionType

FILTERS: tuple[tuple[str, DecisionType | None], ...] = (
    ("All", None),
    ("Plans", DecisionType.PLAN),
    ("Decisions", DecisionType.DECISION),
    ("Assumptions", DecisionType.ASSUMPTION),
    ("Questions", DecisionType.QUESTION),
    ("Rejected", DecisionType.REJECTED),
)


class DecisionBrowserScreen(ModalScreen[None]):
    """Keyboard-first browser for structured engineering decisions."""

    BINDINGS = [
        Binding("escape", "close_or_blur", "Close"),
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
        Binding("left", "previous_filter", "Filter"),
        Binding("right", "next_filter", "Filter"),
        Binding("/", "focus_search", "Search"),
        Binding("d", "dismiss", "Dismiss"),
        Binding("o", "open_files", "Open files"),
        Binding("j", "jump_conversation", "Jump"),
    ]

    def __init__(self, store: DecisionStore, app_ref, *, initial: DecisionFilter | None = None):
        super().__init__()
        self.store = store
        self.app_ref = app_ref
        self.criteria = initial or DecisionFilter()
        self.filter_index = next((i for i, (_, t) in enumerate(FILTERS) if t == self.criteria.type), 0)
        self.selected = 0
        self.records: list[Decision] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="decision-box"):
            yield Label("Decision Browser", id="decision-title")
            yield Static(id="decision-filters")
            yield Input(placeholder="Search decisions, files, reasons…", id="decision-search")
            with Horizontal(id="decision-main"):
                yield Static(id="decision-list")
                yield Static(id="decision-details")
            yield Static("Esc Close · / Search · ←→ Filter · ↑↓ Navigate · O Open files · J Jump · D Dismiss", id="decision-hint")

    def on_mount(self) -> None:
        self._render_browser()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "decision-search":
            self.criteria.search = event.value
            self.selected = 0
            self._render_browser()

    def action_focus_search(self) -> None:
        self.query_one("#decision-search", Input).focus()

    def action_close_or_blur(self) -> None:
        search = self.query_one("#decision-search", Input)
        if search.has_focus:
            search.blur()
            return
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        self.selected = max(0, self.selected - 1)
        self._render_browser()

    def action_cursor_down(self) -> None:
        self.selected = min(max(len(self.records) - 1, 0), self.selected + 1)
        self._render_browser()

    def action_next_filter(self) -> None:
        self.filter_index = (self.filter_index + 1) % len(FILTERS)
        self.criteria.type = FILTERS[self.filter_index][1]
        self.selected = 0
        self._render_browser()

    def action_previous_filter(self) -> None:
        self.filter_index = (self.filter_index - 1) % len(FILTERS)
        self.criteria.type = FILTERS[self.filter_index][1]
        self.selected = 0
        self._render_browser()

    def action_dismiss(self) -> None:
        record = self._selected_record()
        if record:
            self.store.dismiss(record.id)
            self._render_browser()

    def action_open_files(self) -> None:
        record = self._selected_record()
        if record:
            self.app_ref.open_repository_for_decision(record)
            self.dismiss(None)

    def action_jump_conversation(self) -> None:
        record = self._selected_record()
        if record:
            self.app_ref.jump_to_decision_conversation(record)
            self.dismiss(None)

    def _selected_record(self) -> Decision | None:
        if not self.records:
            return None
        self.selected = min(self.selected, len(self.records) - 1)
        return self.records[self.selected]

    def _render_browser(self) -> None:
        self.records = self.store.filter(self.criteria)
        if self.selected >= len(self.records):
            self.selected = max(len(self.records) - 1, 0)
        self._render_filters()
        self._render_list()
        self._render_details(self._selected_record())

    def _render_filters(self) -> None:
        text = Text("Filters  ", style="dim")
        for index, (label, _) in enumerate(FILTERS):
            style = "reverse bold #8fbcbb" if index == self.filter_index else "#8fbcbb"
            text.append(f" {label} ", style=style)
            text.append(" ")
        self.query_one("#decision-filters", Static).update(text)

    def _render_list(self) -> None:
        text = Text("Decision list — newest first\n", style="bold #8fbcbb")
        if not self.records:
            text.append("No workspace memory records match.", style="dim")
        for index, record in enumerate(self.records[:200]):
            selected = index == self.selected
            style = "reverse bold" if selected else ""
            stamp = record.timestamp.strftime("%H:%M:%S")
            text.append(f"{'›' if selected else ' '} {record.type.value.upper()} ", style=style or "bold")
            text.append(f"{record.title}\n", style=style)
            text.append(f"   {stamp} · {record.status.value}", style="reverse" if selected else "dim")
            if record.files:
                text.append(" · " + ", ".join(record.files[:2]), style="reverse" if selected else "yellow")
            text.append("\n")
        self.query_one("#decision-list", Static).update(text)

    def _render_details(self, record: Decision | None) -> None:
        text = Text("Details\n", style="bold #8fbcbb")
        if record is None:
            text.append("No selection", style="dim")
        else:
            confidence = "—" if record.confidence is None else f"{record.confidence:.0%}"
            files = "\n".join(f"  • {path}" for path in record.files) or "  —"
            text.append(f"Title\n{record.title}\n\n", style="bold")
            text.append(f"Reason\n{record.reason or '—'}\n\n")
            text.append(f"Summary\n{record.summary or '—'}\n\n")
            text.append(f"Timestamp\n{record.timestamp.isoformat()}\n\n", style="dim")
            text.append(f"Confidence\n{confidence}\n\n")
            text.append(f"Status\n{record.status.value}\n\n")
            text.append(f"Affected files\n{files}\n\n", style="yellow" if record.files else "")
            text.append(f"Related proposal\n{record.proposal_id or '—'}\n\n")
            text.append(f"Conversation anchor\n{record.conversation_anchor or 'nearest available context'}\n\n", style="dim")
            if record.status != DecisionStatus.DISMISSED:
                text.append("Actions\nO Open file(s) · J Jump to conversation · D Dismiss", style="#8fbcbb")
        self.query_one("#decision-details", Static).update(text)

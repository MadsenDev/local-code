"""Command palette screen for the Rist TUI."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

from local_code.tui.commands import Command, filter_commands


class CommandPaletteScreen(ModalScreen):
    """Centered keyboard-first command palette."""

    BINDINGS = [("escape", "close", "Close"), ("up", "cursor_up", "Up"), ("down", "cursor_down", "Down")]

    def __init__(self, commands: list[Command], app_ref):
        super().__init__()
        self.commands = commands
        self.app_ref = app_ref
        self.filtered: list[Command] = []
        self.selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-box"):
            yield Label("Rist Command Palette", id="palette-title")
            yield Input(placeholder="Search commands…", id="palette-search")
            yield Static(id="palette-list")

    def on_mount(self) -> None:
        self._filter("")
        self.query_one("#palette-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "palette-search":
            self._filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "palette-search":
            self._execute_selected()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        if self.filtered:
            self.selected = max(0, self.selected - 1)
            self._render_list()

    def action_cursor_down(self) -> None:
        if self.filtered:
            self.selected = min(len(self.filtered) - 1, self.selected + 1)
            self._render_list()

    def _filter(self, query: str) -> None:
        self.filtered = filter_commands(self.commands, query, self.app_ref)
        self.selected = min(self.selected, max(0, len(self.filtered) - 1))
        self._render_list()

    def _render_list(self) -> None:
        text = Text()
        if not self.filtered:
            text.append("No commands found", style="dim")
        previous = None
        for index, command in enumerate(self.filtered[:12]):
            if command.category != previous:
                if text.plain:
                    text.append("\n")
                text.append(command.category.upper() + "\n", style="bold #8fbcbb")
                previous = command.category
            prefix = "› " if index == self.selected else "  "
            style = "reverse bold" if index == self.selected else ""
            text.append(prefix + command.title, style=style)
            if command.subtitle:
                text.append(" — " + command.subtitle, style=("reverse" if index == self.selected else "dim"))
            text.append("\n")
        self.query_one("#palette-list", Static).update(text)

    def _execute_selected(self) -> None:
        if not self.filtered:
            return
        self.dismiss(self.filtered[self.selected])

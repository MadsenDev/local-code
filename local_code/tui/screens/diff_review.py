"""Dedicated pending proposal diff review screen."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from local_code.tui.diff_review import ReviewModel


class DiffReviewScreen(ModalScreen[str]):
    """Keyboard-first review workflow for pending code proposals."""

    BINDINGS = [
        Binding("up", "cursor_up", "Previous file"),
        Binding("down", "cursor_down", "Next file"),
        Binding("enter", "apply", "Apply"),
        Binding("escape", "back", "Back"),
        Binding("r", "reject", "Reject"),
    ]

    def __init__(self, model: ReviewModel):
        super().__init__()
        self.model = model
        self.selected = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="review-box"):
            yield Label("Pending Changes", id="review-title")
            yield Static(id="review-summary")
            with Horizontal(id="review-main"):
                yield Static(id="review-files")
                yield Static(id="review-diff")
            with Horizontal(id="review-actions"):
                yield Button("Apply all (Enter)", variant="success", id="apply")
                yield Button("Reject (R)", variant="error", id="reject")
                yield Button("Back (Esc)", id="back")

    def on_mount(self) -> None:
        self._render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(str(event.button.id))

    def action_cursor_up(self) -> None:
        if self.model.files:
            self.selected = max(0, self.selected - 1)
            self._render()

    def action_cursor_down(self) -> None:
        if self.model.files:
            self.selected = min(len(self.model.files) - 1, self.selected + 1)
            self._render()

    def action_apply(self) -> None:
        self.dismiss("apply")

    def action_reject(self) -> None:
        self.dismiss("reject")

    def action_back(self) -> None:
        self.dismiss("back")

    def _render(self) -> None:
        summary = self.model.summary
        summary_text = Text()
        summary_text.append(f"Repository\n{self.model.repository}\n\n", style="dim")
        summary_text.append(f"{summary.files} files   ", style="bold")
        summary_text.append(f"+{summary.added} ", style="green")
        summary_text.append(f"-{summary.removed}   ", style="red")
        summary_text.append(f"Impact: {summary.impact}", style="yellow" if summary.impact == "High" else "dim")
        self.query_one("#review-summary", Static).update(summary_text)

        files_text = Text("Files\n", style="bold #8fbcbb")
        if not self.model.files:
            files_text.append("No file list available", style="dim")
        for index, file in enumerate(self.model.files):
            style = "reverse bold" if index == self.selected else ""
            files_text.append(("› " if index == self.selected else "  ") + file.filename + "\n", style=style)
            files_text.append(f"    +{file.added} ", style="green reverse" if index == self.selected else "green")
            files_text.append(f"-{file.removed}\n", style="red reverse" if index == self.selected else "red")
        self.query_one("#review-files", Static).update(files_text)

        diff_text = Text("Selected file\n", style="bold #8fbcbb")
        if self.model.files:
            file = self.model.files[self.selected]
            diff_text.append(file.filename + "\n\n", style="bold")
            for line in (file.diff or "No diff text available.").splitlines():
                style = ""
                if line.startswith("+") and not line.startswith("+++"):
                    style = "green"
                elif line.startswith("-") and not line.startswith("---"):
                    style = "red"
                elif line.startswith("@@"):
                    style = "#8fbcbb"
                diff_text.append(line + "\n", style=style)
        else:
            diff_text.append("No diff available.", style="dim")
        self.query_one("#review-diff", Static).update(diff_text)

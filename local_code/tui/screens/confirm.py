"""Approval modal for edit and command confirmations."""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


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
        with Vertical(id="confirm-box", classes="rist-modal-card"):
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

"""Conversation rendering widgets."""

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, RichLog, Static


class ConversationView(Vertical):
    """Primary transcript panel with role-specific message cards."""

    def compose(self) -> ComposeResult:
        yield Label("Conversation", classes="panel-title")
        yield RichLog(id="log", wrap=True, markup=True, highlight=False, classes="conversation-log")
        yield Static("", id="live", classes="empty live-card")

    @property
    def log(self) -> RichLog:
        return self.query_one("#log", RichLog)

    def write_welcome(self, mode: str) -> None:
        self.write_system(f"Welcome to **Rist**. Mode: `{mode}`. Type `/help` for commands.", title="Rist ready")

    def write_user(self, text: str) -> None:
        self.log.write(Panel(Text(text), title="user", title_align="left", border_style="cyan", padding=(0, 1)))

    def write_assistant(self, text: str) -> None:
        self.log.write(Panel(Markdown(text), title="assistant", title_align="left", border_style="green", padding=(0, 1)))

    def write_system(self, markdown: str, title: str = "system") -> None:
        self.log.write(Panel(Markdown(markdown), title=title, title_align="left", border_style="bright_black", padding=(0, 1)))

    def write_tool_summary(self, body, title: str, style: str = "cyan") -> None:
        self.log.write(Panel(body, title=title, title_align="left", border_style=style, padding=(0, 1)))

    def write_text(self, text) -> None:
        self.log.write(text)

    def clear(self) -> None:
        self.log.clear()

    def scroll_end(self) -> None:
        self.log.scroll_end(animate=False)

    def update_live(self, text: str) -> None:
        live = self.query_one("#live", Static)
        if text:
            live.remove_class("empty")
            live.update(Text(text))
        else:
            live.update("")
            live.add_class("empty")
        self.scroll_end()

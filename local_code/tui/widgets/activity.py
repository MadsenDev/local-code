"""Structured activity timeline widget."""

from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, RichLog


class ActivityTimeline(Vertical):
    """Compact chronological list of tool and runtime activity."""

    def compose(self) -> ComposeResult:
        yield Label("Activity", classes="panel-title")
        yield RichLog(id="activity-log", wrap=True, markup=True, highlight=False)

    @property
    def log(self) -> RichLog:
        return self.query_one("#activity-log", RichLog)

    def clear(self) -> None:
        self.log.clear()

    def write_event(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "tool":
            line = str(event.get("line", event.get("tool", ""))).strip()
            title, detail = self._split_tool_line(line)
            self.log.write(Panel(Text(detail or line, style="text-muted"), title=title, title_align="left", border_style="blue", padding=(0, 1)))
        elif kind == "milestone":
            self.log.write(Panel(Text(str(event.get("text", "")), style="dim"), title="PLAN", title_align="left", border_style="yellow", padding=(0, 1)))
        elif kind == "transcript":
            body = Text("\n".join(str(line) for line in event.get("lines", [])), style="dim")
            self.log.write(Panel(body, title=str(event.get("title", "TRANSCRIPT")).upper(), title_align="left", border_style="green", padding=(0, 1)))
        elif kind == "trace":
            self.log.write(Panel(Text(str(event.get("text", "")), style="dim"), title="TRACE", title_align="left", border_style="bright_black", padding=(0, 1)))
        elif kind == "command":
            self.log.write(Panel(Text(str(event.get("title", "")), style="text-muted"), title="COMMAND", title_align="left", border_style="magenta", padding=(0, 1)))
        elif kind == "task":
            self.log.write(Panel(Text(str(event.get("text", "")), style="text-muted"), title="TASK", title_align="left", border_style="blue", padding=(0, 1)))
        elif kind == "task_complete":
            self.log.write(Panel(Text(str(event.get("text", "")), style="dim"), title="TASK COMPLETE", title_align="left", border_style="green", padding=(0, 1)))
        elif kind == "task_failed":
            self.log.write(Panel(Text(str(event.get("text", "")), style="red"), title="TASK FAILED", title_align="left", border_style="red", padding=(0, 1)))
        elif kind == "diff":
            body = Text(f"{event.get('files', 0)} files  ", style="dim")
            body.append(f"+{event.get('added', 0)} ", style="green")
            body.append(f"-{event.get('removed', 0)}", style="red")
            self.log.write(Panel(body, title="DIFF", title_align="left", border_style="yellow", padding=(0, 1)))
        elif kind in {"apply", "reject"}:
            title = "APPLY" if kind == "apply" else "REJECT"
            style = "green" if kind == "apply" else "red"
            self.log.write(Panel(Text(str(event.get("text", "")), style="dim"), title=title, title_align="left", border_style=style, padding=(0, 1)))
        self.log.scroll_end(animate=False)

    def write_note(self, text: str, title: str = "NOTE", style: str = "yellow") -> None:
        self.log.write(Panel(Text(text), title=title, title_align="left", border_style=style, padding=(0, 1)))
        self.log.scroll_end(animate=False)

    @staticmethod
    def _split_tool_line(line: str) -> tuple[str, str]:
        if not line:
            return "TOOL", ""
        head, _, tail = line.partition(" ")
        title = head.strip("[]:•⚙").upper() or "TOOL"
        return title, tail.strip()

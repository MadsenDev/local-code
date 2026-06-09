import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time

from .config import MAX_OUTPUT_CHARS, SPINNER_FRAMES

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
except Exception:
    Console = None
    Markdown = None
    Syntax = None


class UI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self.console = Console() if Console is not None else None

    def style(self, text, *styles):
        if not self.enabled:
            return text
        return "".join(styles) + text + self.RESET

    def width(self):
        return shutil.get_terminal_size((100, 20)).columns

    def rule(self, label="", color=CYAN):
        if not self.enabled:
            return f"== {label} ==" if label else "=" * 20
        width = self.width()
        if not label:
            return self.style("─" * min(width, 100), color)
        prefix = f" {label} "
        fill = max(0, min(width, 100) - len(prefix))
        left = fill // 2
        right = fill - left
        return self.style(("─" * left) + prefix + ("─" * right), color)

    def box(self, title, lines=None, color=CYAN, footer=None):
        raw_lines = [str(line) for line in (lines or []) if str(line) != ""]
        wrap_width = max(40, min(self.width(), 100) - 4)
        lines = []
        for line in raw_lines:
            wrapped = textwrap.wrap(line, width=wrap_width, replace_whitespace=False) or [line]
            lines.extend(wrapped)
        if not self.enabled:
            body = [f"== {title} =="]
            body.extend(lines)
            if footer:
                body.append(str(footer))
            return "\n".join(body)
        title_text = f" {title} "
        top = self.style(f"╭─{title_text}", color)
        body = [self.style("│ ", color) + line for line in lines]
        if footer:
            body.append(self.style("│ ", color) + self.style(str(footer), self.DIM))
        bottom = self.style("╰─", color)
        return "\n".join([top, *body, bottom])

    def kv_box(self, title, rows, color=CYAN):
        width = max((len(label) for label, _ in rows), default=0)
        lines = [f"{label + ':':<{width + 1}} {value}" for label, value in rows]
        return self.box(title, lines, color=color)

    def mode_label(self, mode):
        labels = {
            "chat": ("CHAT", "brainstorming only", self.GREEN),
            "hybrid": ("HYBRID", "inspect + execute", self.YELLOW),
            "agent": ("AGENT", "execute with permissions", self.RED),
        }
        name, detail, color = labels.get(mode, (mode.upper(), "", self.CYAN))
        text = f"[{name}] {detail}".strip()
        return self.style(text, self.BOLD, color)

    def compact_status(self, mode, frontend_model, backend_model, repo, branch, changes, edit_perm="ask", cmd_perm="ask"):
        mode_colors = {"chat": self.GREEN, "hybrid": self.YELLOW, "agent": self.RED}
        color = mode_colors.get(mode, self.CYAN)
        perm_note = ""
        if edit_perm != "ask" or cmd_perm != "ask":
            perm_note = self.style(f"  ·  edits:{edit_perm} cmds:{cmd_perm}", self.DIM)
        model_note = ""
        if frontend_model != backend_model:
            model_note = self.style(f"  ·  {frontend_model} / {backend_model}", self.DIM)
        else:
            model_note = self.style(f"  ·  {frontend_model}", self.DIM)
        return (
            self.style("Rist", self.BOLD)
            + "  "
            + self.style(mode, self.BOLD, color)
            + self.style(f"  ·  {repo}/{branch} ({changes})", self.DIM)
            + model_note
            + perm_note
        )

    def tool_line(self, label, color=CYAN):
        if not self.enabled:
            return f"  > {label}"
        return "  " + self.style("⎿ ", color) + label

    def action_card(self, title, summary, detail=None, footer=None, color=CYAN):
        lines = [summary]
        if detail:
            lines.append(self.style(detail, self.DIM))
        return self.box(title, lines, color=color, footer=footer)

    def transcript_item(self, title, lines=None, color=CYAN):
        if not self.enabled:
            return f"  · {title}"
        result = self.tool_line(title, color=color)
        if lines and len(lines[0]) < 80:
            result += self.style(f"  {lines[0]}", self.DIM)
        return result

    def truncate_lines(self, text, limit=8, width=None):
        width = width or max(40, min(self.width(), 120) - 6)
        lines = []
        for raw in (text or "").splitlines():
            if len(lines) >= limit:
                break
            wrapped = textwrap.wrap(raw, width=width, replace_whitespace=False) or [raw]
            for line in wrapped:
                if len(lines) >= limit:
                    break
                lines.append(line)
        extra = max(0, len((text or "").splitlines()) - limit)
        if extra:
            lines.append(self.style(f"... +{extra} lines", self.DIM))
        return lines or [self.style("(no output)", self.DIM)]

    def print_markdown(self, text):
        if self.console is not None and Markdown is not None and self.enabled:
            self.console.print(Markdown(text or ""))
        else:
            print(text or "")

    def render_diff_lines(self, lines):
        text = "\n".join(lines or [])
        if not text:
            return []
        delta = shutil.which("delta")
        if delta and self.enabled:
            try:
                completed = subprocess.run(
                    [delta, "--color-only", "--paging=never", "--line-numbers"],
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=3,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    return completed.stdout.rstrip("\n").splitlines()
            except Exception:
                pass
        if Syntax is None or Console is None or not self.enabled:
            return lines
        console = Console(width=max(40, min(self.width(), 140)), record=True, force_terminal=True)
        console.print(Syntax(text, "diff", theme="ansi_dark", word_wrap=False))
        rendered = console.export_text(styles=True).rstrip("\n")
        return rendered.splitlines()


class Spinner:
    def __init__(self, ui, text):
        self.ui = ui
        self.text = text
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not self.ui.enabled:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        idx = 0
        while not self._stop.is_set():
            frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
            text = self.text
            width = max(20, self.ui.width() - 4)
            if len(text) > width:
                text = text[: width - 1] + "…"
            line = self.ui.style(frame, UI.CYAN, UI.BOLD) + " " + text
            print("\r\033[2K" + line, end="", file=sys.stderr, flush=True)
            time.sleep(0.08)
            idx += 1

    def stop(self, final_text=None):
        if not self.ui.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        print("\r\033[2K", end="", file=sys.stderr, flush=True)
        if final_text:
            print(final_text, file=sys.stderr, flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


def clip(text, limit=MAX_OUTPUT_CHARS):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated to {limit} chars]"


def summarize_text(text, limit=220):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def summarize_action(tool, args):
    if tool == "repo_overview":
        return "inspect repo structure and package metadata"
    if tool == "list_files":
        return f"list files under {args.get('path', '.')}"
    if tool == "search_files":
        return f"search files for {args.get('query', '')!r} in {args.get('path', '.')}"
    if tool == "read_file":
        return f"read {args.get('path', '?')} lines {args.get('start', 1)}-{args.get('end', 200)}"
    if tool == "run_command":
        return f"run shell command: {args.get('command', '')}"
    if tool == "write_file":
        return f"write file {args.get('path', '?')}"
    if tool == "replace_in_file":
        return f"replace text in {args.get('path', '?')}"
    if tool == "insert_after":
        return f"insert text into {args.get('path', '?')} after anchor"
    if tool == "final":
        return "prepare final report"
    return f"{tool} {args}"

import re

from .config import BLOCKED_COMMAND_PATTERNS
from .ui import UI


def command_is_blocked(command):
    return any(re.search(pattern, command) for pattern in BLOCKED_COMMAND_PATTERNS)


def confirm_action(kind, label, content, ui):
    title = "Run command?" if kind == "Command" else "Apply edit?"
    reason = content or f"Requested by the assistant during the current {kind.lower()} step."
    print()
    print(ui.box(title, [label, "", "Reason:", reason], color=UI.YELLOW))
    try:
        while True:
            reply = input(ui.style("[a]llow once  [d]eny  [v]iew details  [q]uit > ", UI.BOLD)).strip().lower()
            if reply in {"a", "allow", "y", "yes"}:
                return True
            if reply in {"d", "deny", "n", "no", ""}:
                return False
            if reply in {"q", "quit"}:
                raise KeyboardInterrupt
            if reply in {"v", "view", "details"}:
                print(ui.box("Details", [label, reason], color=UI.CYAN))
                continue
            print("Use a, d, v, or q.")
    except (EOFError, KeyboardInterrupt):
        print()
        return False

import re

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_FRONTEND_MODEL = "qwen3:8b"
DEFAULT_BACKEND_MODEL = "qwen3:14b"
DEFAULT_MODE = "hybrid"
DEFAULT_VERBOSITY = "normal"
DEFAULT_TOOL_CALLING = "json"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
MAX_TOOL_STEPS = 20
MAX_OUTPUT_CHARS = 12000
MAX_HISTORY_MESSAGES = 40
MEMORY_DIR_NAME = ".local-code"

PROMPT_YES_RE = re.compile(r"^(yes|y|approve|apply|go ahead|do it|continue|ship it)\b", re.I)
CODE_ACTION_RE = re.compile(
    r"\b(inspect|check|search|read|run|edit|change|fix|implement|refactor|debug|repo|repository|codebase|grep|find in code|open file|look in|patch|update|write)\b",
    re.IGNORECASE,
)
EDIT_INTENT_RE = re.compile(
    r"\b(edit|change|fix|implement|refactor|patch|update|write|add|remove|rename|modify|create)\b",
    re.IGNORECASE,
)

HELP_TEXT = """Commands:
/help                         Show this help
/clear                        Clear chat history
/mode NAME                    Set mode: chat, hybrid, agent
/model NAME                   Set both frontend and backend models
/frontend NAME                Change frontend/talker model
/backend NAME                 Change backend/coder model
/planner NAME                 Alias for /frontend
/coder NAME                   Alias for /backend
/permission SCOPE MODE        Set permission mode. Scope: all, command, edit. Mode: ask, allow, deny
/approve on|off               Compatibility alias. on => allow commands, off => ask commands
/verbosity LEVEL              Set output level: quiet, normal, debug
/trace on|off                 Alias. on => debug, off => normal
/raw on|off                   Show or hide raw JSON actions/contracts in debug mode
/tools MODE                    Set backend tool calling: json, native, auto
/paste                        Paste multi-line context; finish with /end
/ask TEXT                     Ask/explain without applying edits
/plan TEXT                    Create a plan/proposal for a task without applying edits
/code TEXT                    Execute a coding task with approval checks
/agent TEXT                   Execute a coding task in agent mode for one turn
/files                        Pick a repo file with fzf
@path                         Reference a file or folder in any prompt
/apply                        Apply the latest approved or pending plan
/undo                         Revert files changed by the last execute run
/status                       Show current settings
/quit                         Exit
"""

BLOCKED_COMMAND_PATTERNS = [
    r"(^|\s)rm\s+-rf\s+/$",
    r"(^|\s)mkfs(\.| )",
    r"(^|\s)dd\s+if=",
    r"(^|\s)(reboot|poweroff|shutdown|halt)(\s|$)",
    r":\(\)\s*\{",
]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

READ_FILE_DEFAULT_END = 500

TEST_COMMAND_PATTERNS = ("pytest", "npm test", "yarn test", "pnpm test", "jest", "vitest", "cargo test", "go test", "rspec")

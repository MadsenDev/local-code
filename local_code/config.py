import re

# Recommended-minimum standard: one shared coder model that fits a 12 GB GPU
# with full context, drives the tool loop reliably, and skips the cross-model
# assessment hop. See MODELS.md. Override with --model / --frontend / --backend.
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_FRONTEND_MODEL = "qwen2.5-coder:7b"
DEFAULT_BACKEND_MODEL = "qwen2.5-coder:7b"
DEFAULT_MODE = "hybrid"
DEFAULT_VERBOSITY = "normal"
DEFAULT_TOOL_CALLING = "json"
DEFAULT_MODEL_ROUTING = "adaptive"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
MAX_TOOL_STEPS = 100
MAX_OUTPUT_CHARS = 12000
MAX_HISTORY_MESSAGES = 40
MEMORY_DIR_NAME = ".rist"
LEGACY_MEMORY_DIR_NAME = ".local-code"

# --- Model generation / reliability layer -------------------------------
# Context window requested from Ollama. 16k fits comfortably alongside a
# 7-8B model on a 12 GB GPU; lower it (LOCAL_CODE_NUM_CTX) for tighter VRAM.
DEFAULT_NUM_CTX = 16384
# Short auxiliary passes (assessment, intent, memory) need far less context.
ASSESS_NUM_CTX = 4096
# Keep the model resident between calls so role switches don't pay a reload.
DEFAULT_KEEP_ALIVE = "30m"
# Deterministic decoding for anything that must return parseable JSON; a
# little warmth for user-facing prose so replies don't read robotically.
STRUCTURED_TEMPERATURE = 0.0
PROSE_TEMPERATURE = 0.7
# Transient-failure retries (connection resets, 5xx, model still loading).
MODEL_MAX_RETRIES = 3
MODEL_RETRY_BACKOFF = 1.5
# Read timeout (seconds) for a single non-streaming generation.
MODEL_REQUEST_TIMEOUT = 300

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
/models                       Show model tiers and the recommended-minimum standard
/decisions ACTION ...         List, add, accept, supersede, or review decisions
/context                      Show context-window accounting
/routing MODE                 Set model routing: single, adaptive, dual
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

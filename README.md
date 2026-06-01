# local-code

A local-first AI coding CLI with a full-screen TUI. Uses a two-model
architecture — a frontend model for conversation/routing and a backend model
for repo inspection and edits — though both roles default to the same model. It
runs against a local [Ollama](https://ollama.com) server or any
OpenAI-compatible provider (OpenRouter, OpenAI, …).

## Requirements

- Python 3.11+
- A model provider: [Ollama](https://ollama.com) running locally, **or** an OpenRouter/OpenAI API key
- `rg` (ripgrep) for file search
- Optional: `delta` for colored diffs, `fzf` for file picker

## Setup

```bash
# Install with the TUI (recommended) and dev tools
pip install -e ".[tui,dev]"
ollama pull qwen2.5-coder:7b
```

The full-screen TUI is the default interactive experience; if `textual` isn't
installed or stdout isn't a TTY, it falls back to the plain line-based REPL.

## Usage

```bash
# Default: full-screen TUI, qwen2.5-coder:7b for both roles (recommended on a 12 GB GPU)
local-code

# Plain line-based REPL instead of the TUI
local-code --no-tui

# Override the model for both roles
local-code --model qwen2.5-coder:14b

# Use different models for frontend and backend (both must fit in VRAM)
local-code --frontend-model qwen3:4b --backend-model qwen2.5-coder:7b

# Use OpenRouter (any OpenAI-compatible provider works the same way)
export OPENROUTER_API_KEY=sk-or-...
local-code --provider openrouter --model qwen/qwen-2.5-coder-32b-instruct

# Run a single prompt non-interactively
local-code --prompt "what does this repo do?"

# Set interaction mode
local-code --mode chat      # conversational only
local-code --mode hybrid    # auto-delegate code tasks (default)
local-code --mode agent     # always delegate to backend

# Choose backend tool protocol
local-code --tool-calling json    # prompt-emitted JSON protocol (default)
local-code --tool-calling native  # require Ollama native tool calls
local-code --tool-calling auto    # try native tools, then fall back to JSON
```

## Modes

| Mode | Behavior |
|------|----------|
| `chat` | Direct single-model replies, no tool loop |
| `hybrid` | Auto-delegates code/repo tasks to the backend tool loop |
| `agent` | All turns go to the backend tool loop |

## TUI

`local-code` opens a full-screen Textual interface by default:

- A scrolling transcript with rendered Markdown replies and `you` / `local-code` panels.
- **Live token streaming** of the answer as it's generated.
- **Tool-activity cards** — every repo read, search, command, and edit (with +/− counts) shows up as it happens.
- **Approval modals** — in execute mode, edits and commands pop a y/n confirmation.
- A status bar (provider · models · mode · permissions · pending proposal) and slash commands.

Slash commands inside the TUI: `/help`, `/status`, `/models`, `/mode`, `/model`,
`/frontend`, `/backend`, `/ask`, `/plan`, `/apply`, `/agent`, `/clear`, `/quit`.
Keys: Enter to send, Ctrl+L to clear, Ctrl+C to quit. Use `--no-tui` for the
plain REPL (also used automatically when piping or when Textual isn't installed).

## Providers

Models are reached through a provider abstraction, selected with `--provider`:

| Provider | Selection | Key |
|----------|-----------|-----|
| Ollama (local, default) | `--provider ollama` (default) | none |
| OpenRouter | `--provider openrouter --model VENDOR/MODEL` | `OPENROUTER_API_KEY` or `--api-key` |
| OpenAI / OpenAI-compatible | `--provider openai [--base-url URL] --model MODEL` | `OPENAI_API_KEY` or `--api-key` |

Any OpenAI-compatible endpoint (Together, Groq, a local vLLM/llama.cpp server,
…) works via `--provider openai --base-url <url>`. The same structured-output
enforcement, retries, and tool loop apply across all providers. Cloud models
skip the local VRAM tiering and few-shot scaffolding (they don't need it).

## Models

The **recommended minimum standard** is `qwen2.5-coder:7b` for both roles — the
sweet spot for a 12 GB GPU (full 16k context, fast, reliable tool calls).
`qwen2.5-coder:14b` is the highest-quality model a 12 GB card can host.

```bash
local-code --model qwen2.5-coder:7b   # recommended, single shared model
local-code --model qwen2.5-coder:14b  # best quality on a 12 GB card
```

Lighter (best-effort — fine for chat/inspection, flaky on multi-file edits):
- `--model qwen3:4b`
- `--model llama3.2:3b`

Splitting roles only helps if **both models fit in VRAM together** (otherwise
every role switch reloads weights). On a 12 GB card:
- `--frontend-model qwen3:4b --backend-model qwen2.5-coder:7b` — both stay resident

`local-code` checks your models at startup and warns about anything below the
standard, not pulled, or too large for the card. Run `/models` any time to see
tiers, or `python -m local_code.eval --model NAME` to measure a model's
tool-loop reliability. **See [MODELS.md](MODELS.md) for the full standard,
tiers, and tuning knobs (`LOCAL_CODE_NUM_CTX`, `LOCAL_CODE_KEEP_ALIVE`).**

### Reliability on weak models

Calls that must return JSON (tool actions, complexity assessment, intent
analysis) are sent to Ollama with a JSON-schema `format` constraint at
temperature 0, so decoding is grammar-constrained — this near-eliminates
malformed tool JSON, the dominant failure mode on small models. Weaker models
also get a few worked tool-call examples in the prompt. Transient Ollama errors
are retried with backoff, and the model is kept resident via `keep_alive`.

## Slash commands

```
/help          Show all commands
/mode NAME     Switch mode: chat, hybrid, agent
/model NAME    Set both models
/tools MODE    Switch backend tool protocol: json, native, auto
/plan TEXT     Propose changes without applying
/apply         Apply a pending proposal
/ask TEXT      Inspect without editing
/files         Pick a file with fzf
/models        Show model tiers and the recommended-minimum standard
/status        Show current config
/clear         Clear chat history
/quit          Exit
```

## Architecture

Before delegated repo work starts, `local-code` now runs a compact intent-analysis pass. This asks the model to restate the user's goal, name non-goals, identify context to inspect before editing, and define success criteria. The resulting `intent_analysis` is attached to the backend contract so even smaller local models get a narrower first step before planning or patching.

Backend tools are defined once in a canonical registry and rendered into both the legacy JSON prompt protocol and Ollama/OpenAI-style function schemas. The default remains `--tool-calling json` for broad local-model compatibility; `native` requires structured tool calls, and `auto` tries native tool calls before falling back to JSON. Tool calls are validated against the registry before dispatch in all modes.

See [CLAUDE.md](CLAUDE.md) for full architecture notes.

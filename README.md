# local-code

A local AI coding CLI backed by [Ollama](https://ollama.com). Uses a two-model architecture — a frontend model for conversation/routing and a backend model for repo inspection and edits — though both roles default to the same model.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- `rg` (ripgrep) for file search
- Optional: `delta` for colored diffs, `fzf` for file picker

## Setup

```bash
pip install -e ".[dev]"
ollama pull qwen2.5-coder:7b
```

## Usage

```bash
# Default: qwen2.5-coder:7b for both roles (recommended on a 12 GB GPU)
local-code

# Override the model for both roles
local-code --model qwen2.5-coder:14b

# Use different models for frontend and backend (both must fit in VRAM)
local-code --frontend-model qwen3:4b --backend-model qwen2.5-coder:7b

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

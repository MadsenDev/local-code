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
ollama pull qwen3:14b
```

## Usage

```bash
# Default: qwen3:14b for both roles
local-code

# Override the model for both roles
local-code --model qwen3:8b

# Use different models for frontend and backend
local-code --frontend-model gemma3:12b --backend-model qwen2.5-coder:14b

# Run a single prompt non-interactively
local-code --prompt "what does this repo do?"

# Set interaction mode
local-code --mode chat      # conversational only
local-code --mode hybrid    # auto-delegate code tasks (default)
local-code --mode agent     # always delegate to backend
```

## Modes

| Mode | Behavior |
|------|----------|
| `chat` | Direct single-model replies, no tool loop |
| `hybrid` | Auto-delegates code/repo tasks to the backend tool loop |
| `agent` | All turns go to the backend tool loop |

## Models

The default model is `qwen3:14b` for both roles. It handles reasoning, code generation, and tool use well.

To run lighter:
- `--model qwen3:8b` — faster, less memory
- `--model qwen3:4b` — minimal hardware

To split roles:
- `--frontend-model gemma3:12b --backend-model qwen2.5-coder:14b` — original dual-model config

## Slash commands

```
/help          Show all commands
/mode NAME     Switch mode: chat, hybrid, agent
/model NAME    Set both models
/plan TEXT     Propose changes without applying
/apply         Apply a pending proposal
/ask TEXT      Inspect without editing
/files         Pick a file with fzf
/status        Show current config
/clear         Clear chat history
/quit          Exit
```

## Architecture

Before delegated repo work starts, `local-code` now runs a compact intent-analysis pass. This asks the model to restate the user's goal, name non-goals, identify context to inspect before editing, and define success criteria. The resulting `intent_analysis` is attached to the backend contract so even smaller local models get a narrower first step before planning or patching.

See [CLAUDE.md](CLAUDE.md) for full architecture notes.

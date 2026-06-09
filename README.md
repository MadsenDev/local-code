# local-code

A local-first AI coding CLI with a full-screen TUI. Uses a two-model
architecture — a frontend model for conversation/routing and a backend model
for repo inspection and edits — though both roles default to the same model. It
runs against a local [Ollama](https://ollama.com) server, an external
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, or a cloud OpenAI-compatible provider.

## Requirements

- Python 3.11+
- A model provider: [Ollama](https://ollama.com), an external llama.cpp `llama-server`, **or** an OpenRouter/OpenAI API key
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

# Let Local-Code choose single vs dual residency from detected VRAM (default)
local-code --model-routing adaptive
local-code --model-routing single   # never switch model weights
local-code --model-routing dual     # explicitly preserve separate role models

# Inspect hardware, loaded models, and the routing recommendation
local-code doctor

# Measure model latency and generation throughput
local-code benchmark --model qwen2.5-coder:7b --benchmark-runs 3

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

Model routing is separate from interaction mode. `adaptive` is the default and only keeps separate frontend/backend models when hardware estimates indicate that both can stay resident. `single` shares one model, while `dual` honors separate role models regardless of estimated reload cost.

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
| llama.cpp (external, local) | `--provider llamacpp --model local` | none |
| OpenRouter | `--provider openrouter --model VENDOR/MODEL` | `OPENROUTER_API_KEY` or `--api-key` |
| OpenAI / OpenAI-compatible | `--provider openai [--base-url URL] --model MODEL` | `OPENAI_API_KEY` or `--api-key` |

Generic OpenAI-compatible endpoints (Together, Groq, local vLLM, …) work via `--provider openai --base-url <url>`. llama.cpp has a first-class `llamacpp` profile with local health checks, benchmarking, command generation, and heavy-backend routing. The same structured-output
enforcement, retries, and tool loop apply across all providers. Cloud models
skip the local VRAM tiering and few-shot scaffolding (they don't need it).

## llama.cpp external backend

> **local-code does not replace llama.cpp. local-code manages and connects to llama.cpp.**

llama.cpp owns all inference responsibilities: GGUF loading, quantization,
GPU/CPU offloading, KV cache management, continuous batching, and model
execution. local-code does not link against llama.cpp, parse GGUF files, compile
llama.cpp, or implement any inference behavior. It can, after an explicit user
command, discover or download a prebuilt `llama-server`, register or download a
GGUF, start and stop the external process, monitor health, and connect through
the OpenAI-compatible HTTP API. Ollama remains supported and is still the
default.

Install llama.cpp separately by following the upstream project, point
`LLAMA_SERVER` at an existing executable, or explicitly download a prebuilt
binary selected by you:

```bash
local-code llama install --url https://example.invalid/llama-server \
  --sha256 EXPECTED_SHA256
```

local-code does not select releases, compile source, unpack platform archives,
or bypass operating-system trust controls. `llama install` expects a direct URL
to a prebuilt executable; verify its origin and checksum.

A GGUF can be registered in place or downloaded from an explicit URL. Downloads
use a temporary `.part` file, support SHA-256 verification, and are atomically
renamed after success. Hugging Face authentication is not managed yet.

```bash
# Keep an existing GGUF where it is.
local-code model register qwen36 --model-path /models/qwen36.gguf

# Or download a user-selected GGUF into ~/.local-code/models/llamacpp/.
local-code model install qwen36 --url https://example.invalid/model.gguf \
  --sha256 EXPECTED_SHA256

local-code model list
```

To print a conservative RTX 3060 12 GB-class starting command without executing it:

```bash
local-code llama command --profile qwen36-35b-a3b --gpu rtx3060 \
  --model-path /path/to/model.gguf
```

The generated command starts from:

```bash
llama-server \
  -m /path/to/model.gguf \
  -c 16384 \
  -ngl 999 \
  --n-cpu-moe 30 \
  -fa on \
  -t 8 \
  -b 1024 \
  -ub 1024 \
  --jinja \
  --host 127.0.0.1 \
  --port 8080
```

Start and manage the external server:

```bash
# Finds llama-server via --llama-server, LLAMA_SERVER, PATH, or local-code's runtime directory.
# Resolves the registered model, starts a detached process, records its PID/log,
# and waits for /v1/models to become healthy.
local-code model start qwen36 --gpu rtx3060

local-code model status
local-code model stop

# First verify /v1/models and a tiny /v1/chat/completions request.
local-code llama doctor

# Run small/medium latency, TTFT, and throughput tests.
local-code bench --provider llamacpp --model local --benchmark-runs 3
local-code bench --provider llamacpp --model local --long-context

# Use the external server for normal local-code work.
local-code --provider llamacpp \
  --base-url http://127.0.0.1:8080/v1 \
  --model local \
  --model-routing adaptive
```

`adaptive` treats llama.cpp profiles as heavy backends: once selected, the
backend handles the whole heavy phase instead of repeatedly switching
small → big → small → big. Large MoE profiles default conceptually to `single`
/`heavy_backend` use. A separate small model may route before the heavy phase
or summarize afterward, but rapid ping-pong is discouraged.

The RTX 3060 command is a starting point, not a promise that every quant or
context fits. After measuring a stable baseline, change one dimension at a
time: context `16384 → 32768 → 65536`, threads `8 → 12 → 16`, batch
`1024 → 2048`, or `n-cpu-moe` `24 → 30 → 32`. Larger contexts consume more RAM
and KV cache; CPU-offloaded MoE weights can require substantial system RAM;
reloading a large model can be slow. Qwen3.6-35B-A3B and Qwen3-Coder-style GGUF
profiles are therefore **experimental heavy backends**, especially on 12 GB
GPUs—not defaults for quick chat or rapid tool loops.

Managed runtime state lives under `~/.local-code/runtimes/llamacpp/` (or
`LOCAL_CODE_HOME`), including `server.json` and `server.log`. local-code only
stops the PID it recorded, verifies that it still appears to be `llama-server`,
and refuses to signal a reused/unrelated PID. An already occupied port is never
silently taken over. `model start` is explicit; normal chat commands do not
implicitly launch or download anything.

If the server is absent, diagnostics and preflight print the configured URL and
lifecycle guidance instead of an Ollama-specific connection error. Optional
llama.cpp `/health`, `/props`, `/slots`, and `/metrics` data is included when the
server build exposes it (metrics may need server-side enablement).

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
/context       Show estimated context usage by source
/routing MODE  Switch single/adaptive/dual model routing
/clear         Clear chat history
/quit          Exit
```

## Hardware diagnostics and context accounting

`local-code doctor` reports system RAM, GPU/VRAM, currently loaded Ollama models, model availability, and a single/dual routing recommendation. `local-code benchmark` makes short measured calls and reports elapsed time plus TTFT and tokens/second when the provider exposes those counters. Add `--json` to either command for machine-readable output.

The interactive `/context` command and `/status` expose an estimated context budget split across conversation history, persistent project memory, repository context, and tool definitions. The estimate intentionally uses a dependency-free conservative heuristic; provider-native token counts remain authoritative when available.

## Architecture

Before delegated repo work starts, `local-code` now runs a compact intent-analysis pass. This asks the model to restate the user's goal, name non-goals, identify context to inspect before editing, and define success criteria. The resulting `intent_analysis` is attached to the backend contract so even smaller local models get a narrower first step before planning or patching.

Backend tools are defined once in a canonical registry and rendered into both the legacy JSON prompt protocol and Ollama/OpenAI-style function schemas. The default remains `--tool-calling json` for broad local-model compatibility; `native` requires structured tool calls, and `auto` tries native tool calls before falling back to JSON. Tool calls are validated against the registry before dispatch in all modes.

See [CLAUDE.md](CLAUDE.md) for full architecture notes.

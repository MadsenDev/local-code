# Rist

Rist is a local-first AI coding agent for real developer hardware.

The old `local-code` command remains available as a temporary compatibility alias.

A local-first AI coding CLI with a full-screen TUI. Uses a two-model
architecture — a frontend model for conversation/routing and a backend model
for repo inspection and edits — though both roles default to the same model. It
defaults to `provider=auto`: Rist first tries a configured managed [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, then an external llama.cpp endpoint, then [Ollama](https://ollama.com) for fallback/compatibility, or a cloud OpenAI-compatible provider when selected.

## Requirements

- Python 3.11+
- `rg` (ripgrep) for file search
- Optional: `delta` for colored diffs, `fzf` for file picker

You do **not** need to manually download a model, install llama.cpp, start a server, or choose a port for the default local setup. Rist manages those implementation details for you.

## Setup

```bash
# Install with the TUI (recommended) and dev tools
pip install -e ".[tui,dev]"
rist setup
rist
```

`rist setup` is the primary onboarding path. It checks your hardware, chooses a recommended local runtime and coding model, downloads only assets listed in Rist's committed manifests, verifies SHA-256 checksums, installs files into the managed Rist data directory, starts the managed runtime, runs diagnostics, and saves configuration. The installer is incremental and resumable: if the managed runtime, model, or configuration already exists and matches the current manifest, setup reuses it instead of downloading it again. If setup is interrupted, downloads use temporary files and verified artifacts are moved into place atomically so a later run can resume cleanly without leaving corrupted models or half-installed binaries. Use `rist setup --force` to discard and reinstall managed assets when you intentionally want a fresh download.

Managed files live under the Rist data directory (`RIST_HOME` when set, otherwise the platform default used by Rist) with a predictable layout:

```text
runtime/     # managed llama-server and install metadata
models/      # verified managed GGUF files
downloads/   # reusable verified archives and temporary .part downloads
cache/
logs/
config/      # saved runtime configuration
```

Returning users can safely run setup again:

```text
✓ Runtime already installed
✓ Model already installed
✓ Configuration already exists
Checking runtime...
Starting runtime...
✓ Ready
Done.
```

### Managed runtime and model commands

```bash
rist model install qwen2.5-coder-7b
rist model list
rist model remove qwen2.5-coder-7b

rist runtime install
rist runtime update
rist runtime status
rist runtime uninstall
```

Rist uses `local_code/manifests/runtime_manifest.json` and `local_code/manifests/model_manifest.json` as the single source of truth for downloadable assets. Each entry includes an id, version, platform, architecture, URL, SHA-256, size, display name, description, license, and source.

### Advanced setup

Advanced users can still bring their own runtime, endpoint, or model path:

```bash
rist setup --yes --model-path /models/qwen2.5-coder-7b.gguf --llama-server /opt/llama.cpp/llama-server --start
rist --provider llamacpp --base-url http://127.0.0.1:8080/v1
rist --provider ollama
```

Manual GGUF files, direct llama-server paths, custom ports, `--base-url`, and cloud/OpenAI-compatible providers remain supported, but they are no longer part of the default onboarding path.

The full-screen TUI is the default interactive experience; if `textual` isn't installed or stdout isn't a TTY, it falls back to the plain line-based REPL.

### Configuration and migration

Rist stores user configuration and managed runtime data in the Rist data directory. On first use, if legacy `~/.local-code/` data exists and the current Rist directory does not, Rist safely copies the legacy directory and leaves the original untouched. Set `RIST_HOME` to use a custom location; `LOCAL_CODE_HOME` remains supported as a compatibility override.

## Usage

```bash
# Default: provider=auto, full-screen TUI, model=local for configured llama.cpp
rist

# Plain line-based REPL instead of the TUI
rist --no-tui

# Override the model for both roles
rist --model qwen2.5-coder:14b

# Use different models for frontend and backend (both must fit in VRAM)
rist --frontend-model qwen3:4b --backend-model qwen2.5-coder:7b

# Use OpenRouter (any OpenAI-compatible provider works the same way)
export OPENROUTER_API_KEY=sk-or-...
rist --provider openrouter --model qwen/qwen-2.5-coder-32b-instruct

# Run a single prompt non-interactively
rist --prompt "what does this repo do?"

# Let Rist choose single vs dual residency from detected VRAM (default)
rist --model-routing adaptive
rist --model-routing single   # never switch model weights
rist --model-routing dual     # explicitly preserve separate role models

# Inspect hardware, loaded models, and the routing recommendation
rist doctor

# Measure model latency and generation throughput
rist benchmark --benchmark-runs 3

# Set interaction mode
rist --mode chat      # conversational only
rist --mode hybrid    # auto-delegate code tasks (default)
rist --mode agent     # always delegate to backend

# Choose backend tool protocol
rist --tool-calling json    # prompt-emitted JSON protocol (default)
rist --tool-calling native  # require Ollama native tool calls
rist --tool-calling auto    # try native tools, then fall back to JSON
```

## Modes

| Mode | Behavior |
|------|----------|
| `chat` | Direct single-model replies, no tool loop |
| `hybrid` | Auto-delegates code/repo tasks to the backend tool loop |
| `agent` | All turns go to the backend tool loop |

Model routing is separate from interaction mode. `adaptive` is the default and only keeps separate frontend/backend models when hardware estimates indicate that both can stay resident. `single` shares one model, while `dual` honors separate role models regardless of estimated reload cost.

## TUI roadmap

Rist TUI v2 roadmap:

| Version | Scope | Status |
| --- | --- | --- |
| v2.3 | Async runtime tasks plus dedicated runtime, doctor, and benchmark result screens | Implemented |
| v2.4 | Repository explorer | Planned |
| v2.5 | Decision browser | Planned |
| v2.6 | Settings screen | Planned |
| v2.7 | Real measured llama.cpp tuning workflow | Planned |

## TUI

Rist opens a full-screen Textual interface by default:

- A scrolling transcript with rendered Markdown replies and `you` / `Rist` panels.
- **Live token streaming** of the answer as it's generated.
- **Tool-activity cards** — every repo read, search, command, and edit (with +/− counts) shows up as it happens.
- **Approval modals** — in execute mode, edits and commands pop a y/n confirmation.
- A status bar (provider · models · assistant busy state · runtime task state · permissions · pending proposal) and slash commands.
- Ctrl+K opens the command palette for runtime, model, routing, permission, proposal, and session actions.
- Runtime management, doctor, and benchmark palette actions run asynchronously so the cockpit remains readable while work is in progress.
- Runtime status, doctor, and benchmark results open in dedicated TUI views instead of being mixed into the command palette.

Slash commands inside the TUI: `/help`, `/status`, `/models`, `/mode`, `/model`,
`/frontend`, `/backend`, `/ask`, `/plan`, `/apply`, `/agent`, `/clear`, `/quit`.
Keys: Enter to send, Ctrl+K to open the command palette, Ctrl+L to clear, Ctrl+C to quit. Use `--no-tui` for the
plain REPL (also used automatically when piping or when Textual isn't installed).

## Providers

Models are reached through a provider abstraction, selected with `--provider`:

| Provider | Selection | Key |
|----------|-----------|-----|
| Auto (default) | `--provider auto` (default): managed llama.cpp → external llama.cpp → Ollama → setup guidance | none |
| llama.cpp (managed/external, local primary path) | `--provider llamacpp --model local` | none |
| Ollama (local fallback/compatibility) | `--provider ollama` | none |
| OpenRouter | `--provider openrouter --model VENDOR/MODEL` | `OPENROUTER_API_KEY` or `--api-key` |
| OpenAI / OpenAI-compatible | `--provider openai [--base-url URL] --model MODEL` | `OPENAI_API_KEY` or `--api-key` |

Generic OpenAI-compatible endpoints (Together, Groq, local vLLM, …) work via `--provider openai --base-url <url>`. llama.cpp has a first-class `llamacpp` profile with local health checks, benchmarking, command generation, and heavy-backend routing. The same structured-output
enforcement, retries, and tool loop apply across all providers. Cloud models
skip the local VRAM tiering and few-shot scaffolding (they don't need it).

## llama.cpp external backend

> **Rist does not replace llama.cpp. Rist manages and connects to llama.cpp.**

llama.cpp owns all inference responsibilities: GGUF loading, quantization,
GPU/CPU offloading, KV cache management, continuous batching, and model
execution. Rist does not link against llama.cpp, parse GGUF files, compile
llama.cpp, or implement any inference behavior. It can, after an explicit user
command, discover or download a prebuilt `llama-server`, register or download a
GGUF, start and stop the external process, monitor health, and connect through
the OpenAI-compatible HTTP API. Ollama remains supported as a fallback and compatibility provider, but the default provider is `auto` and the primary local path is llama.cpp.

Install llama.cpp separately by following the upstream project, point
`LLAMA_SERVER` at an existing executable, or explicitly download a prebuilt
binary selected by you:

```bash
rist llama install --url https://example.invalid/llama-server \
  --sha256 EXPECTED_SHA256
```

Rist does not select releases, compile source, unpack platform archives,
or bypass operating-system trust controls. `llama install` expects a direct URL
to a prebuilt executable; verify its origin and checksum.

A GGUF can be registered in place or downloaded from an explicit URL. Downloads
use a temporary `.part` file, support SHA-256 verification, and are atomically
renamed after success. Hugging Face authentication is not managed yet.

```bash
# Keep an existing GGUF where it is.
rist model register qwen36 --model-path /models/qwen36.gguf

# Or download a user-selected GGUF into ~/.rist/models/llamacpp/.
rist model install qwen36 --url https://example.invalid/model.gguf \
  --sha256 EXPECTED_SHA256

rist model list
```

To print a conservative RTX 3060 12 GB-class starting command without executing it:

```bash
rist llama command --profile qwen36-35b-a3b --gpu rtx3060 \
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
# Finds llama-server via --llama-server, LLAMA_SERVER, PATH, or rist's runtime directory.
# Resolves the registered model, starts a detached process, records its PID/log,
# and waits for /v1/models to become healthy.
rist model start qwen36 --gpu rtx3060

rist model status
rist model stop

# First verify /v1/models and a tiny /v1/chat/completions request.
rist llama doctor

# Run small/medium latency, TTFT, and throughput tests.
rist benchmark
rist benchmark --provider llamacpp --model local --benchmark-runs 3
rist bench --provider llamacpp --model local --long-context

# Use the external server for normal rist work.
rist --provider llamacpp \
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

Managed runtime state lives under `~/.rist/runtimes/llamacpp/` (or a custom
`RIST_HOME`; the legacy `LOCAL_CODE_HOME` override is also accepted), including `server.json` and `server.log`. Rist only
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
rist --model qwen2.5-coder:7b   # recommended, single shared model
rist --model qwen2.5-coder:14b  # best quality on a 12 GB card
```

Lighter (best-effort — fine for chat/inspection, flaky on multi-file edits):
- `--model qwen3:4b`
- `--model llama3.2:3b`

Splitting roles only helps if **both models fit in VRAM together** (otherwise
every role switch reloads weights). On a 12 GB card:
- `--frontend-model qwen3:4b --backend-model qwen2.5-coder:7b` — both stay resident

Rist checks your models at startup and warns about anything below the
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

`rist doctor` reports system RAM, GPU/VRAM, currently loaded Ollama models, model availability, and a single/dual routing recommendation. `rist benchmark` makes short measured calls and reports elapsed time plus TTFT and tokens/second when the provider exposes those counters. Add `--json` to either command for machine-readable output.

The interactive `/context` command and `/status` expose an estimated context budget split across conversation history, persistent project memory, repository context, and tool definitions. The estimate intentionally uses a dependency-free conservative heuristic; provider-native token counts remain authoritative when available.

## Architecture

Before delegated repo work starts, Rist now runs a compact intent-analysis pass. This asks the model to restate the user's goal, name non-goals, identify context to inspect before editing, and define success criteria. The resulting `intent_analysis` is attached to the backend contract so even smaller local models get a narrower first step before planning or patching.

Backend tools are defined once in a canonical registry and rendered into both the legacy JSON prompt protocol and Ollama/OpenAI-style function schemas. The default remains `--tool-calling json` for broad local-model compatibility; `native` requires structured tool calls, and `auto` tries native tool calls before falling back to JSON. Tool calls are validated against the registry before dispatch in all modes.

See [CLAUDE.md](CLAUDE.md) for full architecture notes.

## Managed llama.cpp workflow

llama.cpp remains an **external inference runtime**: it loads GGUF files and
owns quantization, KV cache, batching, and CPU/GPU offload. Rist only
provides a registry, lifecycle convenience commands, diagnostics, routing, and
benchmarks. llama.cpp must be installed separately, or installed from a
user-selected prebuilt binary URL with the helper.

```bash
rist llama install --url https://example.invalid/llama-server
rist model register qwen36 --path ~/Models/qwen.gguf --provider llamacpp
rist model start qwen36
rist llama doctor
rist bench --provider llamacpp --model local
rist llama logs
rist model stop qwen36
```

Use `rist model start qwen36 --dry-run` to inspect the executable, model,
port, generated arguments, log, and state paths without launching anything.
Use `rist llama logs --tail 50` or `rist llama logs --follow` for a
managed server. Logs and state are under
`~/.rist/runtimes/llamacpp/`. `rist model remove qwen36`
unregisters a model but keeps its GGUF; add `--delete-file` (and optionally
`--yes`) only when the file should also be deleted.

Manual mode is equally supported. Start `llama-server` yourself, then select
the external provider for each invocation (or persist equivalent values in
your local wrapper/configuration):

```bash
llama-server -m ~/Models/qwen.gguf --host 127.0.0.1 --port 8080 ...
rist --provider llamacpp --base-url http://127.0.0.1:8080/v1 --model local
```

Rist cannot discover logs for manually started servers. Large MoE models
are heavy, experimental backends; on 8–12 GB GPUs, avoid frequent model
switching and increase context or batch sizes only after measuring a stable
baseline.

## Structured repository intelligence

Rist uses an explicit privacy boundary inside each repository:

```text
.rist/
├── project/                 # reviewable, optionally committed team knowledge
│   ├── intelligence.json
│   ├── dependency-graph.json    # typed static-analysis and manifest graph
│   ├── project.md
│   ├── architecture.md          # concise generated graph view
│   └── decisions.md
├── local/                   # always ignored, machine/user-specific state
│   ├── intelligence/
│   ├── chat_history.jsonl
│   ├── runs.jsonl
│   └── preferences.json
└── .gitignore               # precise scope rules; never a blanket `*`
```

Choose the policy with `--storage-mode local-only|shared|hybrid`, or set
`RIST_STORAGE_MODE`. The default is `hybrid`:

- **`local-only`** ignores both scopes and reads only local intelligence.
- **`shared`** reads reviewable project intelligence and disables automatic
  learning from raw run history.
- **`hybrid`** reads both scopes, while automatically learned facts remain
  local until a person rewrites/reviews them for `.rist/project/`.

Only durable, reviewed project identity, accepted decisions, architecture
components/relationships, conventions, and lifecycle classifications are
eligible for the project scope. Proposed/rejected records and operational facts
remain local. Project-scope writes are rejected if they contain likely secrets,
prompt or provider details, environment values, sensitive fields, or absolute
home-directory paths. Treat this as a safety net, not a substitute for review:
inspect `.rist/project/` before staging it.

The `rist index` command merges filesystem, package-manifest, and language-parser
evidence into `dependency-graph.json`. Python extraction uses the standard
library AST, while JavaScript and TypeScript use a replaceable parser adapter.
Stable node and edge IDs make incremental indexing remove stale relationships;
the generated `architecture.md` summarizes inferred layers and cites graph IDs.

Chat transcripts, raw run logs, caches, local paths, provider configuration, and
personal workflow preferences always belong under `.rist/local/` and are
ignored. Never copy API keys, tokens, prompts, `.env` contents, or personal paths
into reviewable records.

On first use after an upgrade, Rist moves the old fully ignored root-level
`.rist` intelligence and `.rist/private/` state into `.rist/local/`, replaces
the blanket `.rist/.gitignore` rule with precise entries, and removes obsolete
whole-`.rist` rules from `.git/info/exclude`. Migration never runs `git add`,
never stages files, and never commits anything. Review and manually promote only
the minimal knowledge your team intends to share. Existing `.local-code/`
repository memory is copied first and then follows the same private migration.

The versioned JSON store remains authoritative and the Markdown files are
editable views. Rist imports supported bullet, status, and confidence edits and
then renders the views back from structured records. Every record retains a
stable ID, lifecycle status, confidence, source provenance, timestamps, and
supersession links.

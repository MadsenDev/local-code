# Rist User Guide

Rist is a local-first AI coding CLI with a full-screen TUI. It uses a two-model architecture: a frontend model for conversation/routing and a backend model for repository inspection and edits. Both roles default to the same model, and adaptive routing only keeps separate models when the detected hardware makes that practical.

The default provider is `auto`: Rist first tries a configured managed llama.cpp `llama-server`, then an external llama.cpp endpoint, then Ollama for fallback/compatibility, or a cloud OpenAI-compatible provider when selected.

## Requirements

- Python 3.11+
- `rg` (ripgrep) for file search
- Optional: `delta` for colored diffs
- Optional: `fzf` for file picker support

You do **not** need to manually download a model, install llama.cpp, start a server, or choose a port for the default local setup. Rist manages those implementation details for you.

## Installation and setup

```bash
pip install -e ".[tui,dev]"
rist setup
rist
```

`rist setup` is the primary onboarding path. It checks your hardware, chooses a recommended local runtime and coding model, downloads only assets listed in Rist's committed manifests, verifies SHA-256 checksums, installs files into the managed Rist data directory, starts the managed runtime, runs diagnostics, and saves configuration.

The installer is incremental and resumable:

- Existing managed runtime, model, and configuration files are reused when they match the current manifest.
- Downloads use temporary files.
- Verified artifacts are moved into place atomically.
- A later setup run can resume cleanly after interruption.
- `rist setup --force` discards and reinstalls managed assets when you intentionally want a fresh download.

Managed files live under the Rist data directory (`RIST_HOME` when set, otherwise the platform default used by Rist):

```text
runtime/      # managed llama-server and install metadata
models/       # verified managed GGUF files
downloads/    # reusable verified archives and temporary .part downloads
cache/
logs/
config/       # saved runtime configuration
```

Returning users can safely run setup again. A typical no-op setup shows already installed runtime/model/configuration checks, starts or verifies the runtime, and exits ready.

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

Rist uses `local_code/manifests/runtime_manifest.json` and `local_code/manifests/model_manifest.json` as the source of truth for downloadable assets. Each entry includes an id, version, platform, architecture, URL, SHA-256, size, display name, description, license, and source.

### Advanced setup

Advanced users can bring their own runtime, endpoint, or model path:

```bash
rist setup --yes --model-path /models/qwen2.5-coder-7b.gguf --llama-server /opt/llama.cpp/llama-server --start
rist --provider llamacpp --base-url http://127.0.0.1:8080/v1
rist --provider ollama
```

Manual GGUF files, direct llama-server paths, custom ports, `--base-url`, and cloud/OpenAI-compatible providers remain supported, but they are no longer part of the default onboarding path.

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

# Use different models for frontend and backend, when both fit in VRAM
rist --frontend-model qwen3:4b --backend-model qwen2.5-coder:7b

# Use OpenRouter; any OpenAI-compatible provider works similarly
export OPENROUTER_API_KEY=sk-or-...
rist --provider openrouter --model qwen/qwen-2.5-coder-32b-instruct

# Run a single prompt non-interactively
rist --prompt "what does this repo do?"

# Let Rist choose single vs dual residency from detected VRAM
rist --model-routing adaptive
rist --model-routing single
rist --model-routing dual

# Inspect hardware, loaded models, and routing recommendation
rist doctor

# Measure model latency and generation throughput
rist benchmark --benchmark-runs 3

# Set interaction mode
rist --mode chat
rist --mode hybrid
rist --mode agent

# Choose backend tool protocol
rist --tool-calling json
rist --tool-calling native
rist --tool-calling auto
```

## Interaction modes

| Mode | Behavior |
| --- | --- |
| `chat` | Direct single-model replies, no tool loop. |
| `hybrid` | Auto-delegates code/repository tasks to the backend tool loop. |
| `agent` | Sends all turns to the backend tool loop. |

Model routing is separate from interaction mode. `adaptive` is the default and only keeps separate frontend/backend models when hardware estimates indicate that both can stay resident. `single` shares one model, while `dual` honors separate role models regardless of estimated reload cost.

## TUI

Rist opens a full-screen Textual interface by default. It includes:

- A scrolling transcript with rendered Markdown replies and `you` / `Rist` panels.
- Live token streaming as answers are generated.
- Tool-activity cards for repository reads, searches, commands, and edits.
- Approval modals in execute mode before edits and commands.
- A status bar showing provider, models, busy state, runtime task state, permissions, and pending proposals.
- Ctrl+K command palette actions for runtime, model, routing, permission, proposal, and session workflows.
- Dedicated runtime status, doctor, and benchmark result screens.

Slash commands inside the TUI include:

```text
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

Keys: Enter sends, Ctrl+K opens the command palette, Ctrl+L clears, and Ctrl+C quits.

### TUI roadmap

| Version | Scope | Status |
| --- | --- | --- |
| v2.3 | Async runtime tasks plus dedicated runtime, doctor, and benchmark result screens | Implemented |
| v2.4 | Repository explorer | Planned |
| v2.5 | Decision browser | Planned |
| v2.6 | Settings screen | Planned |
| v2.7 | Real measured llama.cpp tuning workflow | Planned |

## Providers

Models are reached through a provider abstraction selected with `--provider`.

| Provider | Selection | Key |
| --- | --- | --- |
| Auto | `--provider auto` | none |
| llama.cpp | `--provider llamacpp --model local` | none |
| Ollama | `--provider ollama` | none |
| OpenRouter | `--provider openrouter --model VENDOR/MODEL` | `OPENROUTER_API_KEY` or `--api-key` |
| OpenAI / OpenAI-compatible | `--provider openai [--base-url URL] --model MODEL` | `OPENAI_API_KEY` or `--api-key` |

Generic OpenAI-compatible endpoints such as Together, Groq, local vLLM, and similar services work through `--provider openai --base-url <url>`. llama.cpp has a first-class profile with local health checks, benchmarking, command generation, and heavy-backend routing. The same structured-output enforcement, retries, and tool loop apply across providers. Cloud models skip the local VRAM tiering and few-shot scaffolding.

## Managed and external llama.cpp

> Rist does not replace llama.cpp. Rist manages and connects to llama.cpp.

llama.cpp owns inference responsibilities: GGUF loading, quantization, GPU/CPU offloading, KV cache management, continuous batching, and model execution. Rist does not link against llama.cpp, parse GGUF files, compile llama.cpp, or implement inference behavior.

After an explicit user command, Rist can discover or download a prebuilt `llama-server`, register or download a GGUF, start and stop the external process, monitor health, and connect through the OpenAI-compatible HTTP API. Ollama remains supported as a fallback and compatibility provider, but the default provider is `auto` and the primary local path is llama.cpp.

Install llama.cpp separately, point `LLAMA_SERVER` at an existing executable, or explicitly download a prebuilt binary selected by you:

```bash
rist llama install --url https://example.invalid/llama-server \
  --sha256 EXPECTED_SHA256
```

A GGUF can be registered in place or downloaded from an explicit URL. Downloads use a temporary `.part` file, support SHA-256 verification, and are atomically renamed after success.

```bash
rist model register qwen36 --model-path /models/qwen36.gguf
rist model install qwen36 --url https://example.invalid/model.gguf \
  --sha256 EXPECTED_SHA256
rist model list
```

To print a conservative RTX 3060 12 GB-class starting command without executing it:

```bash
rist llama command --profile qwen36-35b-a3b --gpu rtx3060 \
  --model-path /path/to/model.gguf
```

Start and manage the external server:

```bash
rist model start qwen36 --gpu rtx3060
rist model status
rist model stop
rist llama doctor
rist benchmark --provider llamacpp --model local --benchmark-runs 3
rist bench --provider llamacpp --model local --long-context
rist --provider llamacpp --base-url http://127.0.0.1:8080/v1 --model local --model-routing adaptive
```

`adaptive` treats llama.cpp profiles as heavy backends: once selected, the backend handles the whole heavy phase instead of repeatedly switching small → big → small → big. Large MoE profiles default conceptually to `single` / `heavy_backend` use.

The RTX 3060 command is a starting point, not a promise that every quant or context fits. After measuring a stable baseline, change one dimension at a time: context, threads, batch, or CPU-offloaded MoE layers. Larger contexts consume more RAM and KV cache, CPU-offloaded MoE weights can require substantial system RAM, and reloading a large model can be slow.

Managed runtime state lives under `~/.rist/runtimes/llamacpp/` or a custom `RIST_HOME`, including `server.json` and `server.log`. Rist only stops the PID it recorded, verifies that it still appears to be `llama-server`, and refuses to signal a reused or unrelated PID. An already occupied port is never silently taken over. Normal chat commands do not implicitly launch or download anything.

## Models

The recommended minimum standard is `qwen2.5-coder:7b` for both roles: it is the sweet spot for a 12 GB GPU, with full 16k context, good speed, and reliable tool calls. `qwen2.5-coder:14b` is the highest-quality model a 12 GB card can typically host.

```bash
rist --model qwen2.5-coder:7b
rist --model qwen2.5-coder:14b
```

Lighter best-effort models can work for chat and inspection, but may be flaky on multi-file edits:

- `--model qwen3:4b`
- `--model llama3.2:3b`

Splitting roles only helps if both models fit in VRAM together. On a 12 GB card, `--frontend-model qwen3:4b --backend-model qwen2.5-coder:7b` is intended to stay resident.

Rist checks models at startup and warns about anything below the standard, not pulled, or too large for the card. Run `/models` any time to see tiers, or `python -m local_code.eval --model NAME` to measure a model's tool-loop reliability. See `MODELS.md` for the full standard, tiers, and tuning knobs such as `LOCAL_CODE_NUM_CTX` and `LOCAL_CODE_KEEP_ALIVE`.

### Reliability on weak models

Calls that must return JSON, including tool actions, complexity assessment, and intent analysis, are sent with a JSON-schema `format` constraint at temperature 0 when supported. This grammar-constrained decoding reduces malformed tool JSON, the dominant failure mode on small models. Weaker models also get worked tool-call examples in the prompt. Transient provider errors are retried with backoff, and local models can be kept resident via `keep_alive`.

## Hardware diagnostics and context accounting

`rist doctor` reports system RAM, GPU/VRAM, currently loaded Ollama models, model availability, and a single/dual routing recommendation. `rist benchmark` makes short measured calls and reports elapsed time plus TTFT and tokens/second when the provider exposes those counters. Add `--json` to either command for machine-readable output.

The interactive `/context` command and `/status` expose an estimated context budget split across conversation history, persistent project memory, repository context, and tool definitions. The estimate intentionally uses a dependency-free conservative heuristic; provider-native token counts remain authoritative when available.

## Architecture and tool loop

Before delegated repository work starts, Rist runs a compact intent-analysis pass. This asks the model to restate the user's goal, name non-goals, identify context to inspect before editing, and define success criteria. The resulting `intent_analysis` is attached to the backend contract so local models get a narrower first step before planning or patching.

Backend tools are defined once in a canonical registry and rendered into both the legacy JSON prompt protocol and Ollama/OpenAI-style function schemas. The default is `--tool-calling json` for broad local-model compatibility; `native` requires structured tool calls, and `auto` tries native tool calls before falling back to JSON. Tool calls are validated against the registry before dispatch in all modes.

See `CLAUDE.md` for full architecture notes.

## Structured repository intelligence

Rist uses an explicit privacy boundary inside each repository:

```text
.rist/
├── project/                 # reviewable, optionally committed team knowledge
│   ├── intelligence.json
│   ├── dependency-graph.json
│   ├── project.md
│   ├── architecture.md
│   └── decisions.md
├── local/                   # always ignored, machine/user-specific state
│   ├── intelligence/
│   ├── chat_history.jsonl
│   ├── runs.jsonl
│   └── preferences.json
└── .gitignore               # precise scope rules
```

Choose the policy with `--storage-mode local-only|shared|hybrid`, or set `RIST_STORAGE_MODE`. The default is `hybrid`:

- `local-only` ignores both scopes and reads only local intelligence.
- `shared` reads reviewable project intelligence and disables automatic learning from raw run history.
- `hybrid` reads both scopes, while automatically learned facts remain local until a person rewrites or reviews them for `.rist/project/`.

Only durable, reviewed project identity, accepted decisions, architecture components/relationships, conventions, and lifecycle classifications are eligible for the project scope. Proposed/rejected records and operational facts remain local. Project-scope writes are rejected if they contain likely secrets, prompt or provider details, environment values, sensitive fields, or absolute home-directory paths.

The `rist index` command merges filesystem, package-manifest, and language-parser evidence into `dependency-graph.json`. Python extraction uses the standard library AST, while JavaScript and TypeScript use a replaceable parser adapter. Stable node and edge IDs make incremental indexing remove stale relationships; the generated `architecture.md` summarizes inferred layers and cites graph IDs.

Chat transcripts, raw run logs, caches, local paths, provider configuration, and personal workflow preferences always belong under `.rist/local/` and are ignored. Never copy API keys, tokens, prompts, `.env` contents, or personal paths into reviewable records.

On first use after an upgrade, Rist moves the old fully ignored root-level `.rist` intelligence and `.rist/private/` state into `.rist/local/`, replaces the blanket `.rist/.gitignore` rule with precise entries, and removes obsolete whole-`.rist` rules from `.git/info/exclude`. Migration never runs `git add`, never stages files, and never commits anything. Review and manually promote only the minimal knowledge your team intends to share.

The versioned JSON store remains authoritative and the Markdown files are editable views. Rist imports supported bullet, status, and confidence edits and then renders the views back from structured records. Every record retains a stable ID, lifecycle status, confidence, source provenance, timestamps, and supersession links.

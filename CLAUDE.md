# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev mode with test deps)
pip install -e ".[dev]"

# Run all tests
.venv/bin/pytest tests/

# Run a single test file or test
.venv/bin/pytest tests/test_contracts.py
.venv/bin/pytest tests/test_contracts.py::TestNormalizeBackendReport -v

# Run the CLI
local-code [--mode chat|hybrid|agent] [--workdir DIR] [--prompt TEXT]

# Measure a model's tool-loop reliability (needs a live Ollama)
python -m local_code.eval --model qwen2.5-coder:7b
```

## Architecture

This is a **two-model AI coding CLI** backed by Ollama. The key architectural idea is a frontend/backend split — two locally-running LLMs with distinct roles, coordinated by `LocalPartner`.

### Dual-model design

- **Frontend** (`qwen2.5-coder:7b` default; same as backend, so the assessment hop is skipped): User-facing conversational model. In hybrid/agent mode, decides whether to answer directly or delegate to the backend. Also acts as a *complexity assessor* — before any backend tool loop runs (when models differ), the frontend makes a lightweight pre-flight call (`ollama_assess`) to decide whether to run the task with itself or escalate to the stronger backend model.
- **Backend** (`qwen2.5-coder:7b` default): Headless code worker. Receives a *contract* (JSON object specifying task scope, permissions, edit policy) and executes a tool loop, emitting `{"tool":"...","args":{...}}` on every turn until it emits `{"tool":"final","args":{...}}`.

### Escalation flow

When `frontend_model != backend_model`, `LocalPartner._run_backend()` calls `_assess_complexity(contract)` before running the tool loop. The frontend model responds with `{"handle":"self"}` or `{"handle":"escalate","reason":"..."}`. Simple tasks (read-only, single-file, chat) run with the frontend model; complex tasks (multi-file edits, refactoring, bootstrapping) escalate to the backend model via `LocalCodeAgent.run_contract_with_model()`. On any parse failure the default is to escalate (safe fallback). When both models are the same, the assessment call is skipped entirely.

### Intent scaffold

Before `LocalPartner._run_backend()` starts delegated repo work, it runs a compact intent-analysis pass with the frontend model. The pass returns JSON describing the user's actual goal, non-goals, needed context, likely files, risks, and success criteria. `LocalPartner._contract_with_intent_scaffold()` attaches that data to the backend contract as `intent_analysis`, resolves likely files into `files_of_interest`, and adds guardrail constraints so the backend inspects relevant context before editing and avoids overreach. If the intent pass fails or returns invalid JSON, the original contract is used unchanged.

### Model reliability layer (`models.py`, `model_profiles.py`)

`models.py` is the single Ollama choke point and where weak-model robustness
lives. Calls that must return JSON (the backend `self.chat` tool/report path,
`ollama_assess` for complexity + intent) are sent with a JSON `format`
constraint at temperature 0, so decoding is grammar-constrained — the dominant
failure mode (malformed tool JSON) largely disappears. `_post_chat_with_format`
degrades a schema → `"json"` → no format on HTTP 400 so it works on any Ollama
version. `_post_chat` retries transient errors with backoff; all calls set
`keep_alive` and an env-overridable `num_ctx` (`LOCAL_CODE_NUM_CTX`,
`LOCAL_CODE_KEEP_ALIVE`). Structured defaults are baked into the model
functions, not the call sites, so the agent/test call signatures are unchanged.

`model_profiles.classify_model()` tiers a model tag (recommended / supported /
best_effort / unsupported) against an **RTX 3060 12 GB** target, parsing param
count (incl. MoE `NNb-aMb`) and family. The profile drives few-shot injection
(`LocalCodeAgent.few_shot_block`) and the startup advisory. The published
standard is `qwen2.5-coder:7b` (ceiling `qwen2.5-coder:14b`). See `MODELS.md`.

`cli.run_preflight()` checks Ollama + pulled models and prints the tier advisory
(also `/models`); it stays silent when everything meets the standard.
`local_code/eval.py` is a runnable harness that scores tool-loop reliability.

### Tool loop (`agent.py`)

`LocalCodeAgent.run_contract()` drives the backend loop. Each iteration:
1. Calls Ollama with the backend system prompt + message history
2. Requests a tool action via the configured backend tool protocol (`json`, `native`, or `auto`)
3. Parses fallback JSON with `parse_action()` when needed (handles JSON with markdown fences, nested strings, garbage prefix/suffix)
4. Validates the tool name and arguments against the canonical registry in `tool_specs.py`
5. Dispatches to `tool_result()` which executes the tool and returns a string
6. Appends `{"role":"assistant","content":...}` and `{"role":"user","content":"Tool result for X:\n..."}` to messages
7. Exits on `tool == "final"`, step limit, or repeated action detection

### Contracts (`contracts.py`)

A *contract* is a plain dict passed from frontend to backend specifying:
- `task`: what to do
- `edit_policy`: `inspect | plan | propose | execute`
- `command_permission` / `edit_permission`: `ask | allow | deny`
- `files_of_interest`, `commands_of_interest`: auto-inferred hints
- `scope`: `narrow | broad`

`normalize_contract()` enriches frontend-delegated contracts by inferring file hints from the prompt text, detecting pasted content, and applying database-task constraints. `normalize_backend_report()` standardizes backend `final` args to a consistent shape — it accepts either a flat dict in `args` or a nested JSON string in `args["message"]`.

### Proposal workflow

When `edit_policy` is `plan` or `propose`, the backend produces a report without writing files. `LocalPartner` stores this as `pending_plan` and prompts the user. On approval (`/apply` or a yes-match), `apply_pending_plan()` re-runs the same contract with `edit_policy="execute"`.

### Modes

| Mode | Behavior |
|------|----------|
| `chat` | Frontend replies directly, no backend delegation |
| `hybrid` | Frontend decides per-turn whether to delegate |
| `agent` | All turns go directly to backend |

### Memory (`.local-code/`)

Per-repo memory lives in `.local-code/` (git-excluded). Files: `project.md`, `decisions.md`, `architecture.md`, `runs.jsonl`. `load_repo_memory()` concatenates these and injects them into the backend system prompt. The directory is auto-created on first run.

### Tools available to the backend

`search_web`, `fetch_url`, `repo_overview`, `list_files`, `search_files`, `read_file`, `run_command`, `write_file`, `replace_in_file`, `replace_lines`, `insert_after`, `final`. The canonical schemas live in `local_code/tool_specs.py`, which also renders prompt-JSON examples and Ollama/OpenAI-style native tool definitions. File ops use ripgrep (`rg`). All paths are validated by `resolve_path()` to stay within workdir. `search_web` uses DuckDuckGo Lite (no API key); `fetch_url` fetches a specific URL.

### Key files

| File | Role |
|------|------|
| `local_code/agent.py` | `LocalCodeAgent` (backend loop) + `LocalPartner` (orchestration) |
| `local_code/contracts.py` | Contract normalization, report parsing, JSON utilities |
| `local_code/tools.py` | All backend tool implementations |
| `local_code/tool_specs.py` | Canonical backend tool schemas, native tool definitions, and argument validation |
| `local_code/models.py` | Ollama HTTP client + reliability layer (structured output, retries, `keep_alive`, capability probes) |
| `local_code/model_profiles.py` | Model capability tiers + recommended-minimum standard (3060 12 GB target) |
| `local_code/eval.py` | Tool-loop reliability eval harness (`python -m local_code.eval`) |
| `local_code/config.py` | All defaults, limits, regex patterns, blocked commands |
| `local_code/cli.py` | REPL, slash commands, at-references, paste handling |
| `local_code/ui.py` | ANSI rendering, `Spinner`, markdown, diff display |
| `local_code/memory.py` | Per-repo memory read/write |
| `local_code/permissions.py` | Dangerous-command blocking, interactive confirm |

## Adding backend tools

1. Add a branch in `LocalCodeAgent.tool_result()` in `agent.py`
2. Add the tool to the schema line in `LocalCodeAgent.system_prompt()`
3. Implement the function in `tools.py`

## Output conventions

- Backend output is clipped to 12 000 chars (`ui.clip()`)
- UI summaries truncate to 220 chars (`summarize_text()`)
- Verbosity levels: `quiet` / `normal` / `debug` — debug shows raw JSON and tool output samples

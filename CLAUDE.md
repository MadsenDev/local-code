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
```

## Architecture

This is a **two-model AI coding CLI** backed by Ollama. The key architectural idea is a frontend/backend split — two locally-running LLMs with distinct roles, coordinated by `LocalPartner`.

### Dual-model design

- **Frontend** (`qwen3:8b` default): User-facing conversational model. In hybrid/agent mode, decides whether to answer directly or delegate to the backend. Also acts as a *complexity assessor* — before any backend tool loop runs (when models differ), the frontend makes a lightweight pre-flight call (`ollama_assess`) to decide whether to run the task with itself or escalate to the stronger backend model.
- **Backend** (`qwen3:14b` default): Headless code worker. Receives a *contract* (JSON object specifying task scope, permissions, edit policy) and executes a tool loop, emitting `{"tool":"...","args":{...}}` on every turn until it emits `{"tool":"final","args":{...}}`.

### Escalation flow

When `frontend_model != backend_model`, `LocalPartner._run_backend()` calls `_assess_complexity(contract)` before running the tool loop. The frontend model responds with `{"handle":"self"}` or `{"handle":"escalate","reason":"..."}`. Simple tasks (read-only, single-file, chat) run with the frontend model; complex tasks (multi-file edits, refactoring, bootstrapping) escalate to the backend model via `LocalCodeAgent.run_contract_with_model()`. On any parse failure the default is to escalate (safe fallback). When both models are the same, the assessment call is skipped entirely.

### Intent scaffold

Before `LocalPartner._run_backend()` starts delegated repo work, it runs a compact intent-analysis pass with the frontend model. The pass returns JSON describing the user's actual goal, non-goals, needed context, likely files, risks, and success criteria. `LocalPartner._contract_with_intent_scaffold()` attaches that data to the backend contract as `intent_analysis`, resolves likely files into `files_of_interest`, and adds guardrail constraints so the backend inspects relevant context before editing and avoids overreach. If the intent pass fails or returns invalid JSON, the original contract is used unchanged.

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
| `local_code/models.py` | Ollama HTTP client (`ollama_chat`) |
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

# Rist

**A local-first AI coding agent that feels at home on real developer hardware.**

Rist gives you a full-screen coding cockpit, repo-aware agent workflows, and a managed local llama.cpp setup without making you become an inference-runtime operator first.

```bash
pip install -e ".[tui,dev]"
rist setup
rist
```

## Why try Rist?

- **Local-first by default.** Rist prefers a managed local llama.cpp runtime, can use an external llama.cpp endpoint, and keeps Ollama plus OpenAI-compatible cloud providers available when you want them.
- **Designed for coding work, not demos.** Ask questions, inspect a repository, plan edits, approve changes, run commands, and keep a durable project memory.
- **A real terminal UI.** The Textual interface streams tokens, shows tool activity as it happens, opens runtime/doctor/benchmark result screens, and gives you approval modals before risky actions.
- **No manual model scavenger hunt for the default path.** `rist setup` checks your machine, picks from committed manifests, downloads verified assets, starts the managed runtime, and saves configuration.
- **Works with the hardware you actually have.** Adaptive routing avoids needless model swaps, diagnostics explain available RAM/VRAM, and benchmarks show real latency and throughput.
- **Portable provider model.** Use local llama.cpp, Ollama, OpenRouter, OpenAI, or another OpenAI-compatible endpoint with the same agent loop.

## What it feels like

Start Rist in a repository and ask it to understand, plan, or change code:

```text
> what does this repo do?
> plan how to add a settings screen
> implement the plan and show me the diff
```

Rist can stay conversational, automatically delegate code tasks to its backend tool loop, or run every turn as an agent task. It keeps the human in charge with explicit modes and approval flows.

## Quick start

### 1. Install

```bash
pip install -e ".[tui,dev]"
```

### 2. Configure the local runtime

```bash
rist setup
```

Setup is incremental and resumable. It reuses already verified runtime/model files and can be forced with `rist setup --force` when you intentionally want a clean reinstall.

### 3. Launch the TUI

```bash
rist
```

If Textual is not installed or stdout is not a TTY, Rist automatically falls back to the plain line-based REPL. You can request that mode directly with:

```bash
rist --no-tui
```

## Common commands

```bash
rist doctor                         # inspect hardware, models, and routing
rist benchmark --benchmark-runs 3   # measure latency and throughput
rist --prompt "what does this repo do?" # one-shot non-interactive prompt
rist --provider ollama              # use Ollama compatibility mode
rist --provider llamacpp --base-url http://127.0.0.1:8080/v1
```

Inside the TUI, use slash commands such as `/help`, `/status`, `/models`, `/mode`, `/plan`, `/apply`, `/ask`, `/context`, and `/quit`. Press Ctrl+K for the command palette.

## Provider options

| Provider | Use it when |
| --- | --- |
| `auto` | You want Rist to try managed llama.cpp, external llama.cpp, then Ollama/setup guidance. |
| `llamacpp` | You want the primary local GGUF path, managed or externally hosted. |
| `ollama` | You already have Ollama models and want compatibility. |
| `openrouter` | You want hosted models through OpenRouter. |
| `openai` | You want OpenAI or any OpenAI-compatible endpoint via `--base-url`. |

## Model recommendation

The recommended minimum standard is **Qwen2.5-Coder 7B** for coding reliability on common 12 GB GPU setups. Rist can also run lighter best-effort models or larger/heavier backends when your hardware and patience allow it.

See the model guide for detailed tiers, routing advice, and tuning knobs.

## Documentation

The README is intentionally short. The detailed docs live here:

- **[User guide](docs/user-guide.md)** — setup, runtime management, providers, TUI, slash commands, diagnostics, managed llama.cpp, and repository intelligence.
- **[Model guide](MODELS.md)** — recommended model tiers and tuning notes.
- **[Architecture notes](CLAUDE.md)** — deeper implementation and design context.

## Project status

Rist is actively evolving. The old `local-code` command remains as a temporary compatibility alias while the project transitions to the Rist name.


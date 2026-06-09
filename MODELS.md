# Model performance requirements

`local-code` is designed to work well across a wide range of local models, but
**below a certain capability it stops being reliable** — the backend tool loop
needs a model that can consistently emit a valid tool call, follow a contract,
and know when to stop. This document defines the recommended-minimum standard,
explains how the app adapts to weaker models, and how to measure a model
yourself.

The reference hardware is a single **NVIDIA RTX 3060 12 GB** at roughly Q4_K_M.
Everything below is sized to that card.

## The standard

| Tier | What it means | Examples (Q4_K_M on a 3060 12 GB) |
|------|---------------|-----------------------------------|
| **Recommended** | The "works properly" bar. Fits with full 16k context, fast, drives the JSON tool loop reliably. | `qwen2.5-coder:7b` ← **the standard**, `qwen3:8b`, `llama3.1:8b` |
| **Supported** | Works well, but near the VRAM ceiling (slower, less context headroom) or slightly less reliable on multi-file edits. | `qwen2.5-coder:14b` ← **ceiling for this card**, `qwen3:14b`, `gemma3:12b`, `qwen3:4b` |
| **Best-effort** | Runs; good for chat and inspection, flaky on multi-step edits. Below the standard. | `qwen3:1.7b`, `llama3.2:3b`, base/non-instruct checkpoints |
| **Unsupported** | Exceeds 12 GB at a usable quant (spills to CPU and crawls) or too small to drive tools. | `qwen2.5-coder:32b`, `gemma3:27b`, `*:70b`, MoE like `qwen3:30b-a3b` |

**Recommended minimum: `qwen2.5-coder:7b` for both roles.** It is the sweet
spot for a 12 GB card — it stays fully resident with a 16k context, responds
quickly, and emits clean tool calls. `qwen2.5-coder:14b` is the best quality the
card can host (single role, reduced context headroom).

Anything below ~4B still runs and is genuinely useful for chat and single-file
inspection, but you should expect occasional tool-loop failures on multi-step
edits — that's why it sits below the standard rather than being "unsupported".

## The dual-model VRAM trap

The architecture supports a small **frontend** (routing/conversation) and a
stronger **backend** (code work). On a 12 GB card this only helps if **both
models fit in VRAM at the same time** — otherwise every frontend↔backend switch
reloads weights from disk and each turn pays seconds of latency.

Two configurations that work on a 3060 12 GB:

- **One shared model** (simplest, recommended): `--model qwen2.5-coder:7b`.
  Same model for both roles, so the complexity-assessment hop is skipped
  entirely and nothing ever reloads.
- **Small frontend + 7B backend** that coexist:
  `--frontend-model qwen3:4b --backend-model qwen2.5-coder:7b`
  (~3 GB + ~5 GB ≈ 8 GB, both stay resident).

Avoid pairing two large distinct models (e.g. `qwen3:8b` + `qwen2.5-coder:14b` ≈
22 GB combined) on this card. `local-code` warns about this at startup.


## Experimental large GGUF/MoE models through llama.cpp

The local Ollama fit tiers above describe models expected to stay mostly on the
reference GPU. Larger GGUF and MoE models have a separate deployment class:
`provider: llamacpp`, `role: heavy_backend`, `recommended_routing: single`, and
`status: experimental`.

The built-in `qwen36-35b-a3b-llamacpp` profile targets repo analysis, debugging
logs, code review, planning, and multi-file reasoning. Its guidance is 8 GB
minimum / 12 GB recommended VRAM, 32 GB recommended / 64 GB ideal system RAM.
Avoid it for quick chat, rapid tool loops, or frequent model switching. Total
weights and KV cache still matter even when only a small subset of MoE experts
is active, so performance depends heavily on quant, RAM bandwidth, offload, and
context length.

**local-code does not replace llama.cpp. local-code manages and connects to
llama.cpp.** llama.cpp remains responsible for GGUF parsing, quantization,
GPU/CPU execution, expert placement, batching, and KV cache. local-code may
manage the external executable's lifecycle and model files after explicit
`model install/register/start/stop` commands. It never compiles llama.cpp or
implements inference. See the README for lifecycle commands, diagnostics,
benchmarks, and conservative RTX 3060 settings.

Adaptive routing pins a llama.cpp heavy backend for the complete heavy phase.
It intentionally avoids repeated `small → big → small → big` transitions. If a
small default model is used outside that phase, use it for initial routing and
optional final summarization—not alternating tool steps.

## How the app adapts to weaker models

These are automatic — you don't configure them:

- **Structured output.** Every call that must return JSON (tool actions,
  complexity assessment, intent analysis) is sent to Ollama with a JSON
  `format` constraint and temperature 0. This grammar-constrains decoding and
  near-eliminates the single biggest failure mode on weak models: malformed or
  chatty tool JSON. Older Ollama builds that reject a schema fall back to
  `format: "json"`, then to plain decoding, so it works on any version.
- **Few-shot tool examples.** Best-effort / medium-tier models get a couple of
  worked tool-call examples appended to the backend prompt; stronger models
  don't (they follow the schema from the description alone).
- **Deterministic decoding** for tool/JSON passes; a little warmth only for
  user-facing prose.
- **Retries with backoff** on transient Ollama errors (model still loading,
  connection blips, 5xx), and **`keep_alive`** so the model stays resident.
- **Bounded recovery.** Invalid tool calls, repeated actions, and step limits
  all return a structured report instead of hanging.

## Knobs

| Flag / env | Effect |
|------------|--------|
| `--model qwen2.5-coder:7b` | Set both roles to one model (recommended on 12 GB) |
| `--frontend-model` / `--backend-model` | Split the roles |
| `LOCAL_CODE_NUM_CTX=8192` | Lower the context window if a 14B model OOMs |
| `LOCAL_CODE_KEEP_ALIVE=30m` | How long Ollama keeps the model resident |
| `--no-preflight` | Skip the startup model/Ollama check |
| `/models` | Show tiers + the standard at any time in the REPL |

## Cloud providers (OpenRouter / OpenAI-compatible)

The tiers above are for **local** models on a 12 GB card. With
`--provider openrouter` (or any OpenAI-compatible endpoint), the model runs in
the cloud, so VRAM fit and the few-shot scaffolding don't apply — local-code
skips them automatically. Structured output, retries, and the tool loop work
identically. Pick any capable instruct/coder model the provider offers, e.g.:

```bash
export OPENROUTER_API_KEY=sk-or-...
local-code --provider openrouter --model qwen/qwen-2.5-coder-32b-instruct
```

## Measuring a model yourself

The eval harness runs a battery of safe, read-only tasks against a throwaway
repo and reports how reliably the model drives the tool loop:

```bash
python -m local_code.eval --model qwen2.5-coder:7b
python -m local_code.eval --model qwen3:4b --runs 3
```

It prints the model's tier, a reliability percentage, and a verdict. A model at
**≥95%** meets the standard; **75–95%** is usable but not flawless; **<75%** will
fail often and is below the bar.

## Managed llama.cpp workflow on RTX 3060-class hardware

llama.cpp is external software and performs all GGUF loading and inference.
local-code can register a GGUF and manage a `llama-server` process for
convenience, but it does not parse the model or perform inference itself.

```bash
local-code llama install --url https://example.invalid/llama-server
local-code model register qwen36 --path ~/Models/qwen.gguf --provider llamacpp
local-code model start qwen36 --gpu rtx3060
local-code llama doctor
local-code bench --provider llamacpp --model local
local-code llama logs --tail 50
local-code model stop qwen36
```

Managed logs and state live in `~/.local-code/runtimes/llamacpp/`. To run
manually instead, launch `llama-server` yourself and invoke local-code with
`--provider llamacpp --base-url http://127.0.0.1:8080/v1 --model local`.
local-code only has log visibility for servers it starts.

For an RTX 3060 12 GB, begin with the conservative 16k-context `rtx3060`
profile. Context above 32k is experimental on 12 GB VRAM, and large MoE GGUFs
may also need substantial system RAM and CPU offload. Do not alternate
frequently between a large llama.cpp backend and another GPU-resident model on
8–12 GB cards: repeated eviction and reloads can dominate request latency.

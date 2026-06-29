"""Hardware/provider diagnostics and repeatable inference benchmarks."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .hardware import detect_hardware, recommend_routing
from .llama_runtime import server_status


@dataclass
class BenchmarkSample:
    elapsed_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    tokens_per_second: float | None
    load_s: float | None


def doctor_report(provider, frontend_model, backend_model, num_ctx=16384):
    is_llamacpp = getattr(provider, "name", "") == "llamacpp"
    base_url = getattr(provider, "base_url", None) if provider.is_local else None
    hardware = detect_hardware(None if is_llamacpp else base_url)
    available = provider.available()
    listed_models = sorted(provider.list_models()) if available and hasattr(provider, "list_models") else []
    models = {}
    for model in dict.fromkeys([frontend_model, backend_model]):
        models[model] = provider.model_available(model) if available else None
    routing = recommend_routing(hardware, frontend_model, backend_model, num_ctx) if provider.is_local else {
        "mode": "single" if frontend_model == backend_model else "dual",
        "reason": "Cloud provider; local residency is not applicable.",
    }
    report = {
        "provider": provider.describe(),
        "provider_type": getattr(provider, "name", "provider"),
        "base_url": base_url,
        "provider_available": available,
        "models_endpoint": available,
        "reported_models": listed_models,
        "models": models,
        "hardware": hardware.to_dict(),
        "routing_recommendation": routing,
        "num_ctx": num_ctx,
    }
    if is_llamacpp:
        report.update(_llamacpp_doctor(provider, backend_model, available, listed_models))
        report["managed_runtime"] = server_status()
        report["configuration"] = {"provider": "llamacpp", "base_url": base_url, "model": backend_model}
    return report


def _llamacpp_doctor(provider, model, available, listed_models):
    chat_ok = False
    chat_error = None
    response_preview = None
    chat_latency_s = None
    if available:
        test_model = model if model in listed_models or not listed_models else listed_models[0]
        try:
            started = time.perf_counter()
            result = provider.chat_result(
                test_model,
                [{"role": "user", "content": "Reply with exactly: ready"}],
                temperature=0.0,
                num_ctx=256,
                timeout=30,
            )
            chat_latency_s = round(time.perf_counter() - started, 3)
            response_preview = result.content[:200]
            chat_ok = bool(result.content or result.tool_calls)
            if not chat_ok:
                chat_error = "The endpoint returned no assistant content or tool calls."
        except Exception as exc:  # noqa: BLE001 - doctor reports errors instead of raising
            chat_error = f"{type(exc).__name__}: {exc}"
    metadata = provider.server_metadata() if available and hasattr(provider, "server_metadata") else {}
    if not available:
        readiness = "unreachable"
    elif not chat_ok:
        readiness = "responding_incorrectly"
    elif model not in listed_models and listed_models and model != "local":
        readiness = "misconfigured"
    else:
        readiness = "ready"
    return {
        "readiness": readiness,
        "chat_completions": chat_ok,
        "chat_error": chat_error,
        "chat_response_preview": response_preview,
        "chat_latency_s": chat_latency_s,
        "server_metadata": metadata,
    }


def _ns_seconds(value):
    return round(value / 1_000_000_000, 3) if isinstance(value, (int, float)) else None


def _estimated_tokens(text):
    # Diagnostics only: OpenAI-compatible streaming does not always return
    # usage. Four characters/token is a transparent, conservative estimate.
    return max(1, round(len(text) / 4)) if text else 0


def _benchmark_once(provider, model, prompt, num_ctx):
    messages = [{"role": "user", "content": prompt}]
    started = time.perf_counter()
    first_token = None
    chunks = []
    if hasattr(provider, "stream"):
        try:
            for chunk in provider.stream(model, messages, temperature=0.0):
                if first_token is None:
                    first_token = time.perf_counter()
                chunks.append(chunk)
        except (NotImplementedError, AttributeError):
            chunks = []
    if chunks:
        finished = time.perf_counter()
        elapsed = finished - started
        ttft = (first_token - started) if first_token else None
        output_tokens = _estimated_tokens("".join(chunks))
        generation_s = max(0.001, finished - (first_token or started))
        tps = output_tokens / generation_s
        return BenchmarkSample(round(elapsed, 3), round(ttft, 3) if ttft is not None else None, _estimated_tokens(prompt), output_tokens, round(tps, 2), None)

    # Non-streaming fallback is useful for providers/tests without SSE. It can
    # report total latency and throughput, but not genuine first-token latency.
    started = time.perf_counter()
    result = provider.chat_result(model, messages, temperature=0.0, num_ctx=num_ctx)
    elapsed = time.perf_counter() - started
    raw = result.raw or {}
    usage = raw.get("usage") or {}
    prompt_tokens = raw.get("prompt_eval_count", usage.get("prompt_tokens", _estimated_tokens(prompt)))
    output_tokens = raw.get("eval_count", usage.get("completion_tokens", _estimated_tokens(result.content)))
    eval_duration = _ns_seconds(raw.get("eval_duration"))
    tps = output_tokens / eval_duration if output_tokens and eval_duration else (output_tokens / elapsed if output_tokens else None)
    total_duration = _ns_seconds(raw.get("total_duration"))
    load_duration = _ns_seconds(raw.get("load_duration"))
    prompt_duration = _ns_seconds(raw.get("prompt_eval_duration"))
    ttft = max(0.0, total_duration - eval_duration) if total_duration is not None and eval_duration is not None else None
    if ttft is None and (load_duration is not None or prompt_duration is not None):
        ttft = (load_duration or 0) + (prompt_duration or 0)
    return BenchmarkSample(round(elapsed, 3), ttft, prompt_tokens, output_tokens, round(tps, 2) if tps else None, load_duration)


def benchmark_model(provider, model, runs=3, num_ctx=16384, long_context=False):
    base_url = getattr(provider, "base_url", None) if provider.is_local and getattr(provider, "name", "") != "llamacpp" else None
    hardware_before = detect_hardware(base_url)
    cases = {
        "small": "Reply in one short sentence: what makes a coding assistant reliable?",
        "medium": ("Review this hypothetical repository architecture and list the three highest-risk integration points. " * 40),
    }
    if long_context:
        cases["long_context"] = ("Context line: parser, router, provider, tests, and documentation must remain consistent. " * 500)
    case_reports = {}
    all_samples = []
    for name, prompt in cases.items():
        samples = [_benchmark_once(provider, model, prompt, num_ctx) for _ in range(runs)]
        all_samples.extend(samples)
        case_reports[name] = {
            "prompt_estimated_tokens": _estimated_tokens(prompt),
            "runs": [asdict(sample) for sample in samples],
            "median_latency_s": round(statistics.median(s.elapsed_s for s in samples), 3),
        }
    tps_values = [sample.tokens_per_second for sample in all_samples if sample.tokens_per_second is not None]
    ttft_values = [sample.ttft_s for sample in all_samples if sample.ttft_s is not None]
    median_elapsed = statistics.median(s.elapsed_s for s in all_samples)
    median_tps = statistics.median(tps_values) if tps_values else None
    assessment = _benchmark_assessment(median_elapsed, median_tps)
    return {
        "model": model,
        "provider": provider.describe(),
        "num_ctx": num_ctx,
        "hardware_before": hardware_before.to_dict(),
        "hardware_after": detect_hardware(base_url).to_dict(),
        "runs": case_reports["small"]["runs"],  # backward-compatible primary samples
        "cases": case_reports,
        "summary": {
            "median_elapsed_s": round(median_elapsed, 3),
            "median_ttft_s": round(statistics.median(ttft_values), 3) if ttft_values else None,
            "median_tokens_per_second": round(median_tps, 2) if median_tps is not None else None,
        },
        "suitability": assessment,
    }


def _benchmark_assessment(latency, tps):
    if tps is None:
        return {"normal_chat": "unknown", "coding_edits": "unknown", "repo_analysis": "unknown", "heavy_reasoning": "unknown"}
    return {
        "normal_chat": "good" if latency <= 5 and tps >= 12 else "usable" if tps >= 6 else "slow",
        "coding_edits": "good" if tps >= 10 else "usable" if tps >= 4 else "slow",
        "repo_analysis": "good" if tps >= 6 else "usable" if tps >= 2 else "slow",
        "heavy_reasoning": "good" if tps >= 4 else "usable" if tps >= 1 else "slow",
    }


def format_doctor(report):
    hardware = report["hardware"]
    lines = ["Rist doctor"]
    if report.get("provider_type") == "llamacpp":
        runtime = report.get("managed_runtime") or {}
        state = runtime.get("state", "stopped")
        model_path = runtime.get("model_path")
        lines.extend([
            "",
            "Managed runtime:",
            f"- state file: {'present' if runtime.get('state_file_present') else 'missing'}",
            f"- PID: {runtime.get('pid', 'none')} ({'running' if runtime.get('pid_running') else state})",
            f"- status: {state}",
            f"- readiness: {report.get('readiness', 'unknown')}",
            f"- executable path: {runtime.get('executable', 'not recorded')}",
            f"- model path: {model_path or 'not recorded'}" + (" (missing)" if model_path and not Path(model_path).is_file() else ""),
            f"- log path: {runtime.get('log_path', 'not recorded')}",
            "",
            "HTTP provider:",
            f"- base URL: {report.get('base_url') or 'not configured'}",
            f"- base URL reachable: {'yes' if report.get('provider_available') else 'no'}",
            f"- /v1/models: {'responded' if report.get('models_endpoint') else 'failed'}",
            "- model name returned: " + (", ".join(report.get("reported_models") or []) or "none"),
            "",
            "Chat completion:",
            f"- tiny test prompt: {'succeeded' if report.get('chat_completions') else 'failed'}",
            f"- latency: {report['chat_latency_s']} s" if report.get("chat_latency_s") is not None else "- latency: not measured",
        ])
        if report.get("chat_error"):
            lines.append(f"- error: {report['chat_error']}")
        config = report.get("configuration") or {}
        lines.extend([
            "",
            "Configuration:",
            f"- configured provider name: {config.get('provider') or 'missing'}",
            f"- configured base_url: {config.get('base_url') or 'missing'}",
            f"- configured model: {config.get('model') or 'missing'}",
        ])
        if state == "stale":
            lines.append("- action: stale PID/state detected; run `rist model stop` to clean it up before restarting.")
        if not report.get("provider_available"):
            lines.extend([
                "",
                "The llama.cpp HTTP endpoint is unreachable. The process may be stopped, still loading, or using a different port/base URL.",
                "Next: run `rist model status`, then `rist llama logs --tail 50`.",
                "To start a registered model: rist model start qwen36 --gpu rtx3060",
            ])
        elif report.get("models_endpoint") and not report.get("chat_completions"):
            lines.append("The models endpoint works, but chat completions fail; inspect the chat error and managed log.")
    else:
        lines.extend([
            f"Provider: {report['provider']} ({'reachable' if report['provider_available'] else 'unreachable'})",
            f"Base URL: {report['base_url']}" if report.get("base_url") else "",
        ])
    lines.append("")
    lines.append(f"System RAM: {hardware['system_ram_gb'] or 'unknown'} GB")
    for gpu in hardware["gpus"]:
        lines.append(f"GPU: {gpu['name']} — {gpu['vram_total_gb'] or 'unknown'} GB VRAM ({gpu['vram_used_gb'] or 0} GB used)")
    if not hardware["gpus"]:
        lines.append("GPU: not detected")
    for model, available in report["models"].items():
        status = "available" if available is True else "missing" if available is False else "unknown"
        lines.append(f"Model: {model} — {status}")
    route = report["routing_recommendation"]
    lines.extend([f"Recommended routing: {route['mode']}", f"Reason: {route['reason']}", f"Context window: {report['num_ctx']:,} tokens"])
    return "\n".join(line for line in lines if line is not None)


def _performance_summary(summary):
    tps = summary.get("median_tokens_per_second")
    ttft = summary.get("median_ttft_s")
    if tps is None:
        return "Needs tuning"
    if tps >= 20 and (ttft is None or ttft <= 2):
        return "Excellent"
    if tps >= 8 and (ttft is None or ttft <= 5):
        return "Good"
    return "Needs tuning"


def format_benchmark(report):
    summary = report["summary"]
    lines = [f"Benchmark: {report['model']}", f"Provider: {report['provider']}", f"Context: {report['num_ctx']:,} tokens"]
    for name, case in report["cases"].items():
        lines.append(f"{name.replace('_', ' ').title()} prompt latency: {case['median_latency_s']} s ({case['prompt_estimated_tokens']} estimated input tokens)")
    if summary["median_ttft_s"] is not None:
        lines.append(f"Median TTFT: {summary['median_ttft_s']} s")
    else:
        lines.append("Median TTFT: unavailable (provider did not expose streaming timing)")
    if summary["median_tokens_per_second"] is not None:
        lines.append(f"Median generation: {summary['median_tokens_per_second']} tok/s")
    lines.append("Suitability: " + ", ".join(f"{name.replace('_', ' ')}={value}" for name, value in report["suitability"].items()))
    lines.append("")
    lines.append("Performance summary")
    lines.append(_performance_summary(summary))
    return "\n".join(lines)


def to_json(report):
    return json.dumps(report, indent=2, sort_keys=True)

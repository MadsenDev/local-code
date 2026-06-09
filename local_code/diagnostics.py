"""Hardware/provider diagnostics and repeatable local inference benchmarks."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass

from .hardware import detect_hardware, recommend_routing


@dataclass
class BenchmarkSample:
    elapsed_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    tokens_per_second: float | None
    load_s: float | None


def doctor_report(provider, frontend_model, backend_model, num_ctx=16384):
    base_url = getattr(provider, "base_url", None) if provider.is_local else None
    hardware = detect_hardware(base_url)
    available = provider.available()
    models = {}
    for model in dict.fromkeys([frontend_model, backend_model]):
        models[model] = provider.model_available(model) if available else None
    routing = recommend_routing(hardware, frontend_model, backend_model, num_ctx) if provider.is_local else {
        "mode": "single" if frontend_model == backend_model else "dual",
        "reason": "Cloud provider; local residency is not applicable.",
    }
    return {
        "provider": provider.describe(),
        "provider_available": available,
        "models": models,
        "hardware": hardware.to_dict(),
        "routing_recommendation": routing,
        "num_ctx": num_ctx,
    }


def _ns_seconds(value):
    return round(value / 1_000_000_000, 3) if isinstance(value, (int, float)) else None


def benchmark_model(provider, model, runs=3, num_ctx=16384):
    base_url = getattr(provider, "base_url", None) if provider.is_local else None
    hardware_before = detect_hardware(base_url)
    samples = []
    messages = [{"role": "user", "content": "Reply with exactly five words describing reliable local software."}]
    for _ in range(runs):
        started = time.perf_counter()
        result = provider.chat_result(model, messages, temperature=0.0, num_ctx=num_ctx)
        elapsed = time.perf_counter() - started
        raw = result.raw or {}
        usage = raw.get("usage") or {}
        prompt_tokens = raw.get("prompt_eval_count", usage.get("prompt_tokens"))
        output_tokens = raw.get("eval_count", usage.get("completion_tokens"))
        eval_duration = _ns_seconds(raw.get("eval_duration"))
        tps = output_tokens / eval_duration if output_tokens and eval_duration else (output_tokens / elapsed if output_tokens else None)
        total_duration = _ns_seconds(raw.get("total_duration"))
        load_duration = _ns_seconds(raw.get("load_duration"))
        prompt_duration = _ns_seconds(raw.get("prompt_eval_duration"))
        ttft = None
        if total_duration is not None and eval_duration is not None:
            ttft = max(0.0, total_duration - eval_duration)
        elif load_duration is not None or prompt_duration is not None:
            ttft = (load_duration or 0) + (prompt_duration or 0)
        samples.append(BenchmarkSample(round(elapsed, 3), ttft, prompt_tokens, output_tokens, round(tps, 2) if tps else None, load_duration))
    tps_values = [sample.tokens_per_second for sample in samples if sample.tokens_per_second is not None]
    ttft_values = [sample.ttft_s for sample in samples if sample.ttft_s is not None]
    hardware_after = detect_hardware(base_url)
    return {
        "model": model,
        "num_ctx": num_ctx,
        "hardware_before": hardware_before.to_dict(),
        "hardware_after": hardware_after.to_dict(),
        "runs": [asdict(sample) for sample in samples],
        "summary": {
            "median_elapsed_s": round(statistics.median(s.elapsed_s for s in samples), 3),
            "median_ttft_s": round(statistics.median(ttft_values), 3) if ttft_values else None,
            "median_tokens_per_second": round(statistics.median(tps_values), 2) if tps_values else None,
        },
    }


def format_doctor(report):
    hardware = report["hardware"]
    lines = [
        "Local-Code doctor",
        f"Provider: {report['provider']} ({'reachable' if report['provider_available'] else 'unreachable'})",
        f"System RAM: {hardware['system_ram_gb'] or 'unknown'} GB",
    ]
    if hardware["gpus"]:
        for gpu in hardware["gpus"]:
            lines.append(f"GPU: {gpu['name']} — {gpu['vram_total_gb'] or 'unknown'} GB VRAM ({gpu['vram_used_gb'] or 0} GB used)")
    else:
        lines.append("GPU: not detected")
    if hardware["loaded_models"]:
        lines.append("Loaded models: " + ", ".join(hardware["loaded_models"]))
    for model, available in report["models"].items():
        state = "available" if available is True else "missing" if available is False else "unknown"
        lines.append(f"Model: {model} — {state}")
    route = report["routing_recommendation"]
    lines.extend([f"Recommended routing: {route['mode']}", f"Reason: {route['reason']}", f"Context window: {report['num_ctx']:,} tokens"])
    return "\n".join(lines)


def format_benchmark(report):
    summary = report["summary"]
    lines = [f"Benchmark: {report['model']}", f"Context: {report['num_ctx']:,} tokens", f"Runs: {len(report['runs'])}", f"Median elapsed: {summary['median_elapsed_s']} s"]
    gpu_after = report["hardware_after"].get("gpus") or []
    if gpu_after:
        used = ", ".join(f"{gpu['name']}: {gpu['vram_used_gb']} / {gpu['vram_total_gb']} GB" for gpu in gpu_after)
        lines.append(f"VRAM after: {used}")
    if summary["median_ttft_s"] is not None:
        lines.append(f"Median TTFT: {summary['median_ttft_s']} s")
    if summary["median_tokens_per_second"] is not None:
        lines.append(f"Median generation: {summary['median_tokens_per_second']} tok/s")
    return "\n".join(lines)


def to_json(report):
    return json.dumps(report, indent=2, sort_keys=True)

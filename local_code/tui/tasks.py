"""Background runtime task helpers for the Rist TUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from local_code.config import load_runtime_config
from local_code.diagnostics import benchmark_model, doctor_report
from local_code.llama_runtime import server_status, start_server, stop_server


class RuntimeTask(str, Enum):
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    STATUS = "status"
    DOCTOR = "doctor"
    BENCHMARK = "benchmark"


@dataclass(frozen=True)
class RuntimeTaskResult:
    task: RuntimeTask
    title: str
    report: dict[str, Any]


_TASK_TITLES = {
    RuntimeTask.START: "Start managed runtime",
    RuntimeTask.STOP: "Stop managed runtime",
    RuntimeTask.RESTART: "Restart managed runtime",
    RuntimeTask.STATUS: "Runtime status",
    RuntimeTask.DOCTOR: "Run doctor",
    RuntimeTask.BENCHMARK: "Run benchmark",
}

_TASK_PROGRESS = {
    RuntimeTask.START: "Starting runtime...",
    RuntimeTask.STOP: "Stopping runtime...",
    RuntimeTask.RESTART: "Restarting runtime...",
    RuntimeTask.STATUS: "Checking runtime status...",
    RuntimeTask.DOCTOR: "Running doctor...",
    RuntimeTask.BENCHMARK: "Benchmarking model...",
}

_TASK_COMPLETE = {
    RuntimeTask.START: "Runtime started",
    RuntimeTask.STOP: "Runtime stopped",
    RuntimeTask.RESTART: "Runtime restarted",
    RuntimeTask.STATUS: "Runtime status ready",
    RuntimeTask.DOCTOR: "Doctor finished",
    RuntimeTask.BENCHMARK: "Benchmark finished",
}


def task_title(task: RuntimeTask | str) -> str:
    return _TASK_TITLES[RuntimeTask(task)]


def task_progress_message(task: RuntimeTask | str) -> str:
    return _TASK_PROGRESS[RuntimeTask(task)]


def task_complete_message(task: RuntimeTask | str) -> str:
    return _TASK_COMPLETE[RuntimeTask(task)]


def run_runtime_task(task: RuntimeTask | str, partner) -> RuntimeTaskResult:
    """Run one runtime/diagnostic action and return structured data."""

    runtime_task = RuntimeTask(task)
    if runtime_task == RuntimeTask.START:
        cfg = load_runtime_config().get("llamacpp", {})
        start_server(cfg.get("profile", "qwen2.5-coder-7b"), cfg.get("gpu_profile", "cpu"), port=int(cfg.get("port", 8080)))
        report = server_status()
    elif runtime_task == RuntimeTask.STOP:
        stop_report = stop_server()
        report = server_status()
        report["message"] = stop_report.get("message", "Managed runtime stopped.")
    elif runtime_task == RuntimeTask.RESTART:
        stop_server()
        cfg = load_runtime_config().get("llamacpp", {})
        start_server(cfg.get("profile", "qwen2.5-coder-7b"), cfg.get("gpu_profile", "cpu"), port=int(cfg.get("port", 8080)))
        report = server_status()
    elif runtime_task == RuntimeTask.STATUS:
        report = server_status()
    elif runtime_task == RuntimeTask.DOCTOR:
        report = doctor_report(partner.provider, partner.frontend_model, partner.backend_model, partner.context_limit)
    else:
        report = benchmark_model(partner.provider, partner.backend_model, num_ctx=partner.context_limit)
    return RuntimeTaskResult(runtime_task, task_title(runtime_task), report)


def benchmark_rating(report: dict[str, Any]) -> str:
    """Return a simple label based only on measured benchmark summary data."""

    summary = report.get("summary") or {}
    tps = summary.get("median_tokens_per_second")
    latency = summary.get("median_elapsed_s")
    ttft = summary.get("median_ttft_s")
    if tps is None and latency is None and ttft is None:
        return "Poor"
    if tps is not None and tps >= 20 and (ttft is None or ttft <= 2) and (latency is None or latency <= 5):
        return "Excellent"
    if tps is not None and tps >= 8 and (ttft is None or ttft <= 5) and (latency is None or latency <= 10):
        return "Good"
    if (tps is not None and tps >= 2) or (latency is not None and latency <= 30):
        return "Needs tuning"
    return "Poor"

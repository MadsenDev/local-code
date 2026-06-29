"""Dedicated runtime, doctor, and benchmark result screens."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from local_code.diagnostics import format_doctor
from local_code.tui.tasks import benchmark_rating


def value_text(value) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def runtime_status_table(report: dict) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("State", value_text(report.get("state")))
    table.add_row("Provider", value_text(report.get("provider", "llamacpp")))
    table.add_row("Base URL", value_text(report.get("base_url")))
    table.add_row("Model/Profile", value_text(report.get("profile") or report.get("model") or report.get("model_path")))
    table.add_row("PID", value_text(report.get("pid")))
    table.add_row("Log path", value_text(report.get("log_path")))
    health = report.get("health") or ("running" if report.get("pid_running") else report.get("state"))
    table.add_row("Health", value_text(health))
    table.add_row("Suggested next action", runtime_next_action(report))
    return table


def runtime_next_action(report: dict) -> str:
    state = report.get("state")
    if state in {"running", "ready"} or report.get("pid_running"):
        return "Run doctor or benchmark from Ctrl+K."
    if state == "stale":
        return "Stop managed runtime, then start it again."
    return "Start managed runtime from Ctrl+K."


def doctor_results_table(report: dict) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Provider availability", "available" if report.get("provider_available") else "unavailable")
    table.add_row("Readiness", value_text(report.get("readiness", "unknown")))
    table.add_row("Model endpoint", "responded" if report.get("models_endpoint") else "failed")
    table.add_row("Chat completions", "succeeded" if report.get("chat_completions") else "failed")
    table.add_row("Latency", f"{report['chat_latency_s']} s" if report.get("chat_latency_s") is not None else "not measured")
    config = report.get("configuration") or {}
    table.add_row("Configuration", ", ".join(f"{k}={v}" for k, v in config.items()) or value_text(report.get("provider")))
    errors = []
    if report.get("chat_error"):
        errors.append(str(report["chat_error"]))
    if not report.get("provider_available"):
        errors.append("Provider endpoint is unreachable; check runtime status and logs.")
    table.add_row("Actionable errors", "\n".join(errors) if errors else "none")
    return table


def benchmark_results_table(report: dict) -> Table:
    summary = report.get("summary") or {}
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Model", value_text(report.get("model")))
    table.add_row("Provider", value_text(report.get("provider")))
    table.add_row("Context", f"{report.get('num_ctx'):,} tokens" if report.get("num_ctx") else "not available")
    table.add_row("Median latency", f"{summary.get('median_elapsed_s')} s" if summary.get("median_elapsed_s") is not None else "not measured")
    table.add_row("Median TTFT", f"{summary.get('median_ttft_s')} s" if summary.get("median_ttft_s") is not None else "not measured")
    table.add_row("Median tokens/sec", value_text(summary.get("median_tokens_per_second")))
    suitability = report.get("suitability") or {}
    table.add_row("Suitability", ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in suitability.items()) or "not measured")
    table.add_row("Final label", benchmark_rating(report))
    return table


def benchmark_cases_text(report: dict) -> Text:
    text = Text()
    for name, case in (report.get("cases") or {}).items():
        text.append(f"{name.replace('_', ' ').title()}: ", style="bold")
        text.append(f"median {case.get('median_latency_s')} s, {case.get('prompt_estimated_tokens')} estimated input tokens\n")
    return text or Text("No per-case results available.", style="dim")


class _ResultScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close"), Binding("q", "close", "Close")]

    def __init__(self, report: dict):
        super().__init__()
        self.report = report

    def action_close(self) -> None:
        self.dismiss(None)


class RuntimeStatusScreen(_ResultScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="result-box"):
            yield Label("Runtime Status", classes="result-title")
            yield Static(runtime_status_table(self.report), classes="result-body")
            yield Static("Esc closes", classes="result-hint")


class DoctorResultsScreen(_ResultScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="result-box"):
            yield Label("Doctor Results", classes="result-title")
            yield Static(doctor_results_table(self.report), classes="result-body")
            yield Static(Text(format_doctor(self.report), style="dim"), classes="result-details")
            yield Static("Esc closes", classes="result-hint")


class BenchmarkResultsScreen(_ResultScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="result-box"):
            yield Label("Benchmark Results", classes="result-title")
            yield Static(benchmark_results_table(self.report), classes="result-body")
            yield Static(benchmark_cases_text(self.report), classes="result-details")
            yield Static("Esc closes", classes="result-hint")

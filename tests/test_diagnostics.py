from local_code.diagnostics import benchmark_model, doctor_report, format_benchmark, format_doctor
from local_code.providers import ChatResult


class FakeProvider:
    is_local = False

    def describe(self):
        return "fake"

    def available(self):
        return True

    def model_available(self, model):
        return True

    def chat_result(self, model, messages, **kwargs):
        return ChatResult("local software stays reliable", raw={"usage": {"prompt_tokens": 8, "completion_tokens": 5}})


def test_doctor_reports_cloud_provider_without_local_routing_probe():
    report = doctor_report(FakeProvider(), "front", "back")
    assert report["provider_available"] is True
    assert report["routing_recommendation"]["mode"] == "dual"


def test_benchmark_reports_elapsed_and_fallback_throughput():
    report = benchmark_model(FakeProvider(), "model", runs=2)
    assert len(report["runs"]) == 2
    assert report["summary"]["median_elapsed_s"] >= 0
    assert report["summary"]["median_tokens_per_second"] is not None
    assert "Benchmark: model" in format_benchmark(report)


class FakeLlamaProvider(FakeProvider):
    name = "llamacpp"
    is_local = True
    base_url = "http://127.0.0.1:8080/v1"

    def list_models(self):
        return {"local"}

    def server_metadata(self):
        return {"health": {"status": "ok"}}


def test_llamacpp_doctor_checks_models_chat_and_metadata():
    report = doctor_report(FakeLlamaProvider(), "local", "local")
    assert report["readiness"] == "ready"
    assert report["models_endpoint"] is True
    assert report["chat_completions"] is True
    assert report["reported_models"] == ["local"]
    assert "health" in report["server_metadata"]


def test_benchmark_includes_small_and_medium_latency():
    report = benchmark_model(FakeProvider(), "model", runs=1, long_context=True)
    assert set(report["cases"]) == {"small", "medium", "long_context"}
    assert set(report["suitability"]) == {"normal_chat", "coding_edits", "repo_analysis", "heavy_reasoning"}


def test_llamacpp_doctor_formats_separate_sections_and_chat_latency(monkeypatch):
    monkeypatch.setattr("local_code.diagnostics.server_status", lambda: {
        "state": "running", "managed": True, "state_file_present": True, "pid_running": True,
        "pid": 22, "executable": "/bin/llama-server", "model_path": "/missing/model.gguf", "log_path": "/tmp/server.log",
    })
    report = doctor_report(FakeLlamaProvider(), "local", "local")
    output = format_doctor(report)
    assert "Managed runtime:" in output
    assert "HTTP provider:" in output
    assert "Chat completion:" in output
    assert "Configuration:" in output
    assert "model path: /missing/model.gguf (missing)" in output
    assert report["chat_latency_s"] is not None


def test_llamacpp_doctor_reports_stale_and_unreachable(monkeypatch):
    class Unreachable(FakeLlamaProvider):
        def available(self):
            return False

    monkeypatch.setattr("local_code.diagnostics.server_status", lambda: {
        "state": "stale", "managed": True, "state_file_present": True, "pid_running": False,
        "pid": 99, "log_path": "/tmp/server.log",
    })
    output = format_doctor(doctor_report(Unreachable(), "wrong-model", "wrong-model"))
    assert "status: stale" in output
    assert "base URL reachable: no" in output
    assert "rist llama logs --tail 50" in output
    assert "stale PID/state detected" in output


def test_llamacpp_doctor_distinguishes_models_from_chat_failure(monkeypatch):
    class BrokenChat(FakeLlamaProvider):
        def chat_result(self, *args, **kwargs):
            raise RuntimeError("template failure")

    monkeypatch.setattr("local_code.diagnostics.server_status", lambda: {"state": "stopped", "managed": False, "state_file_present": False})
    output = format_doctor(doctor_report(BrokenChat(), "local", "local"))
    assert "/v1/models: responded" in output
    assert "tiny test prompt: failed" in output
    assert "models endpoint works, but chat completions fail" in output


def test_llamacpp_doctor_reports_stopped_runtime(monkeypatch):
    monkeypatch.setattr("local_code.diagnostics.server_status", lambda: {
        "state": "stopped", "managed": False, "state_file_present": False, "log_path": "/tmp/server.log",
    })
    output = format_doctor(doctor_report(FakeLlamaProvider(), "local", "local"))
    assert "status: stopped" in output
    assert "state file: missing" in output


def test_llamacpp_doctor_reports_model_misconfiguration(monkeypatch):
    monkeypatch.setattr("local_code.diagnostics.server_status", lambda: {
        "state": "running", "managed": True, "state_file_present": True, "pid_running": True,
    })
    report = doctor_report(FakeLlamaProvider(), "wrong", "wrong")
    assert report["readiness"] == "misconfigured"
    assert "readiness: misconfigured" in format_doctor(report)

from local_code.diagnostics import benchmark_model, doctor_report, format_benchmark
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

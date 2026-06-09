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

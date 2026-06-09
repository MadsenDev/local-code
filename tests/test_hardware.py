from local_code.hardware import GPUInfo, HardwareInfo, estimate_model_vram_gb, recommend_routing
from local_code.routing import resolve_model_routing


def hardware(vram, count=1):
    return HardwareInfo(32.0, [GPUInfo(f"GPU {i}", vram, 0, "nvidia") for i in range(count)])


def test_estimates_q4_model_footprint_from_tag():
    assert estimate_model_vram_gb("qwen2.5-coder:7b") > 5
    assert estimate_model_vram_gb("unknown") is None


def test_adaptive_recommends_single_when_models_do_not_co_reside():
    decision = recommend_routing(hardware(8), "qwen3:4b", "qwen2.5-coder:14b")
    assert decision["mode"] == "single"
    assert "reload cost" in decision["reason"]


def test_adaptive_allows_dual_when_models_fit():
    decision = recommend_routing(hardware(24), "qwen3:4b", "qwen2.5-coder:7b")
    assert decision["mode"] == "dual"


def test_single_mode_shares_frontend_model_when_explicit():
    frontend, backend, decision = resolve_model_routing("single", True, hardware(12), "talker:4b", "coder:7b")
    assert frontend == backend == "coder:7b"
    assert decision["mode"] == "single"


def test_adaptive_single_prefers_backend_capability():
    frontend, backend, decision = resolve_model_routing("adaptive", True, hardware(8), "talker:4b", "coder:14b")
    assert frontend == backend == "coder:14b"
    assert decision["mode"] == "single"

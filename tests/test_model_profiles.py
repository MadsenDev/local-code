from local_code.model_profiles import advisory_lines, classify_model


class TestClassifyModel:
    def test_seven_b_coder_is_recommended(self):
        p = classify_model("qwen2.5-coder:7b")
        assert p.params_b == 7.0
        assert p.is_coder is True
        assert p.fit == "comfortable"
        assert p.reliability == "high"
        assert p.tier == "recommended"
        assert p.meets_standard is True

    def test_eight_b_instruct_is_recommended(self):
        p = classify_model("qwen3:8b")
        assert p.tier == "recommended"
        assert p.reliability == "high"

    def test_fourteen_b_fits_tight_but_supported(self):
        p = classify_model("qwen2.5-coder:14b")
        assert p.fit == "tight"
        assert p.reliability == "high"
        assert p.tier == "supported"
        assert p.meets_standard is True

    def test_four_b_is_best_effort_floor(self):
        p = classify_model("qwen3:4b")
        assert p.reliability == "medium"
        assert p.tier == "supported"

    def test_tiny_model_is_best_effort(self):
        p = classify_model("llama3.2:1b")
        assert p.params_b == 1.0
        assert p.reliability == "low"
        assert p.tier == "best_effort"
        assert p.use_few_shot is True
        assert p.trim_prompt is True

    def test_thirtytwo_b_exceeds_12gb(self):
        p = classify_model("qwen2.5-coder:32b")
        assert p.fit == "exceeds"
        assert p.tier == "unsupported"
        assert p.meets_standard is False

    def test_moe_uses_active_params_for_reliability_and_total_for_fit(self):
        p = classify_model("qwen3:30b-a3b")
        assert p.params_b == 30.0
        assert p.active_b == 3.0
        assert p.fit == "exceeds"  # 30B total won't fit 12 GB
        assert p.tier == "unsupported"

    def test_base_model_flagged_low_reliability(self):
        p = classify_model("llama3.1-base:8b")
        assert p.is_base is True
        assert p.reliability == "low"
        assert p.tier == "best_effort"

    def test_unknown_size_is_handled(self):
        p = classify_model("some-custom-model")
        assert p.params_b is None
        assert p.fit == "unknown"

    def test_recommended_model_skips_few_shot(self):
        assert classify_model("qwen2.5-coder:7b").use_few_shot is False


class TestAdvisory:
    def test_warns_about_dual_model_vram_trap(self):
        lines = advisory_lines("qwen3:8b", "qwen2.5-coder:14b")
        joined = "\n".join(lines)
        assert "won't stay resident" in joined

    def test_no_dual_warning_for_shared_small_model(self):
        lines = advisory_lines("qwen2.5-coder:7b", "qwen2.5-coder:7b")
        assert all("won't stay resident" not in line for line in lines)


def test_llamacpp_heavy_profile_defaults_to_single_routing():
    from local_code.model_profiles import provider_model_profile

    profile = provider_model_profile("qwen36-35b-a3b")
    assert profile["provider"] == "llamacpp"
    assert profile["role"] == "heavy_backend"
    assert profile["recommended_routing"] == "single"
    assert profile["status"] == "experimental"

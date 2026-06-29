from local_code.managed_manifest import load_manifest, select_asset


def test_managed_manifests_validate_and_select_defaults():
    runtime = load_manifest("runtime")
    model = load_manifest("model")
    assert "llama.cpp" in runtime["runtimes"]
    assert "qwen2.5-coder-7b" in model["models"]
    selected = select_asset("model", "qwen2.5-coder-7b")
    assert selected["sha256"]
    assert selected["url"].startswith("https://")

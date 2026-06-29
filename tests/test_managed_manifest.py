from local_code.managed_manifest import load_manifest, select_asset


def test_managed_manifests_validate_and_select_defaults():
    runtime = load_manifest("runtime")
    model = load_manifest("model")
    assert "llama.cpp" in runtime["runtimes"]
    assert "qwen2.5-coder-7b" in model["models"]
    selected = select_asset("model", "qwen2.5-coder-7b")
    assert selected["sha256"]
    assert selected["url"].startswith("https://")


def test_managed_manifests_do_not_ship_placeholder_checksums():
    for kind, section in (("runtime", "runtimes"), ("model", "models")):
        manifest = load_manifest(kind)
        for entries in manifest[section].values():
            for entry in entries:
                assert len(set(entry["sha256"])) > 1

import json
from pathlib import Path

import pytest

import local_code.llama_runtime as runtime
from local_code.llamacpp import generate_llama_server_args, get_llamacpp_profile


def test_friendly_profile_alias_and_argument_generation():
    assert get_llamacpp_profile("qwen36")["id"] == "qwen36-35b-a3b-llamacpp"
    args = generate_llama_server_args("qwen36", "rtx3060", "/models/qwen.gguf", executable="/bin/llama-server")
    assert args[0] == "/bin/llama-server"
    assert args[args.index("-m") + 1] == "/models/qwen.gguf"
    assert args[args.index("-c") + 1] == "16384"


def test_find_llama_server_honors_explicit_and_environment(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    assert runtime.find_llama_server(str(binary)) == str(binary.resolve())
    monkeypatch.setenv("LLAMA_SERVER", str(binary))
    assert runtime.find_llama_server() == str(binary.resolve())


def test_install_and_list_model_from_explicit_url(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    source = tmp_path / "source.gguf"
    source.write_bytes(b"GGUF test bytes")
    report = runtime.install_model("qwen36", source.as_uri())
    assert Path(report["path"]).read_bytes() == b"GGUF test bytes"
    listed = runtime.list_managed_models()
    assert listed[0]["id"] == "qwen36-35b-a3b-llamacpp"
    assert listed[0]["exists"] is True
    assert runtime.resolve_model_path("qwen36") == report["path"]


def test_register_existing_model_without_copying(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    model = tmp_path / "existing.gguf"
    model.write_bytes(b"external")
    entry = runtime.register_model("qwen36", str(model))
    assert entry["path"] == str(model.resolve())
    assert runtime.resolve_model_path("qwen36") == str(model.resolve())


def test_download_rejects_bad_checksum_and_cleans_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    source = tmp_path / "source.gguf"
    source.write_bytes(b"bad checksum")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.install_model("qwen36", source.as_uri(), sha256="0" * 64)
    assert not list((tmp_path / "home").rglob("*.part"))


def test_start_records_external_process_and_waits_for_health(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    binary = tmp_path / "llama-server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return None

    calls = {}

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(runtime, "server_status", lambda: {"state": "stopped", "managed": False})
    monkeypatch.setattr(runtime, "_port_available", lambda host, port: True)
    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime, "probe_server", lambda base_url, timeout=2: {"ready": True, "models": ["local"]})

    report = runtime.start_server("qwen36", "rtx3060", model_path=str(model), executable=str(binary))
    assert report["state"] == "ready"
    assert report["pid"] == 4321
    assert calls["args"][0] == str(binary.resolve())
    assert calls["kwargs"]["start_new_session"] is True
    state = json.loads(runtime.state_path().read_text(encoding="utf-8"))
    assert state["model_path"] == str(model.resolve())


def test_stop_refuses_pid_that_is_not_llama_server(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    runtime._write_state({"pid": 99, "base_url": "http://127.0.0.1:8080/v1"})
    monkeypatch.setattr(runtime, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(runtime, "_pid_is_llama_server", lambda pid: False)
    with pytest.raises(RuntimeError, match="Refusing to stop"):
        runtime.stop_server()


def test_status_removes_stale_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    runtime._write_state({"pid": 123, "profile": "qwen"})
    monkeypatch.setattr(runtime, "_pid_alive", lambda pid: False)
    report = runtime.server_status()
    assert report["state"] == "stale"
    assert not runtime.state_path().exists()

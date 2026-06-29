import json
import io
import tarfile
import zipfile
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
    import hashlib
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = runtime.install_model("qwen36", source.as_uri(), sha256=digest)
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


@pytest.mark.parametrize("archive_type", ["zip", "tar"])
def test_runtime_archive_extracts_server_and_shared_libraries(tmp_path, archive_type):
    archive = tmp_path / ("runtime.zip" if archive_type == "zip" else "runtime.tar.gz")
    files = {
        "llama-build/llama-server": b"server",
        "llama-build/libllama.so.0": b"library",
        "llama-build/llama-cli": b"unneeded",
    }
    if archive_type == "zip":
        with zipfile.ZipFile(archive, "w") as zf:
            for name, contents in files.items():
                zf.writestr(name, contents)
    else:
        with tarfile.open(archive, "w:gz") as tf:
            for name, contents in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                tf.addfile(info, io.BytesIO(contents))

    target = tmp_path / "bin" / "llama-server"
    runtime._extract_llama_server(archive, target)

    assert target.read_bytes() == b"server"
    assert (target.parent / "libllama.so.0").read_bytes() == b"library"
    assert not (target.parent / "llama-cli").exists()


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
    assert report["state"] == "running"
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
    assert runtime.state_path().exists()


def test_recent_logs_handles_missing_empty_and_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    assert runtime.recent_log_lines(100) == []
    runtime.log_path().parent.mkdir(parents=True)
    runtime.log_path().write_text("", encoding="utf-8")
    assert runtime.recent_log_lines(100) == []
    runtime.log_path().write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert runtime.recent_log_lines(2) == ["two\n", "three\n"]


def test_prepare_server_start_is_dry_and_matches_start_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    binary = tmp_path / "llama-server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    called = False

    def forbidden_popen(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dry-run launched a process")

    monkeypatch.setattr(runtime.subprocess, "Popen", forbidden_popen)
    report = runtime.prepare_server_start("qwen36", "rtx3060", model_path=str(model), executable=str(binary), port=9000)
    assert report["args"][0] == str(binary.resolve())
    assert report["args"][report["args"].index("--port") + 1] == "9000"
    assert called is False


def test_start_failure_includes_recent_log_and_original_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    binary = tmp_path / "llama-server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    class Process:
        pid = 44
        returncode = 9
        def poll(self):
            runtime.log_path().write_text("cudaMalloc failed: out of memory\n", encoding="utf-8")
            return 9

    monkeypatch.setattr(runtime, "server_status", lambda: {"state": "stopped", "managed": False})
    monkeypatch.setattr(runtime, "_port_available", lambda host, port: True)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: Process())
    with pytest.raises(RuntimeError) as raised:
        runtime.start_server("qwen36", "rtx3060", model_path=str(model), executable=str(binary))
    message = str(raised.value)
    assert "exited with code 9" in message
    assert "cudaMalloc failed" in message
    assert "lower quantization" in message


def test_health_timeout_includes_recent_log(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    binary = tmp_path / "llama-server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    class Process:
        pid = 45
        returncode = None
        def poll(self):
            runtime.log_path().write_text("still loading model\n", encoding="utf-8")
            return None

    monkeypatch.setattr(runtime, "server_status", lambda: {"state": "stopped", "managed": False})
    monkeypatch.setattr(runtime, "_port_available", lambda host, port: True)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(runtime, "probe_server", lambda *a, **k: {"ready": False, "error": "connection refused"})
    monkeypatch.setattr(runtime.time, "sleep", lambda value: None)
    with pytest.raises(RuntimeError) as raised:
        runtime.start_server("qwen36", "rtx3060", model_path=str(model), executable=str(binary), wait_timeout=0)
    assert "failed to become healthy" in str(raised.value)
    assert "still loading model" in str(raised.value)


def test_remove_model_unregister_delete_missing_and_running_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    runtime.register_model("qwen36", str(model))
    monkeypatch.setattr(runtime, "server_status", lambda: {"state": "stopped", "managed": False})
    kept = runtime.remove_model("qwen36")
    assert kept["unregistered"] and model.exists()

    runtime.register_model("qwen36", str(model))
    monkeypatch.setattr(runtime, "server_status", lambda: {"state": "running", "managed": True, "profile": runtime.model_key("qwen36")})
    with pytest.raises(RuntimeError, match="running"):
        runtime.remove_model("qwen36", delete_file=True, confirmed=True)

    monkeypatch.setattr(runtime, "server_status", lambda: {"state": "stopped", "managed": False})
    deleted = runtime.remove_model("qwen36", delete_file=True, confirmed=True)
    assert deleted["file_deleted"] and not model.exists()

    missing = tmp_path / "missing.gguf"
    registry = {runtime.model_key("qwen36"): {"profile": runtime.model_key("qwen36"), "path": str(missing)}}
    runtime._save_registry(registry)
    report = runtime.remove_model("qwen36", delete_file=True, confirmed=True)
    assert report["file_missing"] and report["unregistered"]


def test_validation_rejects_bad_port_and_non_gguf(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="between 1 and 65535"):
        runtime.validate_port(70000)
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path / "home"))
    binary = tmp_path / "llama-server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match=".gguf"):
        runtime.prepare_server_start("qwen36", "rtx3060", model_path=str(model), executable=str(binary))


def test_generated_numeric_settings_must_be_positive(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(runtime, "generate_llama_server_args", lambda *a, **k: [
        str(binary), "-m", str(model), "-c", "0", "-t", "8", "-b", "1", "-ub", "1", "--host", "127.0.0.1", "--port", "8080",
    ])
    with pytest.raises(ValueError, match="context must be greater than zero"):
        runtime.prepare_server_start("qwen36", "rtx3060", model_path=str(model), executable=str(binary))

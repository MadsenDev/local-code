import sys

from local_code import cli


def test_model_list_command_uses_managed_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["local-code", "model", "list"])
    assert cli.main() == 0
    assert "No managed llama.cpp models" in capsys.readouterr().out


def test_model_start_command_dispatches_profile(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["local-code", "model", "start", "qwen36", "--port", "9000"])
    captured = {}

    def fake_start(profile, gpu, **kwargs):
        captured.update(profile=profile, gpu=gpu, **kwargs)
        return {"state": "ready", "profile": "qwen36-35b-a3b-llamacpp", "base_url": "http://127.0.0.1:9000/v1"}

    monkeypatch.setattr(cli, "start_server", fake_start)
    monkeypatch.setattr(cli, "detect_hardware", lambda: type("Hardware", (), {"gpus": []})())
    assert cli.main() == 0
    assert captured["profile"] == "qwen36"
    assert captured["port"] == 9000
    assert "Status: ready" in capsys.readouterr().out


def test_llama_install_requires_explicit_url(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["local-code", "llama", "install"])
    assert cli.main() == 2
    assert "requires --url" in capsys.readouterr().err


def test_llama_logs_missing_empty_tail_and_follow(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["local-code", "llama", "logs"])
    assert cli.main() == 0
    assert "No managed llama.cpp server log" in capsys.readouterr().out

    path = tmp_path / "runtimes" / "llamacpp" / "server.log"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["local-code", "llama", "logs"])
    assert cli.main() == 0
    assert "log is empty" in capsys.readouterr().out

    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["local-code", "llama", "logs", "--tail", "2"])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "one" not in output and "two" in output and "three" in output

    monkeypatch.setattr(cli.time, "sleep", lambda value: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(sys, "argv", ["local-code", "llama", "logs", "--follow", "--tail", "1"])
    assert cli.main() == 0
    assert "three" in capsys.readouterr().out


def test_model_start_dry_run_never_calls_start(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["local-code", "model", "start", "qwen36", "--dry-run"])
    monkeypatch.setattr(cli, "detect_hardware", lambda: type("Hardware", (), {"gpus": []})())
    monkeypatch.setattr(cli, "prepare_server_start", lambda *a, **k: {
        "executable": "/bin/llama-server", "model_path": "/models/q.gguf", "port": 8080,
        "base_url": "http://127.0.0.1:8080/v1", "command": "/bin/llama-server -m /models/q.gguf",
        "log_path": "/home/u/.local-code/runtimes/llamacpp/server.log", "state_path": "/home/u/.local-code/runtimes/llamacpp/server.json",
    })
    monkeypatch.setattr(cli, "start_server", lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "Dry run" in output and "No process started" in output


def test_status_formats_running_stale_and_stopped(monkeypatch, capsys):
    for report, expected in [
        ({"state": "running", "managed": True, "pid": 12, "profile": "qwen", "port": 8080, "base_url": "http://x/v1", "log_path": "/tmp/log", "health": {"ready": True}}, "Status: running"),
        ({"state": "stale", "managed": True, "pid": 12, "profile": "qwen", "log_path": "/tmp/log"}, "Status: stale"),
        ({"state": "stopped", "managed": False, "log_path": "/tmp/log"}, "No managed llama.cpp server"),
    ]:
        monkeypatch.setattr(cli, "server_status", lambda report=report: report)
        monkeypatch.setattr(sys, "argv", ["local-code", "model", "status"])
        assert cli.main() == 0
        assert expected in capsys.readouterr().out


def test_model_remove_unregister_only_and_confirmed_delete(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    managed = {"id": "qwen36-35b-a3b-llamacpp", "path": str(model), "exists": True}
    monkeypatch.setattr(cli, "list_managed_models", lambda: [managed])
    calls = []
    monkeypatch.setattr(cli, "remove_model", lambda target, **kwargs: calls.append(kwargs) or {
        "id": managed["id"], "path": str(model), "unregistered": True,
        "file_deleted": kwargs["delete_file"], "file_missing": False,
    })
    monkeypatch.setattr(sys, "argv", ["local-code", "model", "remove", "qwen36"])
    assert cli.main() == 0
    assert calls[-1]["delete_file"] is False
    assert "Model file was kept" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["local-code", "model", "remove", "qwen36", "--delete-file", "--yes"])
    assert cli.main() == 0
    assert calls[-1] == {"delete_file": True, "confirmed": True, "force": False}
    assert "Deleted model file" in capsys.readouterr().out

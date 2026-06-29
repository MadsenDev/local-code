import sys

from local_code import cli


def test_model_list_command_uses_managed_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rist", "model", "list"])
    assert cli.main() == 0
    assert "No managed llama.cpp models" in capsys.readouterr().out


def test_model_start_command_dispatches_profile(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rist", "model", "start", "qwen36", "--port", "9000"])
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
    monkeypatch.setattr(sys, "argv", ["rist", "llama", "install"])
    assert cli.main() == 2
    assert "requires --url" in capsys.readouterr().err


def test_llama_logs_missing_empty_tail_and_follow(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_CODE_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rist", "llama", "logs"])
    assert cli.main() == 0
    assert "No managed llama.cpp server log" in capsys.readouterr().out

    path = tmp_path / "runtimes" / "llamacpp" / "server.log"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rist", "llama", "logs"])
    assert cli.main() == 0
    assert "log is empty" in capsys.readouterr().out

    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rist", "llama", "logs", "--tail", "2"])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "one" not in output and "two" in output and "three" in output

    monkeypatch.setattr(cli.time, "sleep", lambda value: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(sys, "argv", ["rist", "llama", "logs", "--follow", "--tail", "1"])
    assert cli.main() == 0
    assert "three" in capsys.readouterr().out


def test_model_start_dry_run_never_calls_start(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rist", "model", "start", "qwen36", "--dry-run"])
    monkeypatch.setattr(cli, "detect_hardware", lambda: type("Hardware", (), {"gpus": []})())
    monkeypatch.setattr(cli, "prepare_server_start", lambda *a, **k: {
        "executable": "/bin/llama-server", "model_path": "/models/q.gguf", "port": 8080,
        "base_url": "http://127.0.0.1:8080/v1", "command": "/bin/llama-server -m /models/q.gguf",
        "log_path": "/home/u/.rist/runtimes/llamacpp/server.log", "state_path": "/home/u/.rist/runtimes/llamacpp/server.json",
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
        monkeypatch.setattr(sys, "argv", ["rist", "model", "status"])
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
    monkeypatch.setattr(sys, "argv", ["rist", "model", "remove", "qwen36"])
    assert cli.main() == 0
    assert calls[-1]["delete_file"] is False
    assert "Model file was kept" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["rist", "model", "remove", "qwen36", "--delete-file", "--yes"])
    assert cli.main() == 0
    assert calls[-1] == {"delete_file": True, "confirmed": True, "force": False}
    assert "Deleted model file" in capsys.readouterr().out


def test_rist_help_uses_new_brand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rist", "--help"])
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr()
    assert "Rist - local-first AI coding agent" in output.out
    assert output.err == ""


def test_rist_doctor_dispatches(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rist", "doctor"])
    monkeypatch.setattr(cli, "build_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "doctor_report", lambda *args, **kwargs: {"readiness": "ready", "provider_available": True})
    monkeypatch.setattr(cli, "format_doctor", lambda report: "Rist doctor")
    assert cli.main() == 0
    assert capsys.readouterr().out == "Rist doctor\n"


def test_legacy_command_warns_then_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RIST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["local-code", "model", "list"])
    assert cli.main() == 0
    output = capsys.readouterr()
    assert "No managed llama.cpp models" in output.out
    assert output.err.startswith("local-code is deprecated and will be removed in a future release. Use `rist` instead.\n")


def test_chat_subcommand_selects_chat_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rist", "chat", "--no-preflight", "--storage-mode", "local-only", "--prompt", "hello"])
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.ui = type("UI", (), {"print_markdown": lambda self, value: None})()
            self.last_report = None

        def run_turn(self, prompt):
            return "ok"

    monkeypatch.setattr(cli, "LocalPartner", FakeAgent)
    monkeypatch.setattr(cli, "build_provider", lambda *args, **kwargs: type("Provider", (), {"is_local": False})())
    monkeypatch.setattr(cli, "detect_hardware", lambda *args: None)
    monkeypatch.setattr(cli, "resolve_model_routing", lambda *args, **kwargs: (args[3], args[4], {"mode": "single"}))
    assert cli.main() == 0
    assert captured["mode"] == "chat"
    assert captured["storage_mode"] == "local-only"


def test_decisions_cli_add_list_accept(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "rist", "decisions", "add", "Use SQLite", "--rationale", "Local-first storage",
        "--component", "storage", "--workdir", str(tmp_path), "--storage-mode", "local-only",
    ])
    assert cli.main() == 0
    added = capsys.readouterr().out
    decision_id = added.split()[1]

    monkeypatch.setattr(sys, "argv", ["rist", "decisions", "accept", decision_id, "--workdir", str(tmp_path), "--storage-mode", "local-only"])
    assert cli.main() == 0
    assert f"Accepted {decision_id}" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["rist", "decisions", "list", "--workdir", str(tmp_path), "--storage-mode", "local-only"])
    assert cli.main() == 0
    assert f"{decision_id}  [accepted]  Use SQLite" in capsys.readouterr().out


def test_setup_writes_config_and_registers_model(tmp_path, monkeypatch, capsys):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setenv("RIST_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", ["rist", "setup", "--model-path", str(model), "--profile", "qwen2.5-coder-7b", "--gpu", "cpu", "--llama-server", "/nope"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "Saved config" in out
    assert "Registered model" in out
    config = cli.load_runtime_config()
    assert config["provider"] == "auto"
    assert config["default_runtime"] == "llamacpp"
    assert config["llamacpp"]["profile"] == "qwen2.5-coder-7b"
    assert any(m["id"] == "qwen2.5-coder-7b-llamacpp" and m["exists"] for m in cli.list_managed_models())


def test_setup_start_calls_managed_start(tmp_path, monkeypatch, capsys):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    monkeypatch.setenv("RIST_HOME", str(tmp_path / "home"))
    calls = []
    monkeypatch.setattr(cli, "start_server", lambda profile, gpu, **kwargs: calls.append((profile, gpu, kwargs)) or {"state": "ready", "managed": True, "base_url": "http://127.0.0.1:8080/v1"})
    monkeypatch.setattr(sys, "argv", ["rist", "setup", "--model-path", str(model), "--start", "--llama-server", str(server), "--gpu", "cpu"])
    assert cli.main() == 0
    assert calls and calls[0][0] == "qwen2.5-coder-7b"
    assert calls[0][2]["executable"] == str(server.resolve())


def test_auto_provider_order_managed_external_ollama(monkeypatch):
    args = cli.parse_args.__globals__["argparse"].Namespace(base_url=None, api_key=None, ollama="http://ollama", llama_server=None, gpu=None, host="127.0.0.1", port=8080, wait_timeout=1)
    monkeypatch.setattr(cli, "load_runtime_config", lambda: {"provider": "auto", "default_runtime": "llamacpp", "llamacpp": {"base_url": "http://llama", "profile": "qwen2.5-coder-7b"}})
    monkeypatch.setattr(cli, "server_status", lambda: {"managed": True, "state": "running", "health": {"ready": True}, "base_url": "http://managed"})
    assert cli._select_auto_provider(args) == ("llamacpp", "http://managed")

    monkeypatch.setattr(cli, "server_status", lambda: {"managed": False, "state": "stopped"})
    monkeypatch.setattr(cli, "_try_autostart_managed_llamacpp", lambda *a: None)
    class P:
        def __init__(self, ok): self.ok = ok
        def available(self): return self.ok
    monkeypatch.setattr(cli, "build_provider", lambda name, **kwargs: P(name == "llamacpp"))
    assert cli._select_auto_provider(args) == ("llamacpp", "http://llama")
    monkeypatch.setattr(cli, "build_provider", lambda name, **kwargs: P(name == "ollama"))
    assert cli._select_auto_provider(args) == ("ollama", "http://ollama")


def test_plain_rist_autostarts_when_config_and_model_ready(monkeypatch):
    args = cli.parse_args.__globals__["argparse"].Namespace(base_url=None, api_key=None, ollama="http://ollama", llama_server=None, gpu=None, host="127.0.0.1", port=8080, wait_timeout=1)
    monkeypatch.setattr(cli, "load_runtime_config", lambda: {"default_runtime": "llamacpp", "llamacpp": {"profile": "qwen2.5-coder-7b", "gpu_profile": "cpu", "llama_server": "/bin/llama-server"}})
    monkeypatch.setattr(cli, "list_managed_models", lambda: [{"id": "qwen2.5-coder-7b-llamacpp", "exists": True}])
    monkeypatch.setattr(cli, "find_llama_server", lambda explicit=None: explicit)
    calls = []
    monkeypatch.setattr(cli, "start_server", lambda profile, gpu, **kwargs: calls.append((profile, gpu, kwargs)) or {"health": {"ready": True}, "base_url": "http://started/v1"})
    started = cli._try_autostart_managed_llamacpp(args, cli.load_runtime_config())
    assert started["base_url"] == "http://started/v1"
    assert calls[0][0] == "qwen2.5-coder-7b"


def test_llama_tune_honestly_reports_conservative_only(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rist", "llama", "tune", "--gpu", "cpu"])
    assert cli.main() == 0
    out = capsys.readouterr().out.lower()
    assert "no measured tuning was run" in out
    assert "conservative" in out


def test_setup_interactive_skip_benchmark(tmp_path, monkeypatch, capsys):
    model = tmp_path / "qwen.gguf"; model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"; server.write_text("#!/bin/sh\n", encoding="utf-8"); server.chmod(0o755)
    monkeypatch.setenv("RIST_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter(["1", "1", str(model), "y", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(cli, "find_llama_server", lambda explicit=None: str(server))
    monkeypatch.setattr(cli, "start_server", lambda *a, **k: {"state": "ready", "base_url": "http://127.0.0.1:8080/v1"})
    monkeypatch.setattr(cli, "doctor_report", lambda *a, **k: {"provider_available": True, "models_endpoint": True, "chat_completions": True})
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: object())
    calls = []
    monkeypatch.setattr(cli, "benchmark_model", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(sys, "argv", ["rist", "setup"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "Welcome to Rist" in out
    assert "✓ Runtime reachable" in out
    assert not calls


def test_setup_interactive_accepts_benchmark(tmp_path, monkeypatch, capsys):
    model = tmp_path / "qwen.gguf"; model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"; server.write_text("#!/bin/sh\n", encoding="utf-8"); server.chmod(0o755)
    monkeypatch.setenv("RIST_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter(["1", "1", str(model), "y", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(cli, "find_llama_server", lambda explicit=None: str(server))
    monkeypatch.setattr(cli, "start_server", lambda *a, **k: {"state": "ready", "base_url": "http://127.0.0.1:8080/v1"})
    monkeypatch.setattr(cli, "doctor_report", lambda *a, **k: {"provider_available": True, "models_endpoint": True, "chat_completions": True})
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "benchmark_model", lambda *a, **k: {"model":"local","provider":"p","num_ctx":1,"cases":{},"summary":{"median_ttft_s":1,"median_tokens_per_second":30},"suitability":{}})
    monkeypatch.setattr(sys, "argv", ["rist", "setup"])
    assert cli.main() == 0
    assert "Performance summary" in capsys.readouterr().out


def test_setup_interactive_missing_llama_server(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RIST_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter(["1", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(cli, "find_llama_server", lambda explicit=None: None)
    monkeypatch.setattr(sys, "argv", ["rist", "setup"])
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "llama-server was not found" in out
    assert "pass --llama-server" in out


def test_setup_start_failure_prints_doctor(tmp_path, monkeypatch, capsys):
    model = tmp_path / "qwen.gguf"; model.write_bytes(b"gguf")
    server = tmp_path / "llama-server"; server.write_text("#!/bin/sh\n", encoding="utf-8"); server.chmod(0o755)
    monkeypatch.setenv("RIST_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    answers = iter(["1", "1", str(model), "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(cli, "find_llama_server", lambda explicit=None: str(server))
    monkeypatch.setattr(cli, "start_server", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: type("P", (), {"is_local": True, "name": "llamacpp", "available": lambda self: False, "describe": lambda self: "llama"})())
    monkeypatch.setattr(sys, "argv", ["rist", "setup"])
    assert cli.main() == 0
    assert "Startup failed: boom" in capsys.readouterr().out

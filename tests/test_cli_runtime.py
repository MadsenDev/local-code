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
    assert "llama.cpp server: ready" in capsys.readouterr().out


def test_llama_install_requires_explicit_url(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["local-code", "llama", "install"])
    assert cli.main() == 2
    assert "requires --url" in capsys.readouterr().err

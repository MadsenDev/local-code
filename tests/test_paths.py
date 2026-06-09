from pathlib import Path

from local_code import paths
from local_code import llama_runtime


def test_migrates_legacy_home_without_deleting_source(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    legacy = home / ".local-code"
    legacy.mkdir(parents=True)
    (legacy / "config.toml").write_text("provider = 'ollama'\n", encoding="utf-8")
    nested_registry = legacy / "models" / "llamacpp" / "models.json"
    nested_registry.parent.mkdir(parents=True)
    nested_registry.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("RIST_HOME", raising=False)
    monkeypatch.delenv("LOCAL_CODE_HOME", raising=False)

    assert paths.rist_home() == home / ".rist"
    assert (home / ".rist" / "config.toml").read_text(encoding="utf-8") == "provider = 'ollama'\n"
    assert (home / ".rist" / "models.json").read_text(encoding="utf-8") == "{}\n"
    assert legacy.is_dir()
    assert capsys.readouterr().err == "Migrated configuration from ~/.local-code to ~/.rist.\n"

    paths.rist_home()
    assert capsys.readouterr().err == ""


def test_existing_rist_home_wins_without_overwrite(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    legacy = home / ".local-code"
    current = home / ".rist"
    legacy.mkdir(parents=True)
    current.mkdir(parents=True)
    (legacy / "config.toml").write_text("legacy\n", encoding="utf-8")
    (current / "config.toml").write_text("current\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("RIST_HOME", raising=False)
    monkeypatch.delenv("LOCAL_CODE_HOME", raising=False)

    assert paths.rist_home() == current
    assert (current / "config.toml").read_text(encoding="utf-8") == "current\n"
    assert capsys.readouterr().err == ""


def test_runtime_paths_use_rist_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("RIST_HOME", raising=False)
    monkeypatch.delenv("LOCAL_CODE_HOME", raising=False)

    assert llama_runtime.runtime_dir() == home / ".rist" / "runtimes" / "llamacpp"
    assert llama_runtime.registry_path() == home / ".rist" / "models.json"


def test_legacy_registry_migrates_with_compatibility_home_override(tmp_path, monkeypatch):
    legacy_home = tmp_path / "custom-home"
    old_registry = legacy_home / "models" / "llamacpp" / "models.json"
    old_registry.parent.mkdir(parents=True)
    old_registry.write_text('{"models": []}\n', encoding="utf-8")
    monkeypatch.setenv("LOCAL_CODE_HOME", str(legacy_home))
    monkeypatch.delenv("RIST_HOME", raising=False)

    assert llama_runtime.registry_path() == legacy_home / "models.json"
    assert llama_runtime.registry_path().read_text(encoding="utf-8") == '{"models": []}\n'


def test_project_memory_migrates_to_rist_without_deleting_legacy(tmp_path):
    from local_code.memory import memory_paths

    legacy = tmp_path / ".local-code"
    legacy.mkdir()
    (legacy / "project.md").write_text("legacy notes\n", encoding="utf-8")

    from local_code.memory import ensure_memory_files

    migrated = ensure_memory_files(tmp_path)
    assert migrated["base"] == tmp_path / ".rist"
    assert migrated["project"].read_text(encoding="utf-8").startswith("# Project Intelligence")
    assert "legacy notes" in migrated["project"].read_text(encoding="utf-8")
    assert legacy.is_dir()

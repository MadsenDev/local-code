import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from local_code import cli
from local_code.intelligence.indexer import ARTIFACT_NAMES, index_repository

NOW = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)


def init_repo(path: Path, files: dict[str, str | bytes] | None = None, commit: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    for relative, content in (files or {}).items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    if commit and files:
        subprocess.run(["git", "add", "."], cwd=path, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return path


def artifact(repo: Path, name: str) -> dict:
    return json.loads((repo / ".rist" / "project" / name).read_text(encoding="utf-8"))


def test_python_repository_detects_manifest_entrypoint_tests_and_commands(tmp_path):
    repo = init_repo(tmp_path / "python", {
        "pyproject.toml": '[project]\nname = "sample"\n[tool.pytest.ini_options]\ntestpaths=["tests"]\n[tool.ruff]\n',
        "src/sample/__main__.py": "def main(): pass\n",
        "tests/test_main.py": "def test_main(): assert True\n",
        ".editorconfig": "root = true\n",
    })

    report = index_repository(repo, now=NOW)
    repo_map = artifact(repo, "repo-map.json")
    manifests = artifact(repo, "manifests.json")
    conventions = artifact(repo, "conventions.json")

    assert report["written"] and report["files_scanned"] == 4
    assert repo_map["languages"] == {"Python": 2}
    assert repo_map["entry_points"] == ["src/sample/__main__.py"]
    assert repo_map["test_locations"] == ["tests"]
    assert manifests["manifests"][0]["name"] == "sample"
    assert manifests["common_commands"] == {"lint": "ruff check .", "test": "pytest"}
    assert ".editorconfig" in conventions["configuration_files"]
    for name in ARTIFACT_NAMES:
        payload = artifact(repo, name)
        assert payload["schema_version"] == 1
        assert payload["repository_revision"]
        assert payload["index_timestamp"] == "2026-06-09T12:00:00Z"
        assert len(payload["content_fingerprint"]) == 64


def test_javascript_typescript_repository_detects_scripts_and_generated_files(tmp_path):
    repo = init_repo(tmp_path / "js", {
        "package.json": json.dumps({"name": "web", "scripts": {"build": "vite build", "test": "vitest"}, "devDependencies": {"vite": "1"}}),
        "src/main.ts": "console.log('hello')\n",
        "src/api.generated.ts": "// generated file; do not edit\nexport {}\n",
        "src/view.test.tsx": "export const View = () => null\n",
        "vite.config.ts": "export default {}\n",
    })

    index_repository(repo, now=NOW)
    repo_map = artifact(repo, "repo-map.json")
    manifests = artifact(repo, "manifests.json")

    assert repo_map["languages"] == {"TypeScript": 4}
    assert repo_map["generated_files"] == ["src/api.generated.ts"]
    assert repo_map["entry_points"] == ["src/main.ts"]
    assert manifests["build_tools"] == ["npm"]
    assert manifests["common_commands"] == {"build": "npm run build", "test": "npm run test"}


def test_monorepo_workspace_boundaries_are_stable(tmp_path):
    repo = init_repo(tmp_path / "mono", {
        "package.json": json.dumps({"name": "root", "workspaces": ["packages/*"]}),
        "packages/api/package.json": json.dumps({"name": "api", "scripts": {"test": "node test.js"}}),
        "packages/api/index.js": "module.exports = {}\n",
        "packages/web/package.json": json.dumps({"name": "web", "scripts": {"build": "vite build"}}),
        "packages/web/index.ts": "export {}\n",
        "pnpm-workspace.yaml": "packages:\n  - packages/*\n",
    })

    index_repository(repo, now=NOW)
    data = artifact(repo, "manifests.json")

    assert data["workspace_boundaries"] == [".", "packages/api", "packages/web"]
    assert [item["path"] for item in data["manifests"]] == [
        "package.json", "packages/api/package.json", "packages/web/package.json", "pnpm-workspace.yaml"
    ]
    assert data["common_commands"]["build"] == "npm --prefix packages/web run build"


def test_empty_repository_produces_valid_deterministic_artifacts(tmp_path):
    repo = init_repo(tmp_path / "empty", commit=False)

    first = index_repository(repo, now=NOW)
    contents = {name: (repo / ".rist" / "project" / name).read_text() for name in ARTIFACT_NAMES}
    second = index_repository(repo, now=NOW)

    assert first["files_scanned"] == 0
    assert first["repository_revision"] is None
    assert second["files_changed"] == 0
    assert second["mode"] == "incremental"
    assert not second["written"]
    assert contents == {name: (repo / ".rist" / "project" / name).read_text() for name in ARTIFACT_NAMES}
    assert all((repo / ".rist" / "project" / name).is_symlink() for name in ARTIFACT_NAMES)


def test_ignored_binary_large_and_noisy_files_are_not_indexed(tmp_path):
    repo = init_repo(tmp_path / "filtered", {
        ".gitignore": "ignored.py\n",
        "tracked.py": "print('ok')\n",
        "node_modules/dependency.js": "ignored by shared exclusion\n",
    })
    (repo / "ignored.py").write_text("secret = True\n")
    (repo / "image.bin").write_bytes(b"\x00\x01binary")
    (repo / "large.py").write_text("x" * 100)

    report = index_repository(repo, max_file_size=32, now=NOW)
    paths = {item["path"] for item in artifact(repo, "repo-map.json")["files"]}

    assert "ignored.py" not in paths
    assert "node_modules/dependency.js" not in paths
    assert "image.bin" not in paths
    assert "large.py" not in paths
    assert report["skip_reasons"] == {"binary": 1, "excluded_directory": 1, "too_large": 1}


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlinks_are_reported_and_never_followed(tmp_path):
    repo = init_repo(tmp_path / "links", {"inside.py": "print('inside')\n"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.py").write_text("print('outside')\n")
    os.symlink(outside / "outside.py", repo / "file-link.py")
    os.symlink(outside, repo / "dir-link")

    report = index_repository(repo, now=NOW)
    repo_map = artifact(repo, "repo-map.json")

    assert report["skip_reasons"]["symlink"] == 2
    assert repo_map["symlinks_skipped"] == ["dir-link", "file-link.py"]
    assert [item["path"] for item in repo_map["files"]] == ["inside.py"]


def test_incremental_status_force_preview_and_cli_json(tmp_path, monkeypatch, capsys):
    repo = init_repo(tmp_path / "incremental", {"app.py": "print('one')\n"})
    first = index_repository(repo, now=NOW)
    assert first["mode"] == "full"

    current = index_repository(repo, status_only=True, now=NOW)
    assert current["status"] == "current" and not current["written"]

    (repo / "app.py").write_text("print('two')\n")
    stale = index_repository(repo, status_only=True, now=NOW)
    assert stale["status"] == "stale" and stale["changed_paths"] == ["app.py"]

    before = (repo / ".rist" / "project" / "repo-map.json").read_text()
    preview = index_repository(repo, preview=True, now=NOW)
    assert preview["mode"] == "preview" and preview["preview_artifacts"]
    assert (repo / ".rist" / "project" / "repo-map.json").read_text() == before

    forced = index_repository(repo, force=True, now=NOW)
    assert forced["mode"] == "full" and forced["files_changed"] == 1

    monkeypatch.setattr("sys.argv", ["rist", "index", "--workdir", str(repo), "--status", "--json"])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "status" and output["status"] == "current"

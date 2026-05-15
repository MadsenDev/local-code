import pytest
from pathlib import Path
from local_code.tools import (
    build_project_profile,
    format_project_profile,
    format_repo_map,
    insert_after,
    read_file,
    replace_in_file,
    replace_lines,
    resolve_path,
    write_file,
)


class TestResolvePath:
    def test_relative_path_within_workdir(self, tmp_path):
        result = resolve_path(str(tmp_path), "src/main.py")
        assert result == tmp_path.resolve() / "src" / "main.py"

    def test_dot_path(self, tmp_path):
        result = resolve_path(str(tmp_path), ".")
        assert result == tmp_path.resolve()

    def test_blocks_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            resolve_path(str(tmp_path), "../../etc/passwd")

    def test_blocks_absolute_outside(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            resolve_path(str(tmp_path), "/etc/passwd")

    def test_blocks_encoded_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            resolve_path(str(tmp_path), "subdir/../../../etc/passwd")


class TestFileOps:
    def test_write_and_read(self, tmp_path):
        write_file(str(tmp_path), "hello.txt", "line1\nline2\n")
        result = read_file(str(tmp_path), "hello.txt", 1, 10)
        assert "line1" in result
        assert "line2" in result

    def test_write_outside_workdir_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="escapes workdir"):
            write_file(str(tmp_path), "../../evil.txt", "bad")

    def test_replace_in_file(self, tmp_path):
        write_file(str(tmp_path), "f.py", "foo = 1\nfoo = 2\n")
        result = replace_in_file(str(tmp_path), "f.py", "foo", "bar", count=1)
        assert "Replaced" in result
        content = read_file(str(tmp_path), "f.py", 1, 10)
        assert "bar = 1" in content
        assert "foo = 2" in content

    def test_replace_missing_text(self, tmp_path):
        write_file(str(tmp_path), "f.py", "hello\n")
        result = replace_in_file(str(tmp_path), "f.py", "nothere", "x")
        assert "not found" in result.lower()

    def test_insert_after(self, tmp_path):
        write_file(str(tmp_path), "f.py", "def foo():\n    pass\n")
        result = insert_after(str(tmp_path), "f.py", "def foo():", "\n    # inserted")
        assert "Inserted" in result
        content = read_file(str(tmp_path), "f.py", 1, 10)
        assert "inserted" in content

    def test_replace_lines(self, tmp_path):
        write_file(str(tmp_path), "f.py", "a = 1\nb = 2\nc = 3\n")
        result = replace_lines(str(tmp_path), "f.py", 2, 2, "b = 99\n")
        assert "Replaced" in result
        content = read_file(str(tmp_path), "f.py", 1, 10)
        assert "b = 99" in content
        assert "a = 1" in content
        assert "c = 3" in content

    def test_replace_lines_out_of_bounds(self, tmp_path):
        write_file(str(tmp_path), "f.py", "a = 1\n")
        result = replace_lines(str(tmp_path), "f.py", 5, 10, "x\n")
        assert "out of bounds" in result


class TestRepoDiscovery:
    def test_detects_tauri_react_vite(self, tmp_path):
        write_file(
            str(tmp_path),
            "package.json",
            '{"name":"desktop-tool","scripts":{"dev":"vite","tauri":"tauri dev"},"dependencies":{"@tauri-apps/api":"1.0.0","react":"18.0.0","react-dom":"18.0.0"},"devDependencies":{"vite":"5.0.0"}}',
        )
        write_file(str(tmp_path), "vite.config.ts", "export default {}\n")
        write_file(str(tmp_path), "src/main.tsx", "import React from 'react'\n")
        write_file(str(tmp_path), "src/App.tsx", "export function App() { return null }\n")
        write_file(str(tmp_path), "src-tauri/tauri.conf.json", "{}\n")
        write_file(str(tmp_path), "src-tauri/Cargo.toml", "[package]\nname='app'\n")

        profile = build_project_profile(str(tmp_path))
        rendered = format_project_profile(profile)

        assert "Tauri" in profile["desktop_runtime"]
        assert "React" in profile["frontend"]
        assert "Vite" in profile["frontend"]
        assert profile["confidence"]["tech_stack"] == "high"
        assert "This appears to be a Tauri desktop app with a React frontend" in rendered

    def test_does_not_claim_electron_from_leftover_directory(self, tmp_path):
        write_file(
            str(tmp_path),
            "package.json",
            '{"name":"desktop-tool","scripts":{"tauri":"tauri dev"},"dependencies":{"@tauri-apps/api":"1.0.0","react":"18.0.0"},"devDependencies":{"vite":"5.0.0"}}',
        )
        write_file(str(tmp_path), "src-tauri/tauri.conf.json", "{}\n")
        write_file(str(tmp_path), "electron/main.ts", "console.log('old')\n")

        profile = build_project_profile(str(tmp_path))

        assert "Tauri" in profile["desktop_runtime"]
        assert "Electron" not in profile["desktop_runtime"]
        assert any("Electron-looking files" in item for item in profile["likely"])

    def test_repo_map_ignores_noisy_directories(self, tmp_path):
        write_file(str(tmp_path), "package.json", '{"dependencies":{"react":"18.0.0"}}')
        write_file(str(tmp_path), "node_modules/pkg/index.js", "noise")
        write_file(str(tmp_path), "dist/app.js", "noise")
        write_file(str(tmp_path), "src/main.tsx", "entry")

        rendered = format_repo_map(str(tmp_path))

        assert "package.json" in rendered
        assert "src/" in rendered
        assert "node_modules" not in rendered
        assert "dist/" not in rendered
        assert "React entrypoint found" in rendered

    def test_extracts_semantic_capabilities_from_source(self, tmp_path):
        write_file(
            str(tmp_path),
            "package.json",
            '{"name":"local-hub","scripts":{"tauri":"tauri dev"},"dependencies":{"@tauri-apps/api":"2.0.0","react":"18.0.0"},"devDependencies":{"vite":"5.0.0"}}',
        )
        write_file(
            str(tmp_path),
            "src/App.tsx",
            "import { ReposView } from './view-repos';\nimport { PortsView } from './view-ports';\nimport { LogsView } from './view-logs';\n",
        )
        write_file(
            str(tmp_path),
            "src/tauri-api.ts",
            "const api = { scanPorts: () => invoke<LivePort[]>('scan_ports'), getGitStatus: () => invoke<GitStatus>('get_git_status') }\ninterface LivePort { port: number }\ninterface GitStatus { branch: string }\n",
        )
        write_file(
            str(tmp_path),
            "src-tauri/src/commands.rs",
            "#[tauri::command]\npub fn scan_workspace_groups() {}\n#[tauri::command]\npub fn kill_process() {}\n",
        )

        profile = build_project_profile(str(tmp_path))
        rendered = format_project_profile(profile)

        assert any("repository view" in item for item in profile["capabilities"])
        assert any("scans localhost ports" in item for item in profile["capabilities"])
        assert any("Git status" in item or "git status" in item for item in profile["source_signals"])
        assert "### Source behavior signals" in rendered
        assert "### Inferred capabilities" in rendered
        assert "for managing local development workspaces, repositories" in rendered

from local_code.contracts import (
    infer_file_hints,
    inspect_workdir_state,
    normalize_backend_report,
    normalize_contract,
)


class TestInferFileHints:
    def test_extracts_file_like_tokens(self):
        hints = infer_file_hints("Check src/main.tsx and package.json for issues.")
        assert "src/main.tsx" in hints
        assert "package.json" in hints

    def test_skips_non_file_tokens(self):
        hints = infer_file_hints("This is e.g. version 1.0 and not a path.")
        assert hints == []


class TestNormalizeBackendReport:
    def test_parses_json_string(self):
        report = normalize_backend_report('{"summary":"ok","findings":["a"]}')
        assert report["summary"] == "ok"
        assert report["findings"] == ["a"]

    def test_fallback_on_bad_input(self):
        report = normalize_backend_report(None, fallback_message="fallback")
        assert report["summary"] == "fallback"


class TestNormalizeContract:
    def test_invalid_edit_policy_falls_back_to_execute_for_bootstrap(self, tmp_path):
        contract = normalize_contract(
            {"edit_policy": "inspect|plan|propose|execute"},
            "set up a Vite React project for me",
            "hybrid",
            workdir_state=inspect_workdir_state(tmp_path),
        )
        assert contract["task_kind"] == "bootstrap_new"
        assert contract["edit_policy"] == "execute"
        assert contract["execution_strategy"] == "inspect_then_execute"
        assert contract["bootstrap_template"] == "vite-react-ts"
        assert contract["verification_checks"] == [
            "package.json exists",
            "src/main.tsx exists",
            "src/App.tsx exists",
        ]

    def test_invalid_edit_policy_falls_back_to_execute_for_edit_in_agent(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"demo"}', encoding="utf-8")
        contract = normalize_contract(
            {"edit_policy": "bogus"},
            "create a new module",
            "agent",
            workdir_state=inspect_workdir_state(tmp_path),
        )
        assert contract["task_kind"] == "edit_existing"
        assert contract["edit_policy"] == "execute"

    def test_invalid_edit_policy_falls_back_to_inspect_for_non_edit_request(self, tmp_path):
        contract = normalize_contract(
            {"edit_policy": "inspect|plan|propose|execute"},
            "check whether package.json exists",
            "hybrid",
            workdir_state=inspect_workdir_state(tmp_path),
        )
        assert contract["task_kind"] == "inspection"
        assert contract["edit_policy"] == "inspect"

    def test_explain_repo_routes_to_inspection_not_bootstrap(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"demo"}', encoding="utf-8")
        contract = normalize_contract(
            {},
            "Explain this repo",
            "hybrid",
            workdir_state=inspect_workdir_state(tmp_path),
        )
        assert contract["task_kind"] == "inspection"
        assert contract["edit_policy"] == "inspect"

    def test_javascript_variant_is_selected_when_asked(self, tmp_path):
        contract = normalize_contract(
            {},
            "Set up a Vite React JavaScript project for me",
            "hybrid",
            workdir_state=inspect_workdir_state(tmp_path),
        )
        assert contract["bootstrap_template"] == "vite-react-js"
        assert contract["target_paths"] == ["package.json", "src/main.jsx", "src/App.jsx"]

    def test_non_empty_non_project_bootstrap_adds_safety_constraint(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        contract = normalize_contract(
            {},
            "Set up a Vite React project for me",
            "hybrid",
            workdir_state=inspect_workdir_state(tmp_path),
        )
        assert contract["task_kind"] == "bootstrap_new"
        assert any("non-empty and not yet a project" in item for item in contract["constraints"])

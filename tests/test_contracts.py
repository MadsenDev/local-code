from local_code.contracts import (
    compute_execution_status,
    infer_file_hints,
    inspect_workdir_state,
    normalize_backend_report,
    normalize_contract,
    validate_plan_report,
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


class TestValidatePlanReport:
    def _report(self, plan, diff=""):
        return {"plan": plan, "diff_summary": diff}

    def test_concrete_plan_with_diff_passes(self, tmp_path):
        (tmp_path / "vite.config.ts").write_text("export default {}\n")
        report = self._report(
            ["Edit vite.config.ts — add host:true to server settings"],
            "--- a/vite.config.ts\n+++ b/vite.config.ts\n@@ -1 +1,2 @@\n",
        )
        assert validate_plan_report(report, str(tmp_path)) == []

    def test_step_without_file_or_command_fails(self, tmp_path):
        report = self._report(["Improve the server settings"], "")
        problems = validate_plan_report(report, str(tmp_path))
        assert len(problems) == 1
        assert "Step 1" in problems[0]

    def test_command_step_passes_without_diff(self, tmp_path):
        report = self._report(["Run npm install to resolve dependencies"], "")
        assert validate_plan_report(report, str(tmp_path)) == []

    def test_shell_command_step_passes(self, tmp_path):
        report = self._report(["Run `git stash` to save local changes"], "")
        assert validate_plan_report(report, str(tmp_path)) == []

    def test_edit_step_without_diff_markers_fails(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        report = self._report(["Edit app.py — set x to 2"], "just a description")
        problems = validate_plan_report(report, str(tmp_path))
        assert len(problems) == 1
        assert "diff_summary" in problems[0]

    def test_create_new_file_step_passes(self, tmp_path):
        report = self._report(
            ["Create src/config.py with the default settings"],
            "--- /dev/null\n+++ b/src/config.py\n@@ -0,0 +1,3 @@\n",
        )
        assert validate_plan_report(report, str(tmp_path)) == []

    def test_empty_plan_fails(self, tmp_path):
        problems = validate_plan_report(self._report([]), str(tmp_path))
        assert problems == ["The plan is empty. Provide one step per concrete change."]

    def test_nonexistent_file_without_create_verb_fails(self, tmp_path):
        report = self._report(
            ["Edit nope.ts — change settings"],
            "--- a/nope.ts\n+++ b/nope.ts\n@@ -1 +1 @@\n",
        )
        problems = validate_plan_report(report, str(tmp_path))
        assert len(problems) == 1
        assert "Step 1" in problems[0]

    def test_mixed_command_and_edit_plan_without_diff_fails(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n")
        report = self._report(
            ["Run npm install to resolve dependencies", "Edit app.py — set x to 2"],
            "",
        )
        problems = validate_plan_report(report, str(tmp_path))
        assert len(problems) == 1
        assert "diff_summary" in problems[0]


class TestComputeExecutionStatus:
    STEPS = [
        "Edit vite.config.ts — add host:true",
        "Run npm install to resolve dependencies",
    ]

    def test_all_steps_covered_is_applied(self):
        status, missed = compute_execution_status(
            self.STEPS, ["vite.config.ts"], ["npm install"]
        )
        assert status == "applied"
        assert missed == []

    def test_file_matched_by_basename(self):
        status, _ = compute_execution_status(
            ["Edit vite.config.ts — add host"], ["/abs/path/vite.config.ts"], []
        )
        assert status == "applied"

    def test_nothing_done_is_plan_not_applied(self):
        status, missed = compute_execution_status(self.STEPS, [], [])
        assert status == "plan_not_applied"
        assert len(missed) == 2

    def test_some_steps_missed_is_partially_applied(self):
        status, missed = compute_execution_status(self.STEPS, ["vite.config.ts"], [])
        assert status == "partially_applied"
        assert missed == ["Run npm install to resolve dependencies"]

    def test_unverifiable_steps_are_excluded(self):
        status, missed = compute_execution_status(
            ["Think about the architecture", "Edit app.py — fix bug"],
            ["app.py"],
            [],
        )
        assert status == "applied"
        assert missed == []

    def test_no_verifiable_steps_falls_back_to_files_changed(self):
        status, _ = compute_execution_status(["Improve things"], ["app.py"], [])
        assert status == "applied"
        status, _ = compute_execution_status(["Improve things"], [], [])
        assert status == "plan_not_applied"

    def test_none_inputs_are_safe(self):
        status, missed = compute_execution_status(None, None, None)
        assert status == "plan_not_applied"
        assert missed == []

    def test_step_covered_when_only_file_side_matches(self):
        status, missed = compute_execution_status(
            ["Edit app.py and run npm install"], ["app.py"], []
        )
        assert status == "applied"
        assert missed == []


class TestExecutionStatusField:
    def test_normalize_passes_execution_status_through(self):
        report = normalize_backend_report({"summary": "x", "execution_status": "applied"})
        assert report["execution_status"] == "applied"

    def test_normalize_defaults_execution_status_to_empty(self):
        report = normalize_backend_report({"summary": "x"})
        assert report["execution_status"] == ""

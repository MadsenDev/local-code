import asyncio

from local_code.agent import LocalPartner
from local_code.providers import OllamaProvider
from local_code.tui import ConfirmScreen, LocalCodeApp
from local_code.tui.commands import build_commands, filter_commands
from local_code.tui.diff_review import ReviewFile, build_review_model, parse_unified_diff, summarize_review
from local_code.tui.repository import RepositoryBadge, RepositoryTree


def _partner(tmp_path, **kw):
    return LocalPartner(
        frontend_model="m",
        backend_model="m",
        provider=OllamaProvider("http://127.0.0.1:9"),
        workdir=str(tmp_path),
        verbosity="quiet",
        mode="chat",
        **kw,
    )


async def _wait_idle(app, pilot, tries=60):
    for _ in range(tries):
        if not app._busy:
            return
        await pilot.pause(0.05)


def test_status_text_renders(tmp_path):
    app = LocalCodeApp(_partner(tmp_path))
    text = app._status_text()
    assert "ollama" in text
    assert "mode chat" in text


def test_normal_turn_runs_partner_and_clears_busy(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        calls = []
        partner.run_turn = lambda text, planning=False: (calls.append(text) or "the answer")
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            app.query_one("#input").value = "hi there"
            await pilot.press("enter")
            await _wait_idle(app, pilot)
            assert calls == ["hi there"]
            assert app._busy is False
            # streaming pane was reset after the turn
            assert app._live_text == ""

    asyncio.run(inner())


def test_slash_mode_command_updates_mode(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            app.query_one("#input").value = "/mode agent"
            await pilot.press("enter")
            await pilot.pause()
            assert partner.mode == "agent"
            assert app._busy is False

    asyncio.run(inner())


def test_slash_model_command_updates_models(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            app.query_one("#input").value = "/model qwen2.5-coder:14b"
            await pilot.press("enter")
            await pilot.pause()
            assert partner.frontend_model == "qwen2.5-coder:14b"
            assert partner.backend_model == "qwen2.5-coder:14b"

    asyncio.run(inner())


def test_confirm_screen_returns_bool_on_keys(tmp_path):
    async def inner():
        app = LocalCodeApp(_partner(tmp_path))
        async with app.run_test() as pilot:
            results = []

            async def push():
                results.append(await app.push_screen_wait(ConfirmScreen("Edit", "src/app.py", "")))

            app.run_worker(push())
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert results == [True]

    asyncio.run(inner())


def test_slash_decisions_lists_records(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        item = partner.decisions.add(title="Use SQLite", rationale="Local storage")
        partner.decisions.accept(item.id)
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            app.query_one("#input").value = "/decisions list"
            await pilot.press("enter")
            await pilot.pause()
            assert app._busy is False
            assert item.id in "".join(strip.text for strip in app.query_one("#log").lines)

    asyncio.run(inner())


def test_command_registry_contains_core_palette_actions():
    commands = build_commands()
    ids = {command.id for command in commands}
    assert "runtime.run_benchmark" in ids
    assert "mode.agent" in ids
    assert "permission.edit.allow" in ids
    assert "session.quit" in ids


def test_command_search_filters_title_keywords_and_category(tmp_path):
    app = LocalCodeApp(_partner(tmp_path))
    commands = build_commands()
    assert [c.title for c in filter_commands(commands, "bench", app)][0] == "Run benchmark"
    assert any(c.id == "mode.agent" for c in filter_commands(commands, "agent", app))
    assert any(c.id == "permission.command.allow" for c in filter_commands(commands, "allow", app))


def test_palette_command_execution_records_activity_and_changes_permission(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        command = next(c for c in app.commands if c.id == "permission.edit.allow")
        async with app.run_test() as pilot:
            app.execute_palette_command(command)
            await pilot.pause()
            assert partner.edit_permission == "allow"
            assert "COMMAND" in "".join(strip.text for strip in app.query_one("#activity-log").lines)

    asyncio.run(inner())


def test_palette_mode_switching_updates_status(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        command = next(c for c in app.commands if c.id == "mode.agent")
        async with app.run_test() as pilot:
            app.execute_palette_command(command)
            await pilot.pause()
            assert partner.mode == "agent"
            assert "mode agent" in app.status.render_status(partner, False)

    asyncio.run(inner())


def test_runtime_command_wiring_uses_existing_status_api(tmp_path, monkeypatch):
    async def inner():
        import local_code.tui.tasks as task_module

        calls = []
        monkeypatch.setattr(task_module, "server_status", lambda: calls.append("status") or {"state": "stopped", "managed": False})
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        command = next(c for c in app.commands if c.id == "runtime.runtime_status")
        async with app.run_test() as pilot:
            app.execute_palette_command(command)
            await pilot.pause()
            assert calls == ["status"]

    asyncio.run(inner())


def test_palette_open_close_with_ctrl_k_and_escape(tmp_path):
    async def inner():
        app = LocalCodeApp(_partner(tmp_path))
        async with app.run_test() as pilot:
            await pilot.press("ctrl+k")
            await pilot.pause(0.1)
            assert app.screen_stack[-1].__class__.__name__ == "CommandPaletteScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen_stack[-1] is app.screen

    asyncio.run(inner())


def _sample_diff():
    return """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,2 +1,3 @@
 old
+new
 keep
-dropped
diff --git a/local_code/cli.py b/local_code/cli.py
--- a/local_code/cli.py
+++ b/local_code/cli.py
@@ -10,0 +11,2 @@
+one
+two
"""


def test_diff_review_model_generation_and_summary(tmp_path):
    partner = _partner(tmp_path)
    partner.last_report = {"needs_approval": True, "diff_summary": _sample_diff(), "plan": ["Update docs"]}
    partner.pending_plan = {"report": partner.last_report, "contract": {}, "original_prompt": "x"}
    model = build_review_model(partner)
    assert model is not None
    assert [file.filename for file in model.files] == ["README.md", "local_code/cli.py"]
    assert model.summary.files == 2
    assert model.summary.added == 3
    assert model.summary.removed == 1
    assert model.plan == ("Update docs",)


def test_diff_summary_parsing_counts_added_removed_lines():
    files = parse_unified_diff(_sample_diff())
    assert files[0] == ReviewFile("README.md", 1, 1, files[0].diff)
    assert files[1].added == 2
    assert files[1].removed == 0
    assert summarize_review(files).impact == "Low"


def test_palette_pending_proposal_commands_enable_disable(tmp_path):
    app = LocalCodeApp(_partner(tmp_path))
    commands = build_commands()
    assert not any(c.id.startswith("proposal.") for c in filter_commands(commands, "proposal", app))
    app.partner.pending_plan = {"report": {"needs_approval": True}, "contract": {}, "original_prompt": "x"}
    ids = {c.id for c in filter_commands(commands, "proposal", app)}
    assert {"proposal.review", "proposal.apply", "proposal.reject"} <= ids


def test_review_apply_and_reject_paths_update_activity_and_status(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        partner.pending_plan = {"report": {"needs_approval": True}, "contract": {}, "original_prompt": "x"}
        partner.apply_pending_plan = lambda: setattr(partner, "pending_plan", None) or "applied"
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            app._review_dismissed("reject")
            await pilot.pause()
            assert partner.pending_plan is None
            assert "Proposal rejected" in "".join(strip.text for strip in app.query_one("#activity-log").lines)
            partner.pending_plan = {"report": {"needs_approval": True}, "contract": {}, "original_prompt": "x"}
            app._review_dismissed("apply")
            await _wait_idle(app, pilot)
            assert partner.pending_plan is None
            text = "".join(strip.text for strip in app.query_one("#activity-log").lines)
            assert "Proposal accepted" in text
            assert "plan:clear" in app.status.render_status(partner, False)

    asyncio.run(inner())


def test_benchmark_rating_helper_uses_measured_summary():
    from local_code.tui.tasks import benchmark_rating

    assert benchmark_rating({"summary": {"median_tokens_per_second": 25, "median_ttft_s": 1, "median_elapsed_s": 4}}) == "Excellent"
    assert benchmark_rating({"summary": {"median_tokens_per_second": 10, "median_ttft_s": 3, "median_elapsed_s": 8}}) == "Good"
    assert benchmark_rating({"summary": {"median_tokens_per_second": 3, "median_elapsed_s": 20}}) == "Needs tuning"
    assert benchmark_rating({"summary": {"median_tokens_per_second": 0.5, "median_elapsed_s": 40}}) == "Poor"


def test_runtime_task_abstraction_status_uses_existing_api(tmp_path, monkeypatch):
    import local_code.tui.tasks as task_module
    from local_code.tui.tasks import RuntimeTask, run_runtime_task

    monkeypatch.setattr(task_module, "server_status", lambda: {"state": "running", "pid": 123})
    result = run_runtime_task(RuntimeTask.STATUS, _partner(tmp_path))
    assert result.task == RuntimeTask.STATUS
    assert result.report["state"] == "running"
    assert result.report["pid"] == 123


def test_runtime_command_schedules_task_instead_of_calling_status_directly(tmp_path, monkeypatch):
    async def inner():
        import local_code.tui.tasks as task_module

        calls = []
        monkeypatch.setattr(task_module, "server_status", lambda: calls.append("status") or {"state": "stopped"})
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        command = next(c for c in app.commands if c.id == "runtime.runtime_status")
        async with app.run_test() as pilot:
            app.execute_palette_command(command)
            assert app._runtime_task_busy is True
            await pilot.pause(0.2)
            assert calls == ["status"]
            assert app._runtime_task_busy is False

    asyncio.run(inner())


def test_runtime_duplicate_task_prevention(tmp_path):
    async def inner():
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            app._runtime_task_busy = True
            assert app.schedule_runtime_task("status") is False
            await pilot.pause()
            assert "TASK FAILED" in "".join(strip.text for strip in app.query_one("#activity-log").lines)

    asyncio.run(inner())


def test_runtime_task_failure_records_activity(tmp_path, monkeypatch):
    async def inner():
        import local_code.tui.app as app_module

        def boom(task, partner):
            raise RuntimeError("runtime exploded")

        monkeypatch.setattr(app_module, "run_runtime_task", boom)
        app = LocalCodeApp(_partner(tmp_path))
        async with app.run_test() as pilot:
            assert app.schedule_runtime_task("status") is True
            await pilot.pause(0.2)
            text = "".join(strip.text for strip in app.query_one("#activity-log").lines)
            assert "TASK FAILED" in text
            assert "runtime exploded" in text
            assert app._runtime_task_busy is False

    asyncio.run(inner())


def test_doctor_results_helper_structures_errors():
    from local_code.tui.screens.runtime_results import doctor_results_table

    table = doctor_results_table({
        "provider_available": False,
        "models_endpoint": False,
        "chat_completions": False,
        "chat_error": "connection refused",
        "configuration": {"provider": "llamacpp"},
    })
    assert table.row_count == 7


def test_runtime_status_screen_data_rendering_helper():
    from local_code.tui.screens.runtime_results import runtime_next_action, runtime_status_table

    report = {"state": "running", "pid": 42, "pid_running": True, "base_url": "http://127.0.0.1:8080", "log_path": "/tmp/log"}
    table = runtime_status_table(report)
    assert table.row_count == 8
    assert runtime_next_action(report) == "Run doctor or benchmark from Ctrl+K."


def test_repository_tree_build_badges_search_selection_and_preview(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\nsecond\n")
    (tmp_path / "README.md").write_text("# Hello\n")
    tree = RepositoryTree(tmp_path)
    tree.build()
    tree.mark_read("README.md")
    tree.mark_edited("src/app.py")
    tree.assign_badges()
    readme = tree.root.find("README.md")
    app = tree.root.find("src/app.py")
    assert readme is not None and RepositoryBadge.READ in readme.badges
    assert app is not None and RepositoryBadge.EDITED in app.badges
    assert [node.path for node in tree.visible_nodes("app") if not node.is_dir] == ["src/app.py"]
    assert "print('hi')" in tree.preview("src/app.py", limit=1)


def test_repository_proposal_badges_apply_and_clear(tmp_path):
    (tmp_path / "a.py").write_text("a\n")
    tree = RepositoryTree(tmp_path)
    tree.build()
    tree.ingest_report({"files_read": ["a.py"], "files_changed": ["a.py"], "needs_approval": True})
    node = tree.root.find("a.py")
    assert node is not None
    assert {RepositoryBadge.READ, RepositoryBadge.EDITED, RepositoryBadge.PROPOSED} <= node.badges
    tree.clear_proposal()
    assert RepositoryBadge.PROPOSED not in node.badges
    tree.set_proposed(["a.py"])
    tree.apply_proposal()
    assert RepositoryBadge.PROPOSED not in node.badges
    assert tree.session["a.py"].edited_count >= 2


def test_repository_palette_commands_and_screen_open(tmp_path):
    async def inner():
        (tmp_path / "README.md").write_text("# Project\n")
        partner = _partner(tmp_path)
        app = LocalCodeApp(partner)
        async with app.run_test() as pilot:
            command = next(c for c in app.commands if c.id == "repository.open")
            app.execute_palette_command(command)
            await pilot.pause()
            assert app.screen_stack[-1].__class__.__name__ == "RepositoryExplorerScreen"
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(inner())


def test_repository_pending_command_disabled_without_proposal(tmp_path):
    app = LocalCodeApp(_partner(tmp_path))
    commands = build_commands()
    assert not any(c.id == "repository.proposed" for c in filter_commands(commands, "pending", app))
    app.partner.pending_plan = {"report": {"needs_approval": True}, "contract": {}, "original_prompt": "x"}
    assert any(c.id == "repository.proposed" for c in filter_commands(commands, "pending", app))

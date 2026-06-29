import asyncio

from local_code.agent import LocalPartner
from local_code.providers import OllamaProvider
from local_code.tui import ConfirmScreen, LocalCodeApp
from local_code.tui.commands import build_commands, filter_commands
from local_code.tui.diff_review import ReviewFile, build_review_model, parse_unified_diff, summarize_review


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
        import local_code.tui.commands as command_module

        calls = []
        monkeypatch.setattr(command_module, "server_status", lambda: calls.append("status") or {"state": "stopped", "managed": False})
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

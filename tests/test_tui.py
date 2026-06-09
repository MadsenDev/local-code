import asyncio

from local_code.agent import LocalPartner
from local_code.providers import OllamaProvider
from local_code.tui import ConfirmScreen, LocalCodeApp


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

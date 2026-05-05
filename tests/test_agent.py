from local_code.agent import LocalCodeAgent, LocalPartner


class TestAllowCommand:
    def test_reuses_approval_for_same_command_within_run(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="ask",
            edit_permission="deny",
            verbosity="quiet",
        )
        calls = []

        def fake_confirm(kind, label, content, ui):
            calls.append((kind, label))
            return True

        monkeypatch.setattr("local_code.agent.confirm_action", fake_confirm)

        allowed, _ = agent.allow_command("npm test", ["npm test"])
        assert allowed is True
        allowed, _ = agent.allow_command("npm test", ["npm test"])
        assert allowed is True
        assert calls == [("Command", "npm test")]

    def test_bootstrap_prefix_reuses_approval_within_run(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="ask",
            edit_permission="deny",
            verbosity="quiet",
        )
        calls = []

        def fake_confirm(kind, label, content, ui):
            calls.append((kind, label))
            return True

        monkeypatch.setattr("local_code.agent.confirm_action", fake_confirm)

        allowed, _ = agent.allow_command(
            "npm create vite@latest . -- --template react-ts",
            ["npm create vite@latest . -- --template react-ts", "npm install"],
            ["npm create vite@latest", "npm install"],
        )
        assert allowed is True
        allowed, _ = agent.allow_command(
            "npm create vite@latest app -- --template react-ts",
            ["npm create vite@latest . -- --template react-ts", "npm install", "npm create vite@latest app -- --template react-ts"],
            ["npm create vite@latest", "npm install"],
        )
        assert allowed is True
        assert calls == [("Command", "npm create vite@latest . -- --template react-ts")]

    def test_cache_resets_between_backend_runs(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="ask",
            edit_permission="deny",
            max_steps=1,
            verbosity="quiet",
        )
        calls = []

        def fake_confirm(kind, label, content, ui):
            calls.append((kind, label))
            return True

        monkeypatch.setattr("local_code.agent.confirm_action", fake_confirm)
        monkeypatch.setattr(agent, "chat", lambda messages: '{"tool":"run_command","args":{"command":"npm test","timeout":1}}')

        report = agent.run_contract({"goal": "test", "scope": ["."], "commands_allowed": ["npm test"], "edit_policy": "execute"}, "")
        assert report["summary"] == "Stopped after reaching the maximum backend tool steps."
        assert calls == [("Command", "npm test")]

        report = agent.run_contract({"goal": "test", "scope": ["."], "commands_allowed": ["npm test"], "edit_policy": "execute"}, "")
        assert report["summary"] == "Stopped after reaching the maximum backend tool steps."
        assert calls == [("Command", "npm test"), ("Command", "npm test")]


class TestParseFrontendAction:
    def test_treats_mixed_question_and_delegate_json_as_reply(self):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=".",
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )
        text = (
            "Before I begin, do you have any specific preferences?\n\n"
            '{"mode":"delegate","message":"Creating a basic React TypeScript project structure.","contract":{"goal":"Initialize a React TypeScript project."}}'
        )

        action = partner.parse_frontend_action(text)

        assert action["mode"] == "reply"
        assert "specific preferences" in action["message"]

    def test_accepts_pure_delegate_json(self):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=".",
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )

        action = partner.parse_frontend_action(
            '{"mode":"delegate","message":"inspecting repo","contract":{"goal":"Inspect repo"}}'
        )

        assert action["mode"] == "delegate"
        assert action["contract"]["goal"] == "Inspect repo"

    def test_accepts_delegate_json_wrapped_in_narration_without_question(self):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=".",
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )
        text = """Okay, let's set up a Vite + React project. I can guide you through the steps, but since this is a new project, I'll need to delegate the initial setup to the backend.

Here's the delegation contract:

```json
{
  "mode": "delegate",
  "message": "Creating a new Vite + React project.",
  "contract": {
    "goal": "Create a new Vite + React project with a basic template.",
    "edit_policy": "execute"
  }
}
```"""

        action = partner.parse_frontend_action(text)

        assert action["mode"] == "delegate"
        assert action["contract"]["goal"] == "Create a new Vite + React project with a basic template."


class TestRoutingAndFailures:
    def test_hybrid_bootstrap_executes_in_empty_directory(self, monkeypatch, tmp_path):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )
        seen = {}

        def fake_run_backend(contract):
            seen["contract"] = contract
            return {"summary": "done", "commands_run": [], "files_read": [], "files_changed": [], "needs_approval": False}

        monkeypatch.setattr(partner, "_run_backend", fake_run_backend)
        monkeypatch.setattr(partner, "frontend_finalize", lambda *args: "completed")

        reply = partner.run_turn("Set up a Vite+React project for me")

        assert reply == "completed"
        assert seen["contract"]["task_kind"] == "bootstrap_new"
        assert seen["contract"]["edit_policy"] == "execute"
        assert seen["contract"]["bootstrap_template"] == "vite-react-ts"

    def test_hybrid_inspection_stays_read_only(self, monkeypatch, tmp_path):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )
        seen = {}

        def fake_run_backend(contract):
            seen["contract"] = contract
            return {"summary": "package missing", "commands_run": [], "files_read": [], "files_changed": [], "needs_approval": False}

        monkeypatch.setattr(partner, "_run_backend", fake_run_backend)
        monkeypatch.setattr(partner, "frontend_finalize", lambda *args: "inspection")

        reply = partner.run_turn("Check whether package.json exists")

        assert reply == "inspection"
        assert seen["contract"]["task_kind"] == "inspection"
        assert seen["contract"]["edit_policy"] == "inspect"

    def test_backend_timeout_is_structured(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )
        calls = {"count": 0}

        def fake_chat(messages):
            calls["count"] += 1
            raise TimeoutError("timed out")

        monkeypatch.setattr(agent, "chat", fake_chat)

        report = agent.run_contract({"goal": "test", "edit_policy": "execute"}, "")

        assert calls["count"] == 2
        assert "timed out" in report["summary"].lower()
        assert "preserved for retry" in report["risks"][0]

    def test_apply_pending_plan_requires_active_pending_proposal(self, tmp_path):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )

        assert partner.apply_pending_plan() is None

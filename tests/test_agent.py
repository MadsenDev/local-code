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

    def test_project_discovery_uses_deterministic_read_only_profile(self, monkeypatch, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"name":"desktop-tool","scripts":{"dev":"vite","tauri":"tauri dev"},"dependencies":{"@tauri-apps/api":"1.0.0","react":"18.0.0","react-dom":"18.0.0"},"devDependencies":{"vite":"5.0.0"}}',
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.tsx").write_text("import React from 'react'\n", encoding="utf-8")
        (tmp_path / "src-tauri").mkdir()
        (tmp_path / "src-tauri" / "tauri.conf.json").write_text("{}\n", encoding="utf-8")
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
        # frontend_finalize makes an LLM call; stub it to return the structured summary so
        # the test can verify the heuristic profile was built correctly without a live model.
        monkeypatch.setattr(partner, "frontend_finalize", lambda prompt, report, **kw: report["summary"])

        reply = partner.run_turn("Without making edits, what can you tell me about this project?")

        assert "## Project overview" in reply
        assert "Tauri" in reply
        assert "React" in reply
        assert partner.last_report["files_changed"] == []
        assert partner.latest_plan is None

    def test_read_only_prompt_forces_inspect_policy_even_with_modify_word(self, monkeypatch, tmp_path):
        partner = LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="allow",
            edit_permission="allow",
            verbosity="quiet",
            mode="agent",
        )
        seen = {}

        def fake_run_backend(contract):
            seen["contract"] = contract
            return {"summary": "inspected", "commands_run": [], "files_read": [], "files_changed": [], "needs_approval": False}

        monkeypatch.setattr(partner, "_run_backend", fake_run_backend)
        monkeypatch.setattr(partner, "frontend_finalize", lambda *args: "inspection")

        reply = partner.run_turn("Do not modify anything, inspect package.json")

        assert reply == "inspection"
        assert seen["contract"]["edit_policy"] == "inspect"
        assert seen["contract"]["read_only"] is True

    def test_invalid_tool_recovery_lists_available_tools(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            max_steps=2,
            verbosity="quiet",
        )
        calls = {"count": 0, "messages": []}

        def fake_chat(messages):
            calls["count"] += 1
            calls["messages"] = list(messages)
            if calls["count"] == 1:
                return '{"tool":"write_file","args":{"path":"x","content":"bad"}}'
            return '{"tool":"final","args":{"summary":"stopped","findings":[]}}'

        monkeypatch.setattr(agent, "chat", fake_chat)

        report = agent.run_contract({"goal": "inspect", "edit_policy": "inspect", "read_only": True}, "")

        assert report["summary"] == "stopped"
        recovery_prompt = calls["messages"][-1]["content"]
        assert "Available tools:" in recovery_prompt
        assert "repo_map" in recovery_prompt
        available_block = recovery_prompt.split("You must respond", 1)[0]
        assert "write_file" not in available_block

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


class TestIntentScaffolding:
    def test_enriches_backend_contract_with_compact_intent_analysis(self, monkeypatch, tmp_path):
        (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
        partner = LocalPartner(
            frontend_model="small",
            backend_model="small",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )

        def fake_assess(model, messages):
            assert model == "small"
            assert "Intent Analyst" in messages[0]["content"]
            return '''{
                "user_goal": "Fix the app startup bug.",
                "not_the_goal": ["Rewrite the app"],
                "needed_context": ["Read app.py before editing"],
                "likely_files": ["app.py"],
                "risks": ["May need a test"],
                "success_criteria": ["app.py still imports"]
            }'''

        monkeypatch.setattr(partner.provider, "assess", fake_assess)

        contract = partner._contract_with_intent_scaffold(
            {
                "goal": "fix startup",
                "task_kind": "edit_existing",
                "edit_policy": "execute",
                "constraints": [],
                "files_of_interest": [],
            }
        )

        assert contract["intent_analysis"]["user_goal"] == "Fix the app startup bug."
        assert contract["intent_analysis"]["not_the_goal"] == ["Rewrite the app"]
        assert contract["files_of_interest"] == ["app.py"]
        assert "Do not overreach beyond the intent analysis not_the_goal list." in contract["constraints"]
        assert "Inspect the intent analysis needed_context before editing when applicable." in contract["constraints"]
        assert "Use the intent analysis success_criteria to decide when to stop." in contract["constraints"]

    def test_intent_scaffold_is_non_fatal_when_model_output_is_invalid(self, monkeypatch, tmp_path):
        partner = LocalPartner(
            frontend_model="small",
            backend_model="small",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )
        monkeypatch.setattr(partner.provider, "assess", lambda *args: "not json")
        original = {"goal": "inspect", "task_kind": "inspection", "edit_policy": "inspect"}

        assert partner._contract_with_intent_scaffold(original) == original


class TestFewShotByModelTier:
    def _agent(self, model, tmp_path):
        return LocalCodeAgent(
            model=model,
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )

    def test_weak_model_gets_few_shot_examples(self, tmp_path):
        agent = self._agent("qwen3:4b", tmp_path)
        block = agent.few_shot_block({"edit_policy": "inspect", "read_only": True})
        assert "Examples" in block
        assert '"tool":"read_file"' in block

    def test_strong_model_skips_few_shot(self, tmp_path):
        agent = self._agent("qwen2.5-coder:14b", tmp_path)
        assert agent.few_shot_block({"edit_policy": "inspect"}) == ""

    def test_system_prompt_includes_examples_for_weak_model(self, tmp_path):
        agent = self._agent("llama3.2:1b", tmp_path)
        prompt = agent.system_prompt({"edit_policy": "inspect", "read_only": True}, "")
        assert "Examples" in prompt


class TestToolCalling:
    def test_invalid_tool_arguments_are_retried_before_dispatch(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            max_steps=2,
            verbosity="quiet",
        )
        calls = {"count": 0, "messages": []}

        def fake_chat(messages):
            calls["count"] += 1
            calls["messages"] = list(messages)
            if calls["count"] == 1:
                return '{"tool":"read_file","args":{"start":1}}'
            return '{"tool":"final","args":{"summary":"validated","findings":[]}}'

        monkeypatch.setattr(agent, "chat", fake_chat)

        report = agent.run_contract({"goal": "inspect", "edit_policy": "inspect", "read_only": True}, "")

        assert report["summary"] == "validated"
        recovery_prompt = calls["messages"][-1]["content"]
        assert "missing required argument" in recovery_prompt
        assert "read_file" in recovery_prompt

    def test_native_tool_call_is_normalized(self, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )

        action = agent.action_from_native_tool_calls(
            [{"function": {"name": "read_file", "arguments": '{"path":"README.md","start":1}'}}]
        )

        assert action == {"tool": "read_file", "args": {"path": "README.md", "start": 1}}

    def test_auto_tool_calling_falls_back_to_json_protocol(self, monkeypatch, tmp_path):
        agent = LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            tool_calling="auto",
        )

        class Result:
            content = ""
            tool_calls = []

        monkeypatch.setattr(agent, "chat_tools", lambda messages, tools: Result())
        monkeypatch.setattr(agent, "chat", lambda messages: '{"tool":"final","args":{"summary":"fallback"}}')

        action, _ = agent.request_action([{"role": "system", "content": "test"}], {"edit_policy": "inspect"})

        assert action == {"tool": "final", "args": {"summary": "fallback"}}

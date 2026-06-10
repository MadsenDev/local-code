# Plan→Execute Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Approved plans actually execute — the execute pass receives the approved plan, vague plans are caught at plan time, no-op finals are rejected once then reported honestly, and the user-facing summary never implies success when no edits happened.

**Architecture:** All changes live in the existing harness (`local_code/contracts.py` and `local_code/agent.py`); no new files. Two pure helpers in `contracts.py` (`validate_plan_report`, `compute_execution_status`) do the mechanical validation; `LocalCodeAgent.run_contract()` uses them to push back inside the tool loop; `LocalPartner.apply_pending_plan()` carries the approved plan into the execute contract; `LocalPartner.frontend_finalize()` is forced to lead with the truth when edits didn't happen.

**Tech Stack:** Python 3, pytest (offline — model calls are monkeypatched with scripted JSON responses, the established pattern in `tests/test_agent.py`).

**Spec:** `docs/superpowers/specs/2026-06-10-plan-execute-reliability-design.md`

**Run tests with:** `.venv/bin/pytest` (the venv lives at `.venv/`).

---

## Background for the implementer (read first)

- A *contract* is a plain dict handed to the backend tool loop. `edit_policy` is one of `inspect|plan|propose|execute`.
- `LocalCodeAgent.run_contract(contract, memory_text)` (agent.py, ~line 681) loops: ask model for a JSON tool action → execute → append result. It ends when the model emits `{"tool":"final","args":{...}}`. The `final` branch (~line 855) builds a report via `normalize_backend_report()`.
- A `tracker` dict (`commands_run` list, `files_read` set, `files_changed` set) records what tools actually did — this is ground truth, independent of what the model claims.
- The existing pushback pattern: append a corrective `{"role":"user", ...}` message and `continue` the loop (see `invalid_action_hint` usage, agent.py ~line 756).
- `LocalPartner` (same file, ~line 1010) orchestrates: stores plan/propose reports as `self.pending_plan`, and `apply_pending_plan()` (~line 1673) re-runs the contract with `edit_policy="execute"`. **The bug:** it discards the plan report, so the execute run starts from scratch.
- Tests monkeypatch the bound method: `monkeypatch.setattr(agent, "chat", lambda messages: '<json string>')`. Default `tool_calling` is `"json"`, so `request_action()` goes straight to `self.chat`.

---

### Task 1: `validate_plan_report()` in contracts.py

Plans must be concrete enough to apply mechanically: every step names a real repo file (or a new file being created, or a runnable command), and file edits come with a real unified diff.

**Files:**
- Modify: `local_code/contracts.py` (add after `normalize_backend_report`, ~line 323)
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contracts.py`:

```python
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
```

Also extend the import at the top of `tests/test_contracts.py` to include `validate_plan_report` (it currently imports from `local_code.contracts`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_contracts.py::TestValidatePlanReport -v`
Expected: FAIL with `ImportError: cannot import name 'validate_plan_report'`

- [ ] **Step 3: Implement `validate_plan_report`**

In `local_code/contracts.py`, add after `normalize_backend_report` (~line 323):

```python
PLAN_CREATE_RE = re.compile(r"\b(create|add|new file|scaffold|generate)\b", re.I)
DIFF_MARKERS = ("--- ", "+++ ", "@@")


def validate_plan_report(report, workdir):
    """Check that a plan/propose report is concrete enough to apply mechanically.

    Returns a list of human-readable problems; an empty list means the plan is
    appliable. A step is concrete when it references an existing repo file, a
    new file it explicitly creates, or a runnable command. If any step is a
    file edit, diff_summary must contain real unified-diff markers.
    """
    problems = []
    steps = report.get("plan") or []
    if not steps:
        return ["The plan is empty. Provide one step per concrete change."]
    has_edit_step = False
    for index, step in enumerate(steps, start=1):
        text = str(step)
        file_hints = infer_file_hints(text)
        existing = resolve_repo_file_hints(workdir, file_hints)
        is_creation = bool(file_hints and PLAN_CREATE_RE.search(text))
        is_command = bool(infer_command_hints(text))
        if existing or is_creation:
            has_edit_step = True
            continue
        if is_command:
            continue
        problems.append(
            f"Step {index} names no existing repo file and no runnable command: "
            f"{text[:120]!r}. Name the exact file and change, or the exact command."
        )
    if has_edit_step:
        diff = str(report.get("diff_summary") or "")
        if not all(marker in diff for marker in DIFF_MARKERS):
            problems.append(
                "diff_summary must contain a real unified diff "
                "(--- a/path, +++ b/path, @@ hunks) for the file edits in the plan."
            )
    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_contracts.py -v`
Expected: all PASS (including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add local_code/contracts.py tests/test_contracts.py
git commit -m "feat: validate plan reports for mechanical appliability"
```

---

### Task 2: `compute_execution_status()` + `execution_status` report field

Harness-derived truth about whether an approved plan was applied. Never trusted from the model.

**Files:**
- Modify: `local_code/contracts.py` (helper after `validate_plan_report`; one line in `normalize_backend_report`; one entry in `VALID_EXECUTION_STRATEGIES` line 9)
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contracts.py` (and add `compute_execution_status` to the import):

```python
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


class TestExecutionStatusField:
    def test_normalize_passes_execution_status_through(self):
        report = normalize_backend_report({"summary": "x", "execution_status": "applied"})
        assert report["execution_status"] == "applied"

    def test_normalize_defaults_execution_status_to_empty(self):
        report = normalize_backend_report({"summary": "x"})
        assert report["execution_status"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_contracts.py::TestComputeExecutionStatus tests/test_contracts.py::TestExecutionStatusField -v`
Expected: FAIL with `ImportError: cannot import name 'compute_execution_status'`

- [ ] **Step 3: Implement**

In `local_code/contracts.py`:

1. Line 9 — add the new strategy:

```python
VALID_EXECUTION_STRATEGIES = {"direct", "inspect_then_execute", "plan_only", "apply_approved_plan"}
```

2. After `validate_plan_report`, add:

```python
def compute_execution_status(plan_steps, files_changed, commands_run):
    """Harness-derived status of an approved plan after an execute run.

    Returns (status, missed_steps) where status is one of "applied",
    "partially_applied", or "plan_not_applied". A step is covered when a file
    it names appears in files_changed (exact or basename match) or a command
    it names matches a commands_run entry by prefix. Steps naming no file or
    command cannot be verified mechanically and are excluded.
    """
    changed = [str(p) for p in files_changed or []]
    changed_names = {Path(p).name for p in changed}
    run = [str(c) for c in commands_run or []]
    missed = []
    verifiable = 0
    for step in plan_steps or []:
        text = str(step)
        file_refs = infer_file_hints(text)
        command_refs = infer_command_hints(text)
        if not file_refs and not command_refs:
            continue
        verifiable += 1
        file_hit = any(ref in changed or Path(ref).name in changed_names for ref in file_refs)
        command_hit = any(
            r == cmd or r.startswith(cmd + " ") or cmd.startswith(r + " ")
            for cmd in command_refs
            for r in run
        )
        if not (file_hit or command_hit):
            missed.append(text)
    if not changed and not run:
        return "plan_not_applied", missed
    if not verifiable:
        return ("applied" if changed else "plan_not_applied"), missed
    if not missed:
        return "applied", missed
    return "partially_applied", missed
```

3. In `normalize_backend_report` (~line 309), add one key to the returned dict, after `"requires_deviation_explanation"`:

```python
        "execution_status": str(report.get("execution_status") or ""),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_contracts.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add local_code/contracts.py tests/test_contracts.py
git commit -m "feat: harness-derived execution_status for approved plans"
```

---

### Task 3: Carry the approved plan into the execute contract

`apply_pending_plan()` attaches the stored plan report; the intent scaffold skips contracts that already carry an approved plan.

**Files:**
- Modify: `local_code/agent.py` — `LocalPartner.apply_pending_plan` (~line 1673), `LocalPartner._contract_with_intent_scaffold` (~line 1527)
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
class TestApprovedPlanCarriage:
    def _partner(self, tmp_path):
        return LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )

    def test_apply_pending_plan_attaches_approved_plan(self, monkeypatch, tmp_path):
        partner = self._partner(tmp_path)
        partner.pending_plan = {
            "original_prompt": "fix the dev server",
            "frontend_message": "",
            "contract": {"goal": "fix the dev server", "edit_policy": "plan"},
            "report": {
                "plan": ["Edit vite.config.ts — add host:true", "Run npm install"],
                "diff_summary": "--- a/vite.config.ts\n+++ b/vite.config.ts\n@@ -1 +1 @@\n",
                "files_read": ["vite.config.ts"],
            },
        }
        seen = {}

        def fake_run_backend(contract):
            seen["contract"] = contract
            return {"summary": "done", "commands_run": [], "files_read": [],
                    "files_changed": ["vite.config.ts"], "needs_approval": False}

        monkeypatch.setattr(partner, "_run_backend", fake_run_backend)
        monkeypatch.setattr(partner, "frontend_finalize", lambda *args, **kw: "applied")

        reply = partner.apply_pending_plan()

        assert reply == "applied"
        contract = seen["contract"]
        assert contract["edit_policy"] == "execute"
        assert contract["execution_strategy"] == "apply_approved_plan"
        assert contract["approved_plan"]["steps"] == [
            "Edit vite.config.ts — add host:true",
            "Run npm install",
        ]
        assert contract["approved_plan"]["diff_summary"].startswith("--- a/")
        assert contract["approved_plan"]["files_read"] == ["vite.config.ts"]

    def test_intent_scaffold_skips_approved_plan_contracts(self, monkeypatch, tmp_path):
        partner = self._partner(tmp_path)

        def fail_intent(contract):
            raise AssertionError("intent analysis must not run for approved plans")

        monkeypatch.setattr(partner, "_intent_analysis", fail_intent)
        contract = {"goal": "x", "approved_plan": {"steps": ["Edit a.py"]}}
        assert partner._contract_with_intent_scaffold(contract) is contract
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agent.py::TestApprovedPlanCarriage -v`
Expected: FAIL — `KeyError: 'approved_plan'` in the first test, `AssertionError: intent analysis must not run` in the second

- [ ] **Step 3: Implement**

In `local_code/agent.py`:

1. `apply_pending_plan` (~line 1673) — replace the contract setup at the top of the method:

```python
    def apply_pending_plan(self):
        if not self.pending_plan:
            return None
        contract = dict(self.pending_plan["contract"])
        contract["edit_policy"] = "execute"
        contract["execution_strategy"] = "apply_approved_plan"
        plan_report = self.pending_plan.get("report") or {}
        contract["approved_plan"] = {
            "steps": list(plan_report.get("plan") or []),
            "diff_summary": str(plan_report.get("diff_summary") or ""),
            "files_read": list(plan_report.get("files_read") or []),
        }
        self.milestone("applying approved plan")
```

(the rest of the method is unchanged)

2. `_contract_with_intent_scaffold` (~line 1527) — add an early return as the first line of the body:

```python
    def _contract_with_intent_scaffold(self, contract):
        if contract.get("approved_plan"):
            return contract
        if contract.get("intent_analysis"):
            return contract
```

(the rest of the method is unchanged)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add local_code/agent.py tests/test_agent.py
git commit -m "feat: carry approved plan into the execute contract"
```

---

### Task 4: Approved-plan rules in the backend system prompt

When the contract carries `approved_plan`, the backend is told to apply steps, not re-investigate.

**Files:**
- Modify: `local_code/agent.py` — `LocalCodeAgent.system_prompt` (~line 228)
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
class TestApprovedPlanPrompt:
    def _agent(self, tmp_path):
        return LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )

    def test_system_prompt_includes_approved_plan_rules(self, tmp_path):
        agent = self._agent(tmp_path)
        contract = {
            "goal": "fix dev server",
            "edit_policy": "execute",
            "approved_plan": {"steps": ["Edit vite.config.ts — add host:true"]},
        }
        prompt = agent.system_prompt(contract, "")
        assert "An approved plan exists" in prompt
        assert "not to re-investigate" in prompt

    def test_system_prompt_omits_block_without_approved_plan(self, tmp_path):
        agent = self._agent(tmp_path)
        prompt = agent.system_prompt({"goal": "x", "edit_policy": "execute"}, "")
        assert "An approved plan exists" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent.py::TestApprovedPlanPrompt -v`
Expected: first test FAILS (`assert "An approved plan exists" in prompt`), second passes

- [ ] **Step 3: Implement**

In `LocalCodeAgent.system_prompt` (~line 228), compute the block before the `return` and interpolate it into the template right after the `Rules:` / `- Follow the contract strictly.` line:

```python
    def system_prompt(self, contract, memory_text):
        approved_plan_rules = ""
        if contract.get("approved_plan"):
            approved_plan_rules = (
                "- An approved plan exists in the contract's approved_plan field. "
                "Your job is to apply each step, not to re-investigate.\n"
                "            - Read a file only if you are about to edit it.\n"
                "            - Finish only when every step is applied, or report exactly "
                "which steps you could not apply and why.\n"
            )
        return textwrap.dedent(
```

and in the template change:

```
            Rules:
            - Follow the contract strictly.
```

to:

```
            Rules:
            {approved_plan_rules}- Follow the contract strictly.
```

(The literal `"            "` indentation inside `approved_plan_rules` matches the template's 12-space indentation so the rendered prompt lines up; `textwrap.dedent` is already a no-op here because interpolated content like the contract JSON breaks the common prefix — match the existing style.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py::TestApprovedPlanPrompt -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add local_code/agent.py tests/test_agent.py
git commit -m "feat: backend prompt rules for applying approved plans"
```

---

### Task 5: Vague-plan pushback in the tool loop

A `final` in plan/propose mode that fails `validate_plan_report` is bounced back into the loop with the problem list, at most twice; then accepted with a vagueness risk.

**Files:**
- Modify: `local_code/agent.py` — imports (~line 31), `run_contract` final branch (~line 855), new helper `vague_plan_hint` (near `invalid_action_hint`, ~line 965)
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
def scripted_chat(monkeypatch, agent, responses, seen=None):
    """Replace agent.chat with a queue of canned JSON responses.

    seen, when given, collects the message list content at each call so tests
    can assert on pushback messages the harness injected.
    """
    queue = list(responses)

    def fake_chat(messages):
        if seen is not None:
            seen.append([m["content"] for m in messages])
        return queue.pop(0)

    monkeypatch.setattr(agent, "chat", fake_chat)


class TestVaguePlanPushback:
    def _agent(self, tmp_path):
        return LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
        )

    VAGUE_FINAL = '{"tool":"final","args":{"summary":"plan ready","plan":["Improve the server settings"],"diff_summary":""}}'

    def test_vague_plan_pushed_back_then_concrete_accepted(self, monkeypatch, tmp_path):
        (tmp_path / "vite.config.ts").write_text("export default {}\n")
        agent = self._agent(tmp_path)
        concrete = (
            '{"tool":"final","args":{"summary":"plan ready",'
            '"plan":["Edit vite.config.ts — add host:true"],'
            '"diff_summary":"--- a/vite.config.ts\\n+++ b/vite.config.ts\\n@@ -1 +1 @@\\n"}}'
        )
        seen = []
        scripted_chat(monkeypatch, agent, [self.VAGUE_FINAL, concrete], seen)

        report = agent.run_contract({"goal": "fix server", "edit_policy": "plan"}, "")

        assert report["plan"] == ["Edit vite.config.ts — add host:true"]
        assert report["needs_approval"] is True
        assert "not concrete enough" in seen[1][-1]

    def test_persistently_vague_plan_accepted_with_risk(self, monkeypatch, tmp_path):
        agent = self._agent(tmp_path)
        scripted_chat(monkeypatch, agent, [self.VAGUE_FINAL] * 3)

        report = agent.run_contract({"goal": "fix server", "edit_policy": "plan"}, "")

        assert report["needs_approval"] is True
        assert any("too vague" in risk for risk in report["risks"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agent.py::TestVaguePlanPushback -v`
Expected: FAIL — first test gets the vague plan back unchanged (no pushback happened), second has no vagueness risk

- [ ] **Step 3: Implement**

In `local_code/agent.py`:

1. Extend the `from .contracts import (...)` block (~line 31) with `compute_execution_status` and `validate_plan_report` (alphabetical order; `compute_execution_status` is used in Task 6 — importing it now avoids touching the import twice).

2. In `run_contract`, next to `invalid_action_count = 0` (~line 699), add:

```python
        plan_pushback_count = 0
        noop_final_count = 0
```

3. In the `final` branch, replace (~line 864):

```python
                if contract.get("edit_policy") in {"plan", "propose"}:
                    report["needs_approval"] = True
```

with:

```python
                if contract.get("edit_policy") in {"plan", "propose"}:
                    report["needs_approval"] = True
                    problems = validate_plan_report(report, self.workdir)
                    if problems and plan_pushback_count < 2:
                        plan_pushback_count += 1
                        self.trace_print("plan final too vague; asking for a concrete plan")
                        messages.append({"role": "user", "content": self.vague_plan_hint(problems)})
                        continue
                    if problems:
                        report["risks"] = list(report["risks"]) + [
                            "Plan may be too vague to apply mechanically: " + "; ".join(problems)[:300]
                        ]
```

4. Add the helper after `invalid_action_hint` (~line 999):

```python
    def vague_plan_hint(self, problems):
        return "\n".join(
            [
                "Your plan is not concrete enough to execute. Problems:",
                *[f"- {p}" for p in problems],
                "",
                "Revise the plan and call final again.",
                "Every plan step must name a real repo file and the exact change, or an exact command.",
                "diff_summary must contain a real unified diff (--- a/path, +++ b/path, @@ hunks).",
            ]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add local_code/agent.py tests/test_agent.py
git commit -m "feat: push back on vague plan finals inside the tool loop"
```

---

### Task 6: No-op rejection and execution_status stamping in execute mode

A `final` in execute mode with an approved plan but no edits/commands is rejected once with the unapplied steps; a second no-op becomes an honest failure report. Status is stamped from the harness's own evidence.

**Files:**
- Modify: `local_code/agent.py` — `run_contract` final branch (directly after the Task 5 block), new helper `unapplied_plan_hint`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py` (uses `scripted_chat` from Task 5):

```python
class TestNoOpExecuteRejection:
    def _agent(self, tmp_path):
        return LocalCodeAgent(
            model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="allow",
            verbosity="quiet",
        )

    def _contract(self, steps):
        return {
            "goal": "fix dev server",
            "edit_policy": "execute",
            "execution_strategy": "apply_approved_plan",
            "approved_plan": {"steps": steps, "diff_summary": "", "files_read": []},
        }

    NOOP_FINAL = '{"tool":"final","args":{"summary":"nothing additional to do"}}'

    def test_noop_final_pushed_back_then_edit_applied(self, monkeypatch, tmp_path):
        (tmp_path / "vite.config.ts").write_text("server: { port: 3000 }\n")
        agent = self._agent(tmp_path)
        seen = []
        scripted_chat(
            monkeypatch,
            agent,
            [
                self.NOOP_FINAL,
                '{"tool":"replace_in_file","args":{"path":"vite.config.ts","old":"port: 3000","new":"host: true, port: 3000"}}',
                '{"tool":"final","args":{"summary":"added host setting","files_changed":["vite.config.ts"]}}',
            ],
            seen,
        )

        report = agent.run_contract(self._contract(["Edit vite.config.ts — add host:true"]), "")

        assert "host: true" in (tmp_path / "vite.config.ts").read_text()
        assert report["execution_status"] == "applied"
        assert "Unapplied steps" in seen[1][-1]
        assert "vite.config.ts" in seen[1][-1]

    def test_double_noop_becomes_honest_failure(self, monkeypatch, tmp_path):
        agent = self._agent(tmp_path)
        scripted_chat(monkeypatch, agent, [self.NOOP_FINAL, self.NOOP_FINAL])

        report = agent.run_contract(self._contract(["Edit vite.config.ts — add host:true"]), "")

        assert report["execution_status"] == "plan_not_applied"
        assert report["summary"] == "The plan was approved but not applied."
        assert any("not applied" in risk for risk in report["risks"])

    def test_partial_application_flagged_with_missed_steps(self, monkeypatch, tmp_path):
        (tmp_path / "vite.config.ts").write_text("server: { port: 3000 }\n")
        (tmp_path / "package.json").write_text("{}\n")
        agent = self._agent(tmp_path)
        scripted_chat(
            monkeypatch,
            agent,
            [
                '{"tool":"replace_in_file","args":{"path":"vite.config.ts","old":"port: 3000","new":"host: true"}}',
                '{"tool":"final","args":{"summary":"done","files_changed":["vite.config.ts"]}}',
            ],
        )

        report = agent.run_contract(
            self._contract(
                ["Edit vite.config.ts — add host:true", "Edit package.json — add dev script"]
            ),
            "",
        )

        assert report["execution_status"] == "partially_applied"
        assert any("package.json" in risk for risk in report["risks"])

    def test_execute_without_approved_plan_unchanged(self, monkeypatch, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        agent = self._agent(tmp_path)
        scripted_chat(
            monkeypatch,
            agent,
            [
                '{"tool":"replace_in_file","args":{"path":"a.py","old":"x = 1","new":"x = 2"}}',
                '{"tool":"final","args":{"summary":"done","files_changed":["a.py"]}}',
            ],
        )

        report = agent.run_contract({"goal": "set x to 2", "edit_policy": "execute"}, "")

        assert report["execution_status"] == "applied"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_agent.py::TestNoOpExecuteRejection -v`
Expected: FAIL — `execution_status` is `""` everywhere and the first no-op final is accepted as-is

- [ ] **Step 3: Implement**

In `run_contract`'s `final` branch, directly after the plan/propose block from Task 5 and **before** the existing `if contract.get("edit_policy") == "execute" and report["files_changed"]:` git-diff block, insert:

```python
                approved_plan = contract.get("approved_plan") or {}
                if contract.get("edit_policy") == "execute" and approved_plan:
                    status, missed = compute_execution_status(
                        approved_plan.get("steps") or [],
                        report["files_changed"],
                        report["commands_run"],
                    )
                    if status == "plan_not_applied" and noop_final_count == 0:
                        noop_final_count += 1
                        self.trace_print("no-op final in execute mode; asking backend to apply the approved plan")
                        messages.append({"role": "user", "content": self.unapplied_plan_hint(approved_plan)})
                        continue
                    report["execution_status"] = status
                    if status == "plan_not_applied":
                        report["summary"] = "The plan was approved but not applied."
                        report["risks"] = list(report["risks"]) + [
                            "Approved plan steps were not applied: "
                            + "; ".join(str(s) for s in (approved_plan.get("steps") or []))[:300]
                        ]
                    elif status == "partially_applied":
                        report["risks"] = list(report["risks"]) + [
                            "Plan steps not verifiably applied: " + "; ".join(missed)[:300]
                        ]
                    if status != "applied":
                        self.transcript_print(
                            "no edits were made" if status == "plan_not_applied" else "plan only partially applied",
                            [str(s)[:220] for s in (missed or approved_plan.get("steps") or [])][:5],
                            color=UI.YELLOW,
                        )
                elif contract.get("edit_policy") == "execute" and report["files_changed"]:
                    report["execution_status"] = "applied"
```

Add the helper after `vague_plan_hint`:

```python
    def unapplied_plan_hint(self, approved_plan):
        steps = approved_plan.get("steps") or []
        return "\n".join(
            [
                "You finished without applying the approved plan. No files were changed and no plan commands were run.",
                "Unapplied steps:",
                *[f"{index}. {step}" for index, step in enumerate(steps, start=1)],
                "",
                "Apply them now using replace_in_file, write_file, replace_lines, insert_after, or run_command.",
                "If a step truly cannot be applied, apply the others and state per step why it cannot be applied.",
            ]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add local_code/agent.py tests/test_agent.py
git commit -m "feat: reject no-op finals and stamp harness-derived execution_status"
```

---

### Task 7: Honest finalize directive

When `execution_status` is anything other than `"applied"`, the frontend model is ordered to lead with the truth.

**Files:**
- Modify: `local_code/agent.py` — `LocalPartner.frontend_finalize` (~line 1376)
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
class TestHonestFinalize:
    def _partner(self, tmp_path):
        return LocalPartner(
            frontend_model="test",
            backend_model="test",
            ollama="http://localhost:11434",
            workdir=str(tmp_path),
            command_permission="deny",
            edit_permission="deny",
            verbosity="quiet",
            mode="hybrid",
        )

    def _capture_chat(self, monkeypatch, partner):
        seen = []

        def fake_chat(model, messages):
            seen.append([m["content"] for m in messages])
            return "summary text"

        monkeypatch.setattr(partner, "chat", fake_chat)
        return seen

    def test_directive_added_when_plan_not_applied(self, monkeypatch, tmp_path):
        partner = self._partner(tmp_path)
        seen = self._capture_chat(monkeypatch, partner)
        report = {"summary": "x", "execution_status": "plan_not_applied"}

        partner.frontend_finalize("fix it", report)

        assert any("Do not imply the task was completed" in c for c in seen[0])

    def test_no_directive_when_applied(self, monkeypatch, tmp_path):
        partner = self._partner(tmp_path)
        seen = self._capture_chat(monkeypatch, partner)
        report = {"summary": "x", "execution_status": "applied", "files_changed": ["a.py"]}

        partner.frontend_finalize("fix it", report)

        assert not any("Do not imply the task was completed" in c for c in seen[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent.py::TestHonestFinalize -v`
Expected: first test FAILS (no directive in messages), second passes

- [ ] **Step 3: Implement**

In `LocalPartner.frontend_finalize` (~line 1376), after the message that appends the backend report and before the `started = time.monotonic()` line, add:

```python
        status = str((backend_report or {}).get("execution_status") or "")
        if status and status != "applied":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "IMPORTANT: No files were changed (or only some plan steps were applied). "
                        "State this plainly as the first sentence of your reply. "
                        "Do not imply the task was completed."
                    ),
                }
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_agent.py::TestHonestFinalize -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add local_code/agent.py tests/test_agent.py
git commit -m "feat: force honest user-facing summary when plan was not applied"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/pytest tests/`
Expected: all tests PASS, no regressions (pay attention to `test_agent.py`, `test_contracts.py`, `test_tui.py`, `test_eval.py` which exercise `run_contract` and report shapes)

- [ ] **Step 2: If anything fails, fix forward**

Most likely regression source: code elsewhere asserting the exact key set of `normalize_backend_report` output. Search with `rg -n "normalize_backend_report" local_code tests` and update any exact-shape assertions to include `execution_status`.

- [ ] **Step 3: Final commit (if fixes were needed)**

```bash
git add -A
git commit -m "test: adjust report-shape assertions for execution_status"
```

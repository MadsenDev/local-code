# Plan→Execute Reliability Design

**Date:** 2026-06-10
**Status:** Approved by user (brainstorming session)
**Scope decision:** Execution reliability first; broader weak-model overhaul and UX polish deferred.
**Model floor:** qwen2.5-coder:7b (the published standard in MODELS.md).
**Approach:** Plan-carrying execute with harness enforcement (Approach A). Step-wise executor (Approach C) noted as future work.

## Problem

When a user approves a plan, `LocalPartner.apply_pending_plan()` (agent.py) copies
the original contract, flips `edit_policy` to `"execute"`, and re-runs the backend
from scratch. The approved plan report — its `plan` steps, `diff_summary`, and the
files already inspected — is discarded. A 7B backend model re-investigates the
repo, finds nothing it is confident about, and emits `final` with zero edits. The
user sees a success-shaped report ("nothing additional to do") even though the
approved work was silently skipped.

Three contributing gaps:

1. The execute contract carries no record of the approved plan.
2. `LocalCodeAgent.run_contract()` accepts an execute-mode `final` with empty
   `files_changed` without pushback, even when a plan was approved.
3. Plan-mode reports are not validated for concreteness — vague steps with no
   real diff reach the user for approval, so even a carried-over plan may not be
   mechanically applicable.

## Design

### 1. Plan-carrying execute contract

`apply_pending_plan()` builds the execute contract with a new field taken from
the stored `pending_plan["report"]`:

```json
"approved_plan": {
  "steps": ["Edit vite.config.ts:server — add host:true ...", "Run npm install"],
  "diff_summary": "--- a/vite.config.ts\n+++ b/vite.config.ts\n@@ ...",
  "files_read": ["vite.config.ts", "package.json"]
}
```

and sets `execution_strategy: "apply_approved_plan"` (replacing the current
`"inspect_then_execute"` for this path).

`_contract_with_intent_scaffold()` returns the contract unchanged when
`approved_plan` is present: intent was analyzed during the plan pass; re-running
it wastes a model call and invites re-interpretation of an already-approved task.

`LocalCodeAgent.system_prompt()` adds a rules block when the contract contains
`approved_plan`:

> An approved plan exists. Your job is to apply each step, not to re-investigate.
> Read a file only if you are about to edit it. Finish only when every step is
> applied, or report exactly which steps you could not apply and why.

### 2. Plan concreteness validation (at plan time)

New function in `contracts.py`:

```python
def validate_plan_report(report, workdir) -> list[str]:
    """Return a list of concreteness problems; empty means the plan is appliable."""
```

Rules:

- Every entry in `report["plan"]` must either reference a file that exists in the
  repo (resolved the same way as `resolve_repo_file_hints`) or be a command step
  (recognizable command text such as `npm install`, `pytest`, `pip install`).
- If any step is a file edit, `report["diff_summary"]` must contain real
  unified-diff markers (`--- `, `+++ `, `@@`).

In `run_contract()`, when a `final` arrives with `edit_policy` in
`{plan, propose}` and validation fails, the harness rejects the final and pushes
the problem list back into the loop as a user message (e.g. "Step 2 names no
existing file — name the file and the exact change"). Maximum **2** pushbacks;
after that the report is accepted but stamped `needs_approval=True` with a risk
entry noting the plan is vague. The tool surfaces weakness honestly instead of
thrashing a 7B model indefinitely.

### 3. No-op rejection in execute mode

In `run_contract()`, when `final` arrives with `edit_policy == "execute"`, an
`approved_plan` present, and both `report["files_changed"]` and the tracker's
`files_changed` empty:

- **First occurrence:** reject the final. Append a user message listing the
  specific unapplied steps:

  > You finished without applying the approved plan. Unapplied steps:
  > 1) Edit vite.config.ts… Apply them now with replace_in_file / write_file /
  > run_command, or state per step why it cannot be applied.

  The loop continues (mirrors the existing `invalid_action_hint` pushback
  pattern).
- **Second occurrence:** accept, but produce an honest failure report with
  `summary: "The plan was approved but not applied."`.

Command-only steps (e.g. `npm install`) count as applied via the tracker's
`commands_run`, not `files_changed`.

### 4. Harness-derived `execution_status`

`normalize_backend_report()` gains an `execution_status` field. It is computed
by the harness from the tracker and the plan steps — never trusted from the
model:

| Value | Condition |
|---|---|
| `applied` | files changed (or commands run) covering every plan step |
| `partially_applied` | some plan steps' files touched, others not (missed steps listed in risks) |
| `plan_not_applied` | no files changed and no plan commands run |

Step-coverage matching: a plan step is *covered* when any repo file path
mentioned in the step text appears in the tracker's `files_changed`, or any
recognizable command text in the step matches a `commands_run` entry by prefix.
Steps containing no recognizable file path or command are excluded from the
status computation (they cannot be verified mechanically).

For execute runs without an `approved_plan`, the field defaults to `applied`
when `files_changed` is non-empty and is omitted otherwise (no behavior change
for non-plan paths).

### 5. Honest user-facing reporting

- `frontend_finalize()` appends a hard directive when `execution_status` is not
  `applied`:

  > No files were changed (or only some plan steps were applied). State this
  > plainly as the first sentence. Do not imply the task was completed.

- The transcript/TUI shows a yellow milestone — `⚠ no edits were made` — emitted
  by the harness, so the truth is visible even if the frontend model softens it.

## Error handling summary

| Failure | Behavior |
|---|---|
| Vague plan final (plan/propose) | Pushback with problem list, max 2 retries, then accept with `needs_approval=True` + vagueness risk |
| No-op final (execute, approved plan) | Pushback listing unapplied steps, 1 retry, then honest failure report |
| Partial application | Accept, `execution_status="partially_applied"`, missed steps in risks, honest finalize directive |
| Model timeout / invalid actions | Unchanged (existing handling) |

## Testing

All offline, following the existing `monkeypatch.setattr(agent, "chat", ...)`
scripted-response pattern in `tests/test_agent.py`:

1. `apply_pending_plan` attaches `approved_plan` + `execution_strategy="apply_approved_plan"` and skips the intent scaffold.
2. Plan-mode final with vague steps → pushback message contains the problem; a second vague final → accepted with `needs_approval=True` and a vagueness risk.
3. Execute no-op final → pushback names unapplied steps; scripted model then edits → `execution_status="applied"`.
4. Execute no-op twice → `execution_status="plan_not_applied"`, summary states the failure plainly.
5. Partial application → `"partially_applied"` with missed steps in risks.
6. `validate_plan_report` unit tests: real file refs pass, missing files fail, command steps pass, missing diff markers fail.
7. `frontend_finalize` receives the no-edit directive when status ≠ `applied`.

## Future work (out of scope)

- **Step-wise executor (Approach C):** execute approved plans one step per
  mini-contract for maximum per-call grounding on weak models. The
  `approved_plan.steps` structure introduced here is the input it needs.
- Broader weak-model reliability overhaul (self-verification, eval coverage).
- UX/output polish beyond the honest-status directive.

"""Reliability eval harness for local models.

Measures how dependably a model drives the backend tool loop — the thing that
actually determines whether Rist "works great" on a given model. It runs
a battery of safe, read-only inspection tasks against a throwaway repo and
scores each outcome.

Run against a live Ollama:

    python -m local_code.eval --model qwen2.5-coder:7b
    python -m local_code.eval --model qwen3:4b --runs 3
    python -m local_code.eval --frontend-model qwen3:4b --backend-model qwen2.5-coder:7b

Outcome categories per task:
    completed     reached `final` with a real, non-stop summary  (good)
    step_limit    ran out of tool steps without finishing
    repeated      looped on the same action without new info
    invalid       never produced a valid tool call
    timeout       the model timed out
    error         the run raised (e.g. Ollama unreachable)
"""

import argparse
import tempfile
from pathlib import Path

STOP_SIGNATURES = {
    "invalid": ("failed to produce a valid", "invalid actions repeatedly"),
    "repeated": ("repeated the same action",),
    "timeout": ("timed out",),
    "step_limit": ("maximum backend tool steps",),
}

# Safe, read-only tasks. Each is a contract the backend loop can satisfy purely
# by inspecting the sandbox repo — no edits, no commands.
EVAL_TASKS = [
    {
        "name": "read_named_file",
        "goal": "Read calculator.py and explain what the add function does.",
        "files_of_interest": ["calculator.py"],
    },
    {
        "name": "search_symbol",
        "goal": "Find where the function 'multiply' is defined in the repo.",
        "files_of_interest": [],
    },
    {
        "name": "describe_repo",
        "goal": "List the Python files in this repo and summarize the project.",
        "files_of_interest": [],
    },
    {
        "name": "locate_constant",
        "goal": "Find the VERSION constant and report its value.",
        "files_of_interest": [],
    },
]


def classify_outcome(report):
    """Map a backend report to an outcome category. Pure; unit-tested."""
    summary = (report.get("summary") or "").lower()
    for category, needles in STOP_SIGNATURES.items():
        if any(needle in summary for needle in needles):
            return category
    return "completed"


def build_sandbox(root):
    root = Path(root)
    (root / "calculator.py").write_text(
        "VERSION = '1.2.3'\n\n\n"
        "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n\n\n"
        "def multiply(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (root / "main.py").write_text(
        "from calculator import add, multiply\n\n"
        "if __name__ == '__main__':\n    print(add(2, 3), multiply(2, 3))\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# sandbox\n\nA tiny calculator used for eval.\n", encoding="utf-8")
    return root


def _make_agent(model, ollama, workdir, max_steps):
    # Imported lazily so `classify_outcome` can be unit-tested without the
    # heavier agent/ollama import chain.
    from .agent import LocalCodeAgent

    return LocalCodeAgent(
        model=model,
        ollama=ollama,
        workdir=workdir,
        command_permission="deny",
        edit_permission="deny",
        max_steps=max_steps,
        verbosity="quiet",
        tool_calling="json",
    )


def run_eval(model, ollama, runs=1, max_steps=8):
    from .model_profiles import classify_model

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        build_sandbox(tmp)
        for _ in range(runs):
            agent = _make_agent(model, ollama, tmp, max_steps)
            for task in EVAL_TASKS:
                contract = {
                    "goal": task["goal"],
                    "scope": ["."],
                    "edit_policy": "inspect",
                    "read_only": True,
                    "files_of_interest": list(task["files_of_interest"]),
                    "task_kind": "inspection",
                }
                try:
                    report = agent.run_contract(contract, "")
                    outcome = classify_outcome(report)
                except Exception as exc:  # noqa: BLE001
                    outcome = "error"
                    report = {"summary": str(exc)}
                results.append({"task": task["name"], "outcome": outcome, "summary": report.get("summary", "")})
    return {"model": model, "profile": classify_model(model), "results": results}


def summarize(eval_result):
    results = eval_result["results"]
    total = len(results) or 1
    completed = sum(1 for r in results if r["outcome"] == "completed")
    counts = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return {
        "reliability": completed / total,
        "completed": completed,
        "total": total,
        "counts": counts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure a model's tool-loop reliability for Rist.")
    parser.add_argument("--model", default=None, help="Model used for both roles")
    parser.add_argument("--backend-model", dest="backend_model", default=None, help="Backend model (defaults to --model)")
    parser.add_argument("--frontend-model", dest="frontend_model", default=None, help="Frontend model (advisory only)")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--runs", type=int, default=1, help="Repeat the task battery N times")
    parser.add_argument("--max-steps", type=int, default=8, help="Max tool steps per task")
    args = parser.parse_args(argv)

    backend = args.backend_model or args.model or "qwen2.5-coder:7b"
    eval_result = run_eval(backend, args.ollama, runs=args.runs, max_steps=args.max_steps)
    stats = summarize(eval_result)
    profile = eval_result["profile"]

    print(f"Model:       {backend}")
    print(f"Tier:        {profile.tier}  (fit={profile.fit}, reliability={profile.reliability})")
    for note in profile.notes:
        print(f"  · {note}")
    print(f"Reliability: {stats['reliability'] * 100:.0f}%  ({stats['completed']}/{stats['total']} tasks completed)")
    print(f"Outcomes:    {stats['counts']}")
    if stats["reliability"] < 0.75:
        print("Verdict:     below the recommended standard — expect frequent failures on this model.")
    elif stats["reliability"] < 0.95:
        print("Verdict:     usable, but not flawless — fine for inspection, watch multi-step edits.")
    else:
        print("Verdict:     meets the standard — drives the tool loop reliably.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

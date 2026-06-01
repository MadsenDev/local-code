from local_code.eval import EVAL_TASKS, build_sandbox, classify_outcome, summarize


class TestClassifyOutcome:
    def test_completed(self):
        assert classify_outcome({"summary": "calculator.py defines add and multiply"}) == "completed"

    def test_invalid(self):
        assert classify_outcome({"summary": "Stopped because the backend failed to produce a valid tool call."}) == "invalid"

    def test_repeated(self):
        assert classify_outcome({"summary": "Stopped because the backend repeated the same action without new information."}) == "repeated"

    def test_step_limit(self):
        assert classify_outcome({"summary": "Stopped after reaching the maximum backend tool steps."}) == "step_limit"

    def test_timeout(self):
        assert classify_outcome({"summary": "Backend model timed out while step 1/8."}) == "timeout"


class TestSummarize:
    def test_reliability_fraction(self):
        result = {
            "results": [
                {"outcome": "completed"},
                {"outcome": "completed"},
                {"outcome": "invalid"},
                {"outcome": "step_limit"},
            ]
        }
        stats = summarize(result)
        assert stats["completed"] == 2
        assert stats["total"] == 4
        assert stats["reliability"] == 0.5
        assert stats["counts"]["completed"] == 2


class TestSandbox:
    def test_builds_expected_files(self, tmp_path):
        build_sandbox(tmp_path)
        assert (tmp_path / "calculator.py").exists()
        assert "def add" in (tmp_path / "calculator.py").read_text()
        assert (tmp_path / "main.py").exists()
        assert len(EVAL_TASKS) >= 3

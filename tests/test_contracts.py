from local_code.contracts import infer_file_hints, normalize_backend_report


class TestInferFileHints:
    def test_finds_real_files(self):
        hints = infer_file_hints("look at src/main.py and config.js")
        assert "src/main.py" in hints
        assert "config.js" in hints

    def test_finds_path_with_directory(self):
        hints = infer_file_hints("edit electron/main.ts")
        assert "electron/main.ts" in hints

    def test_skips_abbreviations(self):
        hints = infer_file_hints("e.g. run the tests, i.e. pytest")
        assert not any(h in ("e.g", "i.e") for h in hints)

    def test_skips_version_numbers(self):
        hints = infer_file_hints("version 1.0 or 2.1.3 was released")
        assert "1.0" not in hints
        assert "2.1" not in hints

    def test_single_char_basename_skipped(self):
        hints = infer_file_hints("see e.go for details")
        # "e.go" has single-char basename — should be filtered
        assert "e.go" not in hints

    def test_real_go_file_kept(self):
        hints = infer_file_hints("edit main.go")
        assert "main.go" in hints


class TestNormalizeBackendReport:
    def test_normalizes_dict(self):
        report = normalize_backend_report({"summary": "ok", "needs_approval": True})
        assert report["summary"] == "ok"
        assert report["needs_approval"] is True
        assert report["findings"] == []

    def test_parses_json_string(self):
        import json
        raw = json.dumps({"summary": "done", "files_read": ["a.py"]})
        report = normalize_backend_report(raw)
        assert report["summary"] == "done"
        assert "a.py" in report["files_read"]

    def test_fallback_on_bad_input(self):
        report = normalize_backend_report(None, fallback_message="fallback")
        assert report["summary"] == "fallback"

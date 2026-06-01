import urllib.error

import local_code.models as models


class TestChatPayload:
    def test_includes_keep_alive_and_options(self):
        payload = models._chat_payload("m", [{"role": "user", "content": "hi"}], temperature=0.0, num_ctx=8192)
        assert payload["keep_alive"]
        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["num_ctx"] == 8192
        assert "format" not in payload

    def test_format_is_attached_when_given(self):
        payload = models._chat_payload("m", [], fmt="json")
        assert payload["format"] == "json"

    def test_env_overrides_num_ctx(self, monkeypatch):
        monkeypatch.setenv("LOCAL_CODE_NUM_CTX", "2048")
        payload = models._chat_payload("m", [], num_ctx=16384)
        assert payload["options"]["num_ctx"] == 2048

    def test_bad_env_num_ctx_falls_back(self, monkeypatch):
        monkeypatch.setenv("LOCAL_CODE_NUM_CTX", "notanint")
        payload = models._chat_payload("m", [], num_ctx=16384)
        assert payload["options"]["num_ctx"] == 16384


class TestTransientRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def flaky(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.URLError("connection reset")
            return {"message": {"content": "ok"}}

        monkeypatch.setattr(models, "_raw_post", flaky)
        monkeypatch.setattr(models.time, "sleep", lambda *_: None)

        data = models._post_chat("http://x", {"model": "m"}, timeout=5)
        assert data["message"]["content"] == "ok"
        assert calls["n"] == 3

    def test_does_not_retry_non_transient(self, monkeypatch):
        calls = {"n": 0}

        def boom(url, payload, timeout):
            calls["n"] += 1
            raise urllib.error.HTTPError("u", 404, "not found", {}, None)

        monkeypatch.setattr(models, "_raw_post", boom)
        monkeypatch.setattr(models.time, "sleep", lambda *_: None)

        try:
            models._post_chat("http://x", {"model": "m"}, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("expected HTTPError")
        assert calls["n"] == 1


class TestFormatFallback:
    def test_schema_400_downgrades_to_json_then_plain(self, monkeypatch):
        seen = []

        def fake_post(url, payload, timeout):
            seen.append(payload.get("format"))
            if isinstance(payload.get("format"), dict):
                raise urllib.error.HTTPError("u", 400, "bad", {}, None)
            if payload.get("format") == "json":
                raise urllib.error.HTTPError("u", 400, "bad", {}, None)
            return {"message": {"content": "ok"}}

        monkeypatch.setattr(models, "_post_chat", fake_post)
        schema = {"type": "object"}
        data = models._post_chat_with_format("http://x", {"model": "m", "format": schema}, timeout=5)
        assert data["message"]["content"] == "ok"
        assert seen == [schema, "json", None]


class TestCapabilityProbes:
    def test_model_is_available_matches_latest(self, monkeypatch):
        monkeypatch.setattr(models, "list_models", lambda *a, **k: {"qwen2.5-coder:7b", "qwen3:latest"})
        assert models.model_is_available("http://x", "qwen2.5-coder:7b") is True
        assert models.model_is_available("http://x", "qwen3") is True
        assert models.model_is_available("http://x", "missing:1b") is False

    def test_model_is_available_unknown_when_empty(self, monkeypatch):
        monkeypatch.setattr(models, "list_models", lambda *a, **k: set())
        assert models.model_is_available("http://x", "anything") is None

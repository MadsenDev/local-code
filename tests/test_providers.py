import urllib.error

import local_code.providers as providers
from local_code.providers import (
    OpenAICompatibleProvider,
    OllamaProvider,
    _response_format,
    build_provider,
)


class TestResponseFormat:
    def test_none(self):
        assert _response_format(None) is None

    def test_json_object(self):
        assert _response_format("json") == {"type": "json_object"}

    def test_json_schema(self):
        rf = _response_format({"type": "object"})
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == {"type": "object"}
        assert rf["json_schema"]["strict"] is True


class TestOpenAIPayload:
    def _provider(self):
        return OpenAICompatibleProvider("https://x/v1", api_key="k", name="openrouter")

    def test_chat_result_parses_choices(self, monkeypatch):
        p = self._provider()
        seen = {}

        def fake_post(payload, timeout):
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "hi there", "tool_calls": []}}]}

        monkeypatch.setattr(p, "_post_with_format", fake_post)
        result = p.chat_result("m", [{"role": "user", "content": "x"}], fmt="json", temperature=0.0)
        assert result.content == "hi there"
        assert seen["payload"]["response_format"] == {"type": "json_object"}
        assert seen["payload"]["temperature"] == 0.0
        assert seen["payload"]["stream"] is False

    def test_chat_tools_passes_tools_and_low_temp(self, monkeypatch):
        p = self._provider()
        seen = {}

        def fake_post(payload, timeout):
            seen["payload"] = payload
            return {"choices": [{"message": {"content": "", "tool_calls": [{"function": {"name": "read_file"}}]}}]}

        monkeypatch.setattr(p, "_post_with_format", fake_post)
        result = p.chat_tools("m", [], [{"type": "function", "function": {"name": "read_file"}}])
        assert seen["payload"]["tools"]
        assert seen["payload"]["temperature"] == 0.0
        assert result.tool_calls[0]["function"]["name"] == "read_file"

    def test_headers_include_auth_and_extra(self):
        p = OpenAICompatibleProvider("https://x/v1", api_key="secret", name="openrouter", extra_headers={"X-Title": "custom-client"})
        headers = p._headers()
        assert headers["Authorization"] == "Bearer secret"
        assert headers["X-Title"] == "custom-client"


class TestFormatFallback:
    def test_schema_400_downgrades_to_json_object_then_plain(self, monkeypatch):
        p = OpenAICompatibleProvider("https://x/v1", api_key="k")
        seen = []

        def fake_post(path, payload, timeout):
            seen.append(payload.get("response_format"))
            rf = payload.get("response_format")
            if rf and rf.get("type") == "json_schema":
                raise urllib.error.HTTPError("u", 400, "bad", {}, None)
            if rf and rf.get("type") == "json_object":
                raise urllib.error.HTTPError("u", 400, "bad", {}, None)
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(p, "_post", fake_post)
        data = p._post_with_format({"response_format": {"type": "json_schema", "json_schema": {}}}, timeout=5)
        assert data["choices"][0]["message"]["content"] == "ok"
        assert [rf and rf.get("type") for rf in seen] == ["json_schema", "json_object", None]


class TestStream:
    def test_sse_parsing(self, monkeypatch):
        p = OpenAICompatibleProvider("https://x/v1", api_key="k")

        class FakeResp:
            def __init__(self, lines):
                self._lines = [ln.encode("utf-8") for ln in lines]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return iter(self._lines)

        lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
            "data: [DONE]\n",
        ]
        monkeypatch.setattr(providers.urllib.request, "urlopen", lambda req, timeout=None: FakeResp(lines))
        chunks = list(p.stream("m", [{"role": "user", "content": "hi"}]))
        assert "".join(chunks) == "Hello"


class TestBuildProvider:
    def test_ollama(self):
        p = build_provider("ollama", base_url="http://127.0.0.1:11434")
        assert isinstance(p, OllamaProvider)
        assert p.is_local is True

    def test_openrouter_sets_base_and_headers(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "rk")
        p = build_provider("openrouter")
        assert isinstance(p, OpenAICompatibleProvider)
        assert p.name == "openrouter"
        assert p.base_url == providers.OPENROUTER_BASE_URL
        assert p.api_key == "rk"
        assert p._headers()["X-Title"] == "Rist"

    def test_openai_uses_env_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "ok")
        p = build_provider("openai")
        assert p.api_key == "ok"
        assert p.base_url == providers.OPENAI_BASE_URL

    def test_unknown_raises(self):
        try:
            build_provider("nope")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_model_available_uses_list(self, monkeypatch):
        p = OpenAICompatibleProvider("https://x/v1", api_key="k")
        monkeypatch.setattr(p, "list_models", lambda: {"a/b", "c/d"})
        assert p.model_available("a/b") is True
        assert p.model_available("z/z") is False
        monkeypatch.setattr(p, "list_models", lambda: set())
        assert p.model_available("a/b") is None

    def test_unavailable_without_key(self):
        p = OpenAICompatibleProvider("https://x/v1", api_key=None)
        assert p.available() is False


def test_llamacpp_reuses_openai_transport_without_api_key(monkeypatch):
    p = build_provider("llamacpp")
    assert isinstance(p, providers.LlamaCppProvider)
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.is_local is True
    assert p.is_heavy_backend is True
    assert p.base_url == "http://127.0.0.1:8080/v1"

    monkeypatch.setattr(p, "_get_json", lambda path, timeout=10: {"data": [{"id": "local"}]})
    assert p.available() is True


def test_llamacpp_failure_message_is_actionable():
    p = build_provider("llamacpp", base_url="http://localhost:9000/v1")
    message = p.failure_message()
    assert "http://localhost:9000/v1" in message
    assert "rist llama command" in message


def test_llamacpp_local_alias_accepts_reported_model_name(monkeypatch):
    p = build_provider("llamacpp")
    monkeypatch.setattr(p, "list_models", lambda: {"/models/qwen.gguf"})
    assert p.model_available("local") is True
    assert p.model_available("different") is False

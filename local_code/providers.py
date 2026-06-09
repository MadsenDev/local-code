"""Model provider abstraction.

`local-code` talks to models through a small provider interface so the same
agent/tool loop can run against a local Ollama server or any OpenAI-compatible
HTTP API (OpenRouter, OpenAI, Together, Groq, a local vLLM/llama.cpp server,
etc.). Two implementations ship here:

- ``OllamaProvider``          — wraps the native Ollama client in ``models.py``.
- ``OpenAICompatibleProvider`` — speaks the OpenAI ``/chat/completions`` shape.

Providers expose a uniform surface (``chat``, ``chat_result``, ``chat_tools``,
``assess``, ``stream``) plus capability probes (``available``, ``list_models``,
``model_available``). Structured-output handling (the big weak-model reliability
lever) is implemented for both: a JSON-schema ``format`` degrades gracefully to
plain JSON mode and then to no constraint, so calls complete on any backend.
"""

import json
import os
import time
import urllib.error
import urllib.request

from .config import (
    ASSESS_NUM_CTX,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA,
    MODEL_MAX_RETRIES,
    MODEL_REQUEST_TIMEOUT,
    MODEL_RETRY_BACKOFF,
    PROSE_TEMPERATURE,
    STRUCTURED_TEMPERATURE,
)
from .llamacpp import DEFAULT_LLAMACPP_BASE_URL
from .models import (
    OllamaChatResult,
    _is_transient,
    list_models as ollama_list_models,
    model_is_available as ollama_model_available,
    ollama_assess,
    ollama_chat,
    ollama_chat_result,
    ollama_chat_tools,
    ollama_stream,
    server_available as ollama_server_available,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"


class ChatResult:
    """Uniform result shape shared by all providers."""

    def __init__(self, content="", tool_calls=None, raw=None):
        self.content = (content or "").strip()
        self.tool_calls = tool_calls or []
        self.raw = raw or {}


class Provider:
    """Interface every provider implements. Subclasses override the verbs."""

    name = "provider"
    is_local = False

    def chat(self, model, messages, *, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX, timeout=MODEL_REQUEST_TIMEOUT):
        raise NotImplementedError

    def chat_result(self, model, messages, *, tools=None, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX, timeout=MODEL_REQUEST_TIMEOUT):
        raise NotImplementedError

    def chat_tools(self, model, messages, tools, *, timeout=MODEL_REQUEST_TIMEOUT):
        raise NotImplementedError

    def assess(self, model, messages, *, timeout=60):
        raise NotImplementedError

    def stream(self, model, messages, *, temperature=PROSE_TEMPERATURE, timeout=MODEL_REQUEST_TIMEOUT):
        raise NotImplementedError

    def available(self):
        return True

    def list_models(self):
        return set()

    def model_available(self, model):
        return None

    def describe(self):
        return self.name


class OllamaProvider(Provider):
    name = "ollama"
    is_local = True

    def __init__(self, base_url=DEFAULT_OLLAMA):
        self.base_url = (base_url or DEFAULT_OLLAMA).rstrip("/")

    def chat(self, model, messages, *, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX, timeout=MODEL_REQUEST_TIMEOUT):
        return ollama_chat(self.base_url, model, messages, fmt=fmt, temperature=temperature, num_ctx=num_ctx, timeout=timeout)

    def chat_result(self, model, messages, *, tools=None, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX, timeout=MODEL_REQUEST_TIMEOUT):
        return ollama_chat_result(self.base_url, model, messages, tools=tools, fmt=fmt, temperature=temperature, num_ctx=num_ctx, timeout=timeout)

    def chat_tools(self, model, messages, tools, *, timeout=MODEL_REQUEST_TIMEOUT):
        return ollama_chat_tools(self.base_url, model, messages, tools, timeout=timeout)

    def assess(self, model, messages, *, timeout=60):
        return ollama_assess(self.base_url, model, messages, timeout=timeout)

    def stream(self, model, messages, *, temperature=PROSE_TEMPERATURE, timeout=MODEL_REQUEST_TIMEOUT):
        yield from ollama_stream(self.base_url, model, messages, timeout=timeout)

    def available(self):
        return ollama_server_available(self.base_url)

    def list_models(self):
        return ollama_list_models(self.base_url)

    def model_available(self, model):
        return ollama_model_available(self.base_url, model)

    def describe(self):
        return f"ollama @ {self.base_url}"


def _response_format(fmt):
    """Map our internal `fmt` to an OpenAI ``response_format`` block."""
    if fmt is None:
        return None
    if fmt == "json":
        return {"type": "json_object"}
    if isinstance(fmt, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": fmt}}
    return None


class OpenAICompatibleProvider(Provider):
    is_local = False

    def __init__(self, base_url, api_key=None, name="openai", extra_headers=None):
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.name = name
        self.extra_headers = extra_headers or {}

    # -- HTTP plumbing --------------------------------------------------
    def _headers(self):
        headers = {"Content-Type": "application/json", "User-Agent": "local-code/0.2"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _post(self, path, payload, timeout):
        url = f"{self.base_url}{path}"
        last_exc = None
        for attempt in range(MODEL_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=self._headers(),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.load(resp)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == MODEL_MAX_RETRIES - 1 or not _is_transient(exc):
                    raise
                time.sleep(MODEL_RETRY_BACKOFF * (2 ** attempt))
        raise last_exc

    def _payload(self, model, messages, *, stream, temperature, tools=None, fmt=None):
        payload = {"model": model, "messages": messages, "stream": stream, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        rf = _response_format(fmt)
        if rf is not None:
            payload["response_format"] = rf
        return payload

    def _post_with_format(self, payload, timeout):
        """POST, degrading response_format json_schema -> json_object -> none on 400."""
        rf = payload.get("response_format")
        try:
            return self._post("/chat/completions", payload, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or rf is None:
                raise
            if rf.get("type") == "json_schema":
                downgraded = dict(payload, response_format={"type": "json_object"})
                try:
                    return self._post("/chat/completions", downgraded, timeout)
                except urllib.error.HTTPError as exc2:
                    if exc2.code != 400:
                        raise
            plain = {k: v for k, v in payload.items() if k != "response_format"}
            return self._post("/chat/completions", plain, timeout)

    # -- Provider verbs -------------------------------------------------
    @staticmethod
    def _result_from(data):
        choices = data.get("choices") or [{}]
        message = (choices[0] or {}).get("message") or {}
        return ChatResult(content=message.get("content") or "", tool_calls=message.get("tool_calls") or [], raw=data)

    def chat_result(self, model, messages, *, tools=None, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX, timeout=MODEL_REQUEST_TIMEOUT):
        payload = self._payload(model, messages, stream=False, temperature=temperature, tools=tools, fmt=fmt)
        data = self._post_with_format(payload, timeout)
        return self._result_from(data)

    def chat(self, model, messages, *, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX, timeout=MODEL_REQUEST_TIMEOUT):
        return self.chat_result(model, messages, fmt=fmt, temperature=temperature, timeout=timeout).content

    def chat_tools(self, model, messages, tools, *, timeout=MODEL_REQUEST_TIMEOUT):
        return self.chat_result(model, messages, tools=tools, temperature=STRUCTURED_TEMPERATURE, timeout=timeout)

    def assess(self, model, messages, *, timeout=60):
        return self.chat_result(model, messages, fmt="json", temperature=STRUCTURED_TEMPERATURE, timeout=timeout).content

    def stream(self, model, messages, *, temperature=PROSE_TEMPERATURE, timeout=MODEL_REQUEST_TIMEOUT):
        payload = self._payload(model, messages, stream=True, temperature=temperature)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or [{}]
                delta = (choices[0] or {}).get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

    def available(self):
        if not self.api_key:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.load(resp)
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_models(self):
        try:
            req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)
        except Exception:  # noqa: BLE001
            return set()
        return {m.get("id", "") for m in (data.get("data") or []) if m.get("id")}

    def model_available(self, model):
        names = self.list_models()
        if not names:
            return None
        return model in names

    def describe(self):
        key = "key set" if self.api_key else "NO API KEY"
        return f"{self.name} @ {self.base_url} ({key})"



class LlamaCppProvider(OpenAICompatibleProvider):
    """Thin adapter for an externally managed llama-server.

    Inference, model loading, quantization, offload, batching, and KV cache
    remain entirely owned by llama.cpp.
    """

    name = "llamacpp"
    is_local = True
    is_heavy_backend = True

    def __init__(self, base_url=DEFAULT_LLAMACPP_BASE_URL, api_key=None):
        super().__init__(base_url or DEFAULT_LLAMACPP_BASE_URL, api_key=api_key, name=self.name)

    def available(self):
        try:
            self._get_json("/models", timeout=5)
            return True
        except Exception:  # noqa: BLE001 - availability probes are intentionally non-fatal
            return False

    def _get_json(self, path, timeout=10):
        req = urllib.request.Request(f"{self.base_url}{path}", headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    @property
    def server_root(self):
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def model_available(self, model):
        names = self.list_models()
        if not names:
            return None
        # ``local`` is the documented single-server alias. llama-server has one
        # loaded model even when /models reports its path or metadata name.
        return bool(names) if model == "local" else model in names

    def server_metadata(self):
        """Best-effort llama-server health, properties, slots, and metrics."""
        metadata = {}
        for label, path in (("health", "/health"), ("properties", "/props"), ("slots", "/slots")):
            try:
                req = urllib.request.Request(f"{self.server_root}{path}", headers=self._headers())
                with urllib.request.urlopen(req, timeout=3) as resp:
                    metadata[label] = json.load(resp)
            except Exception:  # noqa: BLE001 - endpoints vary by llama.cpp build
                continue
        try:
            req = urllib.request.Request(f"{self.server_root}/metrics", headers=self._headers())
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = resp.read(16_384).decode("utf-8", errors="replace").strip()
                if body:
                    metadata["metrics"] = body
        except Exception:  # noqa: BLE001 - metrics must be enabled server-side
            pass
        return metadata

    def describe(self):
        return f"llama.cpp @ {self.base_url} (external llama-server)"

    def failure_message(self):
        return (
            f"llama.cpp provider is configured, but no server responded at {self.base_url}.\n\n"
            "Start llama-server first, or change the base_url in your config.\n"
            "Run: local-code model start qwen36 --gpu rtx3060\n"
            "Or print flags: local-code llama command --profile qwen36-35b-a3b --gpu rtx3060"
        )


def build_provider(kind, *, base_url=None, api_key=None):
    """Construct a provider from a CLI/config selection.

    kind: "ollama" | "llamacpp" | "openrouter" | "openai".
    """
    kind = (kind or "ollama").lower()
    if kind == "ollama":
        return OllamaProvider(base_url or DEFAULT_OLLAMA)
    if kind in {"llamacpp", "llama.cpp", "llama-cpp"}:
        return LlamaCppProvider(base_url or DEFAULT_LLAMACPP_BASE_URL, api_key=api_key)
    if kind == "openrouter":
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        return OpenAICompatibleProvider(
            base_url or OPENROUTER_BASE_URL,
            api_key=key,
            name="openrouter",
            extra_headers={"X-Title": "local-code", "HTTP-Referer": "https://github.com/MadsenDev/local-code"},
        )
    if kind in {"openai", "openai-compatible"}:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleProvider(base_url or OPENAI_BASE_URL, api_key=key, name="openai")
    raise ValueError(f"Unknown provider {kind!r}. Use ollama, llamacpp, openrouter, or openai.")

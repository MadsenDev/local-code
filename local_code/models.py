import json
import os
import time
import urllib.error
import urllib.request

from .config import (
    ASSESS_NUM_CTX,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_NUM_CTX,
    MODEL_MAX_RETRIES,
    MODEL_REQUEST_TIMEOUT,
    MODEL_RETRY_BACKOFF,
    PROSE_TEMPERATURE,
    STRUCTURED_TEMPERATURE,
)


class OllamaChatResult:
    def __init__(self, content="", tool_calls=None, raw=None):
        self.content = (content or "").strip()
        self.tool_calls = tool_calls or []
        self.raw = raw or {}


def _env_num_ctx(default):
    value = os.environ.get("LOCAL_CODE_NUM_CTX")
    if value:
        try:
            return max(512, int(value))
        except ValueError:
            return default
    return default


def _keep_alive():
    return os.environ.get("LOCAL_CODE_KEEP_ALIVE", DEFAULT_KEEP_ALIVE)


def _chat_payload(model, messages, stream=False, num_ctx=DEFAULT_NUM_CTX, temperature=PROSE_TEMPERATURE, tools=None, fmt=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_ctx": _env_num_ctx(num_ctx), "temperature": temperature},
        "keep_alive": _keep_alive(),
    }
    if tools:
        payload["tools"] = tools
    if fmt is not None:
        payload["format"] = fmt
    return payload


def _is_transient(exc):
    """Worth retrying: server still loading the model, connection blips, 5xx."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def _raw_post(ollama_base_url, payload, timeout):
    req = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "rist/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _post_chat(ollama_base_url, payload, timeout):
    """POST with bounded exponential backoff on transient failures."""
    last_exc = None
    for attempt in range(MODEL_MAX_RETRIES):
        try:
            return _raw_post(ollama_base_url, payload, timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == MODEL_MAX_RETRIES - 1 or not _is_transient(exc):
                raise
            time.sleep(MODEL_RETRY_BACKOFF * (2 ** attempt))
    raise last_exc


def _post_chat_with_format(ollama_base_url, payload, timeout):
    """Post a (possibly schema-constrained) request, degrading format gracefully.

    A JSON-schema ``format`` is the single biggest reliability win for weak
    models, but older Ollama builds reject a schema object. On a 400 we fall
    back to plain ``"json"`` mode, then to no format at all, so the call still
    completes on any Ollama version.
    """
    fmt = payload.get("format")
    try:
        return _post_chat(ollama_base_url, payload, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 400 or fmt is None:
            raise
        if isinstance(fmt, dict):
            downgraded = dict(payload, format="json")
            try:
                return _post_chat(ollama_base_url, downgraded, timeout)
            except urllib.error.HTTPError as exc2:
                if exc2.code != 400:
                    raise
        plain = {k: v for k, v in payload.items() if k != "format"}
        return _post_chat(ollama_base_url, plain, timeout)


def ollama_chat_result(ollama_base_url, model, messages, timeout=MODEL_REQUEST_TIMEOUT, tools=None, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX):
    payload = _chat_payload(model, messages, stream=False, tools=tools, fmt=fmt, temperature=temperature, num_ctx=num_ctx)
    data = _post_chat_with_format(ollama_base_url, payload, timeout)
    message = data.get("message") or {}
    return OllamaChatResult(
        content=message.get("content") or "",
        tool_calls=message.get("tool_calls") or [],
        raw=data,
    )


def ollama_chat(ollama_base_url, model, messages, timeout=MODEL_REQUEST_TIMEOUT, fmt=None, temperature=PROSE_TEMPERATURE, num_ctx=DEFAULT_NUM_CTX):
    return ollama_chat_result(
        ollama_base_url, model, messages, timeout=timeout, fmt=fmt, temperature=temperature, num_ctx=num_ctx
    ).content


def ollama_chat_tools(ollama_base_url, model, messages, tools, timeout=MODEL_REQUEST_TIMEOUT):
    # Native tool calls are already structured; decode them deterministically.
    return ollama_chat_result(
        ollama_base_url, model, messages, timeout=timeout, tools=tools, temperature=STRUCTURED_TEMPERATURE
    )


def ollama_assess(ollama_base_url, model, messages, timeout=60):
    # Short JSON-only passes (complexity, intent, memory). Constrain hard.
    payload = _chat_payload(
        model, messages, stream=False, num_ctx=ASSESS_NUM_CTX, temperature=STRUCTURED_TEMPERATURE, fmt="json"
    )
    data = _post_chat_with_format(ollama_base_url, payload, timeout)
    return ((data.get("message") or {}).get("content") or "").strip()


def ollama_stream(ollama_base_url, model, messages, timeout=MODEL_REQUEST_TIMEOUT):
    """Yield text chunks from Ollama's streaming chat endpoint (user-facing prose)."""
    payload = _chat_payload(model, messages, stream=True, temperature=PROSE_TEMPERATURE)
    req = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "rist/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (chunk.get("message") or {}).get("content") or ""
            if content:
                yield content
            if chunk.get("done"):
                break


# --- Capability probes (used by startup preflight; all best-effort) ---------

def server_available(ollama_base_url, timeout=5):
    try:
        req = urllib.request.Request(
            f"{ollama_base_url.rstrip('/')}/api/version",
            headers={"User-Agent": "rist/0.2"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.load(resp)
        return True
    except Exception:  # noqa: BLE001
        return False


def list_models(ollama_base_url, timeout=10):
    """Return the set of locally available model tags, or empty set on failure."""
    try:
        req = urllib.request.Request(
            f"{ollama_base_url.rstrip('/')}/api/tags",
            headers={"User-Agent": "rist/0.2"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception:  # noqa: BLE001
        return set()
    return {m.get("name", "") for m in (data.get("models") or []) if m.get("name")}


def model_is_available(ollama_base_url, model, timeout=10):
    """True if `model` (with or without an explicit :tag) is pulled locally."""
    names = list_models(ollama_base_url, timeout=timeout)
    if not names:
        return None  # could not determine
    if model in names:
        return True
    # Ollama implicitly appends :latest; accept either direction of that.
    base = model.split(":", 1)[0]
    return any(n == model or n.split(":", 1)[0] == base and model == base for n in names) or f"{model}:latest" in names

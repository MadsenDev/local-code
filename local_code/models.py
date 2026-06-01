import json
import urllib.request


class OllamaChatResult:
    def __init__(self, content="", tool_calls=None, raw=None):
        self.content = (content or "").strip()
        self.tool_calls = tool_calls or []
        self.raw = raw or {}


def _chat_payload(model, messages, stream=False, options_ctx=16384, tools=None):
    payload = {"model": model, "messages": messages, "stream": stream, "options": {"num_ctx": options_ctx}}
    if tools:
        payload["tools"] = tools
    return payload


def _post_chat(ollama_base_url, payload, timeout):
    req = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "local-code/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ollama_chat_result(ollama_base_url, model, messages, timeout=300, tools=None):
    payload = _chat_payload(model, messages, stream=False, tools=tools)
    data = _post_chat(ollama_base_url, payload, timeout)
    message = data.get("message") or {}
    return OllamaChatResult(
        content=message.get("content") or "",
        tool_calls=message.get("tool_calls") or [],
        raw=data,
    )


def ollama_chat(ollama_base_url, model, messages, timeout=300):
    return ollama_chat_result(ollama_base_url, model, messages, timeout=timeout).content


def ollama_chat_tools(ollama_base_url, model, messages, tools, timeout=300):
    return ollama_chat_result(ollama_base_url, model, messages, timeout=timeout, tools=tools)


def ollama_assess(ollama_base_url, model, messages, timeout=60):
    payload = _chat_payload(model, messages, stream=False, options_ctx=4096)
    data = _post_chat(ollama_base_url, payload, timeout)
    return ((data.get("message") or {}).get("content") or "").strip()


def ollama_stream(ollama_base_url, model, messages, timeout=300):
    """Yield text chunks from Ollama's streaming chat endpoint."""
    payload = _chat_payload(model, messages, stream=True)
    req = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "local-code/0.2"},
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

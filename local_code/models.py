import json
import urllib.request


def ollama_chat(ollama_base_url, model, messages, timeout=300):
    payload = {"model": model, "messages": messages, "stream": False, "options": {"num_ctx": 32768}}
    req = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "local-code/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return ((data.get("message") or {}).get("content") or "").strip()


def ollama_stream(ollama_base_url, model, messages, timeout=300):
    """Yield text chunks from Ollama's streaming chat endpoint."""
    payload = {"model": model, "messages": messages, "stream": True, "options": {"num_ctx": 32768}}
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

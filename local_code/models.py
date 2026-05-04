import json
import urllib.request


def ollama_chat(ollama_base_url, model, messages, timeout=300):
    payload = {"model": model, "messages": messages, "stream": False}
    req = urllib.request.Request(
        f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "local-code/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return ((data.get("message") or {}).get("content") or "").strip()

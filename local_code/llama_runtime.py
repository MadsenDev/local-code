"""Lifecycle management for an external llama.cpp ``llama-server`` process.

This module orchestrates an executable and model files. It does not link to
llama.cpp or implement inference, GGUF parsing, quantization, offload, batching,
or KV-cache behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .llamacpp import DEFAULT_LLAMACPP_BASE_URL, generate_llama_server_args, get_llamacpp_profile


def local_code_home() -> Path:
    return Path(os.environ.get("LOCAL_CODE_HOME", Path.home() / ".local-code")).expanduser()


def runtime_dir() -> Path:
    return local_code_home() / "runtimes" / "llamacpp"


def models_dir() -> Path:
    return local_code_home() / "models" / "llamacpp"


def state_path() -> Path:
    return runtime_dir() / "server.json"


def registry_path() -> Path:
    return models_dir() / "models.json"


def find_llama_server(explicit: str | None = None) -> str | None:
    """Find llama-server without downloading or compiling it."""
    candidates = [explicit, os.environ.get("LLAMA_SERVER")]
    for command in ("llama-server", "llama-server.exe"):
        path_hit = shutil.which(command)
        if path_hit:
            candidates.append(path_hit)
    candidates.extend([
        str(runtime_dir() / "bin" / "llama-server"),
        str(local_code_home() / "bin" / "llama-server"),
    ])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def _download(url: str, destination: Path, *, sha256: str | None = None, force=False, timeout=60):
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"http", "https", "file"}:
        raise ValueError("Artifact URL must use http, https, or file.")
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to replace it.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    downloaded = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "local-code/0.2"})
        with urllib.request.urlopen(req, timeout=timeout) as response, partial.open("wb") as output:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
        actual = digest.hexdigest()
        if sha256 and actual.lower() != sha256.lower():
            raise ValueError(f"SHA-256 mismatch: expected {sha256.lower()}, got {actual}.")
        partial.replace(destination)
        return {"path": str(destination), "bytes": downloaded, "sha256": actual, "content_length": total}
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def install_llama_server(url: str, *, destination: str | None = None, sha256: str | None = None, force=False):
    """Download a prebuilt llama-server binary selected by the user."""
    target = Path(destination).expanduser() if destination else runtime_dir() / "bin" / "llama-server"
    report = _download(url, target, sha256=sha256, force=force)
    target.chmod(target.stat().st_mode | 0o755)
    report["executable"] = str(target.resolve())
    return report


def _load_registry():
    try:
        data = json.loads(registry_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_registry(registry):
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def model_key(profile_id: str) -> str:
    profile = get_llamacpp_profile(profile_id)
    return profile["id"]


def install_model(profile_id: str, url: str, *, filename: str | None = None, sha256: str | None = None, force=False):
    """Download and register a GGUF selected by profile and explicit URL."""
    profile = get_llamacpp_profile(profile_id)
    name = filename or Path(urllib.parse.urlparse(url).path).name
    if filename and Path(filename).name != filename:
        raise ValueError("--filename must be a filename, not a path.")
    if not name:
        name = profile["id"] + ".gguf"
    if not name.lower().endswith(".gguf"):
        raise ValueError("The model destination must use a .gguf filename.")
    target = models_dir() / profile["id"] / name
    report = _download(url, target, sha256=sha256, force=force, timeout=120)
    registry = _load_registry()
    registry[profile["id"]] = {
        "profile": profile["id"],
        "name": profile["name"],
        "path": str(target.resolve()),
        "source_url": url,
        "sha256": report["sha256"],
        "installed_at": int(time.time()),
    }
    _save_registry(registry)
    report["profile"] = profile["id"]
    return report


def register_model(profile_id: str, model_path: str):
    """Register an existing GGUF without copying or parsing it."""
    profile = get_llamacpp_profile(profile_id)
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    if path.suffix.lower() != ".gguf":
        raise ValueError("The model path must point to a .gguf file.")
    registry = _load_registry()
    registry[profile["id"]] = {
        "profile": profile["id"],
        "name": profile["name"],
        "path": str(path),
        "source_url": None,
        "sha256": None,
        "installed_at": int(time.time()),
    }
    _save_registry(registry)
    return registry[profile["id"]]


def list_managed_models():
    registry = _load_registry()
    result = []
    for key, entry in registry.items():
        item = dict(entry)
        item["id"] = key
        item["exists"] = Path(item.get("path", "")).is_file()
        result.append(item)
    return sorted(result, key=lambda item: item["id"])


def resolve_model_path(profile_id: str, explicit: str | None = None) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        key = model_key(profile_id)
        entry = _load_registry().get(key) or {}
        path = Path(entry.get("path", "")) if entry.get("path") else None
    if not path or not path.is_file():
        raise FileNotFoundError(
            f"No installed GGUF for {profile_id!r}. Run `local-code model install {profile_id} --url URL`, "
            "or pass --model-path /path/to/model.gguf."
        )
    return str(path.resolve())


def _read_state():
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_state(state):
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _pid_is_llama_server(pid):
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            return False
        return "llama-server" in command
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=3, check=False,
            )
        else:
            completed = subprocess.run(
                ["ps", "-p", str(int(pid)), "-o", "command="],
                capture_output=True, text=True, timeout=3, check=False,
            )
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False
    return completed.returncode == 0 and "llama-server" in completed.stdout.lower()


def server_status():
    state = _read_state()
    if not state:
        return {"state": "stopped", "managed": False}
    pid = state.get("pid")
    alive = _pid_alive(pid) and _pid_is_llama_server(pid)
    if not alive:
        state_path().unlink(missing_ok=True)
        return {"state": "stale", "managed": True, **state}
    health = probe_server(state.get("base_url", DEFAULT_LLAMACPP_BASE_URL), timeout=2)
    return {"state": "ready" if health["ready"] else "starting", "managed": True, **state, "health": health}


def probe_server(base_url=DEFAULT_LLAMACPP_BASE_URL, timeout=2):
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"User-Agent": "local-code/0.2"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
        models = [item.get("id") for item in data.get("data", []) if item.get("id")]
        return {"ready": True, "models": models}
    except Exception as exc:  # noqa: BLE001 - status reports failures
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}


def _port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def start_server(
    profile_id: str,
    gpu: str,
    *,
    model_path: str | None = None,
    executable: str | None = None,
    host="127.0.0.1",
    port=8080,
    wait_timeout=120,
):
    """Start an external llama-server after explicit user action."""
    existing = server_status()
    if existing["state"] in {"ready", "starting"}:
        raise RuntimeError(f"A managed llama-server is already {existing['state']} (PID {existing.get('pid')}).")
    binary = find_llama_server(executable)
    if not binary:
        raise FileNotFoundError(
            "llama-server was not found. Install llama.cpp separately, set LLAMA_SERVER, pass --llama-server, "
            "or run `local-code llama install --url URL` for a prebuilt binary."
        )
    path = resolve_model_path(profile_id, model_path)
    if not _port_available(host, port):
        raise RuntimeError(f"{host}:{port} is already in use. Stop the existing server or choose --port.")
    args = generate_llama_server_args(profile_id, gpu, path, host=host, port=port, executable=binary)
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "server.log"
    log = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    base_url = f"http://{connect_host}:{port}/v1"
    state = {
        "pid": process.pid,
        "profile": model_key(profile_id),
        "model_path": path,
        "executable": binary,
        "base_url": base_url,
        "host": host,
        "port": port,
        "gpu": gpu,
        "context": int(args[args.index("-c") + 1]),
        "log_path": str(log_path),
        "started_at": int(time.time()),
        "args": args,
    }
    try:
        _write_state(state)
    except BaseException:
        process.terminate()
        raise
    deadline = time.monotonic() + max(0, wait_timeout)
    last_health = {"ready": False}
    while time.monotonic() <= deadline:
        if process.poll() is not None:
            state_path().unlink(missing_ok=True)
            raise RuntimeError(f"llama-server exited with code {process.returncode}. See {log_path}.")
        last_health = probe_server(base_url)
        if last_health["ready"]:
            return {"state": "ready", "managed": True, **state, "health": last_health}
        time.sleep(0.25)
    return {"state": "starting", "managed": True, **state, "health": last_health}


def stop_server(timeout=10):
    """Stop only the llama-server process recorded as managed by local-code."""
    state = _read_state()
    if not state:
        return {"state": "stopped", "managed": False, "message": "No managed llama-server is running."}
    pid = state.get("pid")
    if not _pid_alive(pid):
        state_path().unlink(missing_ok=True)
        return {"state": "stopped", "managed": True, "message": "Removed stale llama-server state."}
    if not _pid_is_llama_server(pid):
        raise RuntimeError(f"Refusing to stop PID {pid}: it no longer appears to be llama-server.")
    os.killpg(int(pid), signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid):
        os.killpg(int(pid), signal.SIGKILL)
    state_path().unlink(missing_ok=True)
    return {"state": "stopped", "managed": True, "pid": pid, "message": "Managed llama-server stopped."}

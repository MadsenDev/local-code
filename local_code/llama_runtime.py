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
import shlex
import socket
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
import tarfile
from collections import deque
from pathlib import Path

from .llamacpp import DEFAULT_LLAMACPP_BASE_URL, generate_llama_server_args, get_llamacpp_profile
from .paths import migrate_registry_location, rist_home
from .managed_manifest import select_asset


def local_code_home() -> Path:
    """Compatibility name for the Rist user-data directory."""
    return rist_home()


def runtime_dir() -> Path:
    return local_code_home() / "runtime" / "llamacpp"


def downloads_dir() -> Path:
    return local_code_home() / "downloads"


def cache_dir() -> Path:
    return local_code_home() / "cache"


def logs_dir() -> Path:
    return local_code_home() / "logs"


def config_dir() -> Path:
    return local_code_home() / "config"


def log_path() -> Path:
    current = runtime_dir() / "server.log"
    legacy = local_code_home() / "runtimes" / "llamacpp" / "server.log"
    return legacy if legacy.exists() and not current.exists() else current


def models_dir() -> Path:
    return local_code_home() / "models" / "llamacpp"


def state_path() -> Path:
    return runtime_dir() / "server.json"


def registry_path() -> Path:
    home = local_code_home()
    migrate_registry_location(home)
    return home / "models.json"


def find_llama_server(explicit: str | None = None) -> str | None:
    """Find llama-server without downloading or compiling it."""
    candidates = [explicit, os.environ.get("LLAMA_SERVER")]
    for command in ("llama-server", "llama-server.exe"):
        path_hit = shutil.which(command)
        if path_hit:
            candidates.append(path_hit)
    candidates.extend([
        str(runtime_dir() / "bin" / "llama-server"),
        str(local_code_home() / "runtimes" / "llamacpp" / "bin" / "llama-server"),
        str(local_code_home() / "bin" / "llama-server"),
    ])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(path: Path, sha256: str | None) -> bool:
    return path.is_file() and bool(sha256) and _sha256_file(path).lower() == sha256.lower()


def runtime_install_state_path() -> Path:
    return runtime_dir() / "install.json"


def _load_runtime_install_state():
    try:
        data = json.loads(runtime_install_state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def managed_runtime_installed(asset: dict | None = None) -> dict | None:
    asset = asset or select_asset("runtime", "llama.cpp")
    archive = downloads_dir() / "runtime" / Path(urllib.parse.urlparse(asset["url"]).path).name
    executable = runtime_dir() / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    state = _load_runtime_install_state()
    if executable.is_file() and os.access(executable, os.X_OK) and _verified_file(archive, asset.get("sha256")) and state.get("manifest", {}).get("version") == asset.get("version"):
        return {"executable": str(executable.resolve()), "archive": str(archive.resolve()), "sha256": asset.get("sha256"), "manifest": asset, "reused": True}
    return None


def managed_model_installed(profile_id: str, asset: dict | None = None) -> dict | None:
    asset = asset or select_asset("model", profile_id)
    profile = get_llamacpp_profile(profile_id)
    entry = _load_registry().get(profile["id"]) or {}
    path = Path(entry.get("path", "")) if entry.get("path") else None
    if path and _verified_file(path, asset.get("sha256")) and (entry.get("manifest") or {}).get("version") == asset.get("version"):
        return {"profile": profile["id"], "path": str(path.resolve()), "sha256": asset.get("sha256"), "manifest": asset, "reused": True}
    return None


def _download(url: str, destination: Path, *, sha256: str | None = None, force=False, timeout=60, progress=None):
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"http", "https", "file"}:
        raise ValueError("Artifact URL must use http, https, or file.")
    if destination.exists() and not force:
        if sha256 and _verified_file(destination, sha256):
            return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256.lower(), "content_length": destination.stat().st_size, "reused": True}
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    downloaded = 0
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rist/0.2"})
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
                if progress:
                    progress(downloaded, total)
        actual = digest.hexdigest()
        if sha256 and actual.lower() != sha256.lower():
            raise ValueError(f"SHA-256 mismatch: expected {sha256.lower()}, got {actual}.")
        partial.replace(destination)
        return {"path": str(destination), "bytes": downloaded, "sha256": actual, "content_length": total}
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _is_runtime_member(name: str) -> bool:
    """Return whether an archive member is needed to run llama-server."""
    basename = Path(name).name
    lowered = basename.lower()
    return (
        basename in {"llama-server", "llama-server.exe"}
        or lowered.endswith((".dll", ".dylib"))
        or (lowered.startswith("lib") and ".so" in lowered)
    )


def _extract_llama_server(archive: Path, target: Path) -> None:
    """Extract llama-server and its adjacent shared libraries.

    Official llama.cpp release binaries dynamically load libraries shipped in
    the same archive. Flattening the required files into the managed bin
    directory preserves the executable's ``$ORIGIN``/loader-relative lookup.
    """
    extracted_server = False
    target.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir() or not _is_runtime_member(member.filename):
                    continue
                destination = target if Path(member.filename).name in {"llama-server", "llama-server.exe"} else target.parent / Path(member.filename).name
                with zf.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_server |= destination == target
        if extracted_server:
            return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not (member.isfile() or member.islnk() or member.issym()) or not _is_runtime_member(member.name):
                    continue
                src = tf.extractfile(member)
                if src is None:
                    raise FileNotFoundError(f"Runtime archive member was not readable: {member.name}")
                destination = target if Path(member.name).name in {"llama-server", "llama-server.exe"} else target.parent / Path(member.name).name
                with src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_server |= destination == target
        if extracted_server:
            return
    raise FileNotFoundError("The runtime archive did not contain llama-server.")


def install_llama_server(url: str | None = None, *, destination: str | None = None, sha256: str | None = None, force=False, manifest_id="llama.cpp", progress=None):
    """Download a manifest-approved prebuilt llama-server binary."""
    asset = None
    if url is None:
        asset = select_asset("runtime", manifest_id)
        url = asset["url"]
        sha256 = asset["sha256"]
    if asset and not destination and not force:
        installed = managed_runtime_installed(asset)
        if installed:
            return installed
    target = Path(destination).expanduser() if destination else runtime_dir() / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    archive_suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if archive_suffix in {".zip", ".gz", ".tgz", ".tar"}:
        archive = downloads_dir() / "runtime" / Path(urllib.parse.urlparse(url).path).name
        report = _download(url, archive, sha256=sha256, force=force, progress=progress)
        _extract_llama_server(archive, target)
        report["archive"] = str(archive.resolve())
    else:
        report = _download(url, target, sha256=sha256, force=force, progress=progress)
    target.chmod(target.stat().st_mode | 0o755)
    if asset and not destination:
        state = {"executable": str(target.resolve()), "archive": report.get("archive"), "sha256": report.get("sha256"), "manifest": asset, "installed_at": int(time.time())}
        runtime_install_state_path().parent.mkdir(parents=True, exist_ok=True)
        temp = runtime_install_state_path().with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(runtime_install_state_path())
    report.update({"executable": str(target.resolve()), "manifest": asset})
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


def install_model(profile_id: str, url: str | None = None, *, filename: str | None = None, sha256: str | None = None, force=False, progress=None):
    """Download and register a manifest-approved model, or an advanced explicit URL with SHA-256."""
    asset = None
    if url is None:
        asset = select_asset("model", profile_id)
        url = asset["url"]
        sha256 = asset["sha256"]
    if asset and not force:
        installed = managed_model_installed(profile_id, asset)
        if installed:
            return installed
    if not sha256:
        raise ValueError("Model download could not be verified because no SHA-256 was provided.")
    profile = get_llamacpp_profile(profile_id)
    name = filename or Path(urllib.parse.urlparse(url).path).name
    if filename and Path(filename).name != filename:
        raise ValueError("--filename must be a filename, not a path.")
    if not name:
        name = profile["id"] + ".gguf"
    if not name.lower().endswith(".gguf"):
        raise ValueError("The model destination must use a .gguf filename.")
    target = models_dir() / profile["id"] / name
    report = _download(url, target, sha256=sha256, force=force, timeout=120, progress=progress)
    registry = _load_registry()
    registry[profile["id"]] = {
        "profile": profile["id"],
        "name": profile["name"],
        "path": str(target.resolve()),
        "source_url": url,
        "sha256": report["sha256"],
        "manifest": asset,
        "installed_at": int(time.time()),
    }
    _save_registry(registry)
    report["profile"] = profile["id"]
    report["manifest"] = asset
    return report


def register_model(profile_id: str, model_path: str):
    """Register an existing GGUF without copying or parsing it."""
    profile = get_llamacpp_profile(profile_id)
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"GGUF path is a directory, not a file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"GGUF path is not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"GGUF file is not readable: {path}")
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


def remove_model(profile_id: str, *, delete_file=False, confirmed=False, force=False):
    """Unregister a model and optionally delete its registered GGUF file."""
    key = model_key(profile_id)
    registry = _load_registry()
    entry = registry.get(key)
    if not entry:
        raise KeyError(f"Model {profile_id!r} is not registered.")
    status = server_status()
    if status.get("managed") and status.get("profile") == key and status.get("state") in {"running", "starting", "ready"}:
        if not force:
            raise RuntimeError("This model is running in the managed llama.cpp runtime. Stop it first or pass --force.")
        stop_server()
    path = Path(entry.get("path", "")).expanduser()
    existed = path.is_file()
    if delete_file and not confirmed:
        raise PermissionError("Deleting a registered GGUF requires explicit confirmation or --yes.")
    if delete_file and existed:
        path.unlink()
    registry.pop(key, None)
    _save_registry(registry)
    return {
        "id": key,
        "path": str(path),
        "unregistered": True,
        "delete_requested": delete_file,
        "file_deleted": bool(delete_file and existed),
        "file_missing": not existed,
    }


def validate_port(port):
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("Port must be an integer between 1 and 65535.")


def validate_server_args(args):
    """Validate generated numeric llama-server settings without interpreting GGUF."""
    checks = {"-c": "context", "-t": "threads", "-b": "batch", "-ub": "ubatch"}
    for flag, label in checks.items():
        try:
            value = int(args[args.index(flag) + 1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"llama.cpp {label} must be a valid integer.") from exc
        if value <= 0:
            raise ValueError(f"llama.cpp {label} must be greater than zero.")
    port = int(args[args.index("--port") + 1])
    validate_port(port)


def prepare_server_start(
    profile_id: str,
    gpu: str,
    *,
    model_path: str | None = None,
    executable: str | None = None,
    host="127.0.0.1",
    port=8080,
):
    """Resolve and validate exactly the command and paths used by start_server."""
    validate_port(port)
    binary = find_llama_server(executable)
    if not binary:
        raise FileNotFoundError(
            "llama-server was not found. Install llama.cpp separately, set LLAMA_SERVER, pass --llama-server, "
            "or run `rist runtime install` to install the managed runtime."
        )
    path = resolve_model_path(profile_id, model_path)
    if Path(path).suffix.lower() != ".gguf":
        raise ValueError("The model path must point to a .gguf file.")
    args = generate_llama_server_args(profile_id, gpu, path, host=host, port=port, executable=binary)
    validate_server_args(args)
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return {
        "profile": model_key(profile_id),
        "model_path": path,
        "executable": binary,
        "host": host,
        "port": port,
        "base_url": f"http://{connect_host}:{port}/v1",
        "args": args,
        "command": shlex.join(args),
        "log_path": str(log_path()),
        "state_path": str(state_path()),
    }


def resolve_model_path(profile_id: str, explicit: str | None = None) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        key = model_key(profile_id)
        entry = _load_registry().get(key) or {}
        path = Path(entry.get("path", "")) if entry.get("path") else None
    if not path or not path.is_file():
        raise FileNotFoundError(
            f"No installed managed model for {profile_id!r}. Run `rist model install {profile_id}`, "
            "or use Advanced Setup with --model-path."
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
    state_file = state_path()
    state = _read_state()
    if not state:
        return {
            "state": "stopped",
            "managed": False,
            "state_file_present": state_file.exists(),
            "log_path": str(log_path()),
        }
    pid = state.get("pid")
    pid_alive = _pid_alive(pid)
    correct_process = pid_alive and _pid_is_llama_server(pid)
    if not correct_process:
        return {
            "state": "stale",
            "managed": True,
            "state_file_present": True,
            "pid_running": pid_alive,
            **state,
        }
    health = probe_server(state.get("base_url", DEFAULT_LLAMACPP_BASE_URL), timeout=2)
    return {
        "state": "running",
        "managed": True,
        "state_file_present": True,
        "pid_running": True,
        **state,
        "health": health,
    }


def probe_server(base_url=DEFAULT_LLAMACPP_BASE_URL, timeout=2):
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"User-Agent": "rist/0.2"})
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


def recent_log_lines(limit=30, path: str | Path | None = None):
    """Return recent managed log lines without raising for missing/unreadable files."""
    target = Path(path) if path else log_path()
    try:
        if not target.is_file():
            return []
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque(handle, maxlen=max(0, limit)))
    except OSError:
        return []


def startup_failure_message(message, path: str | Path | None = None, limit=30):
    lines = recent_log_lines(limit, path)
    output = [message, "", "Last log lines:"]
    output.extend(line.rstrip("\n") for line in lines)
    if not lines:
        output.append("No managed llama.cpp server log is available.")
    joined = "\n".join(lines).lower()
    suggestions = ["Check that the model path exists.", "Reduce context size.", "Reduce batch size.", "Verify the llama.cpp build matches your CPU/GPU setup."]
    if "out of memory" in joined or "cudamalloc failed" in joined:
        suggestions.extend(["Increase --n-cpu-moe or use a lower quantization."])
    if "unknown argument" in joined:
        suggestions.append("The llama.cpp version may be too old or new for one of these flags.")
    if "failed to load model" in joined:
        suggestions.append("Verify the GGUF path and that the file finished downloading.")
    suggestions.append("Try: rist llama command --profile <profile> --gpu <gpu>")
    output.extend(["", "Suggested fixes:"] + [f"- {item}" for item in dict.fromkeys(suggestions)])
    return "\n".join(output)


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
    if existing["state"] in {"running", "starting", "ready"}:
        raise RuntimeError(f"A managed llama-server is already {existing['state']} (PID {existing.get('pid')}).")
    prepared = prepare_server_start(
        profile_id, gpu, model_path=model_path, executable=executable, host=host, port=port,
    )
    if not _port_available(host, port):
        raise RuntimeError(f"{host}:{port} is already in use. Stop the existing server or choose --port.")
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target_log = Path(prepared["log_path"])
    log = target_log.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            prepared["args"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    finally:
        log.close()
    args = prepared["args"]
    state = {
        **prepared,
        "pid": process.pid,
        "gpu": gpu,
        "context": int(args[args.index("-c") + 1]),
        "started_at": int(time.time()),
    }
    state.pop("command", None)
    try:
        _write_state(state)
    except BaseException:
        process.terminate()
        raise
    deadline = time.monotonic() + max(0, wait_timeout)
    last_health = {"ready": False}
    while True:
        if process.poll() is not None:
            state_path().unlink(missing_ok=True)
            raise RuntimeError(startup_failure_message(
                f"llama-server failed to start. Process exited with code {process.returncode}.", target_log,
            ))
        last_health = probe_server(prepared["base_url"])
        if last_health["ready"]:
            return {"state": "running", "managed": True, **state, "health": last_health}
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    try:
        process.terminate()
    except (AttributeError, OSError):
        pass
    state_path().unlink(missing_ok=True)
    raise RuntimeError(startup_failure_message(
        f"llama-server failed to become healthy within {wait_timeout:g} seconds. Last health error: {last_health.get('error', 'unknown')}",
        target_log,
    ))


def stop_server(timeout=10):
    """Stop only the llama-server process recorded as managed by Rist."""
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

"""llama.cpp orchestration helpers.

Rist never implements inference or loads GGUF itself. This module contains
documented profiles and argument generation for an external llama.cpp runtime.
"""

from __future__ import annotations

import shlex

DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:8080/v1"

LLAMACPP_MODEL_PROFILES = {
    "qwen36-35b-a3b": {
        "id": "qwen36-35b-a3b-llamacpp",
        "name": "Qwen3.6 35B A3B via llama.cpp",
        "provider": "llamacpp",
        "role": "heavy_backend",
        "recommended_routing": "single",
        "hardware": {
            "min_vram_gb": 8,
            "recommended_vram_gb": 12,
            "recommended_ram_gb": 32,
            "ideal_ram_gb": 64,
        },
        "status": "experimental",
        "use_for": ["repo analysis", "debugging logs", "code review", "planning", "multi-file reasoning"],
        "avoid_for": ["quick chat", "rapid tool loops", "frequent model switching"],
        "model_path": "/path/to/model.gguf",
    },
    "qwen3-coder": {
        "id": "qwen3-coder-llamacpp",
        "name": "Qwen3 Coder-style GGUF via llama.cpp",
        "provider": "llamacpp",
        "role": "heavy_backend",
        "recommended_routing": "single",
        "hardware": {
            "min_vram_gb": 8,
            "recommended_vram_gb": 12,
            "recommended_ram_gb": 32,
            "ideal_ram_gb": 64,
        },
        "status": "experimental",
        "use_for": ["repo analysis", "code review", "planning", "multi-file reasoning"],
        "avoid_for": ["quick chat", "rapid tool loops", "frequent model switching"],
        "model_path": "/path/to/model.gguf",
    },
}

LLAMACPP_GPU_PROFILES = {
    "rtx3060": {
        "context": 16384,
        "gpu_layers": 999,
        "cpu_moe": 30,
        "threads": 8,
        "batch": 1024,
        "ubatch": 1024,
    },
}


def get_llamacpp_profile(profile_id: str):
    """Return a command profile, accepting short, friendly, or full ids."""
    normalized = (profile_id or "").lower()
    normalized = {"qwen36": "qwen36-35b-a3b", "qwen3coder": "qwen3-coder"}.get(normalized, normalized)
    if normalized in LLAMACPP_MODEL_PROFILES:
        return LLAMACPP_MODEL_PROFILES[normalized]
    for profile in LLAMACPP_MODEL_PROFILES.values():
        if profile["id"].lower() == normalized:
            return profile
    available = ", ".join(sorted(LLAMACPP_MODEL_PROFILES))
    raise ValueError(f"Unknown llama.cpp profile {profile_id!r}. Available profiles: {available}.")


def generate_llama_server_args(
    profile_id: str,
    gpu: str,
    model_path: str | None = None,
    host="127.0.0.1",
    port=8080,
    executable="llama-server",
):
    """Build argv for an external llama-server process."""
    profile = get_llamacpp_profile(profile_id)
    gpu_key = (gpu or "").lower()
    if gpu_key not in LLAMACPP_GPU_PROFILES:
        available = ", ".join(sorted(LLAMACPP_GPU_PROFILES))
        raise ValueError(f"Unknown GPU profile {gpu!r}. Available GPU profiles: {available}.")
    settings = LLAMACPP_GPU_PROFILES[gpu_key]
    path = model_path or profile["model_path"]
    return [
        executable, "-m", path,
        "-c", str(settings["context"]),
        "-ngl", str(settings["gpu_layers"]),
        "--n-cpu-moe", str(settings["cpu_moe"]),
        "-fa", "on",
        "-t", str(settings["threads"]),
        "-b", str(settings["batch"]),
        "-ub", str(settings["ubatch"]),
        "--jinja", "--host", host, "--port", str(port),
    ]


def generate_llama_server_command(profile_id: str, gpu: str, model_path: str | None = None, host="127.0.0.1", port=8080):
    """Generate, but do not execute, a conservative llama-server command."""
    profile = get_llamacpp_profile(profile_id)
    gpu_key = (gpu or "").lower()
    args = generate_llama_server_args(profile_id, gpu_key, model_path, host, port)
    command_parts = [shlex.quote(args[0])]
    for index in _argument_starts(args):
        width = 1 if args[index] == "--jinja" else 2
        command_parts.append(" ".join(shlex.quote(value) for value in args[index:index + width]))
    command = (" " + "\\" + "\n  ").join(command_parts)
    return {
        "profile": profile,
        "gpu": gpu_key,
        "command": command,
        "args": args,
        "experiments": [
            "context: 16384 -> 32768 -> 65536",
            "threads: 8 -> 12 -> 16",
            "batch: 1024 -> 2048",
            "n-cpu-moe: 24 -> 30 -> 32",
        ],
        "warning": "These are conservative starting values for 12 GB-class hardware. Increase one setting at a time and watch RAM, VRAM, latency, and stability.",
    }


def _argument_starts(args):
    index = 1
    while index < len(args):
        yield index
        index += 1 if args[index] == "--jinja" else 2


def format_llama_server_command(report):
    profile = report["profile"]
    lines = [
        f"# {profile['name']} ({profile['status']}, {profile['role']})",
        "# This command helper does not execute it. `rist model start` can start it explicitly.",
        report["command"],
        "",
        report["warning"],
        "Experiment upward only after the starting configuration is stable:",
    ]
    lines.extend(f"  {item}" for item in report["experiments"])
    return "\n".join(lines)

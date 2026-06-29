"""llama.cpp orchestration helpers.

Rist never implements inference or loads GGUF itself. This module contains
practical coding profiles and argument generation for an external llama.cpp
runtime.
"""

from __future__ import annotations

import shlex

DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:8080/v1"

LLAMACPP_MODEL_PROFILES = {
    "qwen2.5-coder-7b": {
        "id": "qwen2.5-coder-7b-llamacpp",
        "name": "Qwen2.5 Coder 7B Instruct GGUF",
        "provider": "llamacpp",
        "role": "primary_coder",
        "architecture": "dense",
        "recommended_routing": "single",
        "recommended_context": 16384,
        "hardware": {"min_vram_gb": 6, "recommended_vram_gb": 8, "recommended_ram_gb": 16, "ideal_ram_gb": 32},
        "status": "recommended",
        "use_for": ["daily coding", "tool loops", "repo edits", "fast chat"],
        "avoid_for": ["very large repo analysis without retrieval"],
        "model_path": "/path/to/qwen2.5-coder-7b.gguf",
    },
    "qwen2.5-coder-14b": {
        "id": "qwen2.5-coder-14b-llamacpp",
        "name": "Qwen2.5 Coder 14B Instruct GGUF",
        "provider": "llamacpp",
        "role": "strong_coder",
        "architecture": "dense",
        "recommended_routing": "single",
        "recommended_context": 16384,
        "hardware": {"min_vram_gb": 10, "recommended_vram_gb": 12, "recommended_ram_gb": 24, "ideal_ram_gb": 48},
        "status": "recommended",
        "use_for": ["multi-file edits", "code review", "debugging"],
        "avoid_for": ["low-memory laptops"],
        "model_path": "/path/to/qwen2.5-coder-14b.gguf",
    },
    "deepseek-coder": {
        "id": "deepseek-coder-llamacpp",
        "name": "DeepSeek Coder GGUF",
        "provider": "llamacpp",
        "role": "compat_coder",
        "architecture": "dense",
        "recommended_routing": "single",
        "recommended_context": 8192,
        "hardware": {"min_vram_gb": 6, "recommended_vram_gb": 8, "recommended_ram_gb": 16, "ideal_ram_gb": 32},
        "status": "compatible",
        "use_for": ["coding", "legacy GGUF installs"],
        "avoid_for": ["new installs where Qwen2.5 Coder is available"],
        "model_path": "/path/to/deepseek-coder.gguf",
    },
    "qwen3-coder": {
        "id": "qwen3-coder-llamacpp",
        "name": "Qwen3 Coder GGUF",
        "provider": "llamacpp",
        "role": "large_coder",
        "architecture": "dense",
        "recommended_routing": "single",
        "recommended_context": 32768,
        "hardware": {"min_vram_gb": 16, "recommended_vram_gb": 24, "recommended_ram_gb": 32, "ideal_ram_gb": 64},
        "status": "advanced",
        "use_for": ["repo analysis", "code review", "planning", "multi-file reasoning"],
        "avoid_for": ["frequent model switching"],
        "model_path": "/path/to/qwen3-coder.gguf",
    },
    "qwen36-35b-a3b": {
        "id": "qwen36-35b-a3b-llamacpp",
        "name": "Qwen3.6 35B A3B GGUF",
        "provider": "llamacpp",
        "role": "heavy_backend",
        "architecture": "moe",
        "recommended_routing": "single",
        "recommended_context": 32768,
        "hardware": {"min_vram_gb": 8, "recommended_vram_gb": 12, "recommended_ram_gb": 32, "ideal_ram_gb": 64},
        "status": "experimental",
        "use_for": ["repo analysis", "debugging logs", "code review", "planning", "multi-file reasoning"],
        "avoid_for": ["quick chat", "rapid tool loops", "frequent model switching"],
        "model_path": "/path/to/qwen36-a3b.gguf",
    },
}

LLAMACPP_PROFILE_ALIASES = {
    "qwen7": "qwen2.5-coder-7b", "qwen-7b": "qwen2.5-coder-7b", "qwen14": "qwen2.5-coder-14b",
    "qwen-14b": "qwen2.5-coder-14b", "deepseek": "deepseek-coder", "qwen36": "qwen36-35b-a3b",
    "qwen3coder": "qwen3-coder", "qwen3-coder-llamacpp": "qwen3-coder",
}

LLAMACPP_GPU_PROFILES = {
    "cpu": {"context": 8192, "gpu_layers": 0, "threads": 8, "batch": 256, "ubatch": 256},
    "vram8": {"context": 8192, "gpu_layers": 999, "threads": 8, "batch": 512, "ubatch": 512},
    "vram12": {"context": 16384, "gpu_layers": 999, "threads": 8, "batch": 1024, "ubatch": 1024, "cpu_moe": 30},
    "vram16": {"context": 24576, "gpu_layers": 999, "threads": 10, "batch": 1024, "ubatch": 1024, "cpu_moe": 24},
    "vram24": {"context": 32768, "gpu_layers": 999, "threads": 12, "batch": 2048, "ubatch": 1024, "cpu_moe": 12},
    "vram48": {"context": 65536, "gpu_layers": 999, "threads": 16, "batch": 2048, "ubatch": 2048, "cpu_moe": 0},
    # Compatibility alias for existing users/tests.
    "rtx3060": {"context": 16384, "gpu_layers": 999, "threads": 8, "batch": 1024, "ubatch": 1024, "cpu_moe": 30},
}


def is_moe_profile(profile):
    return profile.get("architecture") == "moe"


def get_llamacpp_profile(profile_id: str):
    normalized = (profile_id or "").lower()
    normalized = LLAMACPP_PROFILE_ALIASES.get(normalized, normalized)
    if normalized in LLAMACPP_MODEL_PROFILES:
        return LLAMACPP_MODEL_PROFILES[normalized]
    for profile in LLAMACPP_MODEL_PROFILES.values():
        if profile["id"].lower() == normalized:
            return profile
    available = ", ".join(sorted(LLAMACPP_MODEL_PROFILES))
    raise ValueError(f"Unknown llama.cpp profile {profile_id!r}. Available profiles: {available}.")


def recommend_gpu_profile(hardware):
    max_vram = getattr(hardware, "max_vram_gb", None) or 0
    if max_vram >= 48: return "vram48"
    if max_vram >= 24: return "vram24"
    if max_vram >= 16: return "vram16"
    if max_vram >= 12: return "vram12"
    if max_vram >= 8: return "vram8"
    return "cpu"


def recommend_model_profiles(hardware):
    max_vram = getattr(hardware, "max_vram_gb", None) or 0
    if max_vram >= 24:
        preferred = ["qwen2.5-coder-14b", "qwen3-coder", "qwen2.5-coder-7b"]
    elif max_vram >= 12:
        preferred = ["qwen2.5-coder-7b", "qwen2.5-coder-14b", "qwen36-35b-a3b"]
    elif max_vram >= 8:
        preferred = ["qwen2.5-coder-7b", "deepseek-coder"]
    else:
        preferred = ["qwen2.5-coder-7b"]
    return [get_llamacpp_profile(item) for item in preferred]


def generate_llama_server_args(profile_id: str, gpu: str, model_path: str | None = None, host="127.0.0.1", port=8080, executable="llama-server"):
    profile = get_llamacpp_profile(profile_id)
    gpu_key = (gpu or "").lower()
    if gpu_key not in LLAMACPP_GPU_PROFILES:
        available = ", ".join(sorted(LLAMACPP_GPU_PROFILES))
        raise ValueError(f"Unknown GPU profile {gpu!r}. Available GPU profiles: {available}.")
    settings = LLAMACPP_GPU_PROFILES[gpu_key]
    path = model_path or profile["model_path"]
    args = [executable, "-m", path, "-c", str(settings["context"]), "-ngl", str(settings["gpu_layers"]), "-fa", "on", "-t", str(settings["threads"]), "-b", str(settings["batch"]), "-ub", str(settings["ubatch"]), "--jinja", "--host", host, "--port", str(port)]
    if is_moe_profile(profile) and settings.get("cpu_moe") is not None:
        insert_at = args.index("-fa")
        args[insert_at:insert_at] = ["--n-cpu-moe", str(settings["cpu_moe"])]
    return args


def generate_llama_server_command(profile_id: str, gpu: str, model_path: str | None = None, host="127.0.0.1", port=8080):
    profile = get_llamacpp_profile(profile_id)
    gpu_key = (gpu or "").lower()
    args = generate_llama_server_args(profile_id, gpu_key, model_path, host, port)
    command_parts = [shlex.quote(args[0])]
    for index in _argument_starts(args):
        width = 1 if args[index] == "--jinja" else 2
        command_parts.append(" ".join(shlex.quote(value) for value in args[index:index + width]))
    experiments = ["context: current -> lower if VRAM pressure is high", "threads: cpu_count/2 -> cpu_count", "batch: 512 -> 1024 -> 2048", "ubatch: 256 -> 512 -> 1024"]
    if is_moe_profile(profile): experiments.append("n-cpu-moe: adjust only for MoE models")
    return {"profile": profile, "gpu": gpu_key, "command": (" " + "\\" + "\n  ").join(command_parts), "args": args, "experiments": experiments, "warning": "These are conservative starting values. `rist llama tune` can benchmark alternatives and save the best local configuration."}


def _argument_starts(args):
    index = 1
    while index < len(args):
        yield index
        index += 1 if args[index] == "--jinja" else 2


def format_llama_server_command(report):
    profile = report["profile"]
    lines = [
        f"# {profile['name']} ({profile['status']}, {profile['role']}, {profile.get('architecture', 'unknown')})",
        "# This command helper does not execute it. `rist model start` can start it explicitly.",
        report["command"], "", report["warning"], "Experiment upward only after the starting configuration is stable:",
    ]
    lines.extend(f"  {item}" for item in report["experiments"])
    return "\n".join(lines)

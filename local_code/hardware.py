"""Best-effort local hardware discovery and model residency estimates."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field

from .model_profiles import classify_model


@dataclass
class GPUInfo:
    name: str
    vram_total_gb: float | None = None
    vram_used_gb: float | None = None
    vendor: str = "unknown"


@dataclass
class HardwareInfo:
    system_ram_gb: float | None
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_count: int | None = None
    platform: str = ""
    loaded_models: list[str] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float | None:
        values = [gpu.vram_total_gb for gpu in self.gpus if gpu.vram_total_gb is not None]
        return sum(values) if values else None

    @property
    def max_vram_gb(self) -> float | None:
        values = [gpu.vram_total_gb for gpu in self.gpus if gpu.vram_total_gb is not None]
        return max(values) if values else None

    def to_dict(self):
        data = asdict(self)
        data["total_vram_gb"] = self.total_vram_gb
        data["max_vram_gb"] = self.max_vram_gb
        return data


def _run(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _system_ram_gb():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * pages / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def _nvidia_gpus():
    if not shutil.which("nvidia-smi"):
        return []
    output = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            total, used = float(parts[1]) / 1024, float(parts[2]) / 1024
        except ValueError:
            continue
        gpus.append(GPUInfo(parts[0], round(total, 1), round(used, 1), "nvidia"))
    return gpus


def _amd_gpus():
    if not shutil.which("rocm-smi"):
        return []
    output = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    try:
        data = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return []
    gpus = []
    for card, values in data.items():
        if not isinstance(values, dict):
            continue
        name = values.get("Card series") or values.get("Card model") or card
        total = next((v for k, v in values.items() if "Total Memory" in k), None)
        used = next((v for k, v in values.items() if "Used Memory" in k), None)
        try:
            total_gb = round(float(total) / 1024**3, 1) if total is not None else None
            used_gb = round(float(used) / 1024**3, 1) if used is not None else None
        except (TypeError, ValueError):
            total_gb = used_gb = None
        gpus.append(GPUInfo(str(name), total_gb, used_gb, "amd"))
    return gpus


def loaded_ollama_models(base_url, timeout=2):
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/ps", headers={"User-Agent": "local-code/0.2"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except Exception:  # best-effort diagnostic probe
        return []
    return [model.get("name", "") for model in data.get("models", []) if model.get("name")]


def detect_hardware(ollama_base_url=None):
    gpus = _nvidia_gpus() or _amd_gpus()
    loaded = loaded_ollama_models(ollama_base_url) if ollama_base_url else []
    return HardwareInfo(
        system_ram_gb=_system_ram_gb(),
        gpus=gpus,
        cpu_count=os.cpu_count(),
        platform=platform.platform(),
        loaded_models=loaded,
    )


def estimate_model_vram_gb(model, num_ctx=16384):
    """Estimate Q4 weight + KV/cache headroom from a model tag."""
    profile = classify_model(model)
    if profile.params_b is None:
        return None
    weights = profile.params_b * 0.58
    context = max(num_ctx, 2048) / 16384 * 1.0
    return round(weights + context + 0.6, 1)


def recommend_routing(hardware, frontend_model, backend_model, num_ctx=16384):
    if frontend_model == backend_model:
        return {"mode": "single", "reason": "Both roles already use the same model.", "estimated_vram_gb": estimate_model_vram_gb(frontend_model, num_ctx)}
    front = estimate_model_vram_gb(frontend_model, num_ctx)
    back = estimate_model_vram_gb(backend_model, num_ctx)
    max_vram = hardware.max_vram_gb
    if max_vram is None:
        return {"mode": "single", "reason": "GPU VRAM could not be measured; avoiding unbounded model reload cost.", "estimated_vram_gb": back}
    combined = front + back if front is not None and back is not None else None
    if len(hardware.gpus) > 1 and hardware.total_vram_gb and combined and combined <= hardware.total_vram_gb * 0.9:
        return {"mode": "dual", "reason": "Multiple GPUs have enough estimated aggregate VRAM for both models.", "estimated_vram_gb": combined}
    if combined is not None and combined <= max_vram * 0.85:
        return {"mode": "dual", "reason": "Both models are estimated to remain resident on the GPU.", "estimated_vram_gb": combined}
    return {"mode": "single", "reason": "Estimated dual-model residency exceeds safe VRAM headroom, so reload cost is likely to erase routing gains.", "estimated_vram_gb": combined}

"""Versioned download manifests for managed Rist runtime assets."""
from __future__ import annotations

import json
import platform
from importlib import resources
from pathlib import Path

REQUIRED_FIELDS = {"id", "version", "platform", "architecture", "url", "sha256", "size", "display_name", "description", "license", "source"}


def _norm_arch(machine: str | None = None) -> str:
    value = (machine or platform.machine()).lower()
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    if value in {"aarch64", "arm64"}:
        return "arm64"
    return value


def _norm_os(system: str | None = None) -> str:
    value = (system or platform.system()).lower()
    if value.startswith("darwin"):
        return "macos"
    if value.startswith("windows"):
        return "windows"
    if value.startswith("linux"):
        return "linux"
    return value


def load_manifest(kind: str) -> dict:
    if kind not in {"runtime", "model"}:
        raise ValueError("Manifest kind must be 'runtime' or 'model'.")
    with resources.files("local_code.manifests").joinpath(f"{kind}_manifest.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_manifest(data, kind=kind)
    return data


def validate_manifest(data: dict, *, kind: str | None = None) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Manifest must be an object with schema_version=1.")
    assets_key = "runtimes" if kind == "runtime" else "models" if kind == "model" else None
    keys = [assets_key] if assets_key else ["runtimes", "models"]
    if not any(isinstance(data.get(key), dict) for key in keys):
        raise ValueError("Manifest does not contain downloadable assets.")
    for key in keys:
        assets = data.get(key) or {}
        if not isinstance(assets, dict):
            raise ValueError(f"Manifest section {key} must be an object.")
        for asset_id, entries in assets.items():
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"Manifest asset {asset_id} must contain one or more entries.")
            for entry in entries:
                missing = REQUIRED_FIELDS - set(entry)
                if missing:
                    raise ValueError(f"Manifest asset {asset_id} is missing: {', '.join(sorted(missing))}.")
                if entry["id"] != asset_id:
                    raise ValueError(f"Manifest entry id {entry['id']!r} does not match key {asset_id!r}.")
                if not isinstance(entry.get("size"), int) or entry["size"] <= 0:
                    raise ValueError(f"Manifest asset {asset_id} has an invalid size.")
                sha = str(entry.get("sha256", ""))
                if len(sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
                    raise ValueError(f"Manifest asset {asset_id} has an invalid SHA-256.")


def select_asset(kind: str, asset_id: str, *, os_name: str | None = None, arch: str | None = None) -> dict:
    manifest = load_manifest(kind)
    section = manifest["runtimes" if kind == "runtime" else "models"]
    entries = section.get(asset_id)
    if not entries:
        raise KeyError(f"No managed {kind} asset named {asset_id!r} is available in the manifest.")
    wanted_os = _norm_os(os_name)
    wanted_arch = _norm_arch(arch)
    for entry in entries:
        if entry["platform"] in {wanted_os, "any"} and entry["architecture"] in {wanted_arch, "any"}:
            return dict(entry)
    raise KeyError(f"No managed {kind} asset {asset_id!r} for {wanted_os}-{wanted_arch} is available.")

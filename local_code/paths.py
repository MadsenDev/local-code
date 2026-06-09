"""User data paths and safe migration helpers for Rist."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

RIST_HOME_ENV = "RIST_HOME"
LEGACY_HOME_ENV = "LOCAL_CODE_HOME"
RIST_DIR_NAME = ".rist"
LEGACY_DIR_NAME = ".local-code"


def _default_homes() -> tuple[Path, Path]:
    home = Path.home()
    return home / RIST_DIR_NAME, home / LEGACY_DIR_NAME


def rist_home(*, migrate: bool = True) -> Path:
    """Return Rist's data directory, honoring the legacy override for compatibility."""
    if value := os.environ.get(RIST_HOME_ENV):
        return Path(value).expanduser()
    if value := os.environ.get(LEGACY_HOME_ENV):
        return Path(value).expanduser()

    current, legacy = _default_homes()
    if migrate:
        migrate_legacy_home(current=current, legacy=legacy)
    return current


def migrate_legacy_home(*, current: Path | None = None, legacy: Path | None = None) -> bool:
    """Copy a legacy home into the Rist home without deleting or overwriting data."""
    default_current, default_legacy = _default_homes()
    current = (current or default_current).expanduser()
    legacy = (legacy or default_legacy).expanduser()
    if current.exists() or not legacy.is_dir():
        return False

    staging = Path(tempfile.mkdtemp(prefix=".rist-migrate-", dir=current.parent))
    try:
        shutil.copytree(legacy, staging, dirs_exist_ok=True)
        migrate_registry_location(staging)
        staging.rename(current)
    except FileExistsError:
        shutil.rmtree(staging, ignore_errors=True)
        return False
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    sys.stderr.write("Migrated configuration from ~/.local-code to ~/.rist.\n")
    return True


def migrate_registry_location(home: Path) -> None:
    """Preserve registries created at the pre-Rist nested location."""
    current = home / "models.json"
    legacy = home / "models" / "llamacpp" / "models.json"
    if legacy.is_file() and not current.exists():
        shutil.copy2(legacy, current)

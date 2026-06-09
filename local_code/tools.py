import json
import re
import shlex
import subprocess
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from .ui import clip


NOISY_DIRS = {
    ".cache",
    ".git",
    ".gradle",
    ".mypy_cache",
    ".next",
    ".parcel-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".turbo",
    ".venv",
    ".vite",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "out",
    "target",
    "vendor",
    "venv",
}

DOC_FILES = ("README.md", "AGENTS.md", "CLAUDE.md", "project.md")
PACKAGE_FILES = (
    "package.json",
    "vite.config.ts",
    "vite.config.js",
    "vite.config.mts",
    "tsconfig.json",
    "jsconfig.json",
    "Cargo.toml",
    "src-tauri/Cargo.toml",
    "src-tauri/tauri.conf.json",
    "tauri.conf.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
)
ENTRYPOINTS = (
    "src/main.tsx",
    "src/main.ts",
    "src/main.jsx",
    "src/main.js",
    "src/App.tsx",
    "src/App.ts",
    "src/App.jsx",
    "src/App.js",
    "src-tauri/src/main.rs",
    "electron/main.ts",
    "electron/main.js",
    "electron/preload.ts",
    "electron/preload.js",
)
SEMANTIC_SOURCE_FILES = (
    "src/App.tsx",
    "src/App.ts",
    "src/tauri-api.ts",
    "src/github-auth.ts",
    "src/types.ts",
    "src-tauri/src/commands.rs",
    "src-tauri/src/workspace.rs",
    "src-tauri/src/lib.rs",
)
ARCHITECTURE_DIRS = (
    "docs",
    "src/routes",
    "src/components",
    "src/lib",
    "src/stores",
    "src/hooks",
    "src/services",
    "electron",
    "src-tauri",
)

VIEW_PURPOSE_WORDS = {
    "home": "home/dashboard view",
    "workspace": "workspace management view",
    "workspaces": "workspace management view",
    "repo": "repository view",
    "repos": "repository view",
    "repositories": "repository view",
    "ports": "local port monitoring view",
    "logs": "log viewing/search view",
    "sessions": "session history view",
    "project": "project detail view",
    "projects": "project detail view",
    "onboarding": "onboarding flow",
    "settings": "settings view",
    "palette": "command palette",
}

COMMAND_PURPOSE_HINTS = (
    ("scan_ports", "scans localhost ports"),
    ("get_processes", "lists local development processes"),
    ("kill_process", "can stop local processes"),
    ("get_system_stats", "reads system CPU/memory stats"),
    ("get_git_status", "reads Git status for projects"),
    ("scan_workspaces", "scans workspace folders for projects"),
    ("scan_workspace_groups", "groups discovered repositories/workspaces"),
    ("find_default_workspace_roots", "finds default workspace roots"),
    ("open_in_editor", "opens projects in an editor"),
    ("open_url", "opens URLs"),
    ("read_env_file", "reads and redacts environment files"),
    ("github_request_device_code", "starts GitHub device authentication"),
    ("github_poll_token", "completes GitHub device authentication"),
    ("load_config", "loads local app configuration"),
    ("save_config", "saves local app configuration"),
)

DEFAULT_DISCOVERY_BUDGET = {
    "max_root_files": 50,
    "max_reads": 12,
    "max_searches": 8,
    "max_lines_per_file": 500,
    "stop_when_confidence": "medium",
}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data)


def search_web(query, max_results=5):
    url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    link_re = re.compile(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    snippet_re = re.compile(r'class="result-snippet"[^>]*>(.*?)</td>', re.S)
    tag_re = re.compile(r"<[^>]+>")

    links = link_re.findall(raw)
    snippets = [tag_re.sub("", s).strip() for s in snippet_re.findall(raw)]

    results = []
    for i, (href, title) in enumerate(links[:max_results]):
        title_clean = tag_re.sub("", title).strip()
        snippet = snippets[i] if i < len(snippets) else ""
        results.append(f"{i + 1}. {title_clean}\n   {href}\n   {snippet}")

    return "\n\n".join(results) if results else "(no results found)"


def fetch_url(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "rist/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read().decode("utf-8", errors="replace")
    if "html" in content_type.lower():
        extractor = _TextExtractor()
        extractor.feed(raw)
        return clip("\n".join(extractor.parts))
    return clip(raw)


def run_subprocess(cmd, cwd=None, timeout=30):
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    output = completed.stdout
    if completed.stderr:
        if output:
            output += "\n"
        output += completed.stderr
    return completed.returncode, clip(output.strip())


def git_context(workdir):
    code, root = run_subprocess("git rev-parse --show-toplevel", cwd=workdir, timeout=5)
    if code != 0 or not root:
        return "Not a git repository."
    _, branch = run_subprocess("git branch --show-current", cwd=workdir, timeout=5)
    _, status = run_subprocess("git status --short", cwd=workdir, timeout=5)
    branch = branch.strip() or "(detached HEAD)"
    status = status or "clean"
    return f"Git root: {root.strip()}\nBranch: {branch}\nStatus:\n{status}"


def git_summary(workdir):
    code, root = run_subprocess("git rev-parse --show-toplevel", cwd=workdir, timeout=5)
    if code != 0 or not root:
        return "no git repo", "unknown", "not a git repository"
    _, branch = run_subprocess("git branch --show-current", cwd=workdir, timeout=5)
    _, status = run_subprocess("git status --short", cwd=workdir, timeout=5)
    repo = Path(root.strip()).name
    branch = branch.strip() or "detached"
    changed = len([line for line in status.splitlines() if line.strip()])
    change_text = "clean" if changed == 0 else f"{changed} changed"
    return repo, branch, change_text


def list_files(workdir, path="."):
    cmd = f"rg --files --hidden -g '!.git' {shlex.quote(path)}"
    code, output = run_subprocess(cmd, cwd=workdir, timeout=20)
    if code != 0 and not output:
        return f"list_files failed for {path}"
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[:400]) or "(no files found)"


def _relative(path, root):
    return path.relative_to(root).as_posix()


def _is_noisy(path):
    return any(part in NOISY_DIRS for part in path.parts)


def _read_json_file(path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None


def _read_text_prefix(path, max_lines):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(lines[:max_lines])


def _confidence_from_counts(confirmed, likely=0):
    if confirmed >= 3:
        return "high"
    if confirmed >= 1 or likely >= 2:
        return "medium"
    return "low"


def _humanize_identifier(value):
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = value.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip().lower()


def _add_unique(items, value):
    if value and value not in items:
        items.append(value)


def extract_source_signals(path, text):
    signals = []
    capabilities = []
    path = str(path)

    for view in re.findall(r"['\"]\.\/view-([A-Za-z0-9_-]+)['\"]", text):
        label = VIEW_PURPOSE_WORDS.get(view.lower(), f"{_humanize_identifier(view)} view")
        _add_unique(signals, f"{path}: imports {label}.")
        _add_unique(capabilities, label)

    for command in re.findall(r"invoke<[^>]*>\(['\"]([A-Za-z0-9_:-]+)['\"]", text):
        matched = False
        for needle, description in COMMAND_PURPOSE_HINTS:
            if needle == command:
                _add_unique(signals, f"{path}: frontend invokes `{command}`, which {description}.")
                _add_unique(capabilities, description)
                matched = True
                break
        if not matched:
            _add_unique(signals, f"{path}: frontend invokes Tauri command `{command}`.")

    for command in re.findall(r"invoke_handler!\s*\[(.*?)\]", text, flags=re.S):
        for name in re.findall(r"commands::([A-Za-z0-9_]+)", command):
            for needle, description in COMMAND_PURPOSE_HINTS:
                if needle == name:
                    _add_unique(signals, f"{path}: registers `{name}`, which {description}.")
                    _add_unique(capabilities, description)
                    break

    for name in re.findall(r"#\[tauri::command\]\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)", text):
        for needle, description in COMMAND_PURPOSE_HINTS:
            if needle == name:
                _add_unique(signals, f"{path}: defines `{name}`, which {description}.")
                _add_unique(capabilities, description)
                break

    for type_name in re.findall(r"\b(?:interface|struct)\s+([A-Z][A-Za-z0-9_]+)", text):
        words = _humanize_identifier(type_name)
        if any(token in words for token in ("workspace", "project", "repo", "port", "process", "git", "session", "service")):
            _add_unique(signals, f"{path}: defines `{type_name}` data used for {words}.")

    for line in text.splitlines()[:120]:
        raw = line.strip()
        if not raw.startswith(("//", "///", "/*", "*")):
            continue
        stripped = raw.strip("/").strip("*").strip()
        if not stripped or len(stripped) < 12:
            continue
        lower = stripped.lower()
        if re.search(r"\b(scan|workspace|git|port|process|github|config)\b", lower):
            _add_unique(signals, f"{path}: comment says \"{stripped[:140]}\".")
            break

    return signals, capabilities


def summarize_capability_purpose(capabilities):
    text = " ".join(capabilities).lower()
    parts = []
    if "workspace" in text:
        parts.append("local development workspaces")
    if "repository" in text or "repositories" in text or "git" in text:
        parts.append("repositories")
    if "process" in text or "service" in text:
        parts.append("running services/processes")
    if "port" in text:
        parts.append("localhost ports")
    if "log" in text:
        parts.append("logs")
    if "session" in text:
        parts.append("sessions")
    if "github" in text:
        parts.append("GitHub authentication/integration")
    if not parts:
        return ""
    if len(parts) == 1:
        return f"for managing {parts[0]}"
    return "for managing " + ", ".join(parts[:-1]) + f", and {parts[-1]}"


def repo_map(workdir, max_root_files=50):
    root = Path(workdir).resolve()
    root_files = []
    root_dirs = []
    if not root.exists():
        return {"root": [], "directories": [], "markers": ["Workdir does not exist."]}

    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name in NOISY_DIRS:
            continue
        if child.is_file() and len(root_files) < max_root_files:
            root_files.append(child.name)
        elif child.is_dir():
            root_dirs.append(child.name + "/")

    markers = []
    marker_checks = {
        "Vite config found": any((root / name).exists() for name in ("vite.config.ts", "vite.config.js", "vite.config.mts")),
        "React entrypoint found": any((root / name).exists() for name in ("src/main.tsx", "src/main.jsx", "src/App.tsx", "src/App.jsx")),
        "Tauri config found": (root / "src-tauri/tauri.conf.json").exists() or (root / "tauri.conf.json").exists(),
        "Cargo.toml found in src-tauri": (root / "src-tauri/Cargo.toml").exists(),
        "Electron directory found": (root / "electron").exists(),
        "Next app/pages directory found": (root / "app").exists() or (root / "pages").exists(),
        "Python package config found": (root / "pyproject.toml").exists(),
    }
    for label, found in marker_checks.items():
        if found:
            markers.append(label)

    package = _read_json_file(root / "package.json")
    deps = {}
    if isinstance(package, dict):
        deps.update(package.get("dependencies") or {})
        deps.update(package.get("devDependencies") or {})
        if "react" in deps:
            markers.append("React dependency found")
        if "vite" in deps:
            markers.append("Vite dependency found")
        if "@tauri-apps/api" in deps:
            markers.append("@tauri-apps/api dependency found")
        if "electron" in deps or "electron-builder" in deps:
            markers.append("Electron dependency found")

    return {
        "root": root_files + root_dirs,
        "directories": root_dirs,
        "markers": markers,
    }


def format_repo_map(workdir):
    data = repo_map(workdir)
    root_lines = "\n".join(f"- {item}" for item in data["root"]) or "- (empty)"
    marker_lines = "\n".join(f"- {marker}" for marker in data["markers"]) or "- No common framework markers detected"
    return f"Root:\n{root_lines}\n\nDetected markers:\n{marker_lines}"


def build_project_profile(workdir, budget=None):
    budget = {**DEFAULT_DISCOVERY_BUDGET, **(budget or {})}
    root = Path(workdir).resolve()
    profile = {
        "project_name": None,
        "detected_stack": [],
        "frontend": [],
        "backend": [],
        "desktop_runtime": [],
        "entrypoints": [],
        "package_scripts": {},
        "important_directories": [],
        "documentation_found": [],
        "purpose_clues": [],
        "source_signals": [],
        "capabilities": [],
        "uncertainties": [],
        "confidence": {
            "project_type": "low",
            "tech_stack": "low",
            "purpose": "low",
        },
        "repo_map": repo_map(root, max_root_files=budget["max_root_files"]),
        "files_read": [],
        "confirmed": [],
        "likely": [],
    }
    if not root.exists():
        profile["uncertainties"].append("Working directory does not exist.")
        return profile

    reads_remaining = int(budget["max_reads"])

    def note_read(rel_path):
        nonlocal reads_remaining
        path = root / rel_path
        if reads_remaining <= 0 or not path.exists() or not path.is_file():
            return ""
        reads_remaining -= 1
        rel = rel_path.as_posix() if isinstance(rel_path, Path) else str(rel_path)
        if rel not in profile["files_read"]:
            profile["files_read"].append(rel)
        return _read_text_prefix(path, int(budget["max_lines_per_file"]))

    for doc in DOC_FILES:
        if (root / doc).is_file():
            profile["documentation_found"].append(doc)
            text = note_read(doc)
            for line in text.splitlines():
                stripped = line.strip("# ").strip()
                if stripped and len(stripped) > 4:
                    profile["purpose_clues"].append(f"{doc}: {stripped[:180]}")
                    break
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        profile["documentation_found"].append("docs/")

    package = _read_json_file(root / "package.json")
    deps = {}
    if isinstance(package, dict):
        note_read("package.json")
        profile["project_name"] = package.get("name")
        profile["package_scripts"] = package.get("scripts") or {}
        deps.update(package.get("dependencies") or {})
        deps.update(package.get("devDependencies") or {})
        if profile["project_name"]:
            profile["purpose_clues"].append(f"package.json name: {profile['project_name']}")
        if package.get("description"):
            profile["purpose_clues"].append(f"package.json description: {package['description']}")

    def add_once(key, value):
        if value not in profile[key]:
            profile[key].append(value)

    for rel in PACKAGE_FILES:
        path = root / rel
        if path.exists():
            note_read(rel)

    for rel in ENTRYPOINTS:
        if (root / rel).is_file():
            add_once("entrypoints", rel)
            note_read(rel)

    for rel in SEMANTIC_SOURCE_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = note_read(rel)
        if not text:
            text = _read_text_prefix(path, int(budget["max_lines_per_file"]))
        signals, capabilities = extract_source_signals(rel, text)
        for signal in signals[:12]:
            add_once("source_signals", signal)
        for capability in capabilities[:12]:
            add_once("capabilities", capability)

    for rel in ARCHITECTURE_DIRS:
        path = root / rel
        if path.exists() and not _is_noisy(path.relative_to(root)):
            add_once("important_directories", rel + ("/" if path.is_dir() and not rel.endswith("/") else ""))

    package_scripts = profile["package_scripts"]
    script_blob = "\n".join(str(value) for value in package_scripts.values()).lower()

    tauri_markers = []
    if (root / "src-tauri/tauri.conf.json").exists() or (root / "tauri.conf.json").exists():
        tauri_markers.append("Tauri config")
    if (root / "src-tauri/Cargo.toml").exists():
        tauri_markers.append("src-tauri/Cargo.toml")
    if "@tauri-apps/api" in deps:
        tauri_markers.append("@tauri-apps/api dependency")
    if "tauri" in script_blob:
        tauri_markers.append("package script references tauri")
    if tauri_markers:
        add_once("desktop_runtime", "Tauri")
        add_once("detected_stack", "Tauri")
        profile["confirmed"].append("This repo has Tauri desktop markers: " + ", ".join(tauri_markers) + ".")

    electron_markers = []
    if (root / "electron/main.ts").exists() or (root / "electron/main.js").exists():
        electron_markers.append("electron main file")
    if (root / "electron/preload.ts").exists() or (root / "electron/preload.js").exists():
        electron_markers.append("electron preload file")
    if "electron" in deps or "electron-builder" in deps:
        electron_markers.append("electron dependency")
    if "electron" in script_blob:
        electron_markers.append("package script references electron")
    if electron_markers and ("electron dependency" in electron_markers or "package script references electron" in electron_markers):
        add_once("desktop_runtime", "Electron")
        add_once("detected_stack", "Electron")
        profile["confirmed"].append("Electron appears active: " + ", ".join(electron_markers) + ".")
    elif electron_markers:
        profile["likely"].append("Electron-looking files exist, but dependencies/scripts do not confirm Electron as the active runtime.")
        profile["uncertainties"].append("Electron files may be historical unless package scripts or dependencies confirm they are active.")

    react_markers = []
    if "react" in deps or "react-dom" in deps:
        react_markers.append("React dependency")
    if any((root / rel).exists() for rel in ("src/main.tsx", "src/main.jsx", "src/App.tsx", "src/App.jsx")):
        react_markers.append("React-style TSX/JSX entrypoint")
    if react_markers:
        add_once("frontend", "React")
        add_once("detected_stack", "React")
        profile["confirmed"].append("React frontend markers found: " + ", ".join(react_markers) + ".")

    vite_markers = []
    if "vite" in deps:
        vite_markers.append("Vite dependency")
    if any((root / name).exists() for name in ("vite.config.ts", "vite.config.js", "vite.config.mts")):
        vite_markers.append("Vite config")
    if "vite" in script_blob:
        vite_markers.append("package script references vite")
    if vite_markers:
        add_once("frontend", "Vite")
        add_once("detected_stack", "Vite")
        profile["confirmed"].append("Vite build tooling markers found: " + ", ".join(vite_markers) + ".")

    next_markers = []
    if "next" in deps:
        next_markers.append("Next dependency")
    if (root / "app").exists() or (root / "pages").exists():
        next_markers.append("app/ or pages/ directory")
    if next_markers:
        add_once("frontend", "Next")
        add_once("detected_stack", "Next")
        profile["confirmed"].append("Next.js markers found: " + ", ".join(next_markers) + ".")

    python_markers = []
    py_entry_points = {}
    py_deps = []
    if (root / "pyproject.toml").exists():
        python_markers.append("pyproject.toml")
        try:
            import tomllib  # Python 3.11+
            with open(root / "pyproject.toml", "rb") as _f:
                _toml = tomllib.load(_f)
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
                with open(root / "pyproject.toml", "rb") as _f:
                    _toml = tomllib.load(_f)
            except ImportError:
                _toml = {}
        except Exception:
            _toml = {}
        _proj = _toml.get("project") or {}
        if _proj.get("name"):
            profile["project_name"] = profile["project_name"] or _proj["name"]
            profile["purpose_clues"].append(f"pyproject.toml name: {_proj['name']}")
        if _proj.get("description"):
            profile["purpose_clues"].append(f"pyproject.toml description: {_proj['description'][:180]}")
        py_entry_points = _proj.get("scripts") or (_toml.get("tool", {}).get("poetry", {}).get("scripts") or {})
        _all_deps = list(_proj.get("dependencies") or [])
        for _extra in (_proj.get("optional-dependencies") or {}).values():
            _all_deps.extend(_extra)
        py_deps = [d.split("[")[0].split(">=")[0].split("==")[0].split(">")[0].strip().lower() for d in _all_deps]
        if py_entry_points:
            python_markers.append(f"entry points: {', '.join(list(py_entry_points)[:4])}")
            for _ep in list(py_entry_points.values())[:4]:
                _mod = _ep.split(":")[0].replace(".", "/")
                for _candidate in (f"{_mod}.py", f"{_mod}/__init__.py"):
                    if (root / _candidate).is_file():
                        add_once("entrypoints", _candidate)
                        note_read(_candidate)
        _web_frameworks = {"flask", "django", "fastapi", "starlette", "aiohttp", "tornado", "bottle", "sanic"}
        _cli_frameworks = {"click", "typer", "argparse", "rich", "textual", "prompt-toolkit"}
        _found_web = [f for f in _web_frameworks if f in py_deps]
        _found_cli = [f for f in _cli_frameworks if f in py_deps]
        if _found_web:
            add_once("frontend", "Python web")
            add_once("detected_stack", "Python web")
            python_markers.append(f"web framework(s): {', '.join(_found_web)}")
        if _found_cli or py_entry_points:
            add_once("backend", "Python CLI")
            add_once("detected_stack", "Python CLI")
        _pkg_name = (_proj.get("name") or "").replace("-", "_")
        _pkg_dir = root / _pkg_name if _pkg_name and (root / _pkg_name).is_dir() else None
        if _pkg_dir:
            for _src in sorted(_pkg_dir.glob("*.py"))[:8]:
                _rel = _src.relative_to(root).as_posix()
                if _rel not in profile["files_read"]:
                    text = note_read(_rel)
                    if text:
                        signals, capabilities = extract_source_signals(_rel, text)
                        for s in signals[:6]:
                            add_once("source_signals", s)
                        for c in capabilities[:6]:
                            add_once("capabilities", c)
    if (root / "setup.py").exists() and not python_markers:
        python_markers.append("setup.py")
    if python_markers:
        add_once("backend", "Python")
        add_once("detected_stack", "Python")
        profile["confirmed"].append("Python project: " + "; ".join(python_markers) + ".")
    if (root / "Cargo.toml").exists() or (root / "src-tauri/Cargo.toml").exists():
        add_once("backend", "Rust")
        add_once("detected_stack", "Rust")

    if not profile["documentation_found"]:
        profile["uncertainties"].append("No README.md, AGENTS.md, CLAUDE.md, project.md, or docs/ directory was found.")
    if not profile["purpose_clues"]:
        profile["uncertainties"].append("The exact product purpose is not documented in the inspected files.")
    if profile["capabilities"]:
        capability_text = ", ".join(profile["capabilities"][:8])
        profile["likely"].append(f"Based on source signals, the app likely supports: {capability_text}.")
    if tauri_markers and electron_markers and "Electron" not in profile["desktop_runtime"]:
        profile["confirmed"].append(
            "Tauri appears to be the active desktop runtime; Electron-looking files are present but not confirmed active."
        )

    profile["confidence"]["tech_stack"] = _confidence_from_counts(len(profile["detected_stack"]))
    profile["confidence"]["project_type"] = _confidence_from_counts(
        len(profile["desktop_runtime"]) + len(profile["frontend"]) + len(profile["backend"])
    )
    profile["confidence"]["purpose"] = _confidence_from_counts(len(profile["purpose_clues"]) + len(profile["source_signals"]), len(profile["likely"]))
    return profile


def format_project_profile(profile):
    def lines(items, empty="- None found"):
        return "\n".join(f"- {item}" for item in items) if items else empty

    scripts = profile.get("package_scripts") or {}
    script_lines = [f"{name}: {cmd}" for name, cmd in scripts.items()]
    important = []
    for path in (
        "package.json", "pyproject.toml", "setup.py",
        "src/main.tsx", "src/main.ts", "src/App.tsx",
        "src-tauri/tauri.conf.json", "src-tauri/Cargo.toml",
    ):
        if path in profile.get("files_read", []) or path in profile.get("entrypoints", []):
            important.append(path)
    for path in profile.get("entrypoints", []):
        if path not in important:
            important.append(path)
    important.extend(path for path in profile.get("documentation_found", []) if path not in important)
    overview_parts = []
    if "Tauri" in profile.get("desktop_runtime", []) and "React" in profile.get("frontend", []):
        overview_parts.append("This appears to be a Tauri desktop app with a React frontend")
    elif profile.get("desktop_runtime"):
        overview_parts.append(f"This appears to be a {'/'.join(profile['desktop_runtime'])} desktop app")
    elif profile.get("frontend"):
        overview_parts.append(f"This appears to be a {'/'.join(profile['frontend'])} frontend project")
    elif profile.get("backend"):
        overview_parts.append(f"This appears to be a {'/'.join(profile['backend'])} project")
    else:
        overview_parts.append("The project type is unclear from common framework markers")
    purpose_phrase = summarize_capability_purpose(profile.get("capabilities") or [])
    if purpose_phrase:
        overview_parts[0] += f" {purpose_phrase}"
    name = profile.get("project_name")
    what_it_is = overview_parts[0] + (f" named `{name}`." if name else ".")

    return "\n".join(
        [
            "## Project overview",
            "",
            "### What it is",
            what_it_is,
            "",
            "### Confirmed",
            lines(profile.get("confirmed") or []),
            "",
            "### Likely",
            lines(profile.get("likely") or []),
            "",
            "### Confirmed stack",
            f"- Frontend: {', '.join(profile.get('frontend') or ['unclear'])}",
            f"- Desktop/runtime: {', '.join(profile.get('desktop_runtime') or ['unclear'])}",
            f"- Backend/native: {', '.join(profile.get('backend') or ['unclear'])}",
            f"- Build/tooling: {', '.join([item for item in profile.get('detected_stack', []) if item in {'Vite', 'Next'}] or ['unclear'])}",
            "",
            "### Important files",
            lines(important),
            "",
            "### Architecture",
            "The structure appears split between source entrypoints, framework configuration, and any native/runtime directory markers discovered during inspection.",
            "",
            "### Source behavior signals",
            lines((profile.get("source_signals") or [])[:10]),
            "",
            "### Inferred capabilities",
            lines((profile.get("capabilities") or [])[:10]),
            "",
            "### What is unclear",
            lines(profile.get("uncertainties") or []),
            "",
            "### Confidence",
            f"- Project type: {profile['confidence']['project_type']}",
            f"- Tech stack: {profile['confidence']['tech_stack']}",
            f"- Purpose: {profile['confidence']['purpose']}",
            "",
            "### Suggested next files to inspect",
            lines((profile.get("entrypoints") or [])[:5]),
            "",
            "### Package scripts",
            lines(script_lines),
        ]
    )


def repo_overview(workdir):
    parts = [git_context(workdir)]
    package_path = Path(workdir) / "package.json"
    if package_path.exists():
        try:
            package = json.loads(package_path.read_text(encoding="utf-8"))
            scripts = package.get("scripts") or {}
            deps = sorted((package.get("dependencies") or {}).keys())
            dev_deps = sorted((package.get("devDependencies") or {}).keys())
            parts.append(
                "\n".join(
                    [
                        f"package.json name: {package.get('name', 'unknown')}",
                        f"type: {package.get('type', 'unknown')}",
                        f"main: {package.get('main', 'unknown')}",
                        "scripts:",
                        *(f"  {k}: {v}" for k, v in scripts.items()),
                        "dependencies: " + ", ".join(deps[:30]),
                        "devDependencies: " + ", ".join(dev_deps[:30]),
                    ]
                )
            )
        except Exception as exc:  # noqa: BLE001
            parts.append(f"package.json parse failed: {exc}")

    for dirname in (".", "src", "electron"):
        target = Path(workdir) / dirname
        if not target.exists():
            continue
        _, output = run_subprocess(
            f"find {shlex.quote(str(target))} -maxdepth 2 -type f | sort | sed -n '1,80p'",
            cwd=workdir,
            timeout=20,
        )
        label = "top-level files" if dirname == "." else f"{dirname} files"
        parts.append(f"{label}:\n{output or '(none)'}")

    preferred_entrypoints = []
    for candidate in (
        "package.json",
        "vite.config.ts",
        "src/main.tsx",
        "src/App.tsx",
        "electron/main.ts",
        "electron/preload.ts",
        "electron/ipcHandlers.ts",
    ):
        if (Path(workdir) / candidate).exists():
            preferred_entrypoints.append(candidate)
    if preferred_entrypoints:
        parts.append("preferred source entrypoints for inspection:\n" + "\n".join(preferred_entrypoints))
    return "\n\n".join(parts)


def search_files(workdir, query, path="."):
    cmd = (
        "rg -n --hidden -g '!.git' --max-count 200 --context 2 "
        f"{shlex.quote(query)} {shlex.quote(path)}"
    )
    code, output = run_subprocess(cmd, cwd=workdir, timeout=20)
    if code not in (0, 1):
        return f"search_files failed:\n{output}"
    return output or "(no matches)"


def resolve_path(workdir, path):
    root = Path(workdir).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes workdir: {resolved}")
    return resolved


def read_file(workdir, path, start=1, end=200):
    target = resolve_path(workdir, path)
    if not target.exists():
        return f"File not found: {target}"
    if not target.is_file():
        return f"Not a file: {target}"
    start = max(1, int(start))
    end = max(start, int(end))
    with target.open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    selected = []
    for idx in range(start - 1, min(end, len(lines))):
        selected.append(f"{idx + 1}: {lines[idx].rstrip()}")
    return "\n".join(selected) or "(empty selection)"


def write_file(workdir, path, content):
    target = resolve_path(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        fh.write(content)
    return f"Wrote {len(content)} bytes to {target}"


def replace_in_file(workdir, path, old, new, count=1):
    target = resolve_path(workdir, path)
    if not target.exists():
        return f"File not found: {target}"
    text = target.read_text(encoding="utf-8", errors="replace")
    replacements = text.count(old)
    if old not in text:
        return "Target text not found."
    if count > 0 and replacements < count:
        return f"Found only {replacements} matches, fewer than requested {count}."
    new_text = text.replace(old, new, count if count > 0 else -1)
    target.write_text(new_text, encoding="utf-8")
    actual = replacements if count == 0 else min(replacements, count)
    return f"Replaced {actual} occurrence(s) in {target}"


def replace_lines(workdir, path, start, end, new_content):
    target = resolve_path(workdir, path)
    if not target.exists():
        return f"File not found: {target}"
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    n = len(lines)
    start, end = int(start), int(end)
    if start < 1 or start > n or end < start or end > n:
        return f"Line range {start}-{end} out of bounds (file has {n} lines)."
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    replacement = new_content.splitlines(keepends=True) if new_content else []
    updated = lines[: start - 1] + replacement + lines[end:]
    target.write_text("".join(updated), encoding="utf-8")
    return f"Replaced lines {start}-{end} ({end - start + 1} → {len(replacement)} lines) in {target}"


def insert_after(workdir, path, anchor, content, occurrence=1):
    target = resolve_path(workdir, path)
    if not target.exists():
        return f"File not found: {target}"
    text = target.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(re.escape(anchor), text))
    if not matches:
        return "Anchor not found."
    if occurrence < 1 or occurrence > len(matches):
        return f"Anchor occurrence {occurrence} out of range. Found {len(matches)}."
    match = matches[occurrence - 1]
    pos = match.end()
    updated = text[:pos] + content + text[pos:]
    target.write_text(updated, encoding="utf-8")
    return f"Inserted {len(content)} bytes into {target} after occurrence {occurrence}"

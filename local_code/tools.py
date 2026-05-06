import json
import re
import shlex
import subprocess
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from .ui import clip


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
    req = urllib.request.Request(url, headers={"User-Agent": "local-code/0.2"})
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

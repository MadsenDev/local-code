"""Repository Explorer screen for Rist TUI."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static

from local_code.tui.diff_review import build_review_model
from local_code.tui.repository import RepositoryBadge, RepositoryNode, RepositoryTree


class RepositoryExplorerScreen(ModalScreen[None]):
    """Navigable, read-only project map with Rist session badges."""

    BINDINGS = [
        Binding("up", "cursor_up", "Up"),
        Binding("down", "cursor_down", "Down"),
        Binding("left", "collapse", "Collapse"),
        Binding("right", "expand", "Expand"),
        Binding("enter", "toggle", "Expand/Preview"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "close_or_blur", "Close"),
    ]

    def __init__(self, tree: RepositoryTree, partner, *, badge_filter: RepositoryBadge | None = None):
        super().__init__()
        self.repo_tree = tree
        self.partner = partner
        self.badge_filter = badge_filter
        self.selected = 0
        self.search_query = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="repo-box"):
            yield Label("Repository Explorer", id="repo-title")
            yield Static(id="repo-project")
            yield Input(placeholder="Search files…", id="repo-search")
            with Horizontal(id="repo-main"):
                yield Static(id="repo-tree")
                with Vertical(id="repo-side"):
                    yield Static(id="repo-details")
                    yield Static(id="repo-preview")
            yield Static("Esc Close · / Search · Enter Expand · ↑↓ Navigate", id="repo-hint")

    def on_mount(self) -> None:
        self.repo_tree.build()
        self._render_explorer()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "repo-search":
            self.search_query = event.value
            self.selected = 0
            self._render_explorer()

    def action_focus_search(self) -> None:
        self.query_one("#repo-search", Input).focus()

    def action_close_or_blur(self) -> None:
        if self.query_one("#repo-search", Input).has_focus:
            self.query_one("#repo-search", Input).blur()
            return
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        self.selected = max(0, self.selected - 1)
        self._render_explorer()

    def action_cursor_down(self) -> None:
        nodes = self._visible_nodes()
        self.selected = min(max(len(nodes) - 1, 0), self.selected + 1)
        self._render_explorer()

    def action_expand(self) -> None:
        node = self._selected_node()
        if node and node.is_dir:
            node.expanded = True
            self._render_explorer()

    def action_collapse(self) -> None:
        node = self._selected_node()
        if node and node.is_dir:
            node.expanded = False
            self._render_explorer()

    def action_toggle(self) -> None:
        node = self._selected_node()
        if node and node.is_dir:
            node.expanded = not node.expanded
            self._render_explorer()

    def _visible_nodes(self) -> list[RepositoryNode]:
        return self.repo_tree.visible_nodes(self.search_query, badge=self.badge_filter)

    def _selected_node(self) -> RepositoryNode | None:
        nodes = self._visible_nodes()
        if not nodes:
            return None
        self.selected = min(self.selected, len(nodes) - 1)
        return nodes[self.selected]

    def _render_explorer(self) -> None:
        nodes = self._visible_nodes()
        root = self.repo_tree.root_path
        self.query_one("#repo-project", Static).update(Text(f"Project\n{root.name}/", style="dim"))
        tree_text = Text("Tree\n", style="bold #8fbcbb")
        if not nodes:
            tree_text.append("No files match.", style="dim")
        for index, node in enumerate(nodes):
            depth = node.path.count("/")
            selected = index == self.selected
            style = "reverse bold" if selected else ""
            icon = "▾" if node.is_dir and node.expanded else "▸" if node.is_dir else "·"
            badges = " ".join(f"[{badge.value}]" for badge in sorted(node.badges, key=lambda b: b.value))
            tree_text.append("  " * depth + f"{icon} {node.name}", style=style)
            if badges:
                tree_text.append(f" {badges}", style="yellow reverse" if selected else "yellow")
            tree_text.append("\n")
        self.query_one("#repo-tree", Static).update(tree_text)
        self._render_details(self._selected_node())

    def _render_details(self, node: RepositoryNode | None) -> None:
        details = Text("Details\n", style="bold #8fbcbb")
        preview = Text("Preview\n", style="bold #8fbcbb")
        if node is None:
            details.append("No selection", style="dim")
            preview.append("No preview", style="dim")
        else:
            state = self.repo_tree.session.get(node.path)
            details.append(f"{node.path or node.name}\n", style="bold")
            details.append(f"Type: {'directory' if node.is_dir else 'file'}\n", style="dim")
            if not node.is_dir:
                details.append(f"Size: {node.size or 0} bytes\n", style="dim")
                details.append(f"Language: {node.language}\n", style="dim")
                if node.modified:
                    details.append(f"Last modified: {datetime.fromtimestamp(node.modified).strftime('%Y-%m-%d %H:%M')}\n", style="dim")
            if node.badges:
                details.append("Status: " + ", ".join(b.value for b in sorted(node.badges, key=lambda b: b.value)) + "\n", style="yellow")
            if state:
                details.append(f"Read {state.read_count}× · Edited {state.edited_count}×\n", style="dim")
            diff = self._proposal_diff_for(node.path)
            text = self.repo_tree.preview(node.path, diff=diff)
            for line in text.splitlines():
                style = "green" if line.startswith("+") and not line.startswith("+++") else "red" if line.startswith("-") and not line.startswith("---") else ""
                preview.append(line + "\n", style=style)
        self.query_one("#repo-details", Static).update(details)
        self.query_one("#repo-preview", Static).update(preview)

    def _proposal_diff_for(self, path: str) -> str:
        model = build_review_model(self.partner)
        if not model:
            return ""
        for file in model.files:
            if file.filename == path:
                return file.diff
        return ""

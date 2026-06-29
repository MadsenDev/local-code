"""Intentional Rist prompt input."""

from textual.widgets import Input


class InputBar(Input):
    """Command and prompt input with Rist-specific placeholder."""

    def __init__(self):
        super().__init__(placeholder="Ask Rist to inspect, plan, edit, or explain…", id="input")

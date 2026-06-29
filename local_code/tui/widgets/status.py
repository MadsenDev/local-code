"""Structured status bar widget."""

from textual.widgets import Static


def render_status_text(partner, busy: bool) -> str:
    """Build the structured status line without requiring a mounted widget."""

    models = partner.frontend_model if partner.frontend_model == partner.backend_model else f"{partner.frontend_model}→{partner.backend_model}"
    pending = "pending" if partner.pending_plan else "clear"
    state = "busy" if busy else "idle"
    return (
        f"[b]{partner.provider.name}[/b] | {models} | {state} | mode {partner.mode} | "
        f"routing {partner.routing_decision.get('mode', partner.model_routing)} | "
        f"permissions e:{partner.edit_permission}/c:{partner.command_permission} | plan:{pending}"
    )


class StatusBar(Static):
    """One-line structured runtime status."""

    def render_status(self, partner, busy: bool) -> str:
        return render_status_text(partner, busy)

    def refresh_from(self, partner, busy: bool = False) -> None:
        self.update(self.render_status(partner, busy))

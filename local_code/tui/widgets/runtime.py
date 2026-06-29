"""Runtime confidence panel."""

from rich.table import Table
from textual.widgets import Static


class RuntimePanel(Static):
    """Always-visible compact runtime and context summary."""

    def refresh_from(self, partner) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Provider", partner.provider.name)
        table.add_row("Runtime", partner.provider.describe())
        table.add_row("Profile", getattr(partner, "profile", None) or "default")
        table.add_row("Routing", partner.routing_decision.get("mode", partner.model_routing))
        table.add_row("Context", self._context_label(partner))
        table.add_row("Proposal", "pending" if partner.pending_plan else "clear")
        table.add_row("Managed", "local" if partner.provider.is_local else "cloud")
        self.update(table)

    @staticmethod
    def _context_label(partner) -> str:
        # Keep the always-visible panel cheap: full context accounting can shell
        # out to git and should remain behind the explicit /context command.
        return f"limit {partner.context_limit:,}"

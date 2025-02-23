"""FastAgent validation methods."""

from mcp_agent.core.exceptions import ServerConfigError


def _validate_server_references(self) -> None:
    """
    Validate that all server references in agent configurations exist in config.
    Raises ServerConfigError if any referenced servers are not defined.
    """
    # First check if any agents need servers
    agents_needing_servers = {
        name: agent_data["config"].servers
        for name, agent_data in self.agents.items()
        if agent_data["config"].servers
    }

    if not agents_needing_servers:
        return  # No validation needed

    # If we need servers, verify MCP config exists
    if not hasattr(self.context.config, "mcp"):
        raise ServerConfigError(
            "MCP configuration missing",
            "Agents require server access but no MCP configuration found.\n"
            "Add an 'mcp' section to your configuration file.",
        )

    if not self.context.config.mcp.servers:
        raise ServerConfigError(
            "No MCP servers configured",
            "Agents require server access but no servers are defined.\n"
            "Add server definitions under mcp.servers in your configuration file.",
        )

    # Now check each agent's servers exist
    available_servers = set(self.context.config.mcp.servers.keys())
    for name, servers in agents_needing_servers.items():
        missing = [s for s in servers if s not in available_servers]
        if missing:
            raise ServerConfigError(
                f"Missing server configuration for agent '{name}'",
                f"The following servers are referenced but not defined in config: {', '.join(missing)}",
            )

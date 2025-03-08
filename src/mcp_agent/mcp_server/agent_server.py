# src/mcp_agent/mcp_server/agent_server.py

from mcp.server.fastmcp import FastMCP

# Remove circular import
from mcp_agent.core.agent_app import AgentApp
from mcp.server.fastmcp import Context as MCPContext


class AgentMCPServer:
    """Exposes FastAgent agents as MCP tools through an MCP server."""

    def __init__(
        self,
        agent_app: AgentApp,
        server_name: str = "FastAgent-MCP-Server",
        server_description: str = None,
    ):
        self.agent_app = agent_app
        self.mcp_server = FastMCP(
            name=server_name,
            instructions=server_description
            or f"This server provides access to {len(agent_app.agents)} agents",
        )
        self.setup_tools()

    def setup_tools(self):
        """Register all agents as MCP tools."""
        for agent_name, agent_proxy in self.agent_app._agents.items():
            self.register_agent_tools(agent_name, agent_proxy)

    def register_agent_tools(self, agent_name: str, agent_proxy):
        """Register tools for a specific agent."""

        # Basic send message tool
        @self.mcp_server.tool(
            name=f"{agent_name}.send",
            description=f"Send a message to the {agent_name} agent",
        )
        async def send_message(message: str, ctx: MCPContext) -> str:
            """Send a message to the agent and return its response."""

            # Get the agent's context
            agent_context = None
            if hasattr(agent_proxy, "_agent") and hasattr(
                agent_proxy._agent, "context"
            ):
                agent_context = agent_proxy._agent.context

            # Define the function to execute
            async def execute_send():
                return await agent_proxy.send(message)

            # Execute with bridged context
            if agent_context and ctx:
                return await self.with_bridged_context(agent_context, ctx, execute_send)
            else:
                return await execute_send()

    def run(self, transport: str = "sse", host: str = "0.0.0.0", port: int = 8000):
        """Run the MCP server."""
        if transport == "sse":
            # For running as a web server
            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port

        self.mcp_server.run(transport=transport)

    async def run_async(
        self, transport: str = "sse", host: str = "0.0.0.0", port: int = 8000
    ):
        """Run the MCP server asynchronously."""
        if transport == "sse":
            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port
            await self.mcp_server.run_sse_async()
        else:  # stdio
            await self.mcp_server.run_stdio_async()

    async def with_bridged_context(
        self, agent_context, mcp_context, func, *args, **kwargs
    ):
        """
        Execute a function with bridged context between MCP and agent

        Args:
            agent_context: The agent's context object
            mcp_context: The MCP context from the tool call
            func: The function to execute
            args, kwargs: Arguments to pass to the function
        """
        # Store original progress reporter if it exists
        original_progress_reporter = None
        if hasattr(agent_context, "progress_reporter"):
            original_progress_reporter = agent_context.progress_reporter

        # Store MCP context in agent context for nested calls
        agent_context.mcp_context = mcp_context

        # Create bridged progress reporter
        async def bridged_progress(progress, total=None):
            if mcp_context:
                await mcp_context.report_progress(progress, total)
            if original_progress_reporter:
                await original_progress_reporter(progress, total)

        # Install bridged progress reporter
        if hasattr(agent_context, "progress_reporter"):
            agent_context.progress_reporter = bridged_progress

        try:
            # Call the function
            return await func(*args, **kwargs)
        finally:
            # Restore original progress reporter
            if hasattr(agent_context, "progress_reporter"):
                agent_context.progress_reporter = original_progress_reporter

            # Remove MCP context reference
            if hasattr(agent_context, "mcp_context"):
                delattr(agent_context, "mcp_context")

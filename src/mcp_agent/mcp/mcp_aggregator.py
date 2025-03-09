from asyncio import Lock, gather
from typing import (
    List,
    Dict,
    Optional,
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
)
from mcp import GetPromptResult
from pydantic import BaseModel, ConfigDict
from mcp.client.session import ClientSession
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Tool,
)

from mcp_agent.event_progress import ProgressAction
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.gen_client import gen_client

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

if TYPE_CHECKING:
    from mcp_agent.context import Context


logger = get_logger(
    __name__
)  # This will be replaced per-instance when agent_name is available

SEP = "-"

# Define type variables for the generalized method
T = TypeVar("T")
R = TypeVar("R")


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str


class MCPAggregator(ContextDependent):
    """
    Aggregates multiple MCP servers. When a developer calls, e.g. call_tool(...),
    the aggregator searches all servers in its list for a server that provides that tool.
    """

    initialized: bool = False
    """Whether the aggregator has been initialized with tools and resources from all servers."""

    connection_persistence: bool = False
    """Whether to maintain a persistent connection to the server."""

    server_names: List[str]
    """A list of server names to connect to."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a connection manager to manage persistent connections for this aggregator
        if self.connection_persistence:
            # Try to get existing connection manager from context
            if not hasattr(self.context, "_connection_manager"):
                self.context._connection_manager = MCPConnectionManager(
                    self.context.server_registry
                )
                await self.context._connection_manager.__aenter__()
            self._persistent_connection_manager = self.context._connection_manager

        await self.load_servers()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __init__(
        self,
        server_names: List[str],
        connection_persistence: bool = True,  # Default to True for better stability
        context: Optional["Context"] = None,
        name: str = None,
        **kwargs,
    ):
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to maintain persistent connections to servers (default: True).
        Note: The server names must be resolvable by the gen_client function, and specified in the server registry.
        """
        super().__init__(
            context=context,
            **kwargs,
        )

        self.server_names = server_names
        self.connection_persistence = connection_persistence
        self.agent_name = name
        self._persistent_connection_manager: MCPConnectionManager = None

        # Set up logger with agent name in namespace if available
        global logger
        logger_name = f"{__name__}.{name}" if name else __name__
        logger = get_logger(logger_name)

        # Maps namespaced_tool_name -> namespaced tool info
        self._namespaced_tool_map: Dict[str, NamespacedTool] = {}
        # Maps server_name -> list of tools
        self._server_to_tool_map: Dict[str, List[NamespacedTool]] = {}
        self._tool_map_lock = Lock()

        # TODO: saqadri - add resources and prompt maps as well

    async def close(self):
        """
        Close all persistent connections when the aggregator is deleted.
        """
        if self.connection_persistence and self._persistent_connection_manager:
            try:
                # Only attempt cleanup if we own the connection manager
                if (
                    hasattr(self.context, "_connection_manager")
                    and self.context._connection_manager
                    == self._persistent_connection_manager
                ):
                    logger.info("Shutting down all persistent connections...")
                    await self._persistent_connection_manager.disconnect_all()
                    await self._persistent_connection_manager.__aexit__(
                        None, None, None
                    )
                    delattr(self.context, "_connection_manager")
                self.initialized = False
            except Exception as e:
                logger.error(f"Error during connection manager cleanup: {e}")

    @classmethod
    async def create(
        cls,
        server_names: List[str],
        connection_persistence: bool = False,
    ) -> "MCPAggregator":
        """
        Factory method to create and initialize an MCPAggregator.
        Use this instead of constructor since we need async initialization.
        If connection_persistence is True, the aggregator will maintain a
        persistent connection to the servers for as long as this aggregator is around.
        By default we do not maintain a persistent connection.
        """

        logger.info(f"Creating MCPAggregator with servers: {server_names}")

        instance = cls(
            server_names=server_names,
            connection_persistence=connection_persistence,
        )

        try:
            await instance.__aenter__()

            logger.debug("Loading servers...")
            await instance.load_servers()

            logger.debug("MCPAggregator created and initialized.")
            return instance
        except Exception as e:
            logger.error(f"Error creating MCPAggregator: {e}")
            await instance.__aexit__(None, None, None)

    async def load_servers(self):
        """
        Discover tools from each server in parallel and build an index of namespaced tool names.
        """
        if self.initialized:
            logger.debug("MCPAggregator already initialized.")
            return

        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        for server_name in self.server_names:
            if self.connection_persistence:
                logger.info(
                    f"Creating persistent connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.STARTING,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )
                await self._persistent_connection_manager.get_server(
                    server_name, client_session_factory=MCPAgentClientSession
                )

            logger.info(
                f"MCP Servers initialized for agent '{self.agent_name}'",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "agent_name": self.agent_name,
                },
            )

        async def fetch_tools(client: ClientSession):
            try:
                result: ListToolsResult = await client.list_tools()
                return result.tools or []
            except Exception as e:
                logger.error(f"Error loading tools from server '{server_name}'", data=e)
                return []

        async def load_server_tools(server_name: str):
            tools: List[Tool] = []
            if self.connection_persistence:
                server_connection = (
                    await self._persistent_connection_manager.get_server(
                        server_name, client_session_factory=MCPAgentClientSession
                    )
                )
                tools = await fetch_tools(server_connection.session)
            else:
                async with gen_client(
                    server_name, server_registry=self.context.server_registry
                ) as client:
                    tools = await fetch_tools(client)

            return server_name, tools

        # Gather tools from all servers concurrently
        results = await gather(
            *(load_server_tools(server_name) for server_name in self.server_names),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                continue
            server_name, tools = result

            self._server_to_tool_map[server_name] = []
            for tool in tools:
                namespaced_tool_name = f"{server_name}{SEP}{tool.name}"
                namespaced_tool = NamespacedTool(
                    tool=tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_tool_name,
                )

                self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                self._server_to_tool_map[server_name].append(namespaced_tool)
            logger.debug(
                "MCP Aggregator initialized",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                },
            )
        self.initialized = True

    async def list_servers(self) -> List[str]:
        """Return the list of server names aggregated by this agent."""
        if not self.initialized:
            await self.load_servers()

        return self.server_names

    async def list_tools(self) -> ListToolsResult:
        """
        :return: Tools from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        if not self.initialized:
            await self.load_servers()

        return ListToolsResult(
            tools=[
                namespaced_tool.tool.model_copy(update={"name": namespaced_tool_name})
                for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()
            ]
        )

    async def _execute_on_server(
        self,
        server_name: str,
        operation_type: str,
        operation_name: str,
        method_name: str,
        method_args: Dict[str, Any] = None,
        error_factory: Callable[[str], R] = None,
    ) -> R:
        """
        Generic method to execute operations on a specific server.

        Args:
            server_name: Name of the server to execute the operation on
            operation_type: Type of operation (for logging) e.g., "tool", "prompt"
            operation_name: Name of the specific operation being called (for logging)
            method_name: Name of the method to call on the client session
            method_args: Arguments to pass to the method
            error_factory: Function to create an error return value if the operation fails

        Returns:
            Result from the operation or an error result
        """
        logger.info(
            f"Requesting {operation_type}",
            data={
                "progress_action": ProgressAction.STARTING,
                f"{operation_type}_name": operation_name,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        async def try_execute(client: ClientSession):
            try:
                method = getattr(client, method_name)
                return await method(**method_args) if method_args else await method()
            except Exception as e:
                error_msg = f"Failed to {method_name} '{operation_name}' on server '{server_name}': {e}"
                logger.error(error_msg)
                return error_factory(error_msg) if error_factory else None

        if self.connection_persistence:
            server_connection = await self._persistent_connection_manager.get_server(
                server_name, client_session_factory=MCPAgentClientSession
            )
            return await try_execute(server_connection.session)
        else:
            logger.debug(
                f"Creating temporary connection to server: {server_name}",
                data={
                    "progress_action": ProgressAction.STARTING,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                },
            )
            async with gen_client(
                server_name, server_registry=self.context.server_registry
            ) as client:
                result = await try_execute(client)
                logger.debug(
                    f"Closing temporary connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.SHUTDOWN,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )
                return result

    async def _parse_resource_name(
        self, name: str, resource_type: str
    ) -> tuple[str, str]:
        """
        Parse a possibly namespaced resource name into server name and local resource name.

        Args:
            name: The resource name, possibly namespaced
            resource_type: Type of resource (for error messages), e.g. "tool", "prompt"

        Returns:
            Tuple of (server_name, local_resource_name)
        """
        server_name = None
        local_name = None

        if SEP in name:  # Namespaced resource name
            server_name, local_name = name.split(SEP, 1)
        else:
            # For tools, search all servers for the tool
            if resource_type == "tool":
                for _, tools in self._server_to_tool_map.items():
                    for namespaced_tool in tools:
                        if namespaced_tool.tool.name == name:
                            server_name = namespaced_tool.server_name
                            local_name = name
                            break
                    if server_name:
                        break
            # For other resource types, use the first server
            else:
                local_name = name
                server_name = self.server_names[0] if self.server_names else None

        return server_name, local_name

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        """
        Call a namespaced tool, e.g., 'server_name.tool_name'.
        """
        if not self.initialized:
            await self.load_servers()

        server_name, local_tool_name = await self._parse_resource_name(name, "tool")

        if server_name is None or local_tool_name is None:
            logger.error(f"Error: Tool '{name}' not found")
            return CallToolResult(isError=True, message=f"Tool '{name}' not found")

        return await self._execute_on_server(
            server_name=server_name,
            operation_type="tool",
            operation_name=local_tool_name,
            method_name="call_tool",
            method_args={"name": local_tool_name, "arguments": arguments},
            error_factory=lambda msg: CallToolResult(isError=True, message=msg),
        )

    async def get_prompt(self, prompt_name: str = None) -> GetPromptResult:
        """
        Get a prompt from a server.

        :param prompt_name: Name of the prompt, optionally namespaced with server name
                           using the format 'server_name-prompt_name'
        :return: GetPromptResult containing the prompt description and messages
        """
        if not self.initialized:
            await self.load_servers()

        # Handle the case where prompt_name is None
        if not prompt_name:
            server_name = self.server_names[0] if self.server_names else None
            local_prompt_name = None
        else:
            server_name, local_prompt_name = await self._parse_resource_name(
                prompt_name, "prompt"
            )

        if not server_name:
            logger.error("Error: No servers available for getting prompts")
            return GetPromptResult(
                description="Error: No servers available for getting prompts",
                messages=[],
            )

        return await self._execute_on_server(
            server_name=server_name,
            operation_type="prompt",
            operation_name=local_prompt_name or "default",
            method_name="get_prompt",
            method_args={"name": local_prompt_name} if local_prompt_name else {},
            error_factory=lambda msg: GetPromptResult(description=msg, messages=[]),
        )

    async def list_prompts(self, server_name: str = None):
        """
        List available prompts from one or all servers.

        :param server_name: Optional server name to list prompts from. If not provided,
                           lists prompts from all servers.
        :return: Dictionary mapping server names to lists of available prompts
        """
        if not self.initialized:
            await self.load_servers()

        results = {}

        # If server_name is provided, only list prompts from that server
        if server_name:
            if server_name in self.server_names:
                result = await self._execute_on_server(
                    server_name=server_name,
                    operation_type="prompts-list",
                    operation_name="",
                    method_name="list_prompts",
                    error_factory=lambda _: [],
                )
                results[server_name] = result
            else:
                logger.error(f"Server '{server_name}' not found")
        else:
            # Gather prompts from all servers concurrently
            tasks = [
                self._execute_on_server(
                    server_name=s_name,
                    operation_type="prompts-list",
                    operation_name="",
                    method_name="list_prompts",
                    error_factory=lambda _: [],
                )
                for s_name in self.server_names
            ]
            server_results = await gather(*tasks, return_exceptions=True)

            for i, result in enumerate(server_results):
                if isinstance(result, BaseException):
                    continue
                results[self.server_names[i]] = result

        return results


class MCPCompoundServer(Server):
    """
    A compound server (server-of-servers) that aggregates multiple MCP servers and is itself an MCP server
    """

    def __init__(self, server_names: List[str], name: str = "MCPCompoundServer"):
        super().__init__(name)
        self.aggregator = MCPAggregator(server_names)

        # Register handlers for tools, prompts, and resources
        self.list_tools()(self._list_tools)
        self.call_tool()(self._call_tool)
        self.get_prompt()(self._get_prompt)
        self.list_prompts()(self._list_prompts)

    async def _list_tools(self) -> List[Tool]:
        """List all tools aggregated from connected MCP servers."""
        tools_result = await self.aggregator.list_tools()
        return tools_result.tools

    async def _call_tool(
        self, name: str, arguments: dict | None = None
    ) -> CallToolResult:
        """Call a specific tool from the aggregated servers."""
        try:
            result = await self.aggregator.call_tool(name=name, arguments=arguments)
            return result.content
        except Exception as e:
            return CallToolResult(isError=True, message=f"Error calling tool: {e}")

    async def _get_prompt(self, name: str = None) -> GetPromptResult:
        """Get a prompt from the aggregated servers."""
        try:
            result = await self.aggregator.get_prompt(prompt_name=name)
            return result
        except Exception as e:
            return GetPromptResult(
                description=f"Error getting prompt: {e}", messages=[]
            )

    async def _list_prompts(self, server_name: str = None) -> Dict[str, List[str]]:
        """List available prompts from the aggregated servers."""
        try:
            return await self.aggregator.list_prompts(server_name=server_name)
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return {}

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.run(
                read_stream=read_stream,
                write_stream=write_stream,
                initialization_options=self.create_initialization_options(),
            )

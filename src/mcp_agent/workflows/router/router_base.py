from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, Optional, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp.tools import Tool as FastTool

from mcp_agent.agents.agent import Agent
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)

ResultT = TypeVar("ResultT", bound=str | Agent | Callable)


class RouterResult(BaseModel, Generic[ResultT]):
    """A class that represents the result of a Router.route request"""

    result: ResultT
    """The router returns an MCP server name, an Agent, or a function to route the input to."""

    p_score: float | None = None
    """
    The probability score (i.e. 0->1) of the routing decision. 
    This is optional and may only be provided if the router is probabilistic (e.g. a probabilistic binary classifier).
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class RouterCategory(BaseModel):
    """
    A class that represents a category of routing.
    Used to collect information the router needs to decide.
    """

    name: str
    """The name of the category"""

    description: str | None = None
    """A description of the category"""

    category: str | Agent | Callable
    """The class to route to"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class ServerRouterCategory(RouterCategory):
    """A class that represents a category of routing to an MCP server"""

    tools: List[FastTool] = Field(default_factory=list)


class AgentRouterCategory(RouterCategory):
    """A class that represents a category of routing to an agent"""

    servers: List[ServerRouterCategory] = Field(default_factory=list)


class Router(ABC, ContextDependent):
    """
    Routing classifies an input and directs it to one or more specialized followup tasks.
    This class helps to route an input to a specific MCP server,
    an Agent (an aggregation of MCP servers), or a function (any Callable).

    When to use this workflow:
        - This workflow allows for separation of concerns, and building more specialized prompts.

        - Routing works well for complex tasks where there are distinct categories that
        are better handled separately, and where classification can be handled accurately,
        either by an LLM or a more traditional classification model/algorithm.

    Examples where routing is useful:
        - Directing different types of customer service queries
        (general questions, refund requests, technical support)
        into different downstream processes, prompts, and tools.

        - Routing easy/common questions to smaller models like Claude 3.5 Haiku
        and hard/unusual questions to more capable models like Claude 3.5 Sonnet
        to optimize cost and speed.

    Args:
        routing_instruction: A string that tells the router how to route the input.
        mcp_servers_names: A list of server names to route the input to.
        agents: A list of agents to route the input to.
        functions: A list of functions to route the input to.
    """

    def __init__(
        self,
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        super().__init__(context=context, **kwargs)
        self.routing_instruction = routing_instruction
        self.server_names = server_names or []
        self.agents = agents or []
        self.functions = functions or []
        self.server_registry = self.context.server_registry

        # A dict of categories to route to, keyed by category name.
        # These are populated in the initialize method.
        self.server_categories: Dict[str, ServerRouterCategory] = {}
        self.agent_categories: Dict[str, AgentRouterCategory] = {}
        self.function_categories: Dict[str, RouterCategory] = {}
        self.categories: Dict[str, RouterCategory] = {}
        self.initialized: bool = False

        if not self.server_names and not self.agents and not self.functions:
            raise ValueError(
                "At least one of mcp_servers_names, agents, or functions must be provided."
            )

        if self.server_names and not self.server_registry:
            raise ValueError(
                "server_registry must be provided if mcp_servers_names are provided."
            )

    @abstractmethod
    async def route(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[str | Agent | Callable]]:
        """
        Route the input request to one or more MCP servers, agents, or functions.
        If no routing decision can be made, returns an empty list.

        Args:
            request: The input to route.
            top_k: The maximum number of top routing results to return. May return fewer.
        """

    @abstractmethod
    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[str]]:
        """Route the input to one or more MCP servers."""

    @abstractmethod
    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[Agent]]:
        """Route the input to one or more agents."""

    @abstractmethod
    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[RouterResult[Callable]]:
        """
        Route the input to one or more functions.

        Args:
            input: The input to route.
        """

    async def initialize(self):
        """Initialize the router categories."""

        if self.initialized:
            return

        server_categories = [
            self.get_server_category(server_name) for server_name in self.server_names
        ]
        self.server_categories = {
            category.name: category for category in server_categories
        }

        agent_categories = [self.get_agent_category(agent) for agent in self.agents]
        self.agent_categories = {
            category.name: category for category in agent_categories
        }

        function_categories = [
            self.get_function_category(function) for function in self.functions
        ]
        self.function_categories = {
            category.name: category for category in function_categories
        }

        all_categories = server_categories + agent_categories + function_categories

        self.categories = {category.name: category for category in all_categories}
        self.initialized = True

    def get_server_category(self, server_name: str) -> ServerRouterCategory:
        server_config = self.server_registry.get_server_config(server_name)

        # TODO: saqadri - Currently we only populate the server name and description.
        # To make even more high fidelity routing decisions, we can populate the
        # tools, resources and prompts that the server has access to.
        return ServerRouterCategory(
            category=server_name,
            name=server_config.name if server_config else server_name,
            description=server_config.description,
            tools=[],  # Empty list to avoid validation errors
        )

    def get_agent_category(self, agent: Agent) -> AgentRouterCategory:
        agent_description = (
            agent.instruction({}) if callable(agent.instruction) else agent.instruction
        )

        # Just get server categories without attempting to access tools
        # This is a simpler approach that avoids potential issues with uninitialized agents
        server_categories = [
            self.get_server_category(server_name) for server_name in agent.server_names
        ]

        return AgentRouterCategory(
            category=agent,
            name=agent.name,
            description=agent_description,
            servers=server_categories,
        )

    def get_function_category(self, function: Callable) -> RouterCategory:
        tool = FastTool.from_function(function)

        return RouterCategory(
            category=function,
            name=tool.name,
            description=tool.description,
        )

    def format_category(
        self, category: RouterCategory, index: int | None = None
    ) -> str:
        """Format a category into a readable string."""

        if isinstance(category, ServerRouterCategory):
            category_str = self._format_server_category(category)
        elif isinstance(category, AgentRouterCategory):
            category_str = self._format_agent_category(category)
        else:
            category_str = self._format_function_category(category)

        return category_str

    def _format_tools(self, tools: List[FastTool]) -> str:
        """Format a list of tools into a readable string."""
        if not tools:
            # Return a note about tools within XML tags to maintain structure
            return '<fastagent:tool name="info">No tool information available</fastagent:tool>'

        tool_descriptions = []
        for tool in tools:
            # Access tool name and description safely
            tool_name = getattr(tool, "name", "unnamed-tool")
            tool_description = getattr(tool, "description", "No description available")
            desc = f'<fastagent:tool name="{tool_name}">{tool_description}</fastagent:tool>'
            tool_descriptions.append(desc)

        return "\n".join(tool_descriptions)

    def _format_server_category(self, category: ServerRouterCategory) -> str:
        """Format a server category into a readable string."""
        # Check if we have any content (description or tools)
        has_description = bool(category.description)
        has_tools = bool(category.tools)

        # If no content at all, use self-closing tag
        if not has_description and not has_tools:
            return f'<fastagent:server-category name="{category.name}" />'

        # Otherwise, build the content
        description_section = ""
        if has_description:
            description_section = f"\n<fastagent:description>{category.description}</fastagent:description>"

        # Add tools section if we have tool information
        if has_tools:
            tools = self._format_tools(category.tools)
            return f"""<fastagent:server-category name="{category.name}">{description_section}
<fastagent:tools>
{tools}
</fastagent:tools>
</fastagent:server-category>"""
        else:
            # Just description, no tools
            return f"""<fastagent:server-category name="{category.name}">{description_section}
</fastagent:server-category>"""

    def _format_agent_category(self, category: AgentRouterCategory) -> str:
        """Format an agent category into a readable string."""
        # Check if we have any content (description or servers)
        has_description = bool(category.description)
        has_servers = bool(category.servers)

        # If no content at all, use self-closing tag
        if not has_description and not has_servers:
            return f'<fastagent:agent-category name="{category.name}" />'

        # Build description section if needed
        description_section = ""
        if has_description:
            description_section = f"\n<fastagent:description>{category.description}</fastagent:description>"

        # Handle the case with no servers
        if not has_servers:
            return f"""<fastagent:agent-category name="{category.name}">{description_section}
</fastagent:agent-category>"""

        # Format servers with proper XML tags and include their tools
        server_sections = []
        for server in category.servers:
            # Check if this server has any content
            has_server_description = bool(server.description)
            has_server_tools = bool(server.tools)

            # Use self-closing tag if server has no content
            if not has_server_description and not has_server_tools:
                server_section = f'<fastagent:server name="{server.name}" />'
                server_sections.append(server_section)
                continue

            # Build server description if needed
            server_desc_section = ""
            if has_server_description:
                server_desc_section = f"\n<fastagent:description>{server.description}</fastagent:description>"

            # Format server tools if available
            if has_server_tools:
                tool_items = []
                for tool in server.tools:
                    tool_desc = tool.description if tool.description else ""
                    tool_items.append(
                        f'<fastagent:tool name="{tool.name}">{tool_desc}</fastagent:tool>'
                    )

                tools_section = f"\n<fastagent:tools>\n{chr(10).join(tool_items)}\n</fastagent:tools>"
                server_section = f"""<fastagent:server name="{server.name}">{server_desc_section}{tools_section}
</fastagent:server>"""
            else:
                # Just description, no tools
                server_section = f"""<fastagent:server name="{server.name}">{server_desc_section}
</fastagent:server>"""

            server_sections.append(server_section)

        servers = "\n".join(server_sections)

        return f"""<fastagent:agent-category name="{category.name}">{description_section}
<fastagent:servers>
{servers}
</fastagent:servers>
</fastagent:agent-category>"""

    def _format_function_category(self, category: RouterCategory) -> str:
        """Format a function category into a readable string."""
        # Check if we have a description
        has_description = bool(category.description)

        # If no description, use self-closing tag
        if not has_description:
            return f'<fastagent:function-category name="{category.name}" />'

        # Include description
        return f"""<fastagent:function-category name="{category.name}">
<fastagent:description>{category.description}</fastagent:description>
</fastagent:function-category>"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from mcp.types import TextContent
from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.base_agent import BaseAgent
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.core.request_params import RequestParams
from mcp_agent.workflows.router.router_base import ResultT, RouterResult

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)

# TODO -- reinstate function/server routing
# TODO -- Generate the Example Schema from the Pydantic Model
DEFAULT_ROUTING_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).

<fastagent:data>
<fastagent:categories>
{context}
</fastagent:categories>

<fastagent:request>
{request}
</fastagent:request>
</fastagent:data>

Your task is to analyze the request and determine the most appropriate categories from the options above. Consider:
- The specific capabilities and tools each destination offers
- How well the request matches the category's description
- Whether the request might benefit from multiple categories (up to {top_k})

<fastagent:instruction>
Respond in JSON format. NEVER include Code Fences:
{{
    "categories": [
        {{
            "category": <category name>,
            "confidence": <high, medium or low>,
            "reasoning": <brief explanation>
        }}
    ]
}}

Only include categories that are truly relevant. You may return fewer than {top_k} if appropriate.
If none of the categories are relevant, return an empty list.
</fastagent:instruction>
"""

ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate category.
A category is a specialized destination, such as a Function, an MCP Server (a collection of tools/functions), or an Agent (a collection of servers).

You will analyze requests and choose the most appropriate categories based on their capabilities and descriptions.
You can choose one or more categories, or choose none if no category is appropriate.

Follow these guidelines:
- Carefully match the request's needs with category capabilities
- Consider which tools or servers would best address the request
- If multiple categories could help, select all relevant ones
- Only include truly relevant categories, not tangentially related ones
"""


class ConfidenceRating(BaseModel):
    """Base class for models with confidence ratings and reasoning"""

    """The confidence level of the routing decision."""
    confidence: Literal["high", "medium", "low"]
    """A brief explanation of the routing decision."""
    reasoning: str | None = None  # Make nullable to support both use cases


# Used for LLM output parsing
class StructuredResponseCategory(ConfidenceRating):
    """The name of the category (i.e. MCP server, Agent or function) to route the input to."""

    category: str  # Category name for lookup


class StructuredResponse(BaseModel):
    categories: List[StructuredResponseCategory]


# Used for final router output
class LLMRouterResult(RouterResult[ResultT], ConfidenceRating):
    # Inherits 'result' from RouterResult
    # Inherits 'confidence' and 'reasoning' from ConfidenceRating
    pass


class LLMRouter(BaseAgent):
    """
    A router that uses an LLM to determine the best destination for a request.
    """

    def __init__(
        self,
        config: Union[AgentConfig, str],
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
        **kwargs
    ) -> None:
        """
        Initialize an LLMRouter.
        
        Args:
            config: Agent configuration or name
            server_names: Optional list of server names to route to
            agents: Optional list of agents to route to
            functions: Optional list of functions to route to
            routing_instruction: Optional custom routing instruction
            context: Optional application context
            default_request_params: Optional default request parameters
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(
            config=config,
            server_names=server_names,
            functions=functions,
            context=context,
            **kwargs
        )
        self.agents = agents or []
        self.routing_instruction = routing_instruction
        
        # Override default request params with router-specific params
        router_params = RequestParams(
            systemPrompt=ROUTING_SYSTEM_INSTRUCTION,
            use_history=False,
        )
        
        # Merge with any provided default params
        if default_request_params:
            params_dict = router_params.model_dump()
            params_dict.update(default_request_params.model_dump(exclude_unset=True))
            router_params = RequestParams(**params_dict)
            
        self._default_request_params = router_params
        
        # Initialize category storage
        self.server_categories = {}
        self.agent_categories = {}
        self.function_categories = {}
        self.categories = {}

    async def initialize(self) -> None:
        """Initialize the router and create the LLM instance."""
        if not self.initialized:
            await super().initialize()
            
            # Initialize categories
            await self._initialize_categories()
            
            self.initialized = True
    
    async def _initialize_categories(self) -> None:
        """Initialize the categories for routing."""
        # Implementation would populate self.server_categories, self.agent_categories,
        # self.function_categories and self.categories from server_names, agents, and functions
        pass

    async def route(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[str | Agent | Callable]]:
        """
        Route a request to the most appropriate destination.
        
        Args:
            request: The request to route
            top_k: Maximum number of results to return
            
        Returns:
            List of routing results with confidence ratings
        """
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(request, top_k)

    async def route_to_server(self, request: str, top_k: int = 1) -> List[LLMRouterResult[str]]:
        """
        Route a request to a server.
        
        Args:
            request: The request to route
            top_k: Maximum number of results to return
            
        Returns:
            List of routing results with confidence ratings
        """
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=True,
            include_agents=False,
            include_functions=False,
        )

    async def route_to_agent(self, request: str, top_k: int = 1) -> List[LLMRouterResult[Agent]]:
        """
        Route a request to an agent.
        
        Args:
            request: The request to route
            top_k: Maximum number of results to return
            
        Returns:
            List of routing results with confidence ratings
        """
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=False,
            include_agents=True,
            include_functions=False,
        )

    async def _route_with_llm(
        self,
        request: str,
        top_k: int = 1,
        include_servers: bool = True,
        include_agents: bool = True,
        include_functions: bool = True,
    ) -> List[LLMRouterResult]:
        """
        Route a request using the LLM.
        
        Args:
            request: The request to route
            top_k: Maximum number of results to return
            include_servers: Whether to include servers in routing
            include_agents: Whether to include agents in routing
            include_functions: Whether to include functions in routing
            
        Returns:
            List of routing results with confidence ratings
        """
        if not self.initialized:
            await self.initialize()

        routing_instruction = self.routing_instruction or DEFAULT_ROUTING_INSTRUCTION

        # Generate the categories context
        context = self._generate_context(
            include_servers=include_servers,
            include_agents=include_agents,
            include_functions=include_functions,
        )

        # Format the prompt with all the necessary information
        prompt_text = routing_instruction.format(context=context, request=request, top_k=top_k)
        
        # Create multipart message
        prompt = PromptMessageMultipart(
            role="user",
            content=[TextContent(type="text", text=prompt_text)]
        )

        # Get structured response from LLM
        response = await self.structured(
            [prompt],
            StructuredResponse,
            None
        )

        # Construct the result
        if not response or not response.categories:
            return []

        result: List[LLMRouterResult] = []
        for r in response.categories:
            router_category = self.categories.get(r.category)
            if not router_category:
                # TODO: log or raise an error
                continue

            result.append(
                LLMRouterResult(
                    result=router_category.category,
                    confidence=r.confidence,
                    reasoning=r.reasoning,
                )
            )

        return result[:top_k]

    def _generate_context(
        self,
        include_servers: bool = True,
        include_agents: bool = True,
        include_functions: bool = True,
    ) -> str:
        """
        Generate a formatted context list of categories.
        
        Args:
            include_servers: Whether to include servers in the context
            include_agents: Whether to include agents in the context
            include_functions: Whether to include functions in the context
            
        Returns:
            Formatted context string for the LLM prompt
        """
        context_list = []
        idx = 1

        # Format all categories
        if include_servers:
            for category in self.server_categories.values():
                context_list.append(self.format_category(category, idx))
                idx += 1

        if include_agents:
            for category in self.agent_categories.values():
                context_list.append(self.format_category(category, idx))
                idx += 1

        if include_functions:
            for category in self.function_categories.values():
                context_list.append(self.format_category(category, idx))
                idx += 1

        return "\n\n".join(context_list)
    
    def format_category(self, category: Any, idx: int) -> str:
        """
        Format a category for inclusion in the prompt.
        
        Args:
            category: The category to format
            idx: The index of the category
            
        Returns:
            Formatted category string
        """
        # Implementation would format a category for inclusion in the prompt
        # This is a placeholder for the actual implementation
        return f"{idx}. Category: {category.name} - {category.description}"
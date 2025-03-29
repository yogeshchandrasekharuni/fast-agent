from typing import TYPE_CHECKING, Callable, List, Literal, Optional

from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.augmented_llm import RequestParams
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


class LLMRouter(Agent):
    """
    A router that uses an LLM to route an input to a specific category.
    """

    def __init__(
        self,
        config: AgentConfig,
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
    ) -> None:
        super().__init__(
            config=config,
            server_names=server_names,
            functions=functions,
            routing_instruction=routing_instruction,
            context=context,
        )
        self.agents = agents
        self.default_request_params = default_request_params or RequestParams()

    async def initialize(self) -> None:
        """Initialize the router and create the LLM instance."""
        if not self.initialized:
            await super().initialize()
            router_params = RequestParams(
                systemPrompt=ROUTING_SYSTEM_INSTRUCTION,
                use_history=False,
            )

            # Merge with any provided default params
            if self.default_request_params:
                params_dict = router_params.model_dump()
                params_dict.update(self.default_request_params.model_dump(exclude_unset=True))
                router_params = RequestParams(**params_dict)
            router_params.use_history = False

            self.initialized = True

    async def generate_x(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        return await self._llm.generate_x(multipart_messages, request_params)

    async def route(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[str | Agent | Callable]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(request, top_k)

    async def route_to_server(self, request: str, top_k: int = 1) -> List[LLMRouterResult[str]]:
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
        prompt = routing_instruction.format(context=context, request=request, top_k=top_k)

        response = await self.llm.generate_structured(
            message=prompt,
            response_model=StructuredResponse,
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
        """Generate a formatted context list of categories."""

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

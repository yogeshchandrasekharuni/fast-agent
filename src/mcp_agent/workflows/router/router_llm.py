from typing import Callable, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.workflows.router.router_base import ResultT, Router, RouterResult
from mcp_agent.logging.logger import get_logger
from mcp_agent.event_progress import ProgressAction

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


class LLMRouter(Router):
    """
    A router that uses an LLM to route an input to a specific category.
    """

    def __init__(
        self,
        llm_factory: Callable[..., AugmentedLLM],
        name: str = "LLM Router",
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
        **kwargs,
    ):
        # Extract verb from kwargs to avoid passing it up the inheritance chain
        self._llm_verb = kwargs.pop("verb", None)

        super().__init__(
            server_names=server_names,
            agents=agents,
            functions=functions,
            routing_instruction=routing_instruction,
            context=context,
            **kwargs,
        )

        self.name = name
        self.llm_factory = llm_factory
        self.default_request_params = default_request_params or RequestParams()
        self.llm = None  # Will be initialized in create()

    @classmethod
    async def create(
        cls,
        llm_factory: Callable[..., AugmentedLLM],
        name: str = "LLM Router",
        server_names: List[str] | None = None,
        agents: List[Agent] | None = None,
        functions: List[Callable] | None = None,
        routing_instruction: str | None = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
    ) -> "LLMRouter":
        """
        Factory method to create and initialize a router.
        Use this instead of constructor since we need async initialization.
        """
        instance = cls(
            llm_factory=llm_factory,
            name=name,
            server_names=server_names,
            agents=agents,
            functions=functions,
            routing_instruction=DEFAULT_ROUTING_INSTRUCTION,
            context=context,
            default_request_params=default_request_params,
        )
        await instance.initialize()
        return instance

    async def initialize(self):
        """Initialize the router and create the LLM instance."""
        if not self.initialized:
            await super().initialize()
            router_params = RequestParams(
                systemPrompt=ROUTING_SYSTEM_INSTRUCTION,
                use_history=False,  # Router should be stateless :)
            )

            # Merge with any provided default params
            if self.default_request_params:
                params_dict = router_params.model_dump()
                params_dict.update(
                    self.default_request_params.model_dump(exclude_unset=True)
                )
                router_params = RequestParams(**params_dict)
            # Set up router-specific request params with routing instruction
            router_params.use_history = False
            # Use the stored verb if available, otherwise default to ROUTING
            verb_param = (
                self._llm_verb
                if hasattr(self, "_llm_verb") and self._llm_verb
                else ProgressAction.ROUTING
            )

            self.llm = self.llm_factory(
                agent=None,  # Router doesn't need an agent context
                name=self.name,  # Use the name provided during initialization
                default_request_params=router_params,
                verb=verb_param,  # Use stored verb parameter or default to ROUTING
            )
            self.initialized = True

    async def route(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[str | Agent | Callable]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(request, top_k)

    async def route_to_server(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[str]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=True,
            include_agents=False,
            include_functions=False,
        )

    async def route_to_agent(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[Agent]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=False,
            include_agents=True,
            include_functions=False,
        )

    async def route_to_function(
        self, request: str, top_k: int = 1
    ) -> List[LLMRouterResult[Callable]]:
        if not self.initialized:
            await self.initialize()

        return await self._route_with_llm(
            request,
            top_k,
            include_servers=False,
            include_agents=False,
            include_functions=True,
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
        prompt = routing_instruction.format(
            context=context, request=request, top_k=top_k
        )

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

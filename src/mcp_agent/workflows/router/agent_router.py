from typing import List, Literal, Optional, Dict, Any, Type
from pydantic import BaseModel, Field

from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

logger = get_logger(__name__)

ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate agent
based on their capabilities and specialties.

Your task is to analyze each request and determine which agent would be most effective at handling it,
based on the agent's description, servers, and capabilities.
"""


class AgentInfo(BaseModel):
    """
    Lightweight representation of an agent used for routing decisions.
    Avoids using the complex Agent type directly in models.
    """

    name: str
    """The name of the agent"""

    description: str = ""
    """Description of the agent's capabilities and purpose"""


class AgentCategory(BaseModel):
    """
    Represents an agent available for routing with its metadata.
    Uses a simplified agent representation instead of the full Agent type.
    """

    name: str
    """The name of the agent"""

    description: str
    """Description of the agent's capabilities and purpose"""

    agent_info: AgentInfo
    """Simplified agent information"""

    server_names: List[str] = Field(default_factory=list)
    """List of server names this agent has access to"""

    capabilities: List[str] = Field(default_factory=list)
    """Key capabilities of this agent (e.g., 'code-generation', 'data-analysis')"""

    tags: List[str] = Field(default_factory=list)
    """Optional tags for additional categorization"""

    # Reference to the actual agent kept separately
    _agent: Optional[Agent] = None

    class Config:
        arbitrary_types_allowed = True


class AgentRoutingResult(BaseModel):
    """
    Result of a routing decision, combining confidence information and agent info.
    Uses AgentInfo instead of the full Agent type.
    """

    agent_info: AgentInfo
    """Information about the selected agent"""

    confidence: Literal["high", "medium", "low"]
    """Confidence level of the routing decision"""

    reasoning: Optional[str] = None
    """Optional explanation for why this agent was selected"""

    score: Optional[float] = None
    """Optional numeric score (0-1) if a probabilistic router is used"""

    # Store agent name separately for easy lookup
    agent_name: str

    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Models for structured output parsing
class AgentSelection(BaseModel):
    """Represents a selected agent in the LLM's routing response"""

    agent_name: str
    confidence: Literal["high", "medium", "low"]
    reasoning: Optional[str] = None
    score: Optional[float] = None

    def to_json(self) -> str:
        """Convert to JSON string for testing purposes"""
        return self.model_dump_json(indent=2)


class RoutingResponse(BaseModel):
    """Container for agent selections from the LLM"""

    selections: List[AgentSelection] = Field(default_factory=list)

    def to_json(self) -> str:
        """Convert to JSON string for testing purposes"""
        return self.model_dump_json(indent=2)


class AgentRouter(Agent):
    """
    A router that uses an LLM to route requests to the most appropriate agent.
    Simplified to focus only on agent routing.
    """

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        routing_instruction: Optional[str] = None,
        context=None,
        **kwargs,
    ) -> None:
        super().__init__(config=config, context=context, **kwargs)
        self.agents = agents
        self.routing_instruction = routing_instruction
        self.agent_categories: List[AgentCategory] = []
        self._agent_map: Dict[str, Agent] = {}  # Maps agent names to agent instances
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the router and build agent categories"""
        if not self.initialized:
            await super().initialize()

            # Build agent categories once and populate agent map
            self.agent_categories = []
            self._agent_map = {}

            for agent in self.agents:
                category = self._build_agent_category(agent)
                self.agent_categories.append(category)
                self._agent_map[agent.name] = agent

            # Set up routing LLM with system instruction
            router_params = RequestParams(
                systemPrompt=ROUTING_SYSTEM_INSTRUCTION,
                use_history=False,
            )

            # Merge with any provided default params
            if hasattr(self, "default_request_params") and self.default_request_params:
                params_dict = router_params.model_dump()
                params_dict.update(self.default_request_params.model_dump(exclude_unset=True))
                router_params = RequestParams(**params_dict)

            self.initialized = True

    def _build_agent_category(self, agent: Agent) -> AgentCategory:
        """Convert an agent to a category with appropriate metadata"""
        agent_description = agent.instruction if isinstance(agent.instruction, str) else ""

        # Create simplified agent info
        agent_info = AgentInfo(name=agent.name, description=agent_description)

        # Extract capabilities from description (placeholder for actual implementation)
        capabilities = []

        return AgentCategory(
            name=agent.name,
            description=agent_description,
            agent_info=agent_info,
            server_names=agent.server_names,
            capabilities=capabilities,
            _agent=agent,
        )

    async def route(self, request: str, top_k: int = 1) -> List[AgentRoutingResult]:
        """Route the request to the most appropriate agent(s)"""
        if not self.initialized:
            await self.initialize()

        # Get LLM to analyze the request and match to appropriate agents
        context = self._generate_agent_context()
        prompt = self._format_routing_prompt(context, request, top_k)

        # Use structured output to get LLM's routing decisions
        response = await self._llm.generate_structured(
            message=prompt,
            response_model=RoutingResponse,
        )

        # Convert response to AgentRoutingResult objects
        results = []
        if response and response.selections:
            for selection in response.selections:
                agent_name = selection.agent_name

                # Find the agent category by name
                category = next((c for c in self.agent_categories if c.name == agent_name), None)

                if category:
                    results.append(
                        AgentRoutingResult(
                            agent_info=category.agent_info,
                            agent_name=agent_name,
                            confidence=selection.confidence,
                            reasoning=selection.reasoning,
                            score=selection.score,
                            metadata={"server_names": category.server_names},
                        )
                    )

        return results[:top_k]

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get an agent instance by name"""
        return self._agent_map.get(agent_name)

    def _generate_agent_context(self) -> str:
        """Generate a context string describing all available agents"""
        context_parts = []

        for idx, category in enumerate(self.agent_categories, 1):
            servers = ", ".join(category.server_names) if category.server_names else "none"
            capabilities = ", ".join(category.capabilities) if category.capabilities else "general"

            part = f"""<agent id="{idx}" name="{category.name}">
<description>{category.description}</description>
<servers>{servers}</servers>
<capabilities>{capabilities}</capabilities>
</agent>"""
            context_parts.append(part)

        return "\n\n".join(context_parts)

    def _format_routing_prompt(self, context: str, request: str, top_k: int) -> str:
        """Format the prompt for the LLM to make routing decisions"""
        prompt = f"""<fastagent:data>
<fastagent:agents>
{context}
</fastagent:agents>

<fastagent:request>
{request}
</fastagent:request>
</fastagent:data>

Analyze the request and determine the {top_k} most appropriate agent(s).
Consider each agent's:
- Description and purpose
- Available servers
- Specialized capabilities

<fastagent:instruction>
Respond in JSON format with your routing decision:
{{
  "selections": [
    {{
      "agent_name": "name of the selected agent",
      "confidence": "high|medium|low",
      "reasoning": "brief explanation of why this agent is appropriate",
      "score": optional_numeric_score_between_0_and_1
    }}
  ]
}}
</fastagent:instruction>
"""
        return prompt

    # Helper method for tests to easily create a structured model
    @staticmethod
    def create_test_selection(
        agent_name: str,
        confidence: Literal["high", "medium", "low"] = "high",
        reasoning: Optional[str] = None,
    ) -> AgentSelection:
        """Create a test AgentSelection model that can be converted to JSON"""
        return AgentSelection(
            agent_name=agent_name,
            confidence=confidence,
            reasoning=reasoning or f"This agent is well-suited for the task",
        )

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT | None:
        raise NotImplementedError

    async def send(self, message: str | PromptMessageMultipart) -> str:
        return await self._llm.generate_x([Prompt.user(message)]).first_text()

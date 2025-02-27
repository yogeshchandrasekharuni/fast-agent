from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import asyncio

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)

if TYPE_CHECKING:
    from mcp_agent.context import Context


class ParallelLLM(AugmentedLLM[MessageParamT, MessageT]):
    """
    LLMs can sometimes work simultaneously on a task (fan-out)
    and have their outputs aggregated programmatically (fan-in).
    This workflow performs both the fan-out and fan-in operations using LLMs.
    From the user's perspective, an input is specified and the output is returned.
    """

    def __init__(
        self,
        fan_in_agent: Agent | AugmentedLLM,
        fan_out_agents: List[Agent | AugmentedLLM],
        llm_factory: Callable[[Agent], AugmentedLLM] = None,
        context: Optional["Context"] = None,
        include_request: bool = True,
        **kwargs,
    ):
        super().__init__(context=context, **kwargs)
        self.fan_in_agent = fan_in_agent
        self.fan_out_agents = fan_out_agents
        self.llm_factory = llm_factory
        self.include_request = include_request
        self.history = None  # History tracking is complex in this workflow

    async def ensure_llm(self, agent: Union[Agent, AugmentedLLM]) -> AugmentedLLM:
        """Ensure an agent has an LLM attached, using existing or creating new."""
        if isinstance(agent, AugmentedLLM):
            return agent

        if not hasattr(agent, "_llm") or agent._llm is None:
            return await agent.attach_llm(self.llm_factory)

        return agent._llm

    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT] | Any:
        """Generate responses using parallel fan-out and fan-in."""
        # Ensure all agents have LLMs
        fan_out_llms = []
        for agent in self.fan_out_agents:
            llm = await self.ensure_llm(agent)
            fan_out_llms.append(llm)

        fan_in_llm = await self.ensure_llm(self.fan_in_agent)

        # Run fan-out operations in parallel
        responses = await asyncio.gather(
            *[llm.generate(message, request_params) for llm in fan_out_llms]
        )

        # Get message string for inclusion in formatted output
        message_str = (
            str(message) if isinstance(message, (str, MessageParamT)) else None
        )

        # Run fan-in to aggregate results
        result = await fan_in_llm.generate(
            self._format_responses(responses, message_str),
            request_params=request_params,
        )

        return result

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Generate string responses using parallel fan-out and fan-in."""
        # Ensure all agents have LLMs
        fan_out_llms = []
        for agent in self.fan_out_agents:
            llm = await self.ensure_llm(agent)
            fan_out_llms.append(llm)

        fan_in_llm = await self.ensure_llm(self.fan_in_agent)

        # Run fan-out operations in parallel
        responses = await asyncio.gather(
            *[llm.generate_str(message, request_params) for llm in fan_out_llms]
        )

        # Get message string for inclusion in formatted output
        message_str = (
            str(message) if isinstance(message, (str, MessageParamT)) else None
        )

        # Run fan-in to aggregate results
        result = await fan_in_llm.generate_str(
            self._format_responses(responses, message_str),
            request_params=request_params,
        )

        return result

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Generate structured responses using parallel fan-out and fan-in."""
        # Ensure all agents have LLMs
        fan_out_llms = []
        for agent in self.fan_out_agents:
            llm = await self.ensure_llm(agent)
            fan_out_llms.append(llm)

        fan_in_llm = await self.ensure_llm(self.fan_in_agent)

        # Run fan-out operations in parallel
        responses = await asyncio.gather(
            *[
                llm.generate_structured(message, response_model, request_params)
                for llm in fan_out_llms
            ]
        )

        # Get message string for inclusion in formatted output
        message_str = (
            str(message) if isinstance(message, (str, MessageParamT)) else None
        )

        # Run fan-in to aggregate results
        result = await fan_in_llm.generate_structured(
            self._format_responses(responses, message_str),
            response_model=response_model,
            request_params=request_params,
        )

        return result

    def _format_responses(self, responses: List[Any], message: str = None) -> str:
        """Format a list of responses for the fan-in agent."""
        formatted = []

        # Include the original message if specified
        if self.include_request and message:
            formatted.append("The following request was sent to the agents:")
            formatted.append(f"<fastagent:request>\n{message}\n</fastagent:request>")

        for i, response in enumerate(responses):
            agent_name = self.fan_out_agents[i].name
            formatted.append(
                f'<fastagent:response agent="{agent_name}">\n{response}\n</fastagent:response>'
            )
        return "\n\n".join(formatted)

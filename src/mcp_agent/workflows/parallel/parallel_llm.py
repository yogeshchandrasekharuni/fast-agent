import asyncio
from typing import Any, Dict, List

from mcp_agent.agents.agent import Agent
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.interfaces import AgentProtocol, ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.augmented_llm import (
    RequestParams,
)


class ParallelLLM(AgentProtocol):
    """
    LLMs can sometimes work simultaneously on a task (fan-out)
    and have their outputs aggregated programmatically (fan-in).
    This workflow performs both the fan-out and fan-in operations using LLMs.
    From the user's perspective, an input is specified and the output is returned.
    """

    def __init__(
        self,
        name: str,
        instruction: str,
        fan_in_agent: Agent,
        fan_out_agents: List[Agent],
        include_request: bool = True,
    ) -> None:
        self.name = name
        self.fan_in_agent = fan_in_agent
        self.fan_out_agents = fan_out_agents
        self.include_request = include_request

    async def generate_x(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        responses: list[PromptMessageMultipart] = []

        responses: list[PromptMessageMultipart] = await asyncio.gather(
            *[agent.generate_x(multipart_messages, request_params) for agent in self.fan_out_agents]
        )

        # TODO - we'll just use strings for now, and use multiparts as we add agent-transfer functionality

        received_message: str | None = (
            multipart_messages[-1].all_text() if multipart_messages else None
        )

        string_responses = []
        for response in responses:
            string_responses.append(response.all_text())

        return await self.fan_in_agent.generate_x(
            [Prompt.user(self._format_responses(string_responses, received_message))],
            request_params,
        )

    def _format_responses(self, responses: List[Any], message: str | None = None) -> str:
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

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: type[ModelT],
        request_params: RequestParams | None,
    ) -> ModelT | None:
        raise NotImplementedError

    async def send(self, message: str | PromptMessageMultipart) -> str:
        raise NotImplementedError

    async def prompt(self, default_prompt: str = "") -> str:
        raise NotImplementedError

    async def apply_prompt(self, prompt_name: str, arguments: Dict[str, str] | None = None) -> str:
        raise NotImplementedError

    async def with_resource(
        self, prompt_content: str | PromptMessageMultipart, server_name: str, resource_name: str
    ) -> str:
        raise NotImplementedError

    async def initialize(self) -> None:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError

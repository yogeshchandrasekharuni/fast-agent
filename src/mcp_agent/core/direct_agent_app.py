"""
Direct AgentApp implementation for interacting with agents without proxies.
"""

from typing import Dict, Optional, Union

from mcp_agent.agents.agent import Agent
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class DirectAgentApp:
    """
    Container for active agents that provides a simple API for interacting with them.
    This implementation works directly with Agent instances without proxies.

    The DirectAgentApp provides both attribute-style access (app.agent_name)
    and dictionary-style access (app["agent_name"]) to agents.

    It also implements the AgentProtocol interface, automatically forwarding
    calls to the default agent (the first agent in the container).
    """

    def __init__(self, agents: Dict[str, Agent]) -> None:
        """
        Initialize the DirectAgentApp.

        Args:
            agents: Dictionary of agent instances keyed by name
        """
        self._agents = agents

    def __getitem__(self, key: str) -> Agent:
        """Allow access to agents using dictionary syntax."""
        if key not in self._agents:
            raise KeyError(f"Agent '{key}' not found")
        return self._agents[key]

    def __getattr__(self, name: str) -> Agent:
        """Allow access to agents using attribute syntax."""
        if name in self._agents:
            return self._agents[name]
        raise AttributeError(f"Agent '{name}' not found")

    async def __call__(
        self,
        message: Union[str, PromptMessageMultipart] | None = None,
        agent_name: str | None = None,
    ) -> str:
        """
        Make the object callable to send messages.
        This mirrors the FastAgent implementation that allowed agent("message").

        Args:
            message: The message to send
            agent_name: Optional name of the agent to send to (defaults to first agent or all)

        Returns:
            The agent's response as a string
        """
        if message:
            return await self._agent(agent_name).send(message)

        return await self._agent(agent_name).prompt()

    async def send(self, message: str, agent_name: Optional[str] = None) -> str:
        """
        Send a message to the specified agent (or to all agents).

        Args:
            message: The message to send
            agent_name: Optional name of the agent to send to

        Returns:
            The agent's response as a string
        """
        return await self._agent(agent_name).send(message)

    def _agent(self, agent_name: str | None) -> Agent:
        if agent_name:
            if agent_name not in self._agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            return self._agents[agent_name]

        return next(iter(self.agents.values()))

    async def apply_prompt(
        self,
        prompt_name: str,
        arguments: Dict[str, str] | None = None,
        agent_name: str | None = None,
    ) -> str:
        """
        Apply a prompt template to an agent (default agent if not specified).

        Args:
            prompt_name: Name of the prompt template to apply
            agent_name: Name of the agent to send to
            arguments: Optional arguments for the prompt template

        Returns:
            The agent's response as a string
        """
        return await self._agent(agent_name).apply_prompt(prompt_name, arguments)

    async def with_resource(self, user_prompt: str, server_name: str, resource_name: str) -> str:
        """
        Apply a prompt template to an agent (default agent if not specified).

        Args:
            prompt_name: Name of the prompt template to apply
            agent_name: Name of the agent to send to
            arguments: Optional arguments for the prompt template

        Returns:
            The agent's response as a string
        """
        return await self._agent(None).with_resource(
            prompt_content=user_prompt, server_name=server_name, resource_name=resource_name
        )

    @property
    def agents(self) -> Dict[str, Agent]:
        """Access all agents."""
        return self._agents

    async def close(self) -> None:
        """Shutdown all agents."""
        for agent in self._agents.values():
            try:
                await agent.shutdown()
            except Exception:
                pass

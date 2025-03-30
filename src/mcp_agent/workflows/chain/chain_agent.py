"""
Chain workflow implementation using the clean BaseAgent adapter pattern.

This provides an implementation that delegates operations to a sequence of
other agents, chaining their outputs together.
"""

from typing import Any, List, Optional, Type, Union

from mcp.types import TextContent

from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.core.base_agent import BaseAgent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class ChainAgent(BaseAgent):
    """
    A chain agent that processes requests through a series of specialized agents in sequence.
    Passes the output of each agent to the next agent in the chain.
    """

    def __init__(
        self,
        config: Union[AgentConfig, str],
        agents: List[Agent],
        cumulative: bool = False,
        context: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a ChainAgent.

        Args:
            config: Agent configuration or name
            agents: List of agents to chain together in sequence
            cumulative: Whether each agent sees all previous responses
            context: Optional context object
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config, context=context, **kwargs)
        self.agents = agents
        self.cumulative = cumulative

    async def generate_x(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Chain the request through multiple agents in sequence.

        Args:
            multipart_messages: Initial messages to send to the first agent
            request_params: Optional request parameters

        Returns:
            The response from the final agent in the chain
        """
        if not self.agents:
            # If no agents in the chain, return an empty response
            return PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="No agents available in the chain.")],
            )

        # Get the original user message (last message in the list)
        user_message = multipart_messages[-1] if multipart_messages else None

        # If no user message, return an error
        if not user_message:
            return PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="No input message provided.")],
            )

        # Initialize messages with the input
        current_messages = multipart_messages

        # Track all responses in the chain for cumulative mode
        all_responses: List[PromptMessageMultipart] = []

        # Process through each agent in sequence
        for i, agent in enumerate(self.agents):
            # In cumulative mode, include the original message and all previous responses
            if self.cumulative and all_responses:
                # Create a list with original messages, then all previous responses
                chain_messages = multipart_messages.copy()
                chain_messages.extend(all_responses)
                current_response = await agent.generate_x(chain_messages, request_params)
            else:
                # In sequential mode, just pass the current messages to the next agent
                current_response = await agent.generate_x(current_messages, request_params)

            # Store the response
            all_responses.append(current_response)

            # Prepare for the next agent
            # We create a new context with the user's message followed by the current response
            if i < len(self.agents) - 1:
                current_messages = [user_message, current_response]

        # Return the final response
        return all_responses[-1]

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Optional[ModelT]:
        """
        Chain the request through multiple agents and parse the final response.

        Args:
            prompt: List of messages to send through the chain
            model: Pydantic model to parse the final response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the final agent, or None if parsing fails
        """
        # Generate response through the chain
        response = await self.generate_x(prompt, request_params)

        # Let the last agent in the chain try to parse the response
        if self.agents:
            last_agent = self.agents[-1]
            try:
                return await last_agent.structured([response], model, request_params)
            except Exception as e:
                self.logger.warning(f"Failed to parse response from chain: {str(e)}")
                return None
        return None

    async def initialize(self) -> None:
        """
        Initialize the chain agent and all agents in the chain.
        """
        await super().initialize()

        # Initialize all agents in the chain if not already initialized
        for agent in self.agents:
            if not getattr(agent, "initialized", False):
                await agent.initialize()

    async def shutdown(self) -> None:
        """
        Shutdown the chain agent and all agents in the chain.
        """
        await super().shutdown()

        # Shutdown all agents in the chain
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down agent in chain: {str(e)}")

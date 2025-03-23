"""
Proxy classes for agent interactions.
These proxies provide a consistent interface for interacting with different types of agents.

FOR COMPATIBILITY WITH LEGACY MCP-AGENT CODE

"""

from typing import List, Optional, Dict, Union, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import EmbeddedResource

# Handle circular imports
if TYPE_CHECKING:
    from mcp_agent.core.types import WorkflowType, ProxyDict
else:
    # Define minimal versions for runtime
    from typing import Any

    # Use Any for runtime to avoid circular imports
    WorkflowType = Any
    ProxyDict = Dict[str, "BaseAgentProxy"]


class BaseAgentProxy:
    """Base class for all proxy types"""

    def __init__(self, app: MCPApp, name: str):
        self._app = app
        self._name = name

    async def __call__(self, message: Optional[str] = None) -> str:
        """Allow: agent.researcher('message') or just agent.researcher()"""
        if message is None:
            # When called with no arguments, use prompt() to open the interactive interface
            return await self.prompt()
        return await self.send(message)

    async def send(
        self, message: Optional[Union[str, PromptMessageMultipart]] = None
    ) -> str:
        """
        Allow: agent.researcher.send('message') or agent.researcher.send(Prompt.user('message'))

        Args:
            message: Either a string message or a PromptMessageMultipart object

        Returns:
            The agent's response as a string
        """
        if message is None:
            # For consistency with agent(), use prompt() to open the interactive interface
            return await self.prompt()

        # If a PromptMessageMultipart is passed, use send_prompt
        if isinstance(message, PromptMessageMultipart):
            return await self.send_prompt(message)

        # For string messages, use generate_str (traditional behavior)
        return await self.generate_str(message)

    async def prompt(self, default_prompt: str = "") -> str:
        """Allow: agent.researcher.prompt()"""
        from mcp_agent.core.agent_app import AgentApp

        # First check if _app is directly an AgentApp
        if isinstance(self._app, AgentApp):
            return await self._app.prompt(self._name, default_prompt)

        # If not, check if it's an MCPApp with an _agent_app attribute
        if hasattr(self._app, "_agent_app"):
            agent_app = self._app._agent_app
            if agent_app:
                return await agent_app.prompt(self._name, default_prompt)

        # If we can't find an AgentApp, return an error message
        return "ERROR: Cannot prompt() - AgentApp not found"

    async def generate_str(self, message: str) -> str:
        """Generate response for a message - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_str")

    async def send_prompt(self, prompt: PromptMessageMultipart) -> str:
        """Send a message to the agent and return the response"""
        raise NotImplementedError("Subclasses must implement send(prompt)")

    async def apply_prompt(
        self, prompt_name: str = None, arguments: dict[str, str] = None
    ) -> str:
        """
        Apply a Prompt from an MCP Server - implemented by subclasses.
        This is the preferred method for applying prompts.
        Always returns an Assistant message.

        Args:
            prompt_name: Name of the prompt to apply
            arguments: Optional dictionary of string arguments for prompt templating
        """
        raise NotImplementedError("Subclasses must implement apply_prompt")


class LLMAgentProxy(BaseAgentProxy):
    """Proxy for regular agents that use _llm.generate_str()"""

    def __init__(self, app: MCPApp, name: str, agent: Agent):
        super().__init__(app, name)
        self._agent = agent

    async def generate_str(self, message: str, **kwargs) -> str:
        """Forward message and all kwargs to the agent's LLM"""
        return await self._agent._llm.generate_str(message, **kwargs)

    async def send_prompt(self, prompt: PromptMessageMultipart) -> str:
        """Send a message to the agent and return the response"""
        return await self._agent._llm.generate_prompt(prompt, None)

    async def apply_prompt(
        self, prompt_name: str = None, arguments: dict[str, str] = None
    ) -> str:
        """
        Apply a prompt from an MCP server.
        This is the preferred method for applying prompts.

        Args:
            prompt_name: Name of the prompt to apply
            arguments: Optional dictionary of string arguments for prompt templating

        Returns:
            The assistant's response
        """
        return await self._agent.apply_prompt(prompt_name, arguments)

    # Add the new methods
    async def get_embedded_resources(
        self, server_name: str, resource_name: str
    ) -> List[EmbeddedResource]:
        """
        Get a resource from an MCP server and return it as a list of embedded resources ready for use in prompts.

        Args:
            server_name: Name of the MCP server to retrieve the resource from
            resource_name: Name or URI of the resource to retrieve

        Returns:
            List of EmbeddedResource objects ready to use in a PromptMessageMultipart
        """
        return await self._agent.get_embedded_resources(server_name, resource_name)

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessageMultipart],
        server_name: str,
        resource_name: str,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Either a string message or an existing PromptMessageMultipart
            server_name: Name of the MCP server to retrieve the resource from
            resource_name: Name or URI of the resource to retrieve

        Returns:
            The agent's response as a string
        """
        return await self._agent.with_resource(
            prompt_content, server_name, resource_name
        )


class WorkflowProxy(BaseAgentProxy):
    """Proxy for workflow types that implement generate_str() directly"""

    def __init__(self, app: MCPApp, name: str, workflow: WorkflowType):
        super().__init__(app, name)
        self._workflow = workflow

    async def generate_str(self, message: str, **kwargs) -> str:
        """Forward message and all kwargs to the underlying workflow"""
        return await self._workflow.generate_str(message, **kwargs)


class RouterProxy(BaseAgentProxy):
    """Proxy for LLM Routers"""

    def __init__(self, app: MCPApp, name: str, workflow: WorkflowType):
        super().__init__(app, name)
        self._workflow = workflow

    async def generate_str(self, message: str, **kwargs) -> str:
        """
        Route the message and forward kwargs to the resulting agent if applicable.
        Note: For now, route() itself doesn't accept kwargs.
        """
        results = await self._workflow.route(message)
        if not results:
            return "No appropriate route found for the request."

        # Get the top result
        top_result = results[0]
        if isinstance(top_result.result, Agent):
            # Agent route - delegate to the agent, passing along kwargs
            agent = top_result.result
            return await agent._llm.generate_str(message, **kwargs)
        elif isinstance(top_result.result, str):
            # Server route - use the router directly
            return "Tool call requested by router - not yet supported"

        return f"Routed to: {top_result.result} ({top_result.confidence}): {top_result.reasoning}"


class ChainProxy(BaseAgentProxy):
    """Proxy for chained agent operations"""

    def __init__(
        self, app: MCPApp, name: str, sequence: List[str], agent_proxies: ProxyDict
    ):
        super().__init__(app, name)
        self._sequence = sequence
        self._agent_proxies = agent_proxies
        self._continue_with_final = True  # Default behavior
        self._cumulative = False  # Default to sequential chaining

    async def generate_str(self, message: str, **kwargs) -> str:
        """Chain message through a sequence of agents.

        For the first agent in the chain, pass all kwargs to maintain transparency.

        Two modes of operation:
        1. Sequential (default): Each agent receives only the output of the previous agent
        2. Cumulative: Each agent receives all previous agent responses concatenated
        """
        if not self._sequence:
            return message

        # Process the first agent (same for both modes)
        first_agent = self._sequence[0]
        first_proxy = self._agent_proxies[first_agent]
        first_response = await first_proxy.generate_str(message, **kwargs)

        if len(self._sequence) == 1:
            return first_response

        if self._cumulative:
            # Cumulative mode: each agent gets all previous responses
            cumulative_response = f'<fastagent:response agent="{first_agent}">\n{first_response}\n</fastagent:response>'

            # Process subsequent agents with cumulative results
            for agent_name in self._sequence[1:]:
                proxy = self._agent_proxies[agent_name]
                # Pass all previous responses to next agent
                agent_response = await proxy.generate_str(cumulative_response)
                # Add this agent's response to the cumulative result
                cumulative_response += f'\n\n<fastagent:response agent="{agent_name}">\n{agent_response}\n</fastagent:response>'

            return cumulative_response
        else:
            # Sequential chaining (original behavior)
            current_message = first_response

            # For subsequent agents, just pass the message from previous agent
            for agent_name in self._sequence[1:]:
                proxy = self._agent_proxies[agent_name]
                current_message = await proxy.generate_str(current_message)

            return current_message

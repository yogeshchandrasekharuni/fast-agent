"""
Proxy classes for agent interactions.
These proxies provide a consistent interface for interacting with different types of agents.
"""

from typing import List, Optional, Dict, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp

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
        """Allow: agent.researcher('message')"""
        return await self.send(message)

    async def send(self, message: Optional[str] = None) -> str:
        """Allow: agent.researcher.send('message')"""
        if message is None:
            return await self.prompt()
        return await self.generate_str(message)

    async def prompt(self, default_prompt: str = "") -> str:
        """Allow: agent.researcher.prompt()"""
        return await self._app.prompt(self._name, default_prompt)

    async def generate_str(self, message: str) -> str:
        """Generate response for a message - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_str")


class AgentProxy(BaseAgentProxy):
    """Legacy proxy for individual agent operations"""

    async def generate_str(self, message: str, **kwargs) -> str:
        """Forward only the message to app.send, ignoring kwargs for legacy compatibility"""
        return await self._app.send(self._name, message)


class LLMAgentProxy(BaseAgentProxy):
    """Proxy for regular agents that use _llm.generate_str()"""

    def __init__(self, app: MCPApp, name: str, agent: Agent):
        super().__init__(app, name)
        self._agent = agent

    async def generate_str(self, message: str, **kwargs) -> str:
        """Forward message and all kwargs to the agent's LLM"""
        return await self._agent._llm.generate_str(message, **kwargs)


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

    async def generate_str(self, message: str) -> str:
        """Chain message through a sequence of agents with optional cumulative behavior"""
        if not self._cumulative:
            # Original sequential behavior
            current_message = message
            for agent_name in self._sequence:
                proxy = self._agent_proxies[agent_name]
                current_message = await proxy.generate_str(current_message)
            return current_message
        else:
            # Cumulative behavior
            original_message = message
            agent_responses = {}

            for agent_name in self._sequence:
                proxy = self._agent_proxies[agent_name]

                if not agent_responses:  # First agent
                    response = await proxy.generate_str(original_message)
                else:
                    # Construct context with previous responses
                    context_message = "The following request was sent to the agents:\n"
                    context_message += f"<fastagent:request>\n{original_message}\n</fastagent:request>\n\n"

                    context_message += "Previous agent responses:\n"

                    for prev_name in self._sequence:
                        if prev_name in agent_responses:
                            prev_response = agent_responses[prev_name]
                            context_message += f'<fastagent:response agent="{prev_name}">\n{prev_response}\n</fastagent:response>\n\n'

                    context_message += f"Your task is to build upon this work to address: {original_message}"

                    response = await proxy.generate_str(context_message)

                agent_responses[agent_name] = response

            # Format final output with ALL responses in XML format
            final_output = "The following request was sent to the agents:\n"
            final_output += (
                f"<fastagent:request>\n{original_message}\n</fastagent:request>\n\n"
            )

            for agent_name in self._sequence:
                response = agent_responses[agent_name]
                final_output += f'<fastagent:response agent="{agent_name}">\n{response}\n</fastagent:response>\n\n'

            # Return the XML-structured combination of all responses
            return final_output.strip()

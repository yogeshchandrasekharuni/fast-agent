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
    from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
    from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
    from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import EvaluatorOptimizerLLM
    from mcp_agent.workflows.router.router_llm import LLMRouter
    from typing import Union
    WorkflowType = Union[Orchestrator, ParallelLLM, EvaluatorOptimizerLLM, LLMRouter]
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

    async def generate_str(self, message: str) -> str:
        return await self._app.send(self._name, message)


class LLMAgentProxy(BaseAgentProxy):
    """Proxy for regular agents that use _llm.generate_str()"""

    def __init__(self, app: MCPApp, name: str, agent: Agent):
        super().__init__(app, name)
        self._agent = agent

    async def generate_str(self, message: str) -> str:
        return await self._agent._llm.generate_str(message)


class WorkflowProxy(BaseAgentProxy):
    """Proxy for workflow types that implement generate_str() directly"""

    def __init__(self, app: MCPApp, name: str, workflow: WorkflowType):
        super().__init__(app, name)
        self._workflow = workflow

    async def generate_str(self, message: str) -> str:
        return await self._workflow.generate_str(message)


class RouterProxy(BaseAgentProxy):
    """Proxy for LLM Routers"""

    def __init__(self, app: MCPApp, name: str, workflow: WorkflowType):
        super().__init__(app, name)
        self._workflow = workflow

    async def generate_str(self, message: str) -> str:
        results = await self._workflow.route(message)
        if not results:
            return "No appropriate route found for the request."

        # Get the top result
        top_result = results[0]
        if isinstance(top_result.result, Agent):
            # Agent route - delegate to the agent
            agent = top_result.result

            return await agent._llm.generate_str(message)
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

    async def generate_str(self, message: str) -> str:
        """Chain message through a sequence of agents"""
        current_message = message

        for agent_name in self._sequence:
            proxy = self._agent_proxies[agent_name]
            current_message = await proxy.generate_str(current_message)

        return current_message



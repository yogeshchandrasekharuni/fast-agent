"""
Main application wrapper for interacting with agents.
"""

from typing import Optional, Dict, TYPE_CHECKING

from mcp_agent.app import MCPApp
from mcp_agent.progress_display import progress_display
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
)

# Import proxies directly - they handle their own circular imports
from mcp_agent.core.proxies import (
    BaseAgentProxy,
    AgentProxy,
    LLMAgentProxy,
    RouterProxy,
    ChainProxy,
    WorkflowProxy,
)

# Handle possible circular imports with types
if TYPE_CHECKING:
    from mcp_agent.core.types import ProxyDict
else:
    ProxyDict = Dict[str, BaseAgentProxy]


class AgentApp:
    """Main application wrapper"""

    def __init__(self, app: MCPApp, agents: ProxyDict):
        self._app = app
        self._agents = agents
        # Optional: set default agent for direct calls
        self._default = next(iter(agents)) if agents else None

    async def send(self, agent_name: str, message: Optional[str]) -> str:
        """Core message sending"""
        if agent_name not in self._agents:
            raise ValueError(f"No agent named '{agent_name}'")

        if not message or "" == message:
            return await self.prompt(agent_name)

        proxy = self._agents[agent_name]
        return await proxy.generate_str(message)

    async def prompt(self, agent_name: Optional[str] = None, default: str = "") -> str:
        """
        Interactive prompt for sending messages with advanced features.

        Args:
            agent_name: Optional target agent name (uses default if not specified)
            default: Default message to use when user presses enter
        """
        from mcp_agent.core.enhanced_prompt import (
            get_enhanced_input,
            handle_special_commands,
        )

        agent = agent_name or self._default

        if agent not in self._agents:
            raise ValueError(f"No agent named '{agent}'")

        # Pass all available agent names for auto-completion
        available_agents = list(self._agents.keys())

        # Create agent_types dictionary mapping agent names to their types
        agent_types = {}
        for name, proxy in self._agents.items():
            # Determine agent type based on the proxy type
            if isinstance(proxy, LLMAgentProxy):
                # Convert AgentType.BASIC.value ("agent") to "Agent"
                agent_types[name] = "Agent"
            elif isinstance(proxy, RouterProxy):
                agent_types[name] = "Router"
            elif isinstance(proxy, ChainProxy):
                agent_types[name] = "Chain"
            elif isinstance(proxy, WorkflowProxy):
                # For workflow proxies, check the workflow type
                workflow = proxy._workflow
                if isinstance(workflow, Orchestrator):
                    agent_types[name] = "Orchestrator"
                elif isinstance(workflow, ParallelLLM):
                    agent_types[name] = "Parallel"
                elif isinstance(workflow, EvaluatorOptimizerLLM):
                    agent_types[name] = "Evaluator"
                else:
                    agent_types[name] = "Workflow"

        result = ""
        while True:
            with progress_display.paused():
                # Use the enhanced input method with advanced features
                user_input = await get_enhanced_input(
                    agent_name=agent,
                    default=default,
                    show_default=(default != ""),
                    show_stop_hint=True,
                    multiline=False,  # Default to single-line mode
                    available_agent_names=available_agents,
                    syntax=None,  # Can enable syntax highlighting for code input
                    agent_types=agent_types,  # Pass agent types for display
                )

                # Handle special commands
                command_result = await handle_special_commands(user_input, self)

                # Check if we should switch agents
                if (
                    isinstance(command_result, dict)
                    and "switch_agent" in command_result
                ):
                    agent = command_result["switch_agent"]
                    continue

                # Skip further processing if command was handled
                if command_result:
                    continue

                if user_input.upper() == "STOP":
                    return result
                if user_input == "":
                    continue

            result = await self.send(agent, user_input)

            # Check if current agent is a chain that should continue with final agent
            if agent_types.get(agent) == "Chain":
                proxy = self._agents[agent]
                if isinstance(proxy, ChainProxy) and proxy._continue_with_final:
                    # Get the last agent in the sequence
                    last_agent = proxy._sequence[-1]
                    # Switch to that agent for the next iteration
                    agent = last_agent

        return result

    def __getattr__(self, name: str) -> BaseAgentProxy:
        """Support: agent.researcher"""
        if name not in self._agents:
            raise AttributeError(f"No agent named '{name}'")
        return AgentProxy(self, name)

    def __getitem__(self, name: str) -> BaseAgentProxy:
        """Support: agent['researcher']"""
        if name not in self._agents:
            raise KeyError(f"No agent named '{name}'")
        return AgentProxy(self, name)

    async def __call__(
        self, message: Optional[str] = "", agent_name: Optional[str] = None
    ) -> str:
        """Support: agent('message')"""
        target = agent_name or self._default
        if not target:
            raise ValueError("No default agent available")
        return await self.send(target, message)

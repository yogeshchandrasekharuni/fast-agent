"""
Decorator-based interface for MCP Agent applications.
Provides a simplified way to create and manage agents using decorators.
"""

from typing import List, Optional, Any, Dict, Callable
import yaml
from contextlib import asynccontextmanager

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM  # noqa: F401
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM  # noqa: F401
from mcp_agent.config import Settings
from rich.prompt import Prompt
from rich import print
from mcp_agent.progress_display import progress_display


class MCPAgentDecorator:
    """
    A decorator-based interface for MCP Agent applications.
    Provides a simplified way to create and manage agents using decorators.
    """

    def __init__(self, name: str, config_path: Optional[str] = None):
        """
        Initialize the decorator interface.

        Args:
            name: Name of the application
            config_path: Optional path to config file
        """
        self.name = name
        self.config_path = config_path
        self._load_config()
        self.app = MCPApp(
            name=name,
            settings=Settings(**self.config) if hasattr(self, "config") else None,
        )
        self.agents: Dict[str, Dict[str, Any]] = {}

    def _load_config(self):
        """Load configuration, properly handling YAML without dotenv processing"""
        if self.config_path:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

    def agent(self, name: str, instruction: str, servers: List[str]) -> Callable:
        """
        Decorator to create and register an agent.

        Args:
            name: Name of the agent
            instruction: Base instruction for the agent
            servers: List of server names the agent should connect to
        """

        def decorator(func: Callable) -> Callable:
            # Store the agent configuration for later instantiation
            self.agents[name] = {"instruction": instruction, "servers": servers}

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    @asynccontextmanager
    async def run(self):
        """
        Context manager for running the application.
        Handles setup and teardown of the app and agents.
        """
        async with self.app.run() as agent_app:
            active_agents = {}

            # Create and initialize all registered agents with proper context
            for name, config in self.agents.items():
                agent = Agent(
                    name=name,
                    instruction=config["instruction"],
                    server_names=config["servers"],
                    context=agent_app.context,
                )
                active_agents[name] = agent

            # Start all agents within their context managers
            agent_contexts = []
            for name, agent in active_agents.items():
                ctx = await agent.__aenter__()
                agent_contexts.append((agent, ctx))

                # Attach LLM to each agent
                llm = await agent.attach_llm(AnthropicAugmentedLLM)
                # Store LLM reference on agent
                agent._llm = llm

            # Create a wrapper object with simplified interface
            wrapper = AgentAppWrapper(agent_app, active_agents)
            try:
                yield wrapper
            finally:
                # Cleanup agents
                for agent, _ in agent_contexts:
                    await agent.__aexit__(None, None, None)


class AgentAppWrapper:
    """
    Wrapper class providing a simplified interface to the agent application.
    """

    def __init__(self, app: MCPApp, agents: Dict[str, Agent]):
        self.app = app
        self.agents = agents
        # Store first agent name for default calls
        self._default_agent = next(iter(agents)) if agents else None

    async def send(self, agent_name: str, message: str) -> Any:
        """
        Send a message to a specific agent and get the response.

        Args:
            agent_name: Name of the agent to send message to
            message: Message to send

        Returns:
            Agent's response
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")

        agent = self.agents[agent_name]
        if not hasattr(agent, "_llm") or agent._llm is None:
            raise RuntimeError(f"Agent {agent_name} has no LLM attached")
        result = await agent._llm.generate_str(message)
        return result

    async def __call__(self, message: str, agent_name: Optional[str] = None) -> Any:
        """
        Send a message to an agent using direct call syntax.
        Uses the first registered agent if no agent_name is specified.

        Args:
            message: Message to send to the agent
            agent_name: Optional name of agent to send to. Uses first agent if None.

        Returns:
            Agent's response
        """
        target_agent = agent_name or self._default_agent
        if not target_agent:
            raise ValueError("No agents available")
        return await self.send(target_agent, message)

    async def prompt(self, agent_name: Optional[str] = None, default: str = "") -> None:
        """
        Interactive prompt for sending messages to an agent using rich.

        Args:
            agent_name: Optional name of agent to send to. Uses first agent if None.
            default: Default input value
        """

        target_agent = agent_name or self._default_agent
        if not target_agent:
            raise ValueError("No agents available")

        while True:
            with progress_display.paused():
                if default == "STOP":
                    print("Press <ENTER> to finish")
                elif default != "":
                    print("Enter a prompt, or [red]STOP[/red] to finish")
                    print(
                        f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]"
                    )

                else:
                    print("Enter a prompt, or enter [red]STOP[/red] to finish")

                prompt_text = f"[blue]{target_agent}[/blue] >"
                user_input = Prompt.ask(
                    prompt=prompt_text, default=default, show_default=False
                )

                if user_input.upper() == "STOP":
                    return

            await self.send(target_agent, user_input)

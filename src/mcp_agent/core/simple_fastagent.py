"""
Simplified FastAgent implementation.
This module provides a more straightforward API for creating agent applications.
"""

import argparse
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional, Set, Type, TypeVar

import yaml

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import Settings
from mcp_agent.context import Context
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.error_handling import handle_error
from mcp_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ProviderKeyError,
    ServerConfigError,
    ServerInitializationError,
)
from mcp_agent.core.simple_decorators import agent, orchestrator
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

T = TypeVar("T")  # For the wrapper classes
logger = get_logger(__name__)


class SimpleFastAgent:
    """
    A simplified FastAgent implementation.
    Provides a more direct way to create and manage agents using type-safe decorators.
    """

    def __init__(
        self,
        name: str,
        config_path: Optional[str] = None,
        ignore_unknown_args: bool = False,
    ) -> None:
        """
        Initialize the FastAgent application.

        Args:
            name: Name of the application
            config_path: Optional path to config file
            ignore_unknown_args: Whether to ignore unknown command line arguments
        """
        # Setup command line argument parsing
        parser = argparse.ArgumentParser(description="FastAgent Application")
        parser.add_argument(
            "--model",
            help="Override the default model for all agents",
        )
        parser.add_argument(
            "--agent",
            help="Specify the agent to send a message to (used with --message)",
        )
        parser.add_argument(
            "-m",
            "--message",
            help="Message to send to the specified agent (requires --agent)",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Disable progress display, tool and message logging for cleaner output",
        )

        if ignore_unknown_args:
            known_args, _ = parser.parse_known_args()
            self.args = known_args
        else:
            self.args = parser.parse_args()

        self.name = name
        self.config_path = config_path
        self._load_config()

        # Create the MCPApp with the config
        self.app = MCPApp(
            name=name,
            settings=Settings(**self.config) if hasattr(self, "config") else None,
        )
        
        # Dictionary to store registered agents
        self.agents: Dict[str, Agent] = {}
        
        # Registry for decorator-defined agents
        self.registered_decorators: Set[Callable] = set()
        
        # Bind decorator methods to this instance
        self.agent = agent
        self.orchestrator = orchestrator
        # Add more decorator methods as they're implemented

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if self.config_path:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}

    @property
    def context(self) -> Context:
        """Access the application context"""
        return self.app.context
        
    async def _create_agent_from_decorator(self, decorated_func: Callable) -> Agent:
        """
        Create an actual Agent instance from a decorated function.
        
        Args:
            decorated_func: The decorated function with agent metadata
            
        Returns:
            An initialized Agent instance
        """
        agent_type = getattr(decorated_func, "_agent_type", None)
        agent_config = getattr(decorated_func, "_agent_config", None)
        
        if not agent_type or not agent_config:
            raise ValueError(f"Invalid decorated function: {decorated_func.__name__}")
            
        # Create the agent based on its type
        if agent_type == AgentType.BASIC:
            # Create a basic agent
            agent_instance = Agent(config=agent_config, context=self.context)
            
            # Determine which LLM to use based on model
            llm_class: Type[AugmentedLLMProtocol] = AnthropicAugmentedLLM
            if agent_config.model and "gpt" in agent_config.model:
                llm_class = OpenAIAugmentedLLM
                
            # Initialize the agent
            await agent_instance.initialize()
            
            # Attach the LLM
            await agent_instance.attach_llm(llm_class)
            
            return agent_instance
            
        elif agent_type == AgentType.ORCHESTRATOR:
            # Create an orchestrator (implementation would be added here)
            # For now, return a basic agent as placeholder
            return Agent(config=agent_config, context=self.context)
            
        # Implement other agent types as needed
        
        raise ValueError(f"Unsupported agent type: {agent_type}")

    @asynccontextmanager
    async def run(self):
        """
        Context manager for running the application.
        Initializes all registered agents.
        """
        active_agents: Dict[str, Agent] = {}
        had_error = False
        await self.app.initialize()

        # Handle quiet mode
        quiet_mode = hasattr(self, "args") and self.args.quiet

        try:
            async with self.app.run() as agent_app:
                # Apply quiet mode if requested
                if quiet_mode and hasattr(agent_app.context, "config") and hasattr(agent_app.context.config, "logger"):
                    agent_app.context.config.logger.progress_display = False
                    agent_app.context.config.logger.show_chat = False
                    agent_app.context.config.logger.show_tools = False

                    # Directly disable the progress display singleton
                    from mcp_agent.progress_display import progress_display
                    progress_display.stop()

                # Initialize all registered decorator-based agents
                for func in self.registered_decorators:
                    agent_config = getattr(func, "_agent_config", None)
                    if agent_config:
                        agent_name = agent_config.name
                        agent_instance = await self._create_agent_from_decorator(func)
                        active_agents[agent_name] = agent_instance
                        self.agents[agent_name] = agent_instance
                
                # Handle direct message sending if --agent and --message are provided
                if self.args.agent and self.args.message:
                    agent_name = self.args.agent
                    message = self.args.message

                    if agent_name not in active_agents:
                        available_agents = ", ".join(active_agents.keys())
                        print(f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}")
                        raise SystemExit(1)

                    try:
                        # Get response from the agent
                        response = await active_agents[agent_name]._llm.generate_str(message)

                        # Only print the response in quiet mode
                        if self.args.quiet:
                            print(f"{response}")

                        raise SystemExit(0)
                    except Exception as e:
                        print(f"\n\nError sending message to agent '{agent_name}': {str(e)}")
                        raise SystemExit(1)

                # Yield active agents wrapped in a simple container
                yield SimpleAgentApp(active_agents)
                
        except Exception as e:
            # Handle various error types with friendly messages
            had_error = True
            self._handle_error(e)
            raise SystemExit(1)
            
        finally:
            # Clean up any active agents
            if active_agents and not had_error:
                for agent in active_agents.values():
                    try:
                        await agent.shutdown()
                    except Exception:
                        pass

    def _handle_error(self, e: Exception) -> None:
        """
        Handle errors with consistent formatting and messaging.
        
        Args:
            e: The exception that was raised
        """
        if isinstance(e, ServerConfigError):
            handle_error(
                e,
                "Server Configuration Error",
                "Please check your 'fastagent.config.yaml' configuration file and add the missing server definitions.",
            )
        elif isinstance(e, ProviderKeyError):
            handle_error(
                e,
                "Provider Configuration Error",
                "Please check your 'fastagent.secrets.yaml' configuration file and ensure all required API keys are set.",
            )
        elif isinstance(e, AgentConfigError):
            handle_error(
                e,
                "Workflow or Agent Configuration Error",
                "Please check your agent definition and ensure names and references are correct.",
            )
        elif isinstance(e, ServerInitializationError):
            handle_error(
                e,
                "MCP Server Startup Error",
                "There was an error starting up the MCP Server.",
            )
        elif isinstance(e, ModelConfigError):
            handle_error(
                e,
                "Model Configuration Error",
                "Common models: gpt-4o, o3-mini, sonnet, haiku. for o3, set reasoning effort with o3-mini.high",
            )
        elif isinstance(e, CircularDependencyError):
            handle_error(
                e,
                "Circular Dependency Detected",
                "Check your agent configuration for circular dependencies.",
            )
        elif isinstance(e, PromptExitError):
            handle_error(
                e,
                "User requested exit",
            )
        else:
            handle_error(
                e,
                "Error",
                "An unexpected error occurred."
            )


class SimpleAgentApp:
    """Simple container for active agents, replacing the proxy mechanism"""
    
    def __init__(self, agents: Dict[str, Agent]):
        """
        Initialize the SimpleAgentApp.
        
        Args:
            agents: Dictionary of active agents keyed by name
        """
        self._agents = agents
        
    def __getitem__(self, key: str) -> Agent:
        """Allow access to agents using dictionary syntax"""
        if key not in self._agents:
            raise KeyError(f"Agent '{key}' not found")
        return self._agents[key]
        
    def __getattr__(self, name: str) -> Agent:
        """Allow access to agents using attribute syntax"""
        if name in self._agents:
            return self._agents[name]
        raise AttributeError(f"Agent '{name}' not found")
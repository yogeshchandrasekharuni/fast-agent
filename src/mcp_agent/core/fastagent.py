"""
Direct FastAgent implementation that uses the simplified Agent architecture.
This replaces the traditional FastAgent with a more streamlined approach that
directly creates Agent instances without proxies.
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar

import yaml

from mcp_agent.app import MCPApp
from mcp_agent.config import Settings
from mcp_agent.context import Context
from mcp_agent.core.agent_app import AgentApp
from mcp_agent.core.direct_decorators import (
    agent as agent_decorator,
)
from mcp_agent.core.direct_decorators import (
    chain as chain_decorator,
)
from mcp_agent.core.direct_decorators import (
    evaluator_optimizer as evaluator_optimizer_decorator,
)
from mcp_agent.core.direct_decorators import (
    orchestrator as orchestrator_decorator,
)
from mcp_agent.core.direct_decorators import (
    parallel as parallel_decorator,
)
from mcp_agent.core.direct_decorators import (
    router as router_decorator,
)
from mcp_agent.core.direct_factory import (
    create_agents_in_dependency_order,
    get_model_factory,
)
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
from mcp_agent.core.validation import (
    validate_server_references,
    validate_workflow_references,
)
from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp_agent.agents.agent import Agent

F = TypeVar("F", bound=Callable[..., Any])  # For decorated functions
logger = get_logger(__name__)


class FastAgent:
    """
    A simplified FastAgent implementation that directly creates Agent instances
    without using proxies.
    """

    def __init__(
        self,
        name: str,
        config_path: Optional[str] = None,
        ignore_unknown_args: bool = False,
    ) -> None:
        """
        Initialize the DirectFastAgent application.

        Args:
            name: Name of the application
            config_path: Optional path to config file
            ignore_unknown_args: Whether to ignore unknown command line arguments
        """
        # Setup command line argument parsing
        parser = argparse.ArgumentParser(description="DirectFastAgent Application")
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
        parser.add_argument(
            "--version",
            action="store_true",
            help="Show version and exit",
        )
        parser.add_argument(
            "--server",
            action="store_true",
            help="Run as an MCP server",
        )
        parser.add_argument(
            "--transport",
            choices=["sse", "stdio"],
            default="sse",
            help="Transport protocol to use when running as a server (sse or stdio)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to use when running as a server with SSE transport",
        )
        parser.add_argument(
            "--host",
            default="0.0.0.0",
            help="Host address to bind to when running as a server with SSE transport",
        )

        if ignore_unknown_args:
            known_args, _ = parser.parse_known_args()
            self.args = known_args
        else:
            self.args = parser.parse_args()
            
        # Handle version flag
        if self.args.version:
            try:
                app_version = get_version("fast-agent-mcp")
            except:  # noqa: E722
                app_version = "unknown"
            print(f"fast-agent-mcp v{app_version}")
            sys.exit(0)

        self.name = name
        self.config_path = config_path
        try:
            self._load_config()
        except yaml.parser.ParserError as e:
            handle_error(
                e,
                "YAML Parsing Error",
                "There was an error parsing the config or secrets YAML configuration file.",
            )
            raise SystemExit(1)
        # Create the MCPApp with the config
        self.app = MCPApp(
            name=name,
            settings=Settings(**self.config) if hasattr(self, "config") else None,
        )

        # Dictionary to store agent configurations from decorators
        self.agents: Dict[str, Dict[str, Any]] = {}
        
        # If --server flag is present, automatically start in server mode
        # this ensures that any program can be run as a server without code changes
        if hasattr(self, "args") and self.args.server:
            # Import needed here to avoid circular imports
            import asyncio
            
            # Print info message
            print(f"Starting FastAgent '{name}' in server mode")
            print(f"Transport: {self.args.transport}")
            if self.args.transport == 'sse':
                print(f"Listening on {self.args.host}:{self.args.port}")
            print("Press Ctrl+C to stop")
            
            # We wrap server startup in a task and run it in the event loop
            # This allows us to automatically handle the server at startup time
            async def server_start_task():
                try:
                    await self.start_server(
                        transport=self.args.transport,
                        host=self.args.host,
                        port=self.args.port
                    )
                except KeyboardInterrupt:
                    print("\nServer stopped by user (Ctrl+C)")
                except Exception as e:
                    print(f"\nServer stopped with error: {e}")
                finally:
                    sys.exit(0)
                    
            # Execute the server startup task
            event_loop = asyncio.get_event_loop()
            event_loop.create_task(server_start_task())
            # Note: We don't block here - the program continues execution
            # But when it reaches the run() method, it will yield a dummy object

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        if self.config_path:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}
        elif os.path.exists("fastagent.config.yaml"):
            with open("fastagent.config.yaml") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

    @property
    def context(self) -> Context:
        """Access the application context"""
        return self.app.context

    # Decorator methods with type-safe implementations
    agent = agent_decorator
    orchestrator = orchestrator_decorator
    router = router_decorator
    chain = chain_decorator
    parallel = parallel_decorator
    evaluator_optimizer = evaluator_optimizer_decorator

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
                if (
                    quiet_mode
                    and hasattr(agent_app.context, "config")
                    and hasattr(agent_app.context.config, "logger")
                ):
                    agent_app.context.config.logger.progress_display = False
                    agent_app.context.config.logger.show_chat = False
                    agent_app.context.config.logger.show_tools = False

                    # Directly disable the progress display singleton
                    from mcp_agent.progress_display import progress_display

                    progress_display.stop()

                # Pre-flight validation
                validate_server_references(self.context, self.agents)
                validate_workflow_references(self.agents)

                # Get a model factory function
                def model_factory_func(model=None, request_params=None):
                    return get_model_factory(
                        self.context,
                        model=model,
                        request_params=request_params,
                        cli_model=self.args.model if hasattr(self, "args") else None,
                    )

                # Create all agents in dependency order
                active_agents = await create_agents_in_dependency_order(
                    self.app,
                    self.agents,
                    model_factory_func,
                )

                # Create a wrapper with all agents for simplified access
                wrapper = AgentApp(active_agents)

                # Handle command line options that should be processed after agent initialization
                
                # Handle --server option
                if hasattr(self, "args") and self.args.server:
                    try:
                        # Print info message
                        print(f"Starting FastAgent '{self.name}' in server mode")
                        print(f"Transport: {self.args.transport}")
                        if self.args.transport == 'sse':
                            print(f"Listening on {self.args.host}:{self.args.port}")
                        print("Press Ctrl+C to stop")
                        
                        # Create the MCP server
                        from mcp_agent.mcp_server import AgentMCPServer
                        
                        mcp_server = AgentMCPServer(
                            agent_app=wrapper,
                            server_name=f"{self.name}-MCP-Server",
                        )
                        
                        # Run the server directly (this is a blocking call)
                        await mcp_server.run_async(
                            transport=self.args.transport,
                            host=self.args.host,
                            port=self.args.port
                        )
                    except KeyboardInterrupt:
                        print("\nServer stopped by user (Ctrl+C)")
                    except Exception as e:
                        print(f"\nServer stopped with error: {e}")
                    
                    # Exit after server shutdown
                    raise SystemExit(0)
                
                # Handle direct message sending if --agent and --message are provided
                if hasattr(self, "args") and self.args.agent and self.args.message:
                    agent_name = self.args.agent
                    message = self.args.message

                    if agent_name not in active_agents:
                        available_agents = ", ".join(active_agents.keys())
                        print(
                            f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}"
                        )
                        raise SystemExit(1)

                    try:
                        # Get response from the agent
                        agent = active_agents[agent_name]
                        response = await agent.send(message)

                        # Print the response in quiet mode
                        if self.args.quiet:
                            print(f"{response}")

                        raise SystemExit(0)
                    except Exception as e:
                        print(f"\n\nError sending message to agent '{agent_name}': {str(e)}")
                        raise SystemExit(1)

                yield wrapper

        except (
            ServerConfigError,
            ProviderKeyError,
            AgentConfigError,
            ServerInitializationError,
            ModelConfigError,
            CircularDependencyError,
            PromptExitError,
        ) as e:
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

    def _handle_error(self, e: Exception, error_type: Optional[str] = None) -> None:
        """
        Handle errors with consistent formatting and messaging.

        Args:
            e: The exception that was raised
            error_type: Optional explicit error type
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
        elif isinstance(e, asyncio.CancelledError):
            handle_error(
                e,
                "Cancelled",
                "The operation was cancelled.",
            )
        else:
            handle_error(e, error_type or "Error", "An unexpected error occurred.")

    async def start_server(
        self,
        transport: str = "sse",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
    ) -> None:
        """
        Start the application as an MCP server.
        This method initializes agents and exposes them through an MCP server.
        It is a blocking method that runs until the server is stopped.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
            host: Host address for the server when using SSE
            port: Port for the server when using SSE
            server_name: Optional custom name for the MCP server
            server_description: Optional description for the MCP server
        """
        # This method simply updates the command line arguments and uses run()
        # to ensure we follow the same initialization path for all operations
        
        # Store original args
        original_args = None
        if hasattr(self, "args"):
            original_args = self.args
            
        # Create our own args object with server settings
        from argparse import Namespace
        self.args = Namespace()
        self.args.server = True
        self.args.transport = transport
        self.args.host = host
        self.args.port = port
        self.args.quiet = False
        self.args.model = None
        if hasattr(original_args, "model"):
            self.args.model = original_args.model
            
        # Run the application, which will detect the server flag and start server mode
        async with self.run():
            pass  # This won't be reached due to SystemExit in run()
        
        # Restore original args (if we get here)
        if original_args:
            self.args = original_args
    
    # Keep run_with_mcp_server for backward compatibility
    async def run_with_mcp_server(
        self,
        transport: str = "sse",
        host: str = "0.0.0.0",
        port: int = 8000,
        server_name: Optional[str] = None,
        server_description: Optional[str] = None,
    ) -> None:
        """
        Run the application and expose agents through an MCP server.
        This method is kept for backward compatibility.
        For new code, use start_server() instead.

        Args:
            transport: Transport protocol to use ("stdio" or "sse")
            host: Host address for the server when using SSE
            port: Port for the server when using SSE
            server_name: Optional custom name for the MCP server
            server_description: Optional description for the MCP server
        """
        await self.start_server(
            transport=transport,
            host=host,
            port=port,
            server_name=server_name,
            server_description=server_description,
        )

    async def main(self):
        """
        Helper method for checking if server mode was requested.
        
        Usage:
        ```python
        fast = FastAgent("My App")
        
        @fast.agent(...)
        async def app_main():
            # Check if server mode was requested
            # This doesn't actually do anything - the check happens in run()
            # But it provides a way for application code to know if server mode
            # was requested for conditionals
            is_server_mode = hasattr(self, "args") and self.args.server
            
            # Normal run - this will handle server mode automatically if requested
            async with fast.run() as agent:
                # This code only executes for normal mode
                # Server mode will exit before reaching here
                await agent.send("Hello")
        ```
        
        Returns:
            bool: True if --server flag is set, False otherwise
        """
        # Just check if the flag is set, no action here
        # The actual server code will be handled by run()
        return hasattr(self, "args") and self.args.server
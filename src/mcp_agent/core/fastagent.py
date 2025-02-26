"""
Decorator-based interface for MCP Agent applications.
Provides a simplified way to create and manage agents using decorators.
"""

from typing import (
    List,
    Optional,
    Dict,
    Callable,
    TypeVar,
    Any,
    Union,
    TypeAlias,
    Literal,
)
from enum import Enum
import yaml
import argparse
from contextlib import asynccontextmanager

from mcp_agent.core.exceptions import (
    AgentConfigError,
    ModelConfigError,
    PromptExitError,
    ServerConfigError,
    ProviderKeyError,
    ServerInitializationError,
)

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.event_progress import ProgressAction
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from mcp_agent.workflows.router.router_llm import LLMRouter
from mcp_agent.config import Settings
from rich import print
from mcp_agent.progress_display import progress_display
from mcp_agent.workflows.llm.model_factory import ModelFactory
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams

# TODO -- resintate once Windows&Python 3.13 platform issues are fixed
# import readline  # noqa: F401

# Type aliases for better readability
WorkflowType: TypeAlias = Union[
    Orchestrator, ParallelLLM, EvaluatorOptimizerLLM, LLMRouter
]
AgentOrWorkflow: TypeAlias = Union[Agent, WorkflowType]
ProxyDict: TypeAlias = Dict[str, "BaseAgentProxy"]


class AgentType(Enum):
    """Enumeration of supported agent types."""

    BASIC = "agent"
    ORCHESTRATOR = "orchestrator"
    PARALLEL = "parallel"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    ROUTER = "router"


T = TypeVar("T")  # For the wrapper classes


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
        from .enhanced_prompt import get_enhanced_input, handle_special_commands

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
                    return
                if user_input == "":
                    continue

            result = await self.send(agent, user_input)

        return result

    def __getattr__(self, name: str) -> AgentProxy:
        """Support: agent.researcher"""
        if name not in self._agents:
            raise AttributeError(f"No agent named '{name}'")
        return AgentProxy(self, name)

    def __getitem__(self, name: str) -> AgentProxy:
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


class FastAgent(ContextDependent):
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
        # Initialize ContextDependent
        super().__init__()

        # Setup command line argument parsing
        parser = argparse.ArgumentParser(description="MCP Agent Application")
        parser.add_argument(
            "--model",
            help="Override the default model for all agents. Precedence is default < config_file < command line < constructor",
        )
        self.args = parser.parse_args()

        self.name = name
        self.config_path = config_path
        self._load_config()
        self.app = MCPApp(
            name=name,
            settings=Settings(**self.config) if hasattr(self, "config") else None,
        )
        self.agents: Dict[str, Dict[str, Any]] = {}

    def _create_proxy(
        self, name: str, instance: AgentOrWorkflow, agent_type: str
    ) -> BaseAgentProxy:
        """Create appropriate proxy type based on agent type and validate instance type

        Args:
            name: Name of the agent/workflow
            instance: The agent or workflow instance
            agent_type: Type from AgentType enum values

        Returns:
            Appropriate proxy type wrapping the instance

        Raises:
            TypeError: If instance type doesn't match expected type for agent_type
        """
        if agent_type not in [
            AgentType.PARALLEL.value,
            AgentType.EVALUATOR_OPTIMIZER.value,
        ]:
            self._log_agent_load(name)
        if agent_type == AgentType.BASIC.value:
            if not isinstance(instance, Agent):
                raise TypeError(
                    f"Expected Agent instance for {name}, got {type(instance)}"
                )
            return LLMAgentProxy(self.app, name, instance)
        elif agent_type == AgentType.ORCHESTRATOR.value:
            if not isinstance(instance, Orchestrator):
                raise TypeError(
                    f"Expected Orchestrator instance for {name}, got {type(instance)}"
                )
            return WorkflowProxy(self.app, name, instance)
        elif agent_type == AgentType.PARALLEL.value:
            if not isinstance(instance, ParallelLLM):
                raise TypeError(
                    f"Expected ParallelLLM instance for {name}, got {type(instance)}"
                )
            return WorkflowProxy(self.app, name, instance)
        elif agent_type == AgentType.EVALUATOR_OPTIMIZER.value:
            if not isinstance(instance, EvaluatorOptimizerLLM):
                raise TypeError(
                    f"Expected EvaluatorOptimizerLLM instance for {name}, got {type(instance)}"
                )
            return WorkflowProxy(self.app, name, instance)
        elif agent_type == AgentType.ROUTER.value:
            if not isinstance(instance, LLMRouter):
                raise TypeError(
                    f"Expected LLMRouter instance for {name}, got {type(instance)}"
                )
            return RouterProxy(self.app, name, instance)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    @property
    def context(self):
        """Access the application context"""
        return self.app.context

    def _load_config(self) -> None:
        """Load configuration from YAML file, properly handling without dotenv processing"""
        if self.config_path:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

    def _validate_server_references(self) -> None:
        """
        Validate that all server references in agent configurations exist in config.
        Raises ServerConfigError if any referenced servers are not defined.
        """
        if not self.context.config.mcp or not self.context.config.mcp.servers:
            available_servers = set()
        else:
            available_servers = set(self.context.config.mcp.servers.keys())

        # Check each agent's server references
        for name, agent_data in self.agents.items():
            config = agent_data["config"]
            if config.servers:
                missing = [s for s in config.servers if s not in available_servers]
                if missing:
                    raise ServerConfigError(
                        f"Missing server configuration for agent '{name}'",
                        f"The following servers are referenced but not defined in config: {', '.join(missing)}",
                    )

    def _validate_workflow_references(self) -> None:
        """
        Validate that all workflow references point to valid agents/workflows.
        Also validates that referenced agents have required configuration.
        Raises AgentConfigError if any validation fails.
        """
        available_components = set(self.agents.keys())

        for name, agent_data in self.agents.items():
            agent_type = agent_data["type"]

            if agent_type == AgentType.PARALLEL.value:
                # Check fan_in exists
                fan_in = agent_data["fan_in"]
                if fan_in not in available_components:
                    raise AgentConfigError(
                        f"Parallel workflow '{name}' references non-existent fan_in component: {fan_in}"
                    )

                # Check fan_out agents exist
                fan_out = agent_data["fan_out"]
                missing = [a for a in fan_out if a not in available_components]
                if missing:
                    raise AgentConfigError(
                        f"Parallel workflow '{name}' references non-existent fan_out components: {', '.join(missing)}"
                    )

            elif agent_type == AgentType.ORCHESTRATOR.value:
                # Check all child agents exist and are properly configured
                child_agents = agent_data["child_agents"]
                missing = [a for a in child_agents if a not in available_components]
                if missing:
                    raise AgentConfigError(
                        f"Orchestrator '{name}' references non-existent agents: {', '.join(missing)}"
                    )

                # Validate child agents have required LLM configuration
                for agent_name in child_agents:
                    child_data = self.agents[agent_name]
                    if child_data["type"] == AgentType.BASIC.value:
                        # For basic agents, we'll validate LLM config during creation
                        continue
                    elif not isinstance(child_data["func"], AugmentedLLM):
                        raise AgentConfigError(
                            f"Agent '{agent_name}' used by orchestrator '{name}' lacks LLM capability",
                            "All agents used by orchestrators must be LLM-capable",
                        )

            elif agent_type == AgentType.ROUTER.value:
                # Check all referenced agents exist
                router_agents = agent_data["agents"]
                missing = [a for a in router_agents if a not in available_components]
                if missing:
                    raise AgentConfigError(
                        f"Router '{name}' references non-existent agents: {', '.join(missing)}"
                    )

            elif agent_type == AgentType.EVALUATOR_OPTIMIZER.value:
                # Check both evaluator and optimizer exist
                evaluator = agent_data["evaluator"]
                generator = agent_data["generator"]
                missing = []
                if evaluator not in available_components:
                    missing.append(f"evaluator: {evaluator}")
                if generator not in available_components:
                    missing.append(f"generator: {generator}")
                if missing:
                    raise AgentConfigError(
                        f"Evaluator-Optimizer '{name}' references non-existent components: {', '.join(missing)}"
                    )

    def _get_model_factory(
        self,
        model: Optional[str] = None,
        request_params: Optional[RequestParams] = None,
    ) -> Any:
        """
        Get model factory using specified or default model.
        Model string is parsed by ModelFactory to determine provider and reasoning effort.

        Args:
            model: Optional model specification string
            request_params: Optional RequestParams to configure LLM behavior

        Returns:
            ModelFactory instance for the specified or default model
        """

        # Config has lowest precedence
        model_spec = self.context.config.default_model

        # Command line override has next precedence
        if self.args.model:
            model_spec = self.args.model

        # Model from decorator has highest precedence
        if model:
            model_spec = model

        # Update or create request_params with the final model choice
        if request_params:
            request_params = request_params.model_copy(update={"model": model_spec})
        else:
            request_params = RequestParams(model=model_spec)

        # Let model factory handle the model string parsing and setup
        return ModelFactory.create_factory(model_spec, request_params=request_params)

    def _create_decorator(
        self,
        agent_type: AgentType,
        default_name: str = None,
        default_instruction: str = None,
        default_servers: List[str] = None,
        default_use_history: bool = True,
        wrapper_needed: bool = False,
        **extra_defaults,
    ) -> Callable:
        """
        Factory method for creating agent decorators with common behavior.

        Args:
            agent_type: Type of agent/workflow to create
            default_name: Default name to use if not provided
            default_instruction: Default instruction to use if not provided
            default_servers: Default servers list to use if not provided
            default_use_history: Default history setting
            wrapper_needed: Whether to wrap the decorated function
            **extra_defaults: Additional agent/workflow-specific parameters
        """

        def decorator_wrapper(**kwargs):
            # Apply defaults for common parameters
            name = kwargs.get("name", default_name or f"{agent_type.name.title()}")
            instruction = kwargs.get("instruction", default_instruction or "")
            servers = kwargs.get("servers", default_servers or [])
            model = kwargs.get("model", None)
            use_history = kwargs.get("use_history", default_use_history)
            request_params = kwargs.get("request_params", None)
            human_input = kwargs.get("human_input", False)

            # Create base request params
            def decorator(func: Callable) -> Callable:
                # Create base request params
                if (
                    request_params is not None
                    or model is not None
                    or use_history != default_use_history
                ):
                    max_tokens = 4096 if agent_type == AgentType.BASIC else None
                    params_dict = {"use_history": use_history, "model": model}
                    if max_tokens:
                        params_dict["maxTokens"] = max_tokens
                    if request_params:
                        params_dict.update(request_params)
                    base_params = RequestParams(**params_dict)
                else:
                    base_params = RequestParams(use_history=use_history)

                # Create agent configuration
                config = AgentConfig(
                    name=name,
                    instruction=instruction,
                    servers=servers,
                    model=model,
                    use_history=use_history,
                    default_request_params=base_params,
                    human_input=human_input,
                )

                # Build agent/workflow specific data
                agent_data = {
                    "config": config,
                    "type": agent_type.value,
                    "func": func,
                }

                # Add extra parameters specific to this agent type
                for key, value in kwargs.items():
                    if key not in [
                        "name",
                        "instruction",
                        "servers",
                        "model",
                        "use_history",
                        "request_params",
                        "human_input",
                    ]:
                        agent_data[key] = value

                # Store the configuration under the agent name
                self.agents[name] = agent_data

                # Either wrap or return the original function
                if wrapper_needed:

                    async def wrapper(*args, **kwargs):
                        return await func(*args, **kwargs)

                    return wrapper
                return func

            return decorator

        return decorator_wrapper

    def agent(
        self,
        name: str = "Agent",
        *,
        instruction: str = "You are a helpful agent.",
        servers: List[str] = [],
        model: str | None = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
        human_input: bool = False,
    ) -> Callable:
        """
        Decorator to create and register an agent with configuration.

        Args:
            name: Name of the agent
            instruction: Base instruction for the agent
            servers: List of server names the agent should connect to
            model: Model specification string (highest precedence)
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
            human_input: Whether to enable human input capabilities
        """
        decorator = self._create_decorator(
            AgentType.BASIC,
            default_name="Agent",
            default_instruction="You are a helpful agent.",
            default_use_history=True,
        )(
            name=name,
            instruction=instruction,
            servers=servers,
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
        )
        return decorator

    def orchestrator(
        self,
        name: str = "Orchestrator",
        *,
        instruction: str | None = None,
        agents: List[str],
        model: str | None = None,
        use_history: bool = False,
        request_params: Optional[Dict] = None,
        human_input: bool = False,
        plan_type: Literal["full", "iterative"] = "full",
    ) -> Callable:
        """
        Decorator to create and register an orchestrator.

        Args:
            name: Name of the orchestrator
            instruction: Base instruction for the orchestrator
            agents: List of agent names this orchestrator can use
            model: Model specification string (highest precedence)
            use_history: Whether to maintain conversation history (forced false)
            request_params: Additional request parameters for the LLM
            human_input: Whether to enable human input capabilities
            plan_type: Planning approach - "full" generates entire plan first, "iterative" plans one step at a time
        """
        default_instruction = """
            You are an expert planner. Given an objective task and a list of MCP servers (which are collections of tools)
            or Agents (which are collections of servers), your job is to break down the objective into a series of steps,
            which can be performed by LLMs with access to the servers or agents.
            """

        decorator = self._create_decorator(
            AgentType.ORCHESTRATOR,
            default_name="Orchestrator",
            default_instruction=default_instruction,
            default_servers=[],
            default_use_history=False,
        )(
            name=name,
            instruction=instruction,
            child_agents=agents,
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            plan_type=plan_type,
        )
        return decorator

    def parallel(
        self,
        name: str,
        fan_in: str,
        fan_out: List[str],
        instruction: str = "",
        model: str | None = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
    ) -> Callable:
        """
        Decorator to create and register a parallel executing agent.

        Args:
            name: Name of the parallel executing agent
            fan_in: Name of collecting agent
            fan_out: List of parallel execution agents
            instruction: Optional instruction for the parallel agent
            model: Model specification string
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
        """
        decorator = self._create_decorator(
            AgentType.PARALLEL,
            default_instruction="",
            default_servers=[],
            default_use_history=True,
        )(
            name=name,
            fan_in=fan_in,
            fan_out=fan_out,
            instruction=instruction,
            model=model,
            use_history=use_history,
            request_params=request_params,
        )
        return decorator

    def evaluator_optimizer(
        self,
        name: str,
        generator: str,
        evaluator: str,
        min_rating: str = "GOOD",
        max_refinements: int = 3,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
    ) -> Callable:
        """
        Decorator to create and register an evaluator-optimizer workflow.

        Args:
            name: Name of the workflow
            generator: Name of the generator agent
            evaluator: Name of the evaluator agent
            min_rating: Minimum acceptable quality rating (EXCELLENT, GOOD, FAIR, POOR)
            max_refinements: Maximum number of refinement iterations
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
        """
        decorator = self._create_decorator(
            AgentType.EVALUATOR_OPTIMIZER,
            default_instruction="",
            default_servers=[],
            default_use_history=True,
            wrapper_needed=True,
        )(
            name=name,
            generator=generator,
            evaluator=evaluator,
            min_rating=min_rating,
            max_refinements=max_refinements,
            use_history=use_history,
            request_params=request_params,
        )
        return decorator

    def router(
        self,
        name: str,
        agents: List[str],
        #        servers: List[str] = [],
        model: Optional[str] = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
        human_input: bool = False,
    ) -> Callable:
        """
        Decorator to create and register a router.

        Args:
            name: Name of the router
            agents: List of agent names this router can delegate to
            servers: List of server names the router can use directly (currently not supported)
            model: Model specification string
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
            human_input: Whether to enable human input capabilities
        """
        decorator = self._create_decorator(
            AgentType.ROUTER,
            default_instruction="",
            default_servers=[],
            default_use_history=True,
            wrapper_needed=True,
        )(
            name=name,
            agents=agents,
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
        )
        return decorator

    async def _create_agents_by_type(
        self,
        agent_app: MCPApp,
        agent_type: AgentType,
        active_agents: ProxyDict = None,
        **kwargs,
    ) -> ProxyDict:
        """
        Generic method to create agents of a specific type.

        Args:
            agent_app: The main application instance
            agent_type: Type of agents to create
            active_agents: Dictionary of already created agents/proxies (for dependencies)
            **kwargs: Additional type-specific parameters

        Returns:
            Dictionary of initialized agents wrapped in appropriate proxies
        """
        if active_agents is None:
            active_agents = {}

        # Create a dictionary to store the initialized agents
        result_agents = {}

        # Get all agents of the specified type
        for name, agent_data in self.agents.items():
            if agent_data["type"] == agent_type.value:
                # Get common configuration
                config = agent_data["config"]

                # Type-specific initialization
                if agent_type == AgentType.BASIC:
                    # Create basic agent with configuration
                    agent = Agent(config=config, context=agent_app.context)

                    # Set up LLM with proper configuration
                    async with agent:
                        llm_factory = self._get_model_factory(
                            model=config.model,
                            request_params=config.default_request_params,
                        )
                        agent._llm = await agent.attach_llm(llm_factory)

                    # Store the agent
                    instance = agent

                elif agent_type == AgentType.ORCHESTRATOR:
                    # Get base params configured with model settings
                    base_params = (
                        config.default_request_params.model_copy()
                        if config.default_request_params
                        else RequestParams()
                    )
                    base_params.use_history = False  # Force no history for orchestrator

                    # Get the child agents - need to unwrap proxies and validate LLM config
                    child_agents = []
                    for agent_name in agent_data["child_agents"]:
                        proxy = active_agents[agent_name]
                        instance = self._unwrap_proxy(proxy)
                        # Validate basic agents have LLM
                        if isinstance(instance, Agent):
                            if not hasattr(instance, "_llm") or not instance._llm:
                                raise AgentConfigError(
                                    f"Agent '{agent_name}' used by orchestrator '{name}' missing LLM configuration",
                                    "All agents must be fully configured with LLMs before being used in an orchestrator",
                                )
                        child_agents.append(instance)

                    # Create a properly configured planner agent
                    planner_config = AgentConfig(
                        name=f"{name}",  # Use orchestrator name as prefix
                        instruction=config.instruction
                        or """
                        You are an expert planner. Given an objective task and a list of MCP servers (which are collections of tools)
                        or Agents (which are collections of servers), your job is to break down the objective into a series of steps,
                        which can be performed by LLMs with access to the servers or agents.
                        """,
                        servers=[],  # Planner doesn't need server access
                        model=config.model,
                        default_request_params=base_params,
                    )
                    planner_agent = Agent(
                        config=planner_config, context=agent_app.context
                    )
                    planner_factory = self._get_model_factory(
                        model=config.model,
                        request_params=config.default_request_params,
                    )

                    async with planner_agent:
                        planner = await planner_agent.attach_llm(planner_factory)

                    # Create the orchestrator with pre-configured planner
                    instance = Orchestrator(
                        name=config.name,
                        planner=planner,  # Pass pre-configured planner
                        available_agents=child_agents,
                        context=agent_app.context,
                        request_params=planner.default_request_params,  # Base params already include model settings
                        plan_type=agent_data.get(
                            "plan_type", "full"
                        ),  # Get plan_type from agent_data
                        verb=ProgressAction.PLANNING,
                    )

                elif agent_type == AgentType.EVALUATOR_OPTIMIZER:
                    # Get the referenced agents - unwrap from proxies
                    generator = self._unwrap_proxy(
                        active_agents[agent_data["generator"]]
                    )
                    evaluator = self._unwrap_proxy(
                        active_agents[agent_data["evaluator"]]
                    )

                    if not generator or not evaluator:
                        raise ValueError(
                            f"Missing agents for workflow {name}: "
                            f"generator={agent_data['generator']}, "
                            f"evaluator={agent_data['evaluator']}"
                        )

                    # TODO: Remove legacy - factory usage is only needed for str evaluators
                    # Later this should only be passed when evaluator is a string
                    optimizer_model = (
                        generator.config.model if isinstance(generator, Agent) else None
                    )
                    instance = EvaluatorOptimizerLLM(
                        generator=generator,
                        evaluator=evaluator,
                        min_rating=QualityRating[agent_data["min_rating"]],
                        max_refinements=agent_data["max_refinements"],
                        llm_factory=self._get_model_factory(model=optimizer_model),
                        context=agent_app.context,
                    )

                elif agent_type == AgentType.ROUTER:
                    # Get the router's agents - unwrap proxies
                    router_agents = self._get_agent_instances(
                        agent_data["agents"], active_agents
                    )

                    # Create the router with proper configuration
                    llm_factory = self._get_model_factory(
                        model=config.model,
                        request_params=config.default_request_params,
                    )

                    instance = LLMRouter(
                        name=config.name,
                        llm_factory=llm_factory,
                        agents=router_agents,
                        server_names=config.servers,
                        context=agent_app.context,
                        default_request_params=config.default_request_params,
                        verb=ProgressAction.ROUTING,  # Set verb for progress display
                    )

                elif agent_type == AgentType.PARALLEL:
                    # Get fan-out agents (could be basic agents or other parallels)
                    fan_out_agents = self._get_agent_instances(
                        agent_data["fan_out"], active_agents
                    )

                    # Get fan-in agent - unwrap proxy
                    fan_in_agent = self._unwrap_proxy(
                        active_agents[agent_data["fan_in"]]
                    )

                    # Create the parallel workflow
                    llm_factory = self._get_model_factory(config.model)
                    instance = ParallelLLM(
                        name=config.name,
                        instruction=config.instruction,
                        fan_out_agents=fan_out_agents,
                        fan_in_agent=fan_in_agent,
                        context=agent_app.context,
                        llm_factory=llm_factory,
                        default_request_params=config.default_request_params,
                    )

                else:
                    raise ValueError(f"Unsupported agent type: {agent_type}")

                # Create the appropriate proxy and store in results
                result_agents[name] = self._create_proxy(
                    name, instance, agent_type.value
                )

        return result_agents

    async def _create_basic_agents(self, agent_app: MCPApp) -> ProxyDict:
        """
        Create and initialize basic agents with their configurations.

        Args:
            agent_app: The main application instance

        Returns:
            Dictionary of initialized basic agents wrapped in appropriate proxies
        """
        return await self._create_agents_by_type(agent_app, AgentType.BASIC)

    async def _create_orchestrators(
        self, agent_app: MCPApp, active_agents: ProxyDict
    ) -> ProxyDict:
        """
        Create orchestrator agents.

        Args:
            agent_app: The main application instance
            active_agents: Dictionary of already created agents/proxies

        Returns:
            Dictionary of initialized orchestrator agents wrapped in appropriate proxies
        """
        return await self._create_agents_by_type(
            agent_app, AgentType.ORCHESTRATOR, active_agents
        )

    async def _create_evaluator_optimizers(
        self, agent_app: MCPApp, active_agents: ProxyDict
    ) -> ProxyDict:
        """
        Create evaluator-optimizer workflows.

        Args:
            agent_app: The main application instance
            active_agents: Dictionary of already created agents/proxies

        Returns:
            Dictionary of initialized evaluator-optimizer workflows
        """
        return await self._create_agents_by_type(
            agent_app, AgentType.EVALUATOR_OPTIMIZER, active_agents
        )

    def _get_parallel_dependencies(
        self, name: str, visited: set, path: set
    ) -> List[str]:
        """
        Get dependencies for a parallel agent in topological order.

        Args:
            name: Name of the parallel agent
            visited: Set of already visited agents
            path: Current path for cycle detection

        Returns:
            List of agent names in dependency order

        Raises:
            ValueError: If circular dependency detected
        """
        if name in path:
            path_str = " -> ".join(path)
            raise ValueError(f"Circular dependency detected: {path_str} -> {name}")

        if name in visited:
            return []

        if name not in self.agents:
            return []

        config = self.agents[name]
        if config["type"] != AgentType.PARALLEL.value:
            return []

        path.add(name)
        deps = []

        # Get dependencies from fan-out agents
        for fan_out in config["fan_out"]:
            deps.extend(self._get_parallel_dependencies(fan_out, visited, path))

        # Add this agent after its dependencies
        deps.append(name)
        visited.add(name)
        path.remove(name)

        return deps

    async def _create_parallel_agents(
        self, agent_app: MCPApp, active_agents: ProxyDict
    ) -> ProxyDict:
        """
        Create parallel execution agents in dependency order.

        Args:
            agent_app: The main application instance
            active_agents: Dictionary of already created agents/proxies

        Returns:
            Dictionary of initialized parallel agents
        """
        parallel_agents = {}
        visited = set()

        # Get all parallel agents
        parallel_names = [
            name
            for name, agent_data in self.agents.items()
            if agent_data["type"] == AgentType.PARALLEL.value
        ]

        # Create agents in dependency order
        for name in parallel_names:
            # Get ordered dependencies if not already processed
            if name not in visited:
                try:
                    ordered_agents = self._get_parallel_dependencies(
                        name, visited, set()
                    )
                except ValueError as e:
                    raise ValueError(f"Error creating parallel agent {name}: {str(e)}")

                # Create each agent in order
                for agent_name in ordered_agents:
                    if agent_name not in parallel_agents:
                        # Create one parallel agent at a time using the generic method
                        agent_result = await self._create_agents_by_type(
                            agent_app,
                            AgentType.PARALLEL,
                            active_agents,
                            agent_name=agent_name,
                        )
                        if agent_name in agent_result:
                            parallel_agents[agent_name] = agent_result[agent_name]

        return parallel_agents

    async def _create_routers(
        self, agent_app: MCPApp, active_agents: ProxyDict
    ) -> ProxyDict:
        """
        Create router agents.

        Args:
            agent_app: The main application instance
            active_agents: Dictionary of already created agents

        Returns:
            Dictionary of initialized router agents
        """
        return await self._create_agents_by_type(
            agent_app, AgentType.ROUTER, active_agents
        )

    def _unwrap_proxy(self, proxy: BaseAgentProxy) -> AgentOrWorkflow:
        """
        Unwrap a proxy to get the underlying agent or workflow instance.

        Args:
            proxy: The proxy object to unwrap

        Returns:
            The underlying Agent or workflow instance
        """
        if isinstance(proxy, LLMAgentProxy):
            return proxy._agent
        return proxy._workflow

    def _get_agent_instances(
        self, agent_names: List[str], active_agents: ProxyDict
    ) -> List[AgentOrWorkflow]:
        """
        Get list of actual agent/workflow instances from a list of names.

        Args:
            agent_names: List of agent names to look up
            active_agents: Dictionary of active agent proxies

        Returns:
            List of unwrapped agent/workflow instances
        """
        return [self._unwrap_proxy(active_agents[name]) for name in agent_names]

    @asynccontextmanager
    async def run(self):
        """
        Context manager for running the application.
        Performs validation and provides user-friendly error messages.
        """
        active_agents = {}
        had_error = False
        try:
            async with self.app.run() as agent_app:
                # Pre-flight validation
                self._validate_server_references()
                self._validate_workflow_references()

                # Create all types of agents in dependency order
                active_agents = await self._create_basic_agents(agent_app)
                orchestrators = await self._create_orchestrators(
                    agent_app, active_agents
                )
                parallel_agents = await self._create_parallel_agents(
                    agent_app, active_agents
                )
                evaluator_optimizers = await self._create_evaluator_optimizers(
                    agent_app, active_agents
                )
                routers = await self._create_routers(agent_app, active_agents)

                # Merge all agents into active_agents
                active_agents.update(orchestrators)
                active_agents.update(parallel_agents)
                active_agents.update(evaluator_optimizers)
                active_agents.update(routers)

                # Create wrapper with all agents
                wrapper = AgentApp(agent_app, active_agents)
                yield wrapper

        except ServerConfigError as e:
            had_error = True
            self._handle_error(
                e,
                "Server Configuration Error",
                "Please check your 'fastagent.config.yaml' configuration file and add the missing server definitions.",
            )
            raise SystemExit(1)

        except ProviderKeyError as e:
            had_error = True
            self._handle_error(
                e,
                "Provider Configuration Error",
                "Please check your 'fastagent.secrets.yaml' configuration file and ensure all required API keys are set.",
            )
            raise SystemExit(1)

        except AgentConfigError as e:
            had_error = True
            self._handle_error(
                e,
                "Workflow or Agent Configuration Error",
                "Please check your agent definition and ensure names and references are correct.",
            )
            raise SystemExit(1)

        except ServerInitializationError as e:
            had_error = True
            self._handle_error(
                e,
                "MCP Server Startup Error",
                "There was an error starting up the MCP Server.",
            )
            raise SystemExit(1)

        except ModelConfigError as e:
            had_error = True
            self._handle_error(
                e,
                "Model Configuration Error",
                "Common models: gpt-4o, o3-mini, sonnet, haiku. for o3, set reasoning effort with o3-mini.high",
            )
            raise SystemExit(1)

        except PromptExitError as e:
            had_error = True
            self._handle_error(
                e,
                "User requested exit",
            )
            raise SystemExit(1)

        finally:
            # Clean up any active agents without re-raising errors
            if active_agents and not had_error:
                for name, proxy in active_agents.items():
                    if isinstance(proxy, LLMAgentProxy):
                        try:
                            await proxy._agent.__aexit__(None, None, None)
                        except Exception as e:
                            print(f"DEBUG {e.message}")
                            pass  # Ignore cleanup errors

    def _handle_error(
        self, e: Exception, error_type: str, suggestion: str = None
    ) -> None:
        """
        Handle errors with consistent formatting and messaging.

        Args:
            e: The exception that was raised
            error_type: Type of error to display
            suggestion: Optional suggestion message to display
        """
        print(f"\n[bold red]{error_type}:")
        print(getattr(e, "message", str(e)))
        if hasattr(e, "details") and e.details:
            print("\nDetails:")
            print(e.details)
        if suggestion:
            print(f"\n{suggestion}")

    def _log_agent_load(self, agent_name: str) -> None:
        self.app._logger.info(
            f"Loaded {agent_name}",
            data={
                "progress_action": ProgressAction.LOADED,
                "agent_name": agent_name,
            },
        )

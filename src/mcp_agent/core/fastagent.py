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
    Literal,
)
import yaml
import argparse
from contextlib import asynccontextmanager

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.config import Settings
from mcp_agent.event_progress import ProgressAction
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.workflows.llm.model_factory import ModelFactory
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.router.router_llm import LLMRouter

from mcp_agent.core.agent_app import AgentApp
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.agent_utils import unwrap_proxy, get_agent_instances, log_agent_load
from mcp_agent.core.error_handling import handle_error
from mcp_agent.core.proxies import (
    BaseAgentProxy,
    LLMAgentProxy,
    WorkflowProxy,
    RouterProxy,
    ChainProxy,
)
from mcp_agent.core.types import AgentOrWorkflow, ProxyDict
from mcp_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ModelConfigError,
    PromptExitError,
    ServerConfigError,
    ProviderKeyError,
    ServerInitializationError,
)

# TODO -- reinstate once Windows&Python 3.13 platform issues are fixed
# import readline  # noqa: F401

from rich import print

T = TypeVar("T")  # For the wrapper classes


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
        self.args = parser.parse_args()

        # Quiet mode will be handled in _load_config()

        self.name = name
        self.config_path = config_path
        self._load_config()

        # Create the MCPApp with the config
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
            AgentType.CHAIN.value,
        ]:
            log_agent_load(self.app, name)
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
        elif agent_type == AgentType.CHAIN.value:
            # Chain proxy is directly returned from _create_agents_by_type
            # No need for type checking as it's already a ChainProxy
            return instance
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
                self.config = yaml.safe_load(f) or {}

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
                    # Check if it's a workflow type or has LLM capability
                    # Workflows like EvaluatorOptimizer and Parallel are valid for orchestrator
                    func = child_data["func"]
                    workflow_types = [
                        AgentType.EVALUATOR_OPTIMIZER.value,
                        AgentType.PARALLEL.value,
                        AgentType.ROUTER.value,
                        AgentType.CHAIN.value,
                    ]

                    if not (
                        isinstance(func, AugmentedLLM)
                        or child_data["type"] in workflow_types
                        or (hasattr(func, "_llm") and func._llm is not None)
                    ):
                        raise AgentConfigError(
                            f"Agent '{agent_name}' used by orchestrator '{name}' lacks LLM capability",
                            "All agents used by orchestrators must be LLM-capable (either an AugmentedLLM or have an _llm property)",
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

            elif agent_type == AgentType.CHAIN.value:
                # Check that all agents in the sequence exist
                sequence = agent_data.get("sequence", agent_data.get("agents", []))
                missing = [a for a in sequence if a not in available_components]
                if missing:
                    raise AgentConfigError(
                        f"Chain '{name}' references non-existent agents: {', '.join(missing)}"
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
        instruction_or_kwarg: str = None,
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
            instruction_or_kwarg: Optional positional parameter for instruction
            instruction: Base instruction for the agent (keyword arg)
            servers: List of server names the agent should connect to
            model: Model specification string (highest precedence)
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
            human_input: Whether to enable human input capabilities

        The instruction can be provided either as a second positional argument
        or as a keyword argument. Positional argument takes precedence when both are provided.

        Usage:
            @fast.agent("agent_name", "Your instruction here")  # Using positional arg
            @fast.agent("agent_name", instruction="Your instruction here")  # Using keyword arg
        """
        # Use positional argument if provided, otherwise use keyword argument
        final_instruction = (
            instruction_or_kwarg if instruction_or_kwarg is not None else instruction
        )

        decorator = self._create_decorator(
            AgentType.BASIC,
            default_name="Agent",
            default_instruction="You are a helpful agent.",
            default_use_history=True,
        )(
            name=name,
            instruction=final_instruction,
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
        max_iterations: int = 30,  # Add the max_iterations parameter with default value
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
            max_iterations: Maximum number of planning iterations (default: 10)
        """
        default_instruction = """
            You are an expert planner. Given an objective task and a list of MCP servers (which are collections of tools)
            or Agents (which are collections of servers), your job is to break down the objective into a series of steps,
            which can be performed by LLMs with access to the servers or agents.
            """

        # Handle request_params update with max_iterations
        if request_params is None:
            request_params = {"max_iterations": max_iterations}
        elif isinstance(request_params, dict):
            if "max_iterations" not in request_params:
                request_params["max_iterations"] = max_iterations

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
        fan_out: List[str],
        fan_in: Optional[str] = None,
        instruction: str = "",
        model: str | None = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
        include_request: bool = True,
    ) -> Callable:
        """
        Decorator to create and register a parallel executing agent.

        Args:
            name: Name of the parallel executing agent
            fan_out: List of parallel execution agents
            fan_in: Optional name of collecting agent. If not provided, a passthrough agent
                    will be created automatically with the name "{name}_fan_in"
            instruction: Optional instruction for the parallel agent
            model: Model specification string
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
            include_request: Whether to include the original request in the fan-in message
        """
        # If fan_in is not provided, create a passthrough agent with a derived name
        if fan_in is None:
            passthrough_name = f"{name}_fan_in"

            # Register the passthrough agent directly in self.agents
            self.agents[passthrough_name] = {
                "config": AgentConfig(
                    name=passthrough_name,
                    instruction=f"Passthrough fan-in for {name}",
                    servers=[],
                    use_history=use_history,
                ),
                "type": AgentType.BASIC.value,  # Using BASIC type since we're just attaching a PassthroughLLM
                "func": lambda x: x,  # Simple passthrough function (never actually called)
            }

            # Use this passthrough as the fan-in
            fan_in = passthrough_name

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
            include_request=include_request,
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
        instruction: Optional[str] = None,
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
            instruction: Optional instruction for the workflow (if not provided, uses generator's instruction)
        """
        decorator = self._create_decorator(
            AgentType.EVALUATOR_OPTIMIZER,
            default_instruction="",  # We'll get instruction from generator or override
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
            instruction=instruction,  # Pass through any custom instruction
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
            default_use_history=False,
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

    def chain(
        self,
        name: str = "Chain",
        *,
        sequence: List[str] = None,
        agents: List[str] = None,  # Alias for sequence
        instruction: str = None,
        model: str | None = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
        continue_with_final: bool = True,
    ) -> Callable:
        """
        Decorator to create and register a chain of agents.

        Args:
            name: Name of the chain
            sequence: List of agent names in order of execution (preferred name)
            agents: Alias for sequence (backwards compatibility)
            instruction: Optional custom instruction for the chain (if none provided, will autogenerate based on sequence)
            model: Model specification string (not used directly in chain)
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters
            continue_with_final: When using prompt(), whether to continue with the final agent after processing chain (default: True)
        """
        # Support both parameter names
        agent_sequence = sequence or agents
        if agent_sequence is None:
            raise ValueError("Either 'sequence' or 'agents' parameter must be provided")

        # Auto-generate instruction if not provided
        if instruction is None:
            # We'll generate it later when we have access to the agent configs and can see servers
            instruction = f"Chain of agents: {', '.join(agent_sequence)}"

        decorator = self._create_decorator(
            AgentType.CHAIN,
            default_name="Chain",
            default_instruction=instruction,
            default_use_history=True,
            wrapper_needed=True,
        )(
            name=name,
            sequence=agent_sequence,
            instruction=instruction,
            model=model,
            use_history=use_history,
            request_params=request_params,
            continue_with_final=continue_with_final,
        )
        return decorator

    def passthrough(
        self, name: str = "Passthrough", use_history: bool = True, **kwargs
    ) -> Callable:
        """
        Decorator to create and register a passthrough agent.
        A passthrough agent simply returns any input message without modification.

        This is useful for parallel workflows where no fan-in aggregation is needed
        (the fan-in agent can be a passthrough that simply returns the combined outputs).

        Args:
            name: Name of the passthrough agent
            use_history: Whether to maintain conversation history
            **kwargs: Additional parameters (ignored, for compatibility)
        """
        decorator = self._create_decorator(
            AgentType.BASIC,  # Using BASIC agent type since we'll use a regular agent with PassthroughLLM
            default_name="Passthrough",
            default_instruction="Passthrough agent that returns input without modification",
            default_use_history=use_history,
            wrapper_needed=True,
        )(
            name=name,
            use_history=use_history,
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
                    # Get the agent name for special handling
                    agent_name = agent_data["config"].name

                    # Check if this is an agent that should use the PassthroughLLM
                    if agent_name.endswith("_fan_in") or agent_name.startswith(
                        "passthrough"
                    ):
                        # Import here to avoid circular imports
                        from mcp_agent.workflows.llm.augmented_llm import PassthroughLLM

                        # Create basic agent with configuration
                        agent = Agent(config=config, context=agent_app.context)

                        # Set up a PassthroughLLM directly
                        async with agent:
                            agent._llm = PassthroughLLM(
                                name=f"{config.name}_llm",
                                context=agent_app.context,
                                agent=agent,
                                default_request_params=config.default_request_params,
                            )

                        # Store the agent
                        instance = agent
                    else:
                        # Standard basic agent with LLM
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

                    optimizer_model = (
                        generator.config.model if isinstance(generator, Agent) else None
                    )

                    instance = EvaluatorOptimizerLLM(
                        name=config.name,  # Pass name from config
                        generator=generator,
                        evaluator=evaluator,
                        min_rating=QualityRating[agent_data["min_rating"]],
                        max_refinements=agent_data["max_refinements"],
                        llm_factory=self._get_model_factory(model=optimizer_model),
                        context=agent_app.context,
                        instruction=config.instruction,  # Pass any custom instruction
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

                elif agent_type == AgentType.CHAIN:
                    # Get the sequence from either parameter
                    sequence = agent_data.get("sequence", agent_data.get("agents", []))

                    # Auto-generate instruction if not provided or if it's just the default
                    default_instruction = f"Chain of agents: {', '.join(sequence)}"

                    # If user provided a custom instruction, use that
                    # Otherwise, generate a description based on the sequence and their servers
                    if config.instruction == default_instruction:
                        # Get all agent names in the sequence
                        agent_names = []
                        all_servers = set()

                        # Collect information about the agents and their servers
                        for agent_name in sequence:
                            if agent_name in active_agents:
                                agent_proxy = active_agents[agent_name]
                                if hasattr(agent_proxy, "_agent"):
                                    # For LLMAgentProxy
                                    agent_instance = agent_proxy._agent
                                    agent_names.append(agent_name)
                                    if hasattr(agent_instance, "server_names"):
                                        all_servers.update(agent_instance.server_names)
                                elif hasattr(agent_proxy, "_workflow"):
                                    # For WorkflowProxy
                                    agent_names.append(agent_name)

                        # Generate a better description
                        if agent_names:
                            server_part = (
                                f" with access to servers: {', '.join(sorted(all_servers))}"
                                if all_servers
                                else ""
                            )
                            config.instruction = f"Sequence of agents: {', '.join(agent_names)}{server_part}."

                    # Create a ChainProxy without needing a new instance
                    # Just pass the agent proxies and sequence
                    instance = ChainProxy(self.app, name, sequence, active_agents)
                    # Set continue_with_final behavior from configuration
                    instance._continue_with_final = agent_data.get(
                        "continue_with_final", True
                    )

                # We removed the AgentType.PASSTHROUGH case
                # Passthrough agents are now created as BASIC agents with a special LLM

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
                        include_request=agent_data.get("include_request", True),
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

    def _get_dependencies(
        self, name: str, visited: set, path: set, agent_type: AgentType = None
    ) -> List[str]:
        """
        Get dependencies for an agent in topological order.
        Works for both Parallel and Chain workflows.

        Args:
            name: Name of the agent
            visited: Set of already visited agents
            path: Current path for cycle detection
            agent_type: Optional type filter (e.g., only check Parallel or Chain)

        Returns:
            List of agent names in dependency order

        Raises:
            ValueError: If circular dependency detected
        """
        if name in path:
            path_str = " -> ".join(path)
            raise CircularDependencyError(f"Path: {path_str} -> {name}")

        if name in visited:
            return []

        if name not in self.agents:
            return []

        config = self.agents[name]

        # Skip if not the requested type (when filtering)
        if agent_type and config["type"] != agent_type.value:
            return []

        path.add(name)
        deps = []

        # Handle dependencies based on agent type
        if config["type"] == AgentType.PARALLEL.value:
            # Get dependencies from fan-out agents
            for fan_out in config["fan_out"]:
                deps.extend(self._get_dependencies(fan_out, visited, path, agent_type))
        elif config["type"] == AgentType.CHAIN.value:
            # Get dependencies from sequence agents
            sequence = config.get("sequence", config.get("agents", []))
            for agent_name in sequence:
                deps.extend(
                    self._get_dependencies(agent_name, visited, path, agent_type)
                )

        # Add this agent after its dependencies
        deps.append(name)
        visited.add(name)
        path.remove(name)

        return deps

    def _get_parallel_dependencies(
        self, name: str, visited: set, path: set
    ) -> List[str]:
        """
        Get dependencies for a parallel agent in topological order.
        Legacy function that calls the more general _get_dependencies.

        Args:
            name: Name of the parallel agent
            visited: Set of already visited agents
            path: Current path for cycle detection

        Returns:
            List of agent names in dependency order

        Raises:
            ValueError: If circular dependency detected
        """
        return self._get_dependencies(name, visited, path, AgentType.PARALLEL)

    async def _create_agents_in_dependency_order(
        self, agent_app: MCPApp, active_agents: ProxyDict, agent_type: AgentType
    ) -> ProxyDict:
        """
        Create agents in dependency order to avoid circular references.
        Works for both Parallel and Chain workflows.

        Args:
            agent_app: The main application instance
            active_agents: Dictionary of already created agents/proxies
            agent_type: Type of agents to create (AgentType.PARALLEL or AgentType.CHAIN)

        Returns:
            Dictionary of initialized agents
        """
        result_agents = {}
        visited = set()

        # Get all agents of the specified type
        agent_names = [
            name
            for name, agent_data in self.agents.items()
            if agent_data["type"] == agent_type.value
        ]

        # Create agents in dependency order
        for name in agent_names:
            # Get ordered dependencies if not already processed
            if name not in visited:
                try:
                    ordered_agents = self._get_dependencies(
                        name, visited, set(), agent_type
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Error creating {agent_type.name.lower()} agent {name}: {str(e)}"
                    )

                # Create each agent in order
                for agent_name in ordered_agents:
                    if agent_name not in result_agents:
                        # Create one agent at a time using the generic method
                        agent_result = await self._create_agents_by_type(
                            agent_app,
                            agent_type,
                            active_agents,
                            agent_name=agent_name,
                        )
                        if agent_name in agent_result:
                            result_agents[agent_name] = agent_result[agent_name]

        return result_agents

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
        return await self._create_agents_in_dependency_order(
            agent_app, active_agents, AgentType.PARALLEL
        )

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
        return unwrap_proxy(proxy)

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
        return get_agent_instances(agent_names, active_agents)

    @asynccontextmanager
    async def run(self):
        """
        Context manager for running the application.
        Performs validation and provides user-friendly error messages.
        """
        active_agents = {}
        had_error = False

        # Handle quiet mode by disabling logger settings after initialization
        quiet_mode = hasattr(self, "args") and self.args.quiet

        try:
            async with self.app.run() as agent_app:
                # Apply quiet mode directly to the context's config if needed
                if (
                    quiet_mode
                    and hasattr(agent_app.context, "config")
                    and hasattr(agent_app.context.config, "logger")
                ):
                    # Apply after initialization but before agents are created
                    agent_app.context.config.logger.progress_display = False
                    agent_app.context.config.logger.show_chat = False
                    agent_app.context.config.logger.show_tools = False

                    # Directly disable the progress display singleton
                    from mcp_agent.progress_display import progress_display

                    progress_display.stop()  # This will stop and hide the display

                # Pre-flight validation
                self._validate_server_references()
                self._validate_workflow_references()

                # Create all types of agents in dependency order
                # First create basic agents
                active_agents = await self._create_basic_agents(agent_app)

                # Create workflow types that don't depend on other workflows first
                evaluator_optimizers = await self._create_evaluator_optimizers(
                    agent_app, active_agents
                )
                active_agents.update(evaluator_optimizers)

                # Create parallel agents next as they might be dependencies
                parallel_agents = await self._create_parallel_agents(
                    agent_app, active_agents
                )
                active_agents.update(parallel_agents)

                # Create routers next
                routers = await self._create_routers(agent_app, active_agents)
                active_agents.update(routers)

                # Create chains next
                chains = await self._create_agents_in_dependency_order(
                    agent_app, active_agents, AgentType.CHAIN
                )
                active_agents.update(chains)

                # Create orchestrators last as they might depend on any other agent type
                orchestrators = await self._create_orchestrators(
                    agent_app, active_agents
                )

                # Add orchestrators to active_agents (other types were already added)
                active_agents.update(orchestrators)

                # Create wrapper with all agents
                wrapper = AgentApp(agent_app, active_agents)

                # Handle direct message sending if --agent and --message are provided
                if self.args.agent and self.args.message:
                    agent_name = self.args.agent
                    message = self.args.message

                    if agent_name not in active_agents:
                        available_agents = ", ".join(active_agents.keys())
                        print(
                            f"\n\nError: Agent '{agent_name}' not found. Available agents: {available_agents}"
                        )
                        raise SystemExit(1)

                    try:
                        # Get response
                        response = await wrapper[agent_name].send(message)

                        # Only print the response in quiet mode
                        if self.args.quiet:
                            print(f"{response}")

                        raise SystemExit(0)
                    except Exception as e:
                        print(
                            f"\n\nError sending message to agent '{agent_name}': {str(e)}"
                        )
                        raise SystemExit(1)

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

        except CircularDependencyError as e:
            had_error = True
            self._handle_error(
                e,
                "Circular Dependency Detected",
                "Check your agent configuration for circular dependencies.",
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
        handle_error(e, error_type, suggestion)

    def _log_agent_load(self, agent_name: str) -> None:
        # Using the imported function
        log_agent_load(self.app, agent_name)

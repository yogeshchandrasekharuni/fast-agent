"""
Decorator-based interface for MCP Agent applications.
Provides a simplified way to create and manage agents using decorators.
"""

from typing import List, Optional, Dict, Callable, TypeVar, Any, Union, TypeAlias
from enum import Enum
import yaml
import argparse
from contextlib import asynccontextmanager

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from mcp_agent.workflows.router.router_llm import LLMRouter
from mcp_agent.config import Settings
from rich.prompt import Prompt
from rich import print
from mcp_agent.progress_display import progress_display
from mcp_agent.workflows.llm.model_factory import ModelFactory
from mcp_agent.workflows.llm.augmented_llm import RequestParams

import readline  # noqa: F401


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
            return await top_result.result._llm.generate_str(message)
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
        Interactive prompt for sending messages.

        Args:
            agent_name: Optional target agent name (uses default if not specified)
            default_prompt: Default message to use when user presses enter
        """

        agent = agent_name or self._default

        if agent not in self._agents:
            raise ValueError(f"No agent named '{agent}'")
        result = ""
        while True:
            with progress_display.paused():
                if default == "STOP":
                    print("Press <ENTER> to finish.")
                elif default != "":
                    print("Enter a prompt, or [red]STOP[/red] to finish.")
                    print(
                        f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]"
                    )
                else:
                    print("Enter a prompt, or [red]STOP[/red] to finish")

                prompt_text = f"[blue]{agent}[/blue] >"
                user_input = Prompt.ask(
                    prompt=prompt_text, default=default, show_default=False
                )
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

    @property
    def context(self):
        """Access the application context"""
        return self.app.context

    def _load_config(self) -> None:
        """Load configuration from YAML file, properly handling without dotenv processing"""
        if self.config_path:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)

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

    def agent(
        self,
        name: str = "Agent",
        *,
        instruction: str = "You are a helpful agent.",
        servers: List[str] = [],
        model: Optional[str] = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
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
        """
        #        print(f"\nDecorating agent {name} with model={model}")

        def decorator(func: Callable) -> Callable:
            base_params = RequestParams(
                use_history=use_history,
                model=model,  # Include model in initial params
                maxTokens=4096,  # Default to larger context for agents TODO configurations
                **(request_params or {}),
            )

            # Create agent configuration
            config = AgentConfig(
                name=name,
                instruction=instruction,
                servers=servers,
                model=model,  # Highest precedence
                use_history=use_history,
                default_request_params=base_params,
            )

            # Store the agent configuration
            self.agents[name] = {
                "config": config,
                "type": AgentType.BASIC.value,
                "func": func,
            }

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def orchestrator(
        self,
        name: str,
        instruction: str,
        agents: List[str],
        model: str | None = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
    ) -> Callable:
        """
        Decorator to create and register an orchestrator.

        Args:
            name: Name of the orchestrator
            instruction: Base instruction for the orchestrator
            agents: List of agent names this orchestrator can use
            model: Model specification string (highest precedence)
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
        """

        def decorator(func: Callable) -> Callable:
            # Create base request params
            base_params = RequestParams(
                use_history=use_history, **(request_params or {})
            )

            # Create agent configuration
            config = AgentConfig(
                name=name,
                instruction=instruction,
                servers=[],  # Orchestrators don't need servers
                model=model,  # Highest precedence
                use_history=use_history,
                default_request_params=base_params,
            )

            # Store the orchestrator configuration
            self.agents[name] = {
                "config": config,
                "child_agents": agents,
                "type": AgentType.ORCHESTRATOR.value,
                "func": func,
            }

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

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

        def decorator(func: Callable) -> Callable:
            # Create request params with history setting
            params = RequestParams(**(request_params or {}))
            params.use_history = use_history

            # Create agent configuration
            config = AgentConfig(
                name=name,
                instruction=instruction,
                servers=[],  # Parallel agents don't need servers
                model=model,
                use_history=use_history,
                default_request_params=params,
            )

            # Store the parallel configuration
            self.agents[name] = {
                "config": config,
                "fan_out": fan_out,
                "fan_in": fan_in,
                "type": AgentType.PARALLEL.value,
                "func": func,
            }

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def evaluator_optimizer(
        self,
        name: str,
        optimizer: str,
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
            optimizer: Name of the optimizer agent
            evaluator: Name of the evaluator agent
            min_rating: Minimum acceptable quality rating (EXCELLENT, GOOD, FAIR, POOR)
            max_refinements: Maximum number of refinement iterations
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
        """

        def decorator(func: Callable) -> Callable:
            # Create workflow configuration
            config = AgentConfig(
                name=name,
                instruction="",  # Uses optimizer's instruction
                servers=[],  # Uses agents' server access
                use_history=use_history,
                default_request_params=request_params,
            )

            # Store the workflow configuration
            self.agents[name] = {
                "config": config,
                "optimizer": optimizer,
                "evaluator": evaluator,
                "min_rating": min_rating,
                "max_refinements": max_refinements,
                "type": AgentType.EVALUATOR_OPTIMIZER.value,
                "func": func,
            }

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def router(
        self,
        name: str,
        agents: List[str],
        servers: List[str] = [],
        model: Optional[str] = None,
        use_history: bool = True,
        request_params: Optional[Dict] = None,
    ) -> Callable:
        """
        Decorator to create and register a router.

        Args:
            name: Name of the router
            agents: List of agent names this router can delegate to
            servers: List of server names the router can use directly
            model: Model specification string
            use_history: Whether to maintain conversation history
            request_params: Additional request parameters for the LLM
        """

        def decorator(func: Callable) -> Callable:
            # Create base request params
            base_params = RequestParams(
                use_history=use_history, **(request_params or {})
            )

            # Create agent configuration
            config = AgentConfig(
                name=name,
                instruction="",  # Router uses its own routing instruction
                servers=servers,
                model=model,
                use_history=use_history,
                default_request_params=base_params,
            )

            # Store the router configuration
            self.agents[name] = {
                "config": config,
                "agents": agents,
                "type": AgentType.ROUTER.value,
                "func": func,
            }

            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    async def _create_basic_agents(self, agent_app: MCPApp) -> ProxyDict:
        """
        Create and initialize basic agents with their configurations.

        Args:
            agent_app: The main application instance

        Returns:
            Dictionary of initialized basic agents wrapped in appropriate proxies
        """
        active_agents = {}

        for name, agent_data in self.agents.items():
            if agent_data["type"] == AgentType.BASIC.value:
                config = agent_data["config"]

                # Create agent with configuration
                agent = Agent(config=config, context=agent_app.context)

                # Set up LLM with proper configuration
                async with agent:
                    llm_factory = self._get_model_factory(
                        model=config.model,
                        request_params=config.default_request_params,
                    )
                    agent._llm = await agent.attach_llm(llm_factory)

                # Create proxy for the agent
                active_agents[name] = self._create_proxy(
                    name, agent, AgentType.BASIC.value
                )

        return active_agents

    def _create_orchestrators(
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
        orchestrators = {}
        for name, agent_data in self.agents.items():
            if agent_data["type"] == AgentType.ORCHESTRATOR.value:
                config = agent_data["config"]

                # TODO: Remove legacy - This model/params setup should be in Agent class
                # Resolve model alias if present
                model_config = ModelFactory.parse_model_string(config.model)
                resolved_model = model_config.model_name

                # Start with existing params if available
                if config.default_request_params:
                    base_params = config.default_request_params.model_copy()
                    # Update with orchestrator-specific settings
                    base_params.use_history = config.use_history
                    base_params.model = resolved_model
                else:
                    base_params = RequestParams(
                        use_history=config.use_history, model=resolved_model
                    )

                llm_factory = self._get_model_factory(
                    model=config.model,  # Use original model string for factory creation
                    request_params=base_params,
                )

                # Get the child agents - need to unwrap proxies
                child_agents = []
                for agent_name in agent_data["child_agents"]:
                    proxy = active_agents[agent_name]
                    if isinstance(proxy, LLMAgentProxy):
                        child_agents.append(proxy._agent)  # Get the actual Agent
                    else:
                        # Handle case where it might be another workflow
                        child_agents.append(proxy._workflow)

                orchestrator = Orchestrator(
                    name=config.name,
                    instruction=config.instruction,
                    available_agents=child_agents,
                    context=agent_app.context,
                    llm_factory=llm_factory,
                    request_params=base_params,  # Use our base params that include model
                    plan_type="full",
                )

                # Use factory to create appropriate proxy
                orchestrators[name] = self._create_proxy(
                    name, orchestrator, AgentType.ORCHESTRATOR.value
                )
        return orchestrators

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
        workflows = {}
        for name, agent_data in self.agents.items():
            if agent_data["type"] == AgentType.EVALUATOR_OPTIMIZER.value:
                # Get the referenced agents - unwrap from proxies
                optimizer = self._unwrap_proxy(active_agents[agent_data["optimizer"]])
                evaluator = self._unwrap_proxy(active_agents[agent_data["evaluator"]])

                if not optimizer or not evaluator:
                    raise ValueError(
                        f"Missing agents for workflow {name}: "
                        f"optimizer={agent_data['optimizer']}, "
                        f"evaluator={agent_data['evaluator']}"
                    )

                # TODO: Remove legacy - factory usage is only needed for str evaluators
                # Later this should only be passed when evaluator is a string
                optimizer_model = (
                    optimizer.config.model if isinstance(optimizer, Agent) else None
                )
                workflow = EvaluatorOptimizerLLM(
                    optimizer=optimizer,
                    evaluator=evaluator,
                    min_rating=QualityRating[agent_data["min_rating"]],
                    max_refinements=agent_data["max_refinements"],
                    llm_factory=self._get_model_factory(model=optimizer_model),
                    context=agent_app.context,
                )

                workflows[name] = self._create_proxy(
                    name, workflow, AgentType.EVALUATOR_OPTIMIZER.value
                )

        return workflows

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

    def _create_parallel_agents(
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
                        agent_data = self.agents[agent_name]
                        config = agent_data["config"]

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
                        parallel = ParallelLLM(
                            name=config.name,
                            instruction=config.instruction,
                            fan_out_agents=fan_out_agents,
                            fan_in_agent=fan_in_agent,
                            context=agent_app.context,
                            llm_factory=llm_factory,
                            default_request_params=config.default_request_params,
                        )

                        parallel_agents[agent_name] = self._create_proxy(
                            name, parallel, AgentType.PARALLEL.value
                        )

        return parallel_agents

    def _create_routers(self, agent_app: MCPApp, active_agents: ProxyDict) -> ProxyDict:
        """
        Create router agents.

        Args:
            agent_app: The main application instance
            active_agents: Dictionary of already created agents

        Returns:
            Dictionary of initialized router agents
        """
        routers = {}
        for name, agent_data in self.agents.items():
            if agent_data["type"] == AgentType.ROUTER.value:
                config = agent_data["config"]

                # Get the router's agents - unwrap proxies
                router_agents = self._get_agent_instances(
                    agent_data["agents"], active_agents
                )

                # Create the router with proper configuration
                llm_factory = self._get_model_factory(
                    model=config.model,
                    request_params=config.default_request_params,
                )

                router = LLMRouter(
                    name=config.name,  # Add the name parameter
                    llm_factory=llm_factory,
                    agents=router_agents,
                    server_names=config.servers,
                    context=agent_app.context,
                    default_request_params=config.default_request_params,
                )

                routers[name] = self._create_proxy(name, router, AgentType.ROUTER.value)

        return routers

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
        Handles setup and teardown of the app and agents.

        Yields:
            AgentAppWrapper instance with all initialized agents
        """
        async with self.app.run() as agent_app:
            # Create all types of agents
            active_agents = await self._create_basic_agents(agent_app)
            orchestrators = self._create_orchestrators(agent_app, active_agents)
            parallel_agents = self._create_parallel_agents(agent_app, active_agents)
            evaluator_optimizers = await self._create_evaluator_optimizers(
                agent_app, active_agents
            )
            routers = self._create_routers(agent_app, active_agents)

            # Merge all agents into active_agents
            active_agents.update(orchestrators)
            active_agents.update(parallel_agents)
            active_agents.update(evaluator_optimizers)
            active_agents.update(routers)

            # Create wrapper with all agents
            wrapper = AgentApp(agent_app, active_agents)
            try:
                yield wrapper
            finally:
                # Clean up basic agents - need to get the actual agent from the proxy
                for name, proxy in active_agents.items():
                    if isinstance(proxy, LLMAgentProxy):
                        await proxy._agent.__aexit__(None, None, None)

        # async def send(self, agent_name: str, message: str) -> Any:
        # """
        # Send a message to a specific agent and get the response.

        # Args:
        #     agent_name: Name of the target agent
        #     message: Message to send

        # Returns:
        #     Agent's response

        # Raises:
        #     ValueError: If agent not found
        #     RuntimeError: If agent has no LLM attached
        # """
        # if agent_name not in self.agents:
        #     raise ValueError(f"Agent {agent_name} not found")

        # agent = self.agents[agent_name]

        # # Special handling for routers
        # if isinstance(agent._llm, LLMRouter):
        #     # Route the message and get results
        #     results = await agent._llm.route(message)
        #     if not results:
        #         return "No appropriate route found for the request."

        #     # Get the top result
        #     top_result = results[0]
        #     if isinstance(top_result.result, Agent):
        #         # Agent route - delegate to the agent
        #         return await top_result.result._llm.generate_str(message)
        #     elif isinstance(top_result.result, str):
        #         # Server route - use the router directly
        #         return "Tool call requested by router - not yet supported"
        #     else:
        #         return f"Routed to: {top_result.result} ({top_result.confidence}): {top_result.reasoning}"

        # # Normal agent handling
        # if not hasattr(agent, "_llm") or agent._llm is None:
        #     raise RuntimeError(f"Agent {agent_name} has no LLM attached")

        # return await agent._llm.generate_str(message)

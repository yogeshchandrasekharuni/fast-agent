"""
Decorators for FastAgent applications.
Contains decorator definitions extracted from fastagent.py.
"""

from typing import Callable, Dict, List, Optional, TypeVar, Literal
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.workflows.llm.augmented_llm import RequestParams

T = TypeVar("T")  # For the wrapper classes


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
                model="passthrough",
                instruction=f"This agent combines the results from the fan-out agents verbatim. {name}",
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
    #    servers: List[str] = [],
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
    cumulative: bool = False,
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
        cumulative: When True, each agent receives all previous agent responses concatenated (default: False)
                    When False, each agent only gets the output of the previous agent (default behavior)
    """
    # Support both parameter names
    agent_sequence = sequence or agents
    if agent_sequence is None:
        raise ValueError("Either 'sequence' or 'agents' parameter must be provided")

    # Auto-generate instruction if not provided
    if instruction is None:
        # Generate an appropriate instruction based on mode
        if cumulative:
            instruction = f"Cumulative chain of agents: {', '.join(agent_sequence)}"
        else:
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
        cumulative=cumulative,
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

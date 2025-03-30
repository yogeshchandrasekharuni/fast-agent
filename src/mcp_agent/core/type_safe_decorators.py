"""
Enhanced type-safe decorator system for FastAgent applications.
This module provides type-safe decorators for creating agents with improved type annotations
while preserving backward compatibility with the existing API.
"""

import inspect
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from mcp_agent.agents.agent import AgentConfig
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.request_params import RequestParams

# Type variables for the decorated function
P = ParamSpec("P")  # Parameters
R = TypeVar("R")  # Return type
T = TypeVar("T")  # Generic type parameter

# Type for agent function - can be async or sync
AgentCallable = Callable[P, Union[Awaitable[R], R]]


# Protocol for decorated agent functions
class DecoratedAgentProtocol(Protocol[P, R]):
    """Protocol defining the interface of a decorated agent function."""

    _agent_type: AgentType
    _agent_config: AgentConfig

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Union[Awaitable[R], R]: ...


# Protocol for orchestrator functions with additional metadata
class DecoratedOrchestratorProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated orchestrator functions with additional metadata."""

    _child_agents: List[str]
    _plan_type: Literal["full", "iterative"]


# Protocol for router functions
class DecoratedRouterProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated router functions with additional metadata."""

    _router_agents: List[str]
    _router_type: Literal["llm", "embedding"]


# Protocol for chain functions
class DecoratedChainProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated chain functions with additional metadata."""

    _chain_agents: List[str]


# Protocol for parallel functions
class DecoratedParallelProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated parallel functions with additional metadata."""

    _parallel_agents: List[str]


# Type-safe request parameters dictionary
class RequestParamsDict(TypedDict, total=False):
    """Type-safe dictionary for request parameters."""

    temperature: float
    maxTokens: int
    topP: float
    topK: int
    frequencyPenalty: float
    presencePenalty: float
    stopSequences: List[str]
    systemPrompt: str
    model: Optional[str]
    use_history: bool
    max_iterations: int
    parallel_tool_calls: bool


# Base decorator implementation
def _create_agent_decorator(
    agent_type: AgentType,
    name: str,
    instruction: str,
    *,
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
    **additional_metadata: Any,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Base implementation for agent decorators.

    Args:
        agent_type: Type of agent to create
        name: Name of the agent
        instruction: Base instruction for the agent
        servers: List of server names the agent should connect to
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        additional_metadata: Additional metadata to attach to the agent

    Returns:
        A decorator that registers the agent with appropriate metadata
    """

    def decorator(func: AgentCallable[P, R]) -> DecoratedAgentProtocol[P, R]:
        is_async = inspect.iscoroutinefunction(func)

        # Handle both async and sync functions consistently
        if is_async:

            @wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Create the agent configuration
                config = AgentConfig(
                    name=name,
                    instruction=instruction,
                    servers=servers,
                    model=model,
                    use_history=use_history,
                    human_input=human_input,
                )

                # Handle request params
                if request_params:
                    if isinstance(request_params, RequestParams):
                        config.default_request_params = request_params
                    else:
                        config.default_request_params = RequestParams(**request_params)

                # Execute the original function
                return await func(*args, **kwargs)
        else:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Create the agent configuration
                config = AgentConfig(
                    name=name,
                    instruction=instruction,
                    servers=servers,
                    model=model,
                    use_history=use_history,
                    human_input=human_input,
                )

                # Handle request params
                if request_params:
                    if isinstance(request_params, RequestParams):
                        config.default_request_params = request_params
                    else:
                        config.default_request_params = RequestParams(**request_params)

                # Execute the original function
                return func(*args, **kwargs)

        # Store metadata on the wrapper function
        setattr(wrapper, "_agent_type", agent_type)
        setattr(
            wrapper,
            "_agent_config",
            AgentConfig(
                name=name,
                instruction=instruction,
                servers=servers,
                model=model,
                use_history=use_history,
                human_input=human_input,
            ),
        )

        # Add additional metadata
        for key, value in additional_metadata.items():
            setattr(wrapper, key, value)

        # Return the wrapped function with proper type annotation
        return cast(DecoratedAgentProtocol[P, R], wrapper)

    return decorator


# Enhanced agent decorator with proper typing
def agent(
    name: str,
    instruction: str,
    *,
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Decorator to create and register a standard agent.

    Args:
        name: Name of the agent
        instruction: Base instruction for the agent
        servers: List of server names the agent should connect to
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities

    Returns:
        A decorator that registers the agent with proper type annotations
    """
    return _create_agent_decorator(
        AgentType.BASIC,
        name=name,
        instruction=instruction,
        servers=servers,
        model=model,
        use_history=use_history,
        request_params=request_params,
        human_input=human_input,
    )


# Enhanced orchestrator decorator with proper typing
def orchestrator(
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
    plan_type: Literal["full", "iterative"] = "full",
    max_iterations: int = 30,
) -> Callable[[AgentCallable[P, R]], DecoratedOrchestratorProtocol[P, R]]:
    """
    Decorator to create and register an orchestrator agent.

    Args:
        name: Name of the orchestrator
        agents: List of agent names this orchestrator can use
        instruction: Base instruction for the orchestrator
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        plan_type: Planning approach - "full" or "iterative"
        max_iterations: Maximum number of planning iterations

    Returns:
        A decorator that registers the orchestrator with proper type annotations
    """
    default_instruction = """
    You are an expert planner. Given an objective task and a list of Agents 
    (which are collections of capabilities), your job is to break down the objective 
    into a series of steps, which can be performed by these agents.
    """

    final_request_params: Dict[str, Any] = {}
    if request_params:
        if isinstance(request_params, RequestParams):
            final_request_params = request_params.model_dump()
        else:
            final_request_params = dict(request_params)

    final_request_params["max_iterations"] = max_iterations

    return cast(
        Callable[[AgentCallable[P, R]], DecoratedOrchestratorProtocol[P, R]],
        _create_agent_decorator(
            AgentType.ORCHESTRATOR,
            name=name,
            instruction=instruction or default_instruction,
            servers=[],  # Orchestrators don't directly connect to servers
            model=model,
            use_history=use_history,
            request_params=final_request_params,
            human_input=human_input,
            _child_agents=agents,
            _plan_type=plan_type,
        ),
    )


# Enhanced router decorator with proper typing
def router(
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
    router_type: Literal["llm", "embedding"] = "llm",
) -> Callable[[AgentCallable[P, R]], DecoratedRouterProtocol[P, R]]:
    """
    Decorator to create and register a router agent.

    Args:
        name: Name of the router
        agents: List of agent names this router can route to
        instruction: Base instruction for the router
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        router_type: Routing approach - "llm" or "embedding"

    Returns:
        A decorator that registers the router with proper type annotations
    """
    default_instruction = """
    You are a router that determines which specialized agent should handle a given query.
    Analyze the query and select the most appropriate agent to handle it.
    """

    return cast(
        Callable[[AgentCallable[P, R]], DecoratedRouterProtocol[P, R]],
        _create_agent_decorator(
            AgentType.ROUTER,
            name=name,
            instruction=instruction or default_instruction,
            servers=[],  # Routers don't directly connect to servers
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            _router_agents=agents,
            _router_type=router_type,
        ),
    )


# Enhanced chain decorator with proper typing
def chain(
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedChainProtocol[P, R]]:
    """
    Decorator to create and register a chain agent.

    Args:
        name: Name of the chain
        agents: List of agent names in the chain, executed in sequence
        instruction: Base instruction for the chain
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities

    Returns:
        A decorator that registers the chain with proper type annotations
    """
    default_instruction = """
    You are a chain that processes requests through a series of specialized agents in sequence.
    Pass the output of each agent to the next agent in the chain.
    """

    return cast(
        Callable[[AgentCallable[P, R]], DecoratedChainProtocol[P, R]],
        _create_agent_decorator(
            AgentType.CHAIN,
            name=name,
            instruction=instruction or default_instruction,
            servers=[],  # Chains don't directly connect to servers
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            _chain_agents=agents,
        ),
    )


# Enhanced parallel decorator with proper typing
def parallel(
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedParallelProtocol[P, R]]:
    """
    Decorator to create and register a parallel agent.

    Args:
        name: Name of the parallel agent
        agents: List of agent names to execute in parallel
        instruction: Base instruction for the parallel agent
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities

    Returns:
        A decorator that registers the parallel agent with proper type annotations
    """
    default_instruction = """
    You are a parallel processor that executes multiple agents simultaneously 
    and aggregates their results.
    """

    return cast(
        Callable[[AgentCallable[P, R]], DecoratedParallelProtocol[P, R]],
        _create_agent_decorator(
            AgentType.PARALLEL,
            name=name,
            instruction=instruction or default_instruction,
            servers=[],  # Parallel agents don't directly connect to servers
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            _parallel_agents=agents,
        ),
    )


# Enhanced evaluator_optimizer decorator with proper typing
def evaluator_optimizer(
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: Optional[Union[RequestParams, RequestParamsDict]] = None,
    human_input: bool = False,
    optimization_rounds: int = 3,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Decorator to create and register an evaluator-optimizer agent.

    Args:
        name: Name of the evaluator-optimizer
        agents: List of agent names this evaluator-optimizer will evaluate/optimize
        instruction: Base instruction for the evaluator-optimizer
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        optimization_rounds: Number of optimization iterations to perform

    Returns:
        A decorator that registers the evaluator-optimizer with proper type annotations
    """
    default_instruction = """
    You are an evaluator and optimizer. Your job is to evaluate the outputs 
    from other agents and suggest improvements to optimize their performance.
    """

    final_request_params: Dict[str, Any] = {}
    if request_params:
        if isinstance(request_params, RequestParams):
            final_request_params = request_params.model_dump()
        else:
            final_request_params = dict(request_params)

    final_request_params["optimization_rounds"] = optimization_rounds

    return _create_agent_decorator(
        AgentType.EVALUATOR_OPTIMIZER,
        name=name,
        instruction=instruction or default_instruction,
        servers=[],
        model=model,
        use_history=use_history,
        request_params=final_request_params,
        human_input=human_input,
        _eval_optimizer_agents=agents,
        _optimization_rounds=optimization_rounds,
    )

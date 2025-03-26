"""
Simplified decorator system for FastAgent applications.
This module provides type-safe decorators for creating agents without the proxy layer.
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, cast

from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.interfaces import AgentProtocol, AugmentedLLMProtocol

# Type variable for the decorated function
F = TypeVar('F', bound=Callable[..., Any])

def agent(
    name: str, 
    instruction: str,
    *,
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: Optional[Dict[str, Any]] = None,
    human_input: bool = False,
) -> Callable[[F], F]:
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
        A decorator that registers the agent
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create the agent configuration
            config = AgentConfig(
                name=name,
                instruction=instruction,
                servers=servers,
                model=model,
                use_history=use_history,
                human_input=human_input
            )
            
            # Create request params if provided
            if request_params:
                config.default_request_params = RequestParams(**request_params)
                
            # Initialize the agent
            # Note: This is where agent creation logic would go
            # For now, we return a placeholder to illustrate the approach
            return f"Agent {name} created with model {model}"
            
        # Store metadata on the wrapper function
        setattr(wrapper, "_agent_type", AgentType.BASIC)
        setattr(wrapper, "_agent_config", AgentConfig(
            name=name,
            instruction=instruction,
            servers=servers,
            model=model,
            use_history=use_history,
            human_input=human_input
        ))
        
        # Return the wrapped function
        return cast(F, wrapper)
    
    return decorator


def orchestrator(
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: Optional[Dict[str, Any]] = None,
    human_input: bool = False,
    plan_type: Literal["full", "iterative"] = "full",
    max_iterations: int = 30,
) -> Callable[[F], F]:
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
        A decorator that registers the orchestrator
    """
    default_instruction = """
    You are an expert planner. Given an objective task and a list of Agents 
    (which are collections of capabilities), your job is to break down the objective 
    into a series of steps, which can be performed by these agents.
    """
    
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create final request params
            final_request_params = request_params or {}
            final_request_params["max_iterations"] = max_iterations
            
            # Create the agent configuration
            config = AgentConfig(
                name=name,
                instruction=instruction or default_instruction,
                servers=[],  # Orchestrators don't directly connect to servers
                model=model,
                use_history=use_history,
                human_input=human_input
            )
            
            if final_request_params:
                config.default_request_params = RequestParams(**final_request_params)
                
            # Initialize the orchestrator
            # Note: This is where orchestrator creation logic would go
            return f"Orchestrator {name} created with agents {', '.join(agents)}"
            
        # Store metadata on the wrapper function
        setattr(wrapper, "_agent_type", AgentType.ORCHESTRATOR)
        setattr(wrapper, "_agent_config", AgentConfig(
            name=name,
            instruction=instruction or default_instruction,
            servers=[],
            model=model,
            use_history=use_history,
            human_input=human_input
        ))
        setattr(wrapper, "_child_agents", agents)
        setattr(wrapper, "_plan_type", plan_type)
        
        # Return the wrapped function
        return cast(F, wrapper)
    
    return decorator


# Add similar implementations for other decorator types (router, chain, parallel, etc.)
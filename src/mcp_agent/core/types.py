"""
Type definitions for fast-agent core module.
"""

from typing import TYPE_CHECKING, Dict, TypeAlias, Union

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
)
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_agent import ParallelAgent
from mcp_agent.workflows.router.router_agent import RouterAgent

# Avoid circular imports
if TYPE_CHECKING:
    from mcp_agent.core.proxies import BaseAgentProxy

# Type aliases for better readability
WorkflowType: TypeAlias = Union[Orchestrator, ParallelAgent, EvaluatorOptimizerLLM, RouterAgent]
AgentOrWorkflow: TypeAlias = Union[Agent, WorkflowType]
ProxyDict: TypeAlias = Dict[str, "BaseAgentProxy"]  # Forward reference as string

"""
Type definitions for fast-agent core module.
"""

from typing import Dict, Union, TypeAlias, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
)
from mcp_agent.workflows.router.router_llm import LLMRouter

# Avoid circular imports
if TYPE_CHECKING:
    from mcp_agent.core.proxies import BaseAgentProxy

# Type aliases for better readability
WorkflowType: TypeAlias = Union[
    Orchestrator, ParallelLLM, EvaluatorOptimizerLLM, LLMRouter
]
AgentOrWorkflow: TypeAlias = Union[Agent, WorkflowType]
ProxyDict: TypeAlias = Dict[str, "BaseAgentProxy"]  # Forward reference as string

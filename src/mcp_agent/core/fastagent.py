"""
FastAgent implementation using the direct pattern.

This module re-exports the DirectFastAgent implementation as FastAgent
to provide backward compatibility for importers.
"""

# Re-export DirectFastAgent as FastAgent for backward compatibility
from mcp_agent.core.direct_fastagent import DirectFastAgent as FastAgent

# Re-export other types users might import from fastagent
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.direct_agent_app import DirectAgentApp as AgentApp

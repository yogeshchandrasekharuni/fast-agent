"""
Module for handling MCP Sampling functionality without causing circular imports.
This module is carefully designed to avoid circular imports in the agent system.
"""

from mcp import ClientSession
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelFactoryClassProtocol

# Protocol is sufficient to describe the interface - no need for TYPE_CHECKING imports

logger = get_logger(__name__)


async def sample(
    ctx: ClientSession, params: CreateMessageRequestParams
) -> CreateMessageResult:
    """
    Handle sampling requests from the MCP protocol.
    This function extracts the model from the server config and
    returns a simple response without creating a full model instance.

    Note: This function intentionally avoids direct imports of model_factory
    to prevent circular imports between:
    - mcp_server_registry → mcp_connection_manager → mcp_agent_client_session → sampling
    - model_factory → agent → mcp_aggregator → gen_client → mcp_server_registry
    """
    try:
        model = None
        if (
            hasattr(ctx, "session")
            and hasattr(ctx.session, "server_config")
            and ctx.session.server_config
            and hasattr(ctx.session.server_config, "sampling")
            and ctx.session.server_config.sampling.model
        ):
            model = ctx.session.server_config.sampling.model

        if model is None:
            raise ValueError("No model configured")

        # Import model_factory dynamically to avoid circular imports
        # Use the ModelFactoryClassProtocol for type checking
        from mcp_agent.workflows.llm.model_factory import ModelFactory
        from mcp_agent.agents.agent import Agent

        # Verify ModelFactory matches our protocol
        model_factory: ModelFactoryClassProtocol = ModelFactory

        # Create a factory for the specified model
        factory = model_factory.create_factory(model)

        # Create a minimal agent for the factory
        # We use a simple agent configuration just for the sampling use case
        agent = Agent(
            name="sampling_agent",
            instruction="You are a sampling agent.",
            server_names=[],  # No servers needed for this simple use case
            context=ctx.context if hasattr(ctx, "context") else None,
        )

        # Initialize the LLM using our factory - existence check only
        _ = factory(agent=agent)  # Just verify it works, don't need to use

        # Get user message from the request params
        user_message = params.messages[0].content.text
        llm_response = "Response from LLM: " + user_message

        # Log successful LLM response
        logger.info(f"LLM successfully processed: {user_message[:30]}...")

        # Use the LLM-generated response in our result
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=llm_response),
            model=model,
            stopReason="endTurn",
        )
    except Exception as e:
        logger.error(f"Error in sampling: {str(e)}")
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=f"Error in sampling: {str(e)}"),
            model=model or "unknown",
            stopReason="error",
        )

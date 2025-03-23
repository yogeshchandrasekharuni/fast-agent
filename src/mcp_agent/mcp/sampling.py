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
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Protocol is sufficient to describe the interface - no need for TYPE_CHECKING imports

logger = get_logger(__name__)


def create_sampling_llm(
    mcp_ctx: ClientSession, model_string: str
) -> AugmentedLLMProtocol:
    """
    Create an LLM instance for sampling without tools support.
    This utility function creates a minimal LLM instance based on the model string.

    Args:
        mcp_ctx: The MCP ClientSession
        model_string: The model to use (e.g. "passthrough", "claude-3-5-sonnet-latest")

    Returns:
        An initialized LLM instance ready to use
    """
    from mcp_agent.workflows.llm.model_factory import ModelFactory
    from mcp_agent.agents.agent import Agent, AgentConfig

    # Get application context from global state if available
    # We don't try to extract it from mcp_ctx as they're different contexts
    app_context = None
    try:
        from mcp_agent.context import get_current_context

        app_context = get_current_context()
    except Exception:
        logger.warning("App context not available for sampling call")

    # Create a minimal agent configuration
    agent_config = AgentConfig(
        name="sampling_agent",
        instruction="You are a sampling agent.",
        servers=[],  # No servers needed
    )

    # Create agent with our application context (not the MCP context)
    # Set connection_persistence=False to avoid server connections
    agent = Agent(
        config=agent_config,
        context=app_context,
        server_names=[],  # Make sure no server connections are attempted
        connection_persistence=False,  # Avoid server connection management
    )

    # Create the LLM using the factory
    factory = ModelFactory.create_factory(model_string)
    llm = factory(agent=agent)

    # Attach the LLM to the agent
    agent._llm = llm

    return llm


async def sample(
    mcp_ctx: ClientSession, params: CreateMessageRequestParams
) -> CreateMessageResult:
    """
    Handle sampling requests from the MCP protocol.
    This function extracts the model from the server config and
    returns a simple response using the specified model.
    """
    model = None
    try:
        # Extract model from server config
        if (
            hasattr(mcp_ctx, "session")
            and hasattr(mcp_ctx.session, "server_config")
            and mcp_ctx.session.server_config
            and hasattr(mcp_ctx.session.server_config, "sampling")
            and mcp_ctx.session.server_config.sampling.model
        ):
            model = mcp_ctx.session.server_config.sampling.model

        if model is None:
            raise ValueError("No model configured")

        # Create an LLM instance using our utility function
        llm = create_sampling_llm(mcp_ctx, model)

        # Get user message from the request params
        user_message = params.messages[0].content.text

        # Create a multipart prompt message with the user's input
        prompt = PromptMessageMultipart(
            role="user", content=[TextContent(type="text", text=user_message)]
        )

        try:
            # Use the LLM to generate a response
            logger.info(f"Processing input: {user_message[:50]}...")
            llm_response = await llm.generate_prompt(prompt, None)
            logger.info(f"Generated response: {llm_response[:50]}...")
        except Exception as e:
            # If there's an error in LLM processing, fall back to echo
            logger.error(f"Error generating response: {str(e)}")
            llm_response = f"Echo response: {user_message}"

        # Return the LLM-generated response
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

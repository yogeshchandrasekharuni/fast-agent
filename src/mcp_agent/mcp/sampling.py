"""
Module for handling MCP Sampling functionality without causing circular imports.
This simplified implementation directly converts between MCP types and PromptMessageMultipart.
"""

from mcp import ClientSession
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol

# Import the converter since we've fixed the circular import
from mcp_agent.workflows.llm.sampling_converter import SamplingConverter

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

    app_context = None
    try:
        from mcp_agent.context import get_current_context

        app_context = get_current_context()
    except Exception:
        logger.warning("App context not available for sampling call")

    # Create a minimal agent configuration
    agent_config = AgentConfig(
        name="sampling_agent", instruction="You are a helpful AI Agent.", servers=[]
    )

    agent = Agent(
        config=agent_config,
        context=app_context,
        connection_persistence=False,
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
    Handle sampling requests from the MCP protocol using SamplingConverter.

    This function:
    1. Extracts the model from the request
    2. Uses SamplingConverter to convert types
    3. Calls the LLM's generate_prompt method
    4. Returns the result as a CreateMessageResult

    Args:
        mcp_ctx: The MCP ClientSession
        params: The sampling request parameters

    Returns:
        A CreateMessageResult containing the LLM's response
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

        # Create an LLM instance
        llm = create_sampling_llm(mcp_ctx, model)

        # Extract all messages from the request params
        if not params.messages:
            raise ValueError("No messages provided")

        # Convert all SamplingMessages to PromptMessageMultipart objects
        conversation = SamplingConverter.convert_messages(params.messages)

        # Extract request parameters using our converter
        request_params = SamplingConverter.extract_request_params(params)

        # Use the new public apply_prompt method which is cleaner than calling the protected method
        llm_response = await llm.apply_prompt(conversation, request_params)
        logger.info(f"Complete sampling request : {llm_response[:50]}...")

        # Create result using our converter
        return SamplingConverter.create_message_result(
            response=llm_response, model=model
        )
    except Exception as e:
        logger.error(f"Error in sampling: {str(e)}")
        return SamplingConverter.error_result(
            error_message=f"Error in sampling: {str(e)}", model=model
        )

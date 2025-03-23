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

        from mcp_agent.workflows.llm.model_factory import ModelFactory
        from mcp_agent.agents.agent import Agent, AgentConfig
        from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

        # Verify ModelFactory matches our protocol
        model_factory: ModelFactoryClassProtocol = ModelFactory

        factory = model_factory.create_factory("passthrough")

        agent_config = AgentConfig(
            name="sampling_agent",
            instruction="You are a sampling agent.",
            servers=[],  # No servers needed for this simple use case
        )

        # Create the agent using the config
        agent = Agent(
            config=agent_config,
            context=ctx.context if hasattr(ctx, "context") else None,
        )

        # Initialize the LLM using our factory and attach it to the agent
        llm = factory(agent=agent)
        agent._llm = llm  # Attach the LLM directly

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

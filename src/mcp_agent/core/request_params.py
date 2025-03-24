"""
Request parameters definitions for LLM interactions.
"""

from pydantic import Field
from mcp.types import CreateMessageRequestParams


class RequestParams(CreateMessageRequestParams):
    """
    Parameters to configure the AugmentedLLM 'generate' requests.
    """

    messages: None = Field(exclude=True, default=None)
    """
    Ignored. 'messages' are removed from CreateMessageRequestParams 
    to avoid confusion with the 'message' parameter on 'generate' method.
    """

    maxTokens: int = 2048
    """The maximum number of tokens to sample, as requested by the server."""

    model: str | None = None
    """
    The model to use for the LLM generation.
    If specified, this overrides the 'modelPreferences' selection criteria.
    """

    use_history: bool = True
    """
    Include the message history in the generate request.
    """

    max_iterations: int = 10
    """
    The maximum number of iterations to run the LLM for.
    """

    parallel_tool_calls: bool = True
    """
    Whether to allow multiple tool calls per iteration.
    Also known as multi-step tool use.
    """
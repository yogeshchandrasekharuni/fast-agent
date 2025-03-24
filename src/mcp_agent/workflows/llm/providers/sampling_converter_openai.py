from typing import Dict, Any

from openai.types.chat import (
    ChatCompletionMessage,
)

from mcp.types import (
    PromptMessage,
)

from mcp_agent.workflows.llm.sampling_format_converter import (
    SamplingFormatConverter,
)

from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class OpenAISamplingConverter(
    SamplingFormatConverter[Dict[str, Any], ChatCompletionMessage]
):
    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> Dict[str, Any]:
        """Convert an MCP PromptMessage to an OpenAI message dict."""
        from mcp_agent.workflows.llm.providers.multipart_converter_openai import (
            OpenAIConverter,
        )

        # Use the full-featured OpenAI converter for consistent handling
        return OpenAIConverter.convert_prompt_message_to_openai(message)

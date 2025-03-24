from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
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
    SamplingFormatConverter[ChatCompletionMessageParam, ChatCompletionMessage]
):
    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> ChatCompletionMessageParam:
        """Convert an MCP PromptMessage to an OpenAI ChatCompletionMessageParam."""
        content_text = (
            message.content.text
            if hasattr(message.content, "text")
            else str(message.content)
        )

        return {
            "role": message.role,
            "content": content_text,
        }

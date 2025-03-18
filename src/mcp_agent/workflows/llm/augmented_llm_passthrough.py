from typing import Any, List, Optional, Type, Union

from mcp import GetPromptResult
from mcp.types import PromptMessage
from pydantic_core import from_json
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)
from mcp_agent.logging.logger import get_logger


class PassthroughLLM(AugmentedLLM):
    """
    A specialized LLM implementation that simply passes through input messages without modification.

    This is useful for cases where you need an object with the AugmentedLLM interface
    but want to preserve the original message without any processing, such as in a
    parallel workflow where no fan-in aggregation is needed.
    """

    def __init__(self, name: str = "Passthrough", context=None, **kwargs):
        super().__init__(name=name, context=context, **kwargs)
        self.provider = "fast-agent"
        # Initialize logger - keep it simple without name reference
        self.logger = get_logger(__name__)
        self._messages = [PromptMessage]

    async def generate(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        request_params: Optional[RequestParams] = None,
    ) -> Union[List[MessageT], Any]:
        """Simply return the input message as is."""
        # Return in the format expected by the caller
        return [message] if isinstance(message, list) else message

    async def generate_str(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        request_params: Optional[RequestParams] = None,
    ) -> str:
        """Return the input message as a string."""
        self.show_user_message(message, model="fastagent-passthrough", chat_turn=0)
        await self.show_assistant_message(message, title="ASSISTANT/PASSTHROUGH")

        return str(message)

    async def generate_structured(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        response_model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> ModelT:
        """
        Return the input message as the requested model type.
        This is a best-effort implementation - it may fail if the
        message cannot be converted to the requested model.
        """
        if isinstance(message, response_model):
            return message
        elif isinstance(message, dict):
            return response_model(**message)
        elif isinstance(message, str):
            return response_model.model_validate(from_json(message, allow_partial=True))

    async def generate_prompt(
        self, prompt: "PromptMessageMultipart", request_params: RequestParams | None
    ) -> str:
        return await self.generate_str(prompt.content[0].text, request_params)

    async def apply_prompt_template(
        self, prompt_result: GetPromptResult, prompt_name: str
    ) -> str:
        """
        Apply a prompt template by adding it to the conversation history.
        If the last message in the prompt is from a user, automatically
        generate an assistant response.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the assistant's response if generated,
            or the last assistant message in the prompt
        """
        prompt_messages: List[PromptMessage] = prompt_result.messages

        # Extract arguments if they were stored in the result
        arguments = getattr(prompt_result, "arguments", None)

        # Display information about the loaded prompt
        await self.show_prompt_loaded(
            prompt_name=prompt_name,
            description=prompt_result.description,
            message_count=len(prompt_messages),
            arguments=arguments,
        )
        self._messages = prompt_messages

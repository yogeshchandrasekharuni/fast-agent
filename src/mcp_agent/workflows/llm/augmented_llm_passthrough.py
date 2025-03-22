from typing import Any, List, Optional, Type, Union
import json
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
        # Check if this is a special command to call a tool
        if isinstance(message, str) and message.startswith("***CALL_TOOL "):
            return await self._call_tool_and_return_result(message)

        self.show_user_message(message, model="fastagent-passthrough", chat_turn=0)
        await self.show_assistant_message(message, title="ASSISTANT/PASSTHROUGH")

        return str(message)

    async def _call_tool_and_return_result(self, command: str) -> str:
        """
        Call a tool based on the command and return its result as a string.

        Args:
            command: The command string, expected format: "***CALL_TOOL <server>-<tool_name> [arguments_json]"

        Returns:
            Tool result as a string
        """
        try:
            # Parse the tool name and optional arguments
            parts = command.split(" ", 2)
            if len(parts) < 2:
                return "Error: Invalid format. Expected '***CALL_TOOL <tool_name> [arguments_json]'"

            tool_name = parts[1].strip()
            arguments = None

            # Parse optional JSON arguments if provided
            if len(parts) > 2:
                try:
                    arguments = json.loads(parts[2])
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON arguments: {parts[2]}"

            # Call the tool and get the result
            self.logger.info(f"Calling tool {tool_name} with arguments {arguments}")
            result = await self.aggregator.call_tool(tool_name, arguments)

            # Format the result as a string
            if result.isError:
                return f"Error calling tool '{tool_name}': {result.message}"

            # Extract text content from result
            result_text = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    result_text.append(content_item.text)
                else:
                    result_text.append(str(content_item))

            return "\n".join(result_text)

        except Exception as e:
            self.logger.error(f"Error calling tool: {str(e)}")
            return f"Error calling tool: {str(e)}"

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
        message = prompt.content[0].text if prompt.content else ""
        if isinstance(message, str) and message.startswith("***CALL_TOOL "):
            return await self._call_tool_and_return_result(message)
        return await self.generate_str(message, request_params)

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

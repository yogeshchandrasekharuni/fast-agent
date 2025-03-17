from typing import List, Optional, Union
from mcp import GetPromptResult
from mcp.types import PromptMessage
from mcp_agent.workflows.llm.augmented_llm import MessageParamT, RequestParams
from mcp_agent.workflows.llm.augmented_llm_passthrough import PassthroughLLM


# TODO -- support tool calling
class PlaybackLLM(PassthroughLLM):
    """
    A specialized LLM implementation that plays back assistant messages when loaded with prompts.

    Unlike the PassthroughLLM which simply passes through messages without modification,
    PlaybackLLM is designed to simulate a conversation by playing back prompt messages
    in sequence when loaded with prompts through apply_prompt_template.

    After apply_prompts has been called, each call to generate_str returns the next
    "ASSISTANT" message in the loaded messages. If no messages are set or all messages have
    been played back, it returns a message indicating that messages are exhausted.
    """

    def __init__(self, name: str = "Playback", **kwargs):
        super().__init__(name=name, **kwargs)
        self._messages: List[PromptMessage] = []
        self._current_index = 0

    async def generate_str(
        self,
        message: Union[str, MessageParamT, List[MessageParamT]],
        request_params: Optional[RequestParams] = None,
    ) -> str:
        """
        Return the next ASSISTANT message in the loaded messages list.
        If no messages are available or all have been played back,
        returns a message indicating messages are exhausted.

        Note: Only assistant messages are returned; user messages are skipped.
        """
        self.show_user_message(message, model="fastagent-playback", chat_turn=0)

        if not self._messages or self._current_index >= len(self._messages):
            size = len(self._messages) if self._messages else 0
            response = f"MESSAGES EXHAUSTED (list size {size})"
        else:
            response = self._get_next_assistant_message()

        await self.show_assistant_message(response, title="ASSISTANT/PLAYBACK")
        return response

    def _get_next_assistant_message(self) -> str:
        """
        Get the next assistant message from the loaded messages.
        Increments the current message index and skips user messages.
        """
        # Find next assistant message
        while self._current_index < len(self._messages):
            message = self._messages[self._current_index]
            self._current_index += 1

            # Skip non-assistant messages
            if getattr(message, "role", None) != "assistant":
                continue

            # Get content as string
            content = message.content
            if hasattr(content, "text"):
                return content.text
            return str(content)

        # If we get here, we've run out of assistant messages
        return f"MESSAGES EXHAUSTED (list size {len(self._messages)})"

    async def apply_prompt_template(
        self, prompt_result: GetPromptResult, prompt_name: str
    ) -> str:
        """
        Apply a prompt template by adding its messages to the playback queue.

        Args:
            prompt_result: The GetPromptResult containing prompt messages
            prompt_name: The name of the prompt being applied

        Returns:
            String representation of the first message or an indication that no messages were added
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

        # Add new messages to the end of the existing messages list
        self._messages.extend(prompt_messages)

        if not prompt_messages:
            return "Prompt contains no messages"

        # Reset current index if this is the first time loading messages
        if len(self._messages) == len(prompt_messages):
            self._current_index = 0

        return f"Added {len(prompt_messages)} messages to playback queue"

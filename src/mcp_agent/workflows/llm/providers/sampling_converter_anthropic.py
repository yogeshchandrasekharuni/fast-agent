import json
from typing import Iterable, List
from mcp import CreateMessageResult, SamplingMessage, StopReason
from pydantic import BaseModel
from mcp_agent.workflows.llm.sampling_format_converter import SamplingFormatConverter

from mcp.types import (
    PromptMessage,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from anthropic.types import (
    ContentBlock,
    DocumentBlockParam,
    Message,
    MessageParam,
    ImageBlockParam,
    TextBlock,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class AnthropicSamplingConverter(SamplingFormatConverter[MessageParam, Message]):
    """
    Convert between Anthropic and MCP types.
    """

    @classmethod
    def from_sampling_result(cls, result: CreateMessageResult) -> Message:
        #  -> Message
        if result.role != "assistant":
            raise ValueError(
                f"Expected role to be 'assistant' but got '{result.role}' instead."
            )

        return Message(
            role="assistant",
            type="message",
            content=[mcp_content_to_anthropic_content(result.content)],
            stop_reason=mcp_stop_reason_to_anthropic_stop_reason(result.stopReason),
            model=result.model,
            usage={"input_tokens": 0, "output_tokens": 0},
            id="sampling_id",
            # TODO -- incorporate usage info and message identity
        )

    @classmethod
    def to_sampling_result(cls, result: Message) -> CreateMessageResult:
        contents = anthropic_content_to_mcp_content(result.content)
        if len(contents) > 1:
            raise NotImplementedError(
                "Multiple content elements in a single message are not supported in MCP yet"
            )
        mcp_content = contents[0]

        # Create a dictionary with required fields
        result_dict = {
            "role": result.role,
            "content": mcp_content,
            "model": result.model,
            "stopReason": anthropic_stop_reason_to_mcp_stop_reason(result.stop_reason),
        }

        # Add any other fields from the original message that might be needed
        extras = result.model_dump(exclude={"role", "content", "model", "stop_reason"})
        if extras:
            # Only include compatible fields to avoid validation errors
            # Skip fields that would cause validation issues with CreateMessageResult
            safe_extras = {
                k: v for k, v in extras.items() if k in CreateMessageResult.model_fields
            }
            result_dict.update(safe_extras)

        return CreateMessageResult(**result_dict)

    @classmethod
    def from_sampling_message(cls, param: SamplingMessage) -> MessageParam:
        extras = param.model_dump(exclude={"role", "content"})
        return MessageParam(
            role=param.role,
            content=[mcp_content_to_anthropic_content(param.content)],
            **extras,
        )

    @classmethod
    def to_sampling_message(cls, param: MessageParam) -> SamplingMessage:
        # Implement the conversion from ChatCompletionMessage to MCP message param

        contents = anthropic_content_to_mcp_content(param["content"])

        # TODO: saqadri - the mcp_content can have multiple elements
        # while sampling message content has a single content element
        # Right now we error out if there are > 1 elements in mcp_content
        # We need to handle this case properly going forward
        if len(contents) > 1:
            raise NotImplementedError(
                "Multiple content elements in a single message are not supported"
            )
        mcp_content = contents[0]

        # Only include fields that are valid for SamplingMessage
        extras = {
            k: v
            for k, v in param.items()
            if k not in ["role", "content"] and k in SamplingMessage.model_fields
        }

        return SamplingMessage(
            role=param["role"],
            content=mcp_content,
            **extras,
        )

    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> MessageParam:
        """Convert an MCP PromptMessage to an Anthropic MessageParam."""

        # Extract content text
        content_text = (
            message.content.text
            if hasattr(message.content, "text")
            else str(message.content)
        )

        # Extract extras for flexibility
        extras = message.model_dump(exclude={"role", "content"})

        # Handle based on role
        if message.role == "user":
            return {"role": "user", "content": content_text, **extras}
        elif message.role == "assistant":
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
                **extras,
            }
        else:
            # Fall back to user for any unrecognized role, including "system"
            _logger.warning(
                f"Unsupported role '{message.role}' in PromptMessage. Falling back to 'user' role."
            )
            return {
                "role": "user",
                "content": f"[{message.role.upper()}] {content_text}",
                **extras,
            }


def anthropic_content_to_mcp_content(
    content: str
    | Iterable[
        TextBlockParam
        | ImageBlockParam
        | ToolUseBlockParam
        | ToolResultBlockParam
        | DocumentBlockParam
        | ContentBlock
    ],
) -> List[TextContent | ImageContent | EmbeddedResource]:
    mcp_content = []

    if isinstance(content, str):
        mcp_content.append(TextContent(type="text", text=content))
    else:
        for block in content:
            if block.type == "text":
                mcp_content.append(TextContent(type="text", text=block.text))
            elif block.type == "image":
                raise NotImplementedError("Image content conversion not implemented")
            elif block.type == "tool_use":
                # Best effort to convert a tool use to text (since there's no ToolUseContent)
                mcp_content.append(
                    TextContent(
                        type="text",
                        text=to_string(block),
                    )
                )
            elif block.type == "tool_result":
                # Best effort to convert a tool result to text (since there's no ToolResultContent)
                mcp_content.append(
                    TextContent(
                        type="text",
                        text=to_string(block),
                    )
                )
            elif block.type == "document":
                raise NotImplementedError("Document content conversion not implemented")
            else:
                # Last effort to convert the content to a string
                mcp_content.append(TextContent(type="text", text=str(block)))

    return mcp_content


def mcp_stop_reason_to_anthropic_stop_reason(stop_reason: StopReason):
    if not stop_reason:
        return None
    elif stop_reason == "endTurn":
        return "end_turn"
    elif stop_reason == "maxTokens":
        return "max_tokens"
    elif stop_reason == "stopSequence":
        return "stop_sequence"
    elif stop_reason == "toolUse":
        return "tool_use"
    else:
        return stop_reason


def anthropic_stop_reason_to_mcp_stop_reason(stop_reason: str) -> StopReason:
    if not stop_reason:
        return None
    elif stop_reason == "end_turn":
        return "endTurn"
    elif stop_reason == "max_tokens":
        return "maxTokens"
    elif stop_reason == "stop_sequence":
        return "stopSequence"
    elif stop_reason == "tool_use":
        return "toolUse"
    else:
        return stop_reason


def mcp_content_to_anthropic_content(
    content: TextContent | ImageContent | EmbeddedResource,
) -> ContentBlock:
    if isinstance(content, TextContent):
        return TextBlock(type=content.type, text=content.text)
    elif isinstance(content, ImageContent):
        # Best effort to convert an image to text (since there's no ImageBlock)
        return TextBlock(type="text", text=f"{content.mimeType}:{content.data}")
    elif isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return TextBlock(type="text", text=content.resource.text)
        else:  # BlobResourceContents
            return TextBlock(
                type="text", text=f"{content.resource.mimeType}:{content.resource.blob}"
            )
    else:
        # Last effort to convert the content to a string
        return TextBlock(type="text", text=str(content))


def to_string(obj: BaseModel | dict) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    else:
        return json.dumps(obj)

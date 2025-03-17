from typing import Iterable
from mcp import CreateMessageResult, SamplingMessage
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
)

from mcp.types import (
    PromptMessage,
    TextContent,
    ImageContent,
    EmbeddedResource,
    TextResourceContents,
)

from mcp_agent.workflows.llm.sampling_format_converter import (
    SamplingFormatConverter,
    typed_dict_extras,
)

from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class OpenAISamplingConverter(
    SamplingFormatConverter[ChatCompletionMessageParam, ChatCompletionMessage]
):
    """
    Convert between OpenAI and MCP types.
    """

    @classmethod
    def from_sampling_result(cls, result: CreateMessageResult) -> ChatCompletionMessage:
        """Convert an MCP message result to an OpenAI ChatCompletionMessage."""
        # Basic implementation - would need to be expanded

        if result.role != "assistant":
            raise ValueError(
                f"Expected role to be 'assistant' but got '{result.role}' instead."
            )
        # TODO -- add image support for sampling
        return ChatCompletionMessage(
            role=result.role,
            content=result.content.text or "image",
        )

    @classmethod
    def to_sampling_result(cls, result: ChatCompletionMessage) -> CreateMessageResult:
        """Convert an OpenAI ChatCompletionMessage to an MCP message result."""
        content = result.content
        if content is None:
            content = ""

        return CreateMessageResult(
            role=result.role,
            content=TextContent(type="text", text=content),
            model="unknown",  # Model is required by CreateMessageResult
        )

    @classmethod
    def from_sampling_message(
        cls, param: SamplingMessage
    ) -> ChatCompletionMessageParam:
        if param.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=mcp_to_openai_blocks(param.content),
            )
        elif param.role == "user":
            return ChatCompletionUserMessageParam(
                role="user",
                content=mcp_to_openai_blocks(param.content),
            )
        else:
            raise ValueError(
                f"Unexpected role: {param.role}, MCP only supports 'assistant' and 'user'"
            )

    @classmethod
    def to_sampling_message(cls, param: ChatCompletionMessageParam) -> SamplingMessage:
        contents = openai_to_mcp_blocks(param)

        # TODO: saqadri - the mcp_content can have multiple elements
        # while sampling message content has a single content element
        # Right now we error out if there are > 1 elements in mcp_content
        # We need to handle this case properly going forward
        if len(contents) > 1:
            raise NotImplementedError(
                "Multiple content elements in a single message are not supported"
            )
        mcp_content: TextContent | ImageContent | EmbeddedResource = contents[0]

        if param["role"] == "assistant":
            return SamplingMessage(
                role="assistant",
                content=mcp_content,
                **typed_dict_extras(param, ["role", "content"]),
            )
        elif param["role"] == "user":
            return SamplingMessage(
                role="user",
                content=mcp_content,
                **typed_dict_extras(param, ["role", "content"]),
            )
        elif param.role == "tool":
            raise NotImplementedError(
                "Tool messages are not supported in SamplingMessage yet"
            )
        elif param.role == "system":
            raise NotImplementedError(
                "System messages are not supported in SamplingMessage yet"
            )
        elif param.role == "developer":
            raise NotImplementedError(
                "Developer messages are not supported in SamplingMessage yet"
            )
        elif param.role == "function":
            raise NotImplementedError(
                "Function messages are not supported in SamplingMessage yet"
            )
        else:
            raise ValueError(
                f"Unexpected role: {param.role}, MCP only supports 'assistant', 'user', 'tool', 'system', 'developer', and 'function'"
            )

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


def mcp_to_openai_blocks(
    content: TextContent | ImageContent | EmbeddedResource,
) -> ChatCompletionContentPartTextParam:
    if isinstance(content, list):
        # Handle list of content items
        return ChatCompletionContentPartTextParam(
            type="text",
            text="\n".join(mcp_to_openai_blocks(c) for c in content),
        )

    if isinstance(content, TextContent):
        return ChatCompletionContentPartTextParam(type="text", text=content.text)
    elif isinstance(content, ImageContent):
        # Best effort to convert an image to text
        return ChatCompletionContentPartTextParam(
            type="text", text=f"{content.mimeType}:{content.data}"
        )
    elif isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return ChatCompletionContentPartTextParam(
                type="text", text=content.resource.text
            )
        else:  # BlobResourceContents
            return ChatCompletionContentPartTextParam(
                type="text", text=f"{content.resource.mimeType}:{content.resource.blob}"
            )
    else:
        # Last effort to convert the content to a string
        return ChatCompletionContentPartTextParam(type="text", text=str(content))


def openai_to_mcp_blocks(
    content: str
    | Iterable[ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam],
) -> Iterable[TextContent | ImageContent | EmbeddedResource]:
    mcp_content = []

    if isinstance(content, str):
        mcp_content = [TextContent(type="text", text=content)]

    else:
        mcp_content = [TextContent(type="text", text=content["content"])]

    return mcp_content

    # # TODO: saqadri - this is a best effort conversion, we should handle all possible content types
    # for c in content["content"]:
    #     # TODO: evalstate, need to go through all scenarios here
    #     if isinstance(c, str):
    #         mcp_content.append(TextContent(type="text", text=c))
    #         break

    #     if c.type == "text":  # isinstance(c, ChatCompletionContentPartTextParam):
    #         mcp_content.append(
    #             TextContent(
    #                 type="text", text=c.text, **typed_dict_extras(c, ["text"])
    #             )
    #         )
    #     elif (
    #         c.type == "image_url"
    #     ):  # isinstance(c, ChatCompletionContentPartImageParam):
    #         raise NotImplementedError("Image content conversion not implemented")
    #         # TODO: saqadri - need to download the image into a base64-encoded string
    #         # Download image from c.image_url
    #         # return ImageContent(
    #         #     type="image",
    #         #     data=downloaded_image,
    #         #     **c
    #         # )
    #     elif (
    #         c.type == "input_audio"
    #     ):  # isinstance(c, ChatCompletionContentPartInputAudioParam):
    #         raise NotImplementedError("Audio content conversion not implemented")
    #     elif (
    #         c.type == "refusal"
    #     ):  # isinstance(c, ChatCompletionContentPartRefusalParam):
    #         mcp_content.append(
    #             TextContent(
    #                 type="text", text=c.refusal, **typed_dict_extras(c, ["refusal"])
    #             )
    #         )
    #     else:
    #         raise ValueError(f"Unexpected content type: {c.type}")

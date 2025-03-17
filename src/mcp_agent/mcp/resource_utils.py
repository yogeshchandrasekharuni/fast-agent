import base64
from pathlib import Path
from typing import List, Optional, Tuple
from mcp.types import (
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
    ImageContent,
)
import mcp_agent.mcp.mime_utils as mime_utils

HTTP_TIMEOUT = 10  # Default timeout for HTTP requests

# Define a type alias for resource content results
ResourceContent = Tuple[str, str, bool]


def find_resource_file(resource_path: str, prompt_files: List[Path]) -> Optional[Path]:
    """Find a resource file relative to one of the prompt files"""
    for prompt_file in prompt_files:
        potential_path = prompt_file.parent / resource_path
        if potential_path.exists():
            return potential_path
    return None


# TODO -- decide how to deal with this. Both Anthropic and OpenAI allow sending URLs in
# input message
# TODO -- used?
# async def fetch_remote_resource(
#     url: str, timeout: int = HTTP_TIMEOUT
# ) -> ResourceContent:
#     """
#     Fetch a remote resource from a URL

#     Returns:
#         Tuple of (content, mime_type, is_binary)
#         - content: Text content or base64-encoded binary content
#         - mime_type: The MIME type of the resource
#         - is_binary: Whether the content is binary (and base64-encoded)
#     """

#     async with httpx.AsyncClient(timeout=timeout) as client:
#         response = await client.get(url)
#         response.raise_for_status()

#         # Get the content type or guess from URL
#         mime_type = response.headers.get("content-type", "").split(";")[0]
#         if not mime_type:
#             mime_type = mime_utils.guess_mime_type(url)

#         # Check if this is binary content
#         is_binary = mime_utils.is_binary_content(mime_type)

#         if is_binary:
#             # For binary responses, get the binary content and base64 encode it
#             content = base64.b64encode(response.content).decode("utf-8")
#         else:
#             # For text responses, just get the text
#             content = response.text

#         return content, mime_type, is_binary


def load_resource_content(
    resource_path: str, prompt_files: List[Path]
) -> ResourceContent:
    """
    Load a resource's content and determine its mime type

    Args:
        resource_path: Path to the resource file
        prompt_files: List of prompt files (to find relative paths)

    Returns:
        Tuple of (content, mime_type, is_binary)
        - content: String content for text files, base64-encoded string for binary files
        - mime_type: The MIME type of the resource
        - is_binary: Whether the content is binary (and base64-encoded)

    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Try to locate the resource file
    resource_file = find_resource_file(resource_path, prompt_files)
    if resource_file is None:
        raise FileNotFoundError(f"Resource not found: {resource_path}")

    # Determine mime type
    mime_type = mime_utils.guess_mime_type(str(resource_file))
    is_binary = mime_utils.is_binary_content(mime_type)

    if is_binary:
        # For binary files, read as binary and base64 encode
        with open(resource_file, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
    else:
        # For text files, read as text
        with open(resource_file, "r", encoding="utf-8") as f:
            content = f.read()

    return content, mime_type, is_binary


# Create a safe way to generate resource URIs that Pydantic accepts
def create_resource_uri(path: str) -> str:
    """Create a resource URI from a path"""
    return f"resource://{Path(path).name}"


def create_embedded_resource(
    resource_path: str, content: str, mime_type: str, is_binary: bool = False
) -> EmbeddedResource:
    """Create an embedded resource content object"""
    # Format a valid resource URI string
    resource_uri_str = create_resource_uri(resource_path)

    # Create common resource args dict to reduce duplication
    resource_args = {
        "uri": resource_uri_str,  # type: ignore
        "mimeType": mime_type,
    }

    if is_binary:
        return EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                **resource_args,
                blob=content,
            ),
        )
    else:
        return EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                **resource_args,
                text=content,
            ),
        )


def create_image_content(data: str, mime_type: str) -> ImageContent:
    """Create an image content object from base64-encoded data"""
    return ImageContent(
        type="image",
        data=data,
        mimeType=mime_type,
    )

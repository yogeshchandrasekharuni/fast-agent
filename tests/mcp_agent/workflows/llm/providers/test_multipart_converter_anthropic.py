import unittest
import base64
from typing import List, Union

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    Role,
    TextResourceContents,
    BlobResourceContents,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

from anthropic.types import (
    MessageParam,
    ContentBlockParam,
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
)

from mcp_agent.workflows.llm.providers.multipart_converter_anthropic import (
    AnthropicConverter,
    normalize_uri,
)


class TestAnthropicConverter(unittest.TestCase):
    """Test cases for conversion from MCP message types to Anthropic API."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a test message"
        self.sample_image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")

    def test_text_content_conversion(self):
        """Test conversion of TextContent to Anthropic text block."""
        # Create a text content message
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageMultipart(role="user", content=[text_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access, not attribute access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_image_content_conversion(self):
        """Test conversion of ImageContent to Anthropic image block."""
        # Create an image content message
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        multipart = PromptMessageMultipart(role="user", content=[image_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "image")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["media_type"], "image/jpeg"
        )
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["data"], self.sample_image_base64
        )

    def test_embedded_resource_text_conversion(self):
        """Test conversion of text-based EmbeddedResource to Anthropic document block."""
        # Create a text resource
        text_resource = TextResourceContents(
            uri="test://example.com/document.txt",
            mimeType="text/plain",
            text=self.sample_text,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=text_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["title"], "document.txt")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["media_type"], "text/plain"
        )
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["data"], self.sample_text
        )

    def test_embedded_resource_pdf_conversion(self):
        """Test conversion of PDF EmbeddedResource to Anthropic document block."""
        # Create a PDF resource
        pdf_base64 = base64.b64encode(b"fake_pdf_data").decode("utf-8")
        pdf_resource = BlobResourceContents(
            uri="test://example.com/document.pdf",
            mimeType="application/pdf",
            blob=pdf_base64,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=pdf_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["media_type"], "application/pdf"
        )
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], pdf_base64)

    def test_embedded_resource_image_url_conversion(self):
        """Test conversion of image URL in EmbeddedResource to Anthropic image block."""
        # Create an image resource with URL
        image_resource = BlobResourceContents(
            uri="https://example.com/image.jpg",
            mimeType="image/jpeg",
            blob=self.sample_image_base64,  # This should be ignored for URL
        )
        embedded_resource = EmbeddedResource(type="resource", resource=image_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "image")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "url")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["url"],
            "https://example.com/image.jpg",
        )

    def test_assistant_role_restrictions(self):
        """Test that assistant messages can only contain text blocks."""
        # Create mixed content for assistant
        text_content = TextContent(type="text", text=self.sample_text)
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        multipart = PromptMessageMultipart(
            role="assistant", content=[text_content, image_content]
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - only text should remain
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_multiple_content_blocks(self):
        """Test conversion of messages with multiple content blocks."""
        # Create multiple content blocks
        text_content1 = TextContent(type="text", text="First text")
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        text_content2 = TextContent(type="text", text="Second text")

        multipart = PromptMessageMultipart(
            role="user", content=[text_content1, image_content, text_content2]
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 3)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], "First text")
        self.assertEqual(anthropic_msg["content"][1]["type"], "image")
        self.assertEqual(anthropic_msg["content"][2]["type"], "text")
        self.assertEqual(anthropic_msg["content"][2]["text"], "Second text")

    def test_unsupported_mime_type_handling(self):
        """Test handling of unsupported MIME types."""
        # Create an image with unsupported mime type
        image_content = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/bmp",  # Unsupported in Anthropic API
        )
        text_content = TextContent(type="text", text="This is some text")
        multipart = PromptMessageMultipart(
            role="user", content=[text_content, image_content]
        )

        # Convert to Anthropic format - should convert unsupported image to text fallback
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have kept the text content and added a fallback text for the image
        self.assertEqual(len(anthropic_msg["content"]), 2)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], "This is some text")
        self.assertEqual(anthropic_msg["content"][1]["type"], "text")
        self.assertTrue(
            "[Image with unsupported format: image/bmp]"
            in anthropic_msg["content"][1]["text"]
        )

    def test_svg_resource_conversion(self):
        """Test handling of SVG resources - should convert to code block."""
        # Create an embedded SVG resource
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        )
        svg_resource = TextResourceContents(
            uri="test://example.com/image.svg",
            mimeType="image/svg+xml",
            text=svg_content,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=svg_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format - should extract SVG as text
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should be converted to a text block with the SVG code
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertIn("```xml", anthropic_msg["content"][0]["text"])
        self.assertIn(svg_content, anthropic_msg["content"][0]["text"])

    def test_empty_content_list(self):
        """Test conversion with empty content list."""
        multipart = PromptMessageMultipart(role="user", content=[])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have empty content list
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 0)

    def test_embedded_resource_pdf_url_conversion(self):
        """Test conversion of PDF URL in EmbeddedResource to Anthropic document block."""
        # Create a PDF resource with URL
        pdf_resource = BlobResourceContents(
            uri="https://example.com/document.pdf",
            mimeType="application/pdf",
            blob=base64.b64encode(b"fake_pdf_data").decode("utf-8"),
        )
        embedded_resource = EmbeddedResource(type="resource", resource=pdf_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "url")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["url"],
            "https://example.com/document.pdf",
        )

    def test_mixed_content_with_unsupported_formats(self):
        """Test conversion of mixed content where some items are unsupported."""
        # Create mixed content with supported and unsupported items
        text_content = TextContent(type="text", text=self.sample_text)
        unsupported_image = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/bmp",  # Unsupported
        )
        supported_image = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/jpeg",  # Supported
        )

        multipart = PromptMessageMultipart(
            role="user", content=[text_content, unsupported_image, supported_image]
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have kept the text, created fallback for unsupported, and kept supported image
        self.assertEqual(len(anthropic_msg["content"]), 3)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)
        self.assertEqual(
            anthropic_msg["content"][1]["type"], "text"
        )  # Fallback text for unsupported
        self.assertEqual(
            anthropic_msg["content"][2]["type"], "image"
        )  # Supported image kept
        self.assertEqual(
            anthropic_msg["content"][2]["source"]["media_type"], "image/jpeg"
        )

    def test_conversion_error_handling(self):
        """Test handling of conversion errors."""

        # Create a problematic embedded resource (missing required attribute)
        # We'll mock this with a custom class since actual errors depend on implementation
        class ProblemResource(TextResourceContents):
            def __init__(self):
                super().__init__(uri="test://problem", mimeType="text/plain", text="")
                # Force an attribute error during conversion
                delattr(self, "uri")

        problem_resource = ProblemResource()
        embedded_resource = EmbeddedResource(type="resource", resource=problem_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format - should create error fallback
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have a fallback text block for the error
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertTrue(
            "Content conversion error" in anthropic_msg["content"][0]["text"]
        )

    def test_code_file_as_text_document_with_filename(self):
        """Test handling of code files using a simple filename."""
        code_text = "def hello_world():\n    print('Hello, world!')"

        # Use the helper function with simple filename
        code_resource = create_text_resource(
            text=code_text, filename_or_uri="example.py", mime_type="text/x-python"
        )

        embedded_resource = EmbeddedResource(type="resource", resource=code_resource)

        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Check that title is set correctly
        self.assertEqual(anthropic_msg["content"][0]["title"], "example.py")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], code_text)

    def test_code_file_as_text_document_with_uri(self):
        """Test handling of code files using a proper URI."""
        code_text = "def hello_world():\n    print('Hello, world!')"

        # Use the helper function with full URI
        code_resource = create_text_resource(
            text=code_text,
            filename_or_uri="file:///projects/example.py",
            mime_type="text/x-python",
        )

        embedded_resource = EmbeddedResource(type="resource", resource=code_resource)

        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should extract just the filename from the path
        self.assertEqual(anthropic_msg["content"][0]["title"], "example.py")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], code_text)


def create_text_resource(
    text: str, filename_or_uri: str, mime_type: str = None
) -> TextResourceContents:
    """
    Helper function to create a TextResourceContents with proper URI handling.

    Args:
        text: The text content
        filename_or_uri: A filename or URI
        mime_type: Optional MIME type

    Returns:
        A properly configured TextResourceContents
    """
    # Normalize the URI
    uri = normalize_uri(filename_or_uri)

    return TextResourceContents(uri=uri, mimeType=mime_type, text=text)


class TestUriNormalization(unittest.TestCase):
    """Tests for URI normalization functionality."""

    def test_already_valid_uri(self):
        """Test that already valid URIs are left unchanged."""
        uri = "https://example.com/path/file.txt"
        result = normalize_uri(uri)
        self.assertEqual(result, uri)

    def test_file_uri(self):
        """Test that file:// URIs are left unchanged."""
        uri = "file:///path/to/file.txt"
        result = normalize_uri(uri)
        self.assertEqual(result, uri)

    def test_simple_filename(self):
        """Test that simple filenames are converted to file:// URIs."""
        filename = "example.py"
        result = normalize_uri(filename)
        self.assertEqual(result, "file:///example.py")

    def test_relative_path(self):
        """Test that relative paths are converted to file:// URIs."""
        path = "path/to/file.txt"
        result = normalize_uri(path)
        self.assertEqual(result, "file:///path/to/file.txt")

    def test_absolute_path(self):
        """Test that absolute paths are converted to file:// URIs."""
        path = "/path/to/file.txt"
        result = normalize_uri(path)
        self.assertEqual(result, "file:///path/to/file.txt")

    def test_windows_path(self):
        """Test that Windows paths are normalized and converted."""
        path = "C:\\path\\to\\file.txt"
        result = normalize_uri(path)
        self.assertEqual(result, "file:///C:/path/to/file.txt")

    def test_empty_string(self):
        """Test handling of empty strings."""
        result = normalize_uri("")
        self.assertEqual(result, "")

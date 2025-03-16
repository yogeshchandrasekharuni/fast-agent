"""
Tests for the prompt_format_utils module, focusing on resource handling.
"""

import pytest
import tempfile
import os
from pathlib import Path

from mcp.types import (
    TextContent,
    EmbeddedResource,
    TextResourceContents,
    ImageContent,
)

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.prompt_format_utils import (
    multipart_messages_to_delimited_format,
    delimited_format_to_multipart_messages,
    save_messages_to_delimited_file,
    load_messages_from_delimited_file,
    guess_mime_type,
)


class TestPromptFormatUtils:
    """Tests for the prompt_format_utils module."""

    def test_multipart_with_resources_to_delimited(self):
        """Test converting multipart messages with resources to delimited format."""
        # Create messages with resources
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Here's a code sample:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://code.py",
                            mimeType="text/x-python",
                            text='print("Hello, World!")'
                        )
                    )
                ]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[
                    TextContent(type="text", text="I've analyzed your code and made improvements:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://improved_code.py",
                            mimeType="text/x-python",
                            text='def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()'
                        )
                    )
                ]
            )
        ]

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format(
            messages, 
            user_delimiter="---USER",
            assistant_delimiter="---ASSISTANT",
            resource_delimiter="---RESOURCE"
        )

        # Verify structure
        assert len(delimited) == 8  # 2 role delimiters + 2 content blocks + 4 resource-related entries
        assert delimited[0] == "---USER"
        assert "Here's a code sample:" in delimited[1]
        assert 'print("Hello, World!")' in delimited[1]
        assert delimited[2] == "---RESOURCE"
        assert delimited[3] == "code.py"
        assert delimited[4] == "---ASSISTANT"
        assert "I've analyzed your code" in delimited[5]

    def test_delimited_with_resources_to_multipart(self):
        """Test converting delimited format with resources to multipart messages."""
        # Create delimited content with resources
        delimited_content = """---USER
Here's a CSS file I want to improve:

---RESOURCE
styles.css

---ASSISTANT
I've reviewed your CSS and made it more efficient:

---RESOURCE
improved_styles.css"""

        # Convert to multipart messages
        messages = delimited_format_to_multipart_messages(
            delimited_content,
            resource_delimiter="---RESOURCE"
        )

        # Verify structure
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert len(messages[0].content) == 2  # Text and resource
        assert messages[0].content[0].type == "text"
        assert "Here's a CSS file" in messages[0].content[0].text
        assert messages[0].content[1].type == "resource"
        assert str(messages[0].content[1].resource.uri) == "resource://styles.css"

        assert messages[1].role == "assistant"
        assert len(messages[1].content) == 2  # Text and resource
        assert messages[1].content[0].type == "text"
        assert "I've reviewed your CSS" in messages[1].content[0].text
        assert messages[1].content[1].type == "resource"
        assert str(messages[1].content[1].resource.uri) == "resource://improved_styles.css"

    def test_multiple_resources_in_one_message(self):
        """Test handling multiple resources in a single message."""
        # Create a message with multiple resources
        message = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="I need to analyze these files:"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="resource://data1.csv",
                        mimeType="text/csv",
                        text="id,name,value\n1,A,10\n2,B,20"
                    )
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="resource://data2.csv",
                        mimeType="text/csv",
                        text="id,name,value\n3,C,30\n4,D,40"
                    )
                )
            ]
        )

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format([message])

        # Verify structure - should have user delimiter, text, and two resource references
        assert len(delimited) == 6
        assert delimited[0] == "---USER"
        assert "I need to analyze these files:" in delimited[1]
        assert delimited[2] == "---RESOURCE"
        assert delimited[3] == "data1.csv"
        assert delimited[4] == "---RESOURCE"

        # Convert back to multipart
        messages = delimited_format_to_multipart_messages("\n".join(delimited))

        # Verify round-trip conversion
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert len(messages[0].content) == 3  # Text and two resources
        assert messages[0].content[0].type == "text"
        assert messages[0].content[1].type == "resource"
        assert messages[0].content[2].type == "resource"

    def test_image_handling(self):
        """Test handling image content in multipart messages."""
        # Create a message with an image
        message = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Look at this image:"),
                ImageContent(
                    type="image",
                    data="base64EncodedImageData",
                    mimeType="image/png"
                )
            ]
        )

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format([message])

        # In the current implementation, images get a placeholder
        assert len(delimited) == 2
        assert delimited[0] == "---USER"
        assert "Look at this image:" in delimited[1]
        assert "[IMAGE]" in delimited[1]

    @pytest.fixture
    def temp_resource_file(self):
        """Create a temporary file for testing resource handling."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Here's a file with resources:

---RESOURCE
file1.js

---RESOURCE
file2.css

---ASSISTANT
I've analyzed both files.

---RESOURCE
analysis.md""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    def test_save_and_load_with_resources(self, temp_resource_file):
        """Test saving and loading multipart messages with resources."""
        # Create messages with resources
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Check this JSON file:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://config.json",
                            mimeType="application/json",
                            text='{"key": "value"}'
                        )
                    )
                ]
            )
        ]

        # Save to file
        save_messages_to_delimited_file(messages, str(temp_resource_file))

        # Load from file
        loaded_messages = load_messages_from_delimited_file(str(temp_resource_file))

        # Verify structure
        assert len(loaded_messages) == 1
        assert loaded_messages[0].role == "user"
        assert len(loaded_messages[0].content) == 2  # Text and resource
        assert loaded_messages[0].content[0].type == "text"
        assert loaded_messages[0].content[1].type == "resource"
        assert str(loaded_messages[0].content[1].resource.uri) == "resource://config.json"

    def test_guess_mime_type(self):
        """Test guessing MIME types from file extensions."""
        assert guess_mime_type("file.txt") == "text/plain"
        assert guess_mime_type("file.py") == "text/x-python"
        assert guess_mime_type("file.js") in ["application/javascript", "text/javascript"]
        assert guess_mime_type("file.json") == "application/json"
        assert guess_mime_type("file.html") == "text/html"
        assert guess_mime_type("file.css") == "text/css"
        assert guess_mime_type("file.png") == "image/png"
        assert guess_mime_type("file.jpg") == "image/jpeg"
        assert guess_mime_type("file.jpeg") == "image/jpeg"
        
        # Unknown extension should default to text/plain
        assert guess_mime_type("file.unknown") == "text/plain"

    def test_round_trip_with_mime_types(self):
        """Test round-trip conversion preserving MIME type information."""
        # Original message with different MIME types
        original_messages = [
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Here are some files:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://script.js",
                            mimeType="application/javascript",
                            text="function hello() { return 'Hello!'; }"
                        )
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://style.css",
                            mimeType="text/css",
                            text="body { color: blue; }"
                        )
                    )
                ]
            )
        ]
        
        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(original_messages)
        delimited_text = "\n".join(delimited_content)
        
        # Convert back to multipart
        result_messages = delimited_format_to_multipart_messages(delimited_text)
        
        # Verify structure
        assert len(result_messages) == 1
        assert result_messages[0].role == "user"
        assert len(result_messages[0].content) == 3  # Text and two resources
        
        # The resource URIs should be preserved
        resources = [content for content in result_messages[0].content if content.type == "resource"]
        assert len(resources) == 2
        
        # Resource URIs should be preserved
        resource_uris = [str(resource.resource.uri) for resource in resources]
        assert "resource://script.js" in resource_uris
        assert "resource://style.css" in resource_uris
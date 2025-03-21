import unittest
import asyncio
import tempfile
import base64
import os
from pathlib import Path

from mcp.server.lowlevel.helper_types import ReadResourceContents

from mcp_agent.mcp.prompts.prompt_server import (
    create_resource_handler,
    create_resource_uri,
)
from mcp_agent.mcp import mime_utils


class TestPromptServerResourceHandling(unittest.TestCase):
    """Tests for resource handling in the prompt server."""

    def test_create_resource_uri(self):
        """Test that resource URIs are created correctly."""
        # Create from a simple filename
        uri = create_resource_uri("test.txt")
        self.assertEqual(uri, "resource://fast-agent/test.txt")
        
        # Create from a path
        uri = create_resource_uri("/path/to/file.pdf")
        self.assertEqual(uri, "resource://fast-agent/file.pdf")
        
        # Create from a Path object
        uri = create_resource_uri(Path("/path/to/image.png"))
        self.assertEqual(uri, "resource://fast-agent/image.png")

    def test_text_resource_handler(self):
        """Test that text resources are handled correctly."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("This is a test text file.")
            temp_txt_path = Path(f.name)
        
        try:
            # Get the resource handler
            mime_type = mime_utils.guess_mime_type(str(temp_txt_path))
            handler = create_resource_handler(temp_txt_path, mime_type)
            
            # Run the handler
            result = asyncio.run(handler())
            
            # Check that we get an iterable of ReadResourceContents
            self.assertTrue(hasattr(result, '__iter__'))
            result_list = list(result)
            self.assertEqual(len(result_list), 1)
            
            content_item = result_list[0]
            self.assertIsInstance(content_item, ReadResourceContents)
            self.assertEqual(content_item.mime_type, "text/plain")
            self.assertEqual(content_item.content, "This is a test text file.")
        finally:
            # Clean up
            os.unlink(temp_txt_path)

    def test_binary_resource_handler(self):
        """Test that binary resources are handled correctly."""
        # Create a temporary binary file (a simple PDF-like content)
        pdf_content = b"%PDF-1.0\nThis is a test PDF file.\n%%EOF"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as f:
            f.write(pdf_content)
            temp_pdf_path = Path(f.name)
        
        try:
            # Get the resource handler
            mime_type = mime_utils.guess_mime_type(str(temp_pdf_path))
            handler = create_resource_handler(temp_pdf_path, mime_type)
            
            # Run the handler
            result = asyncio.run(handler())
            
            # Check that we get an iterable of ReadResourceContents
            self.assertTrue(hasattr(result, '__iter__'))
            result_list = list(result)
            self.assertEqual(len(result_list), 1)
            
            content_item = result_list[0]
            self.assertIsInstance(content_item, ReadResourceContents)
            self.assertEqual(content_item.mime_type, "application/pdf")
            
            # Verify the content is binary and matches
            self.assertEqual(content_item.content, pdf_content)
        finally:
            # Clean up
            os.unlink(temp_pdf_path)

    def test_image_resource_handler(self):
        """Test that image resources are handled correctly."""
        # Create a temporary "image" file (not a real image, just binary data)
        img_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode="wb") as f:
            f.write(img_content)
            temp_img_path = Path(f.name)
        
        try:
            # Get the resource handler
            mime_type = mime_utils.guess_mime_type(str(temp_img_path))
            handler = create_resource_handler(temp_img_path, mime_type)
            
            # Run the handler
            result = asyncio.run(handler())
            
            # Check that we get an iterable of ReadResourceContents
            self.assertTrue(hasattr(result, '__iter__'))
            result_list = list(result)
            self.assertEqual(len(result_list), 1)
            
            content_item = result_list[0]
            self.assertIsInstance(content_item, ReadResourceContents)
            self.assertEqual(content_item.mime_type, "image/png")
            
            # Verify the content is binary and matches
            self.assertEqual(content_item.content, img_content)
        finally:
            # Clean up
            os.unlink(temp_img_path)


if __name__ == "__main__":
    unittest.main()
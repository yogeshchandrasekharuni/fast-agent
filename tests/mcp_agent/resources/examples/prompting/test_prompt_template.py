"""
Unit tests for the prompt template module.
"""

import os
import pytest
import tempfile
from pathlib import Path

from mcp_agent.resources.examples.prompting.prompt_template import (
    PromptTemplate,
    PromptContent,
    PromptTemplateLoader,
    PromptMetadata,
)


class TestPromptContent:
    """Tests for the PromptContent class"""

    def test_apply_substitutions(self):
        """Test substituting variables in content"""
        content = PromptContent(
            text="Hello {{name}}! Your age is {{age}}.", role="user"
        )
        context = {"name": "Alice", "age": 30}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Alice! Your age is 30."
        assert result.role == "user"

    def test_apply_substitutions_missing_var(self):
        """Test substituting with missing variables"""
        content = PromptContent(
            text="Hello {{name}}! Your age is {{age}}.", role="user"
        )
        context = {"name": "Bob"}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Bob! Your age is {{age}}."
        assert result.role == "user"

    def test_apply_substitutions_no_vars(self):
        """Test substituting with no variables in content"""
        content = PromptContent(text="Hello world!", role="assistant")
        context = {"name": "Charlie"}

        result = content.apply_substitutions(context)

        assert result.text == "Hello world!"
        assert result.role == "assistant"


class TestPromptTemplate:
    """Tests for the PromptTemplate class"""

    def test_simple_mode(self):
        """Test parsing a simple template with no delimiters"""
        template_text = "Hello {{name}}!\nHow are you?"
        template = PromptTemplate(template_text)

        assert len(template.content_sections) == 1
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].text == template_text
        assert template.template_variables == {"name"}

    def test_delimited_mode(self):
        """Test parsing a template with delimiters"""
        template_text = """---USER
Hello {{name}}!

---ASSISTANT
Hi {{name}}! How can I help you today?

---USER
Tell me about {{topic}}.
"""
        template = PromptTemplate(template_text)

        assert len(template.content_sections) == 3
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].text == "Hello {{name}}!"
        assert template.content_sections[1].role == "assistant"
        assert (
            template.content_sections[1].text
            == "Hi {{name}}! How can I help you today?"
        )
        assert template.content_sections[2].role == "user"
        assert template.content_sections[2].text == "Tell me about {{topic}}."
        assert template.template_variables == {"name", "topic"}

    def test_custom_delimiters(self):
        """Test parsing a template with custom delimiters"""
        template_text = """#USER
Hello {{name}}!

#ASSISTANT
Hi there!
"""
        delimiter_map = {"#USER": "user", "#ASSISTANT": "assistant"}
        template = PromptTemplate(template_text, delimiter_map)

        assert len(template.content_sections) == 2
        assert template.content_sections[0].role == "user"
        assert template.content_sections[1].role == "assistant"
        assert template.template_variables == {"name"}

    def test_apply_substitutions(self):
        """Test applying substitutions to an entire template"""
        template_text = """---USER
Hello {{name}}!

---ASSISTANT
Hi {{name}}! How can I help you today?

---USER
Tell me about {{topic}}.
"""
        template = PromptTemplate(template_text)
        context = {"name": "Dave", "topic": "Python"}

        result = template.apply_substitutions(context)

        assert len(result) == 3
        assert result[0].text == "Hello Dave!"
        assert result[1].text == "Hi Dave! How can I help you today?"
        assert result[2].text == "Tell me about Python."


class TestPromptTemplateLoader:
    """Tests for the PromptTemplateLoader class"""

    @pytest.fixture
    def temp_template_file(self):
        """Create a temporary template file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""Hello {{name}}!
            
This is a test prompt with {{variable}} substitution.""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    @pytest.fixture
    def temp_delimited_file(self):
        """Create a temporary delimited template file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Hello {{name}}!

---ASSISTANT
Nice to meet you, {{name}}!

---RESOURCE
some_resource.txt""")
            tf_path = Path(tf.name)
            
        # Create the resource file in the same directory
        resource_path = tf_path.parent / "some_resource.txt"
        with open(resource_path, "w", encoding="utf-8") as rf:
            rf.write("This is some resource content")

        yield tf_path

        # Cleanup
        os.unlink(tf_path)
        if resource_path.exists():
            os.unlink(resource_path)

    def test_load_from_file(self, temp_template_file):
        """Test loading a template from a file"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_template_file)

        assert isinstance(template, PromptTemplate)
        assert len(template.content_sections) == 1
        assert template.template_variables == {"name", "variable"}

    def test_get_metadata_simple(self, temp_template_file):
        """Test getting metadata from a simple template file"""
        loader = PromptTemplateLoader()
        metadata = loader.get_metadata(temp_template_file)

        assert isinstance(metadata, PromptMetadata)
        assert metadata.name == temp_template_file.stem
        # The description format can be either the first line or "Simple prompt: filename"
        # so we just check that we got a reasonable description
        assert (
            metadata.description.startswith("Simple prompt:")
            or "Hello" in metadata.description
        )
        assert metadata.template_variables == {"name", "variable"}
        assert metadata.resource_paths == []
        assert metadata.file_path == temp_template_file

    def test_get_metadata_delimited(self, temp_delimited_file):
        """Test getting metadata from a delimited template file"""
        loader = PromptTemplateLoader()
        metadata = loader.get_metadata(temp_delimited_file)

        assert isinstance(metadata, PromptMetadata)
        assert metadata.name == temp_delimited_file.stem
        # Check for the new format with role in brackets
        assert "[USER]" in metadata.description
        assert "Hello" in metadata.description
        # Make sure filename is not in the description
        assert temp_delimited_file.stem not in metadata.description
        assert metadata.template_variables == {"name"}
        assert metadata.resource_paths == ["some_resource.txt"]
        assert metadata.file_path == temp_delimited_file
        
    def test_resource_inclusion(self, temp_delimited_file):
        """Test that resources are properly included in the template"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_delimited_file)
        
        # Check that we have the right number of sections
        assert len(template.content_sections) == 3
        
        # Check that the resource section contains the content of the file, not just the path
        resource_section = template.content_sections[2]
        assert resource_section.role == "resource"
        assert resource_section.text == "This is some resource content"
        assert "some_resource.txt" not in resource_section.text


# Integration test with realistic examples
class TestIntegration:
    """Integration tests with realistic examples"""

    @pytest.fixture
    def simple_prompt_file(self):
        """Create a simple prompt file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""Hello, World.

This is {{blah}} foo""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    @pytest.fixture
    def delimited_prompt_file(self):
        """Create a delimited prompt file for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
I want to learn about {{topic}}.

---ASSISTANT
I'd be happy to tell you about {{topic}}!

Here are some key points about {{topic}}:
1. It's very interesting
2. It has a rich history
3. Many people study it

Would you like to know more about any specific aspect of {{topic}}?""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    def test_simple_prompt_substitution(self, simple_prompt_file):
        """Test substituting variables in a simple prompt file"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(simple_prompt_file)

        # Verify template variables
        assert template.template_variables == {"blah"}

        # Apply substitutions
        context = {"blah": "substituted"}
        result = template.apply_substitutions(context)

        # Verify result
        assert len(result) == 1
        assert result[0].role == "user"
        assert "This is substituted foo" in result[0].text

    def test_delimited_prompt_substitution(self, delimited_prompt_file):
        """Test substituting variables in a delimited prompt file"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(delimited_prompt_file)

        # Verify template variables
        assert template.template_variables == {"topic"}

        # Apply substitutions
        context = {"topic": "Python programming"}
        result = template.apply_substitutions(context)

        # Verify result
        assert len(result) == 2
        assert result[0].role == "user"
        assert "I want to learn about Python programming." in result[0].text

        assert result[1].role == "assistant"
        assert "I'd be happy to tell you about Python programming!" in result[1].text
        assert "Here are some key points about Python programming:" in result[1].text

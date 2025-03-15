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
        assert result.resources == []

    def test_apply_substitutions_missing_var(self):
        """Test substituting with missing variables"""
        content = PromptContent(
            text="Hello {{name}}! Your age is {{age}}.", role="user"
        )
        context = {"name": "Bob"}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Bob! Your age is {{age}}."
        assert result.role == "user"

    def test_apply_substitutions_with_resources(self):
        """Test substituting variables in content with resources"""
        content = PromptContent(
            text="Hello {{name}}! Your age is {{age}}.",
            role="user",
            resources=["data_{{name}}.txt", "profile_{{age}}.json"]
        )
        context = {"name": "Alice", "age": 30}

        result = content.apply_substitutions(context)

        assert result.text == "Hello Alice! Your age is 30."
        assert result.role == "user"
        assert result.resources == ["data_Alice.txt", "profile_30.json"]


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
        assert template.content_sections[0].resources == []

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

    def test_resources_in_template(self):
        """Test parsing a template with resources"""
        template_text = """---USER
Hello! Check out this resource:

---RESOURCE
sample.txt

What do you think?

---ASSISTANT
I've analyzed the resource and created a response:

---RESOURCE
response.txt

Let me know if you need more details.
"""
        template = PromptTemplate(template_text)

        # Should have 2 sections (user and assistant), each with a resource
        assert len(template.content_sections) == 2
        
        # Check user section
        assert template.content_sections[0].role == "user"
        assert "Hello! Check out this resource:" in template.content_sections[0].text
        assert "What do you think?" in template.content_sections[0].text
        assert template.content_sections[0].resources == ["sample.txt"]
        
        # Check assistant section
        assert template.content_sections[1].role == "assistant"
        assert "I've analyzed the resource" in template.content_sections[1].text
        assert "Let me know if you need more details." in template.content_sections[1].text
        assert template.content_sections[1].resources == ["response.txt"]

    def test_multiple_resources_in_template(self):
        """Test parsing a template with multiple resources per section"""
        template_text = """---USER
Let me share some files with you:

---RESOURCE
file1.txt

---RESOURCE
file2.txt

What do you think of these?

---ASSISTANT
I've analyzed both files:

---RESOURCE
analysis1.txt

---RESOURCE
analysis2.txt

Here are my thoughts.
"""
        template = PromptTemplate(template_text)

        # Should have 2 sections (user and assistant), each with 2 resources
        assert len(template.content_sections) == 2
        
        # Check user section
        assert template.content_sections[0].role == "user"
        assert template.content_sections[0].resources == ["file1.txt", "file2.txt"]
        
        # Check assistant section
        assert template.content_sections[1].role == "assistant"
        assert template.content_sections[1].resources == ["analysis1.txt", "analysis2.txt"]

    def test_apply_substitutions(self):
        """Test applying substitutions to an entire template"""
        template_text = """---USER
Hello {{name}}!

---RESOURCE
data_{{name}}.txt

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
        assert result[0].resources == ["data_Dave.txt"]
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

---RESOURCE
some_resource.txt

---ASSISTANT
Nice to meet you, {{name}}!

---RESOURCE
another_resource.txt""")
            tf_path = Path(tf.name)
            
        # Create the resource files in the same directory
        resource_path1 = tf_path.parent / "some_resource.txt"
        resource_path2 = tf_path.parent / "another_resource.txt"
        with open(resource_path1, "w", encoding="utf-8") as rf:
            rf.write("This is some resource content")
        with open(resource_path2, "w", encoding="utf-8") as rf:
            rf.write("This is another resource content")

        yield tf_path

        # Cleanup
        os.unlink(tf_path)
        if resource_path1.exists():
            os.unlink(resource_path1)
        if resource_path2.exists():
            os.unlink(resource_path2)

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
        # Should find both resources
        assert set(metadata.resource_paths) == {"some_resource.txt", "another_resource.txt"}
        assert metadata.file_path == temp_delimited_file
        
    def test_load_template_with_resources(self, temp_delimited_file):
        """Test loading a template with resources"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(temp_delimited_file)
        
        # Check that we have the right number of sections
        assert len(template.content_sections) == 2
        
        # Check that resources are properly associated with their sections
        user_section = template.content_sections[0]
        assistant_section = template.content_sections[1]
        
        assert user_section.role == "user"
        assert "Hello {{name}}!" in user_section.text
        assert user_section.resources == ["some_resource.txt"]
        
        assert assistant_section.role == "assistant"
        assert "Nice to meet you, {{name}}!" in assistant_section.text
        assert assistant_section.resources == ["another_resource.txt"]


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

    @pytest.fixture
    def resource_prompt_file(self):
        """Create a prompt file with resources for testing"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Can you analyze this {{language}} code?

---RESOURCE
sample_{{language}}.txt

---ASSISTANT
Here's my analysis of your {{language}} code:

---RESOURCE
analysis_{{language}}.txt

Would you like me to explain anything in more detail?""")
            tf_path = Path(tf.name)
            
        # Create sample resource files
        sample_path = tf_path.parent / "sample_python.txt"
        analysis_path = tf_path.parent / "analysis_python.txt"
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("def hello():\n    print('Hello, world!')")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write("# Analysis\nYour function looks good but could use a docstring.")

        yield tf_path

        # Cleanup
        os.unlink(tf_path)
        if sample_path.exists():
            os.unlink(sample_path)
        if analysis_path.exists():
            os.unlink(analysis_path)

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

    def test_resource_prompt_substitution(self, resource_prompt_file):
        """Test substituting variables in a prompt file with resources"""
        loader = PromptTemplateLoader()
        template = loader.load_from_file(resource_prompt_file)

        # Verify template variables
        assert template.template_variables == {"language"}

        # Apply substitutions
        context = {"language": "python"}
        result = template.apply_substitutions(context)

        # Verify result
        assert len(result) == 2
        
        # Check user section
        assert result[0].role == "user"
        assert "Can you analyze this python code?" in result[0].text
        assert result[0].resources == ["sample_python.txt"]

        # Check assistant section
        assert result[1].role == "assistant"
        assert "Here's my analysis of your python code:" in result[1].text
        assert result[1].resources == ["analysis_python.txt"]
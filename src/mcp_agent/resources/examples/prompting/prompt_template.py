"""
Prompt Template Module

Handles prompt templating, variable extraction, and substitution for the prompt server.
Provides clean, testable classes for managing template substitution.
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

from pydantic import BaseModel


class PromptMetadata(BaseModel):
    """Metadata about a prompt file"""

    name: str
    description: str
    template_variables: Set[str] = set()
    resource_paths: List[str] = []
    file_path: Path


class PromptContent(BaseModel):
    """Content of a prompt, which may include template variables"""

    text: str
    role: str = "user"

    def apply_substitutions(self, context: Dict[str, Any]) -> "PromptContent":
        """Apply variable substitutions to the text"""
        result = self.text
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))

        return PromptContent(text=result, role=self.role)


class PromptTemplate:
    """
    A template for a prompt that can have variables substituted.
    """

    def __init__(
        self,
        template_text: str,
        delimiter_map: Optional[Dict[str, str]] = None,
        template_file_path: Optional[Path] = None,
    ):
        """
        Initialize a prompt template.

        Args:
            template_text: The text of the template
            delimiter_map: Optional map of delimiters to roles (e.g. {"---USER": "user"})
            template_file_path: Optional path to the template file (for resource resolution)
        """
        self.template_text = template_text
        self.template_file_path = template_file_path
        self.delimiter_map = delimiter_map or {
            "---USER": "user",
            "---ASSISTANT": "assistant",
            "---RESOURCE": "resource",
        }
        self._template_variables = self._extract_template_variables(template_text)
        self._parsed_content = self._parse_template()

    @property
    def template_variables(self) -> Set[str]:
        """Get the template variables in this template"""
        return self._template_variables

    @property
    def content_sections(self) -> List[PromptContent]:
        """Get the parsed content sections"""
        return self._parsed_content

    def apply_substitutions(self, context: Dict[str, Any]) -> List[PromptContent]:
        """
        Apply variable substitutions to the template.

        Args:
            context: Dictionary of variable names to values

        Returns:
            List of PromptContent with substitutions applied
        """
        result = []
        for section in self._parsed_content:
            result.append(section.apply_substitutions(context))
        return result

    def _extract_template_variables(self, text: str) -> Set[str]:
        """Extract template variables from text using regex"""
        variable_pattern = r"{{([^}]+)}}"
        matches = re.findall(variable_pattern, text)
        return set(matches)

    def _parse_template(self) -> List[PromptContent]:
        """
        Parse the template into sections based on delimiters.
        If no delimiters are found, treat the entire template as a single user message.

        Resources are handled specially - if a resource section references a file,
        the file's contents are loaded and included in the template.
        """
        lines = self.template_text.split("\n")

        # Check if we're in simple mode (no delimiters)
        first_non_empty_line = next((line for line in lines if line.strip()), "")
        delimiter_values = set(self.delimiter_map.keys())

        is_simple_mode = (
            first_non_empty_line and first_non_empty_line not in delimiter_values
        )

        if is_simple_mode:
            # Simple mode: treat the entire content as a single user message
            return [PromptContent(text=self.template_text, role="user")]

        # Standard mode with delimiters
        sections = []
        current_role = None
        current_content = ""

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() in self.delimiter_map:
                # Save previous section if there was one
                if current_role is not None and current_content:
                    sections.append(
                        PromptContent(text=current_content.strip(), role=current_role)
                    )

                # Start new section
                current_role = self.delimiter_map[line.strip()]
                current_content = ""

                # Special handling for resource sections
                if current_role == "resource" and i + 1 < len(lines):
                    resource_path = lines[i + 1].strip()
                    # Try to load the resource file
                    try:
                        # First look for the file in the same directory as the template
                        if (
                            hasattr(self, "template_file_path")
                            and self.template_file_path
                        ):
                            resource_file = (
                                Path(self.template_file_path).parent / resource_path
                            )
                            if resource_file.exists() and resource_file.is_file():
                                with open(resource_file, "r", encoding="utf-8") as f:
                                    current_content = f.read()
                                # Skip the resource path line
                                i += 1
                    except Exception:
                        # If there's an error loading the resource, fall back to just using the path
                        current_content = resource_path
            elif current_role is not None:
                current_content += line + "\n"
            i += 1

        # Add the last section if there is one
        if current_role is not None and current_content:
            sections.append(
                PromptContent(text=current_content.strip(), role=current_role)
            )

        return sections


class PromptTemplateLoader:
    """
    Loads and processes prompt templates from files.
    """

    def __init__(self, delimiter_map: Optional[Dict[str, str]] = None):
        """
        Initialize the loader with optional custom delimiters.

        Args:
            delimiter_map: Optional map of delimiters to roles
        """
        self.delimiter_map = delimiter_map or {
            "---USER": "user",
            "---ASSISTANT": "assistant",
            "---RESOURCE": "resource",
        }

    def load_from_file(self, file_path: Path) -> PromptTemplate:
        """
        Load a prompt template from a file.

        Args:
            file_path: Path to the template file

        Returns:
            A PromptTemplate object
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return PromptTemplate(content, self.delimiter_map, template_file_path=file_path)

    def get_metadata(self, file_path: Path) -> PromptMetadata:
        """
        Analyze a prompt file to extract metadata and template variables.

        Args:
            file_path: Path to the template file

        Returns:
            PromptMetadata with information about the template
        """
        template = self.load_from_file(file_path)

        # Generate a description based on content
        lines = template.template_text.split("\n")
        first_non_empty_line = next((line for line in lines if line.strip()), "")

        # Check if we're in simple mode
        is_simple_mode = (
            first_non_empty_line and first_non_empty_line not in self.delimiter_map
        )

        if is_simple_mode:
            # In simple mode, use first line as description if it seems like one
            first_line = lines[0].strip() if lines else ""
            if (
                len(first_line) < 60
                and "{{" not in first_line
                and "}}" not in first_line
            ):
                description = first_line
            else:
                description = f"Simple prompt: {file_path.stem}"
        else:
            # Regular mode - find text after first delimiter for the description
            description = file_path.stem

            # Look for first delimiter and role
            first_role = None
            first_content_index = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped in self.delimiter_map:
                    first_role = self.delimiter_map[stripped]
                    first_content_index = i + 1
                    break

            if first_role and first_content_index and first_content_index < len(lines):
                # Get up to 3 non-empty lines after the delimiter for a preview
                preview_lines = []
                for j in range(
                    first_content_index, min(first_content_index + 10, len(lines))
                ):
                    stripped = lines[j].strip()
                    if stripped and stripped not in self.delimiter_map:
                        preview_lines.append(stripped)
                        if len(preview_lines) >= 3:
                            break

                if preview_lines:
                    preview = " ".join(preview_lines)
                    if len(preview) > 50:
                        preview = preview[:47] + "..."
                    # Include role in the description but not the filename
                    description = f"[{first_role.upper()}] {preview}"

        # Extract resource paths (after RESOURCE delimiters)
        resource_paths = []
        for i, line in enumerate(lines):
            if line.strip() == "---RESOURCE":
                if i + 1 < len(lines) and lines[i + 1].strip():
                    resource_paths.append(lines[i + 1].strip())

        return PromptMetadata(
            name=file_path.stem,
            description=description,
            template_variables=template.template_variables,
            resource_paths=resource_paths,
            file_path=file_path,
        )

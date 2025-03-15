"""
FastMCP Prompt Server V2

A server that loads prompts from text files with simple delimiters and serves them via MCP.
Uses the prompt_template module for clean, testable handling of prompt templates.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List

import httpx

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage, Message
from mcp.types import TextContent, EmbeddedResource, TextResourceContents

from mcp_agent.resources.examples.prompting.prompt_template import (
    PromptTemplateLoader,
    PromptMetadata,
    PromptContent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prompt_server")

# Create FastMCP server
mcp = FastMCP("Prompt Server")


class PromptConfig(PromptMetadata):
    """Configuration for the prompt server"""

    prompt_files: List[Path] = []
    user_delimiter: str = "---USER"
    assistant_delimiter: str = "---ASSISTANT"
    resource_delimiter: str = "---RESOURCE"
    http_timeout: float = 10.0
    transport: str = "stdio"
    port: int = 8000


# Will be initialized with command line args
config = None

# We'll maintain registries of all exposed resources and prompts
exposed_resources = {}
prompt_registry = {}


def guess_mime_type(file_path: str) -> str:
    """Guess the MIME type based on the file extension"""
    extension = Path(file_path).suffix.lower()
    mime_types = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".html": "text/html",
        ".css": "text/css",
        ".csv": "text/csv",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
    }
    return mime_types.get(extension, "application/octet-stream")


async def fetch_remote_resource(url: str) -> tuple[str, str]:
    """Fetch a remote resource from a URL"""
    async with httpx.AsyncClient(timeout=config.http_timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.text

        # Get the content type or guess from URL
        mime_type = response.headers.get("content-type", "").split(";")[0]
        if not mime_type:
            mime_type = guess_mime_type(url)

        return content, mime_type


def load_resource_content(
    resource_path: str, prompt_files: List[Path]
) -> tuple[str, str]:
    """
    Load a resource's content and determine its mime type

    Args:
        resource_path: Path to the resource file
        prompt_files: List of prompt files (to find relative paths)

    Returns:
        Tuple of (content, mime_type)

    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Try to locate the resource file
    resource_file = None
    for prompt_file in prompt_files:
        potential_path = prompt_file.parent / resource_path
        if potential_path.exists():
            resource_file = potential_path
            break

    if resource_file is None or not resource_file.exists():
        raise FileNotFoundError(f"Resource not found: {resource_path}")

    # Load the content and determine mime type
    mime_type = guess_mime_type(str(resource_file))
    with open(resource_file, "r", encoding="utf-8") as f:
        content = f.read()

    return content, mime_type


def create_embedded_resource(
    resource_path: str, content: str, mime_type: str
) -> EmbeddedResource:
    """Create an embedded resource content object"""
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=f"resource://{Path(resource_path).name}",
            text=content,
            mimeType=mime_type,
        ),
    )


def create_messages_with_resources(
    content_sections: List[PromptContent], prompt_files: List[Path]
) -> List[Message]:
    """
    Create a list of messages from content sections, with resources properly handled.
    
    This implementation produces one message for each content section's text,
    followed by separate messages for each resource (with the same role type
    as the section they belong to).

    Args:
        content_sections: List of PromptContent objects
        prompt_files: List of prompt files (to help locate resource files)

    Returns:
        List of Message objects
    """
    messages = []

    for section in content_sections:
        # Determine the message class based on the section role
        message_class = UserMessage if section.role == "user" else AssistantMessage
        
        # Add the text message
        text_message = message_class(
            content=TextContent(type="text", text=section.text)
        )
        messages.append(text_message)

        # Add resource messages if any, with the same role type as the section
        for resource_path in section.resources:
            try:
                resource_content, mime_type = load_resource_content(
                    resource_path, prompt_files
                )
                embedded_resource = create_embedded_resource(
                    resource_path, resource_content, mime_type
                )
                # Resources inherit the role of their section
                resource_message = message_class(content=embedded_resource)
                messages.append(resource_message)
            except Exception as e:
                logger.error(f"Error loading resource {resource_path}: {e}")

    return messages


def prompt_content_to_message(content: PromptContent) -> List[Message]:
    """
    Convert PromptContent to Message objects.

    This version returns the text content and any resources as separate messages,
    each with the same role type (user or assistant).
    
    Returns:
        List of Message objects (one for text content, plus one for each resource)
    """
    messages = []
    
    # Determine message class based on role
    message_class = UserMessage if content.role == "user" else AssistantMessage
    
    # Create and add text message
    text_content = TextContent(type="text", text=content.text)
    messages.append(message_class(content=text_content))
    
    # Add resource messages if any, with the same role
    for resource_path in content.resources:
        try:
            # Here we'd load the resource, but for simple conversion without loading,
            # we'll just create a placeholder. For actual use, use create_messages_with_resources
            resource_content = f"Resource: {resource_path} (not loaded in this method)"
            embedded_resource = EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=f"resource://{Path(resource_path).name}",
                    text=resource_content,
                    mimeType="text/plain",
                ),
            )
            messages.append(message_class(content=embedded_resource))
        except Exception as e:
            logger.error(f"Error creating resource message for {resource_path}: {e}")
    
    return messages


def register_prompt(file_path: Path):
    """Register a prompt file"""
    try:
        # Use our prompt template loader to analyze the file
        loader = PromptTemplateLoader(
            {
                config.user_delimiter: "user",
                config.assistant_delimiter: "assistant",
                config.resource_delimiter: "resource",
            }
        )

        # Get metadata and load the template
        metadata = loader.get_metadata(file_path)
        template = loader.load_from_file(file_path)

        # Ensure unique name
        prompt_name = metadata.name
        if prompt_name in prompt_registry:
            base_name = prompt_name
            suffix = 1
            while prompt_name in prompt_registry:
                prompt_name = f"{base_name}_{suffix}"
                suffix += 1
            metadata.name = prompt_name

        prompt_registry[metadata.name] = metadata
        logger.info(f"Registered prompt: {metadata.name} ({file_path})")

        # Get template variables
        template_vars = list(metadata.template_variables)

        # Handle prompts with template variables
        if template_vars:
            # Create a prompt handler factory that captures template, template_vars and other needed objects
            def create_prompt_handler(template, template_vars, prompt_files):
                # The docstring for our generated function
                docstring = (
                    f"Prompt with template variables: {', '.join(template_vars)}"
                )

                # Define a generic prompt handler that accepts **kwargs
                async def prompt_handler(**kwargs) -> List[Message]:
                    # Build context from parameters
                    context = {}
                    for var in template_vars:
                        if var in kwargs and kwargs[var] is not None:
                            context[var] = kwargs[var]

                    # Apply substitutions to the template
                    content_sections = template.apply_substitutions(context)

                    # Convert to MCP Message objects, handling resources properly
                    return create_messages_with_resources(
                        content_sections, prompt_files
                    )

                # Set the docstring
                prompt_handler.__doc__ = docstring
                return prompt_handler

            # Create the handler using our factory function
            handler = create_prompt_handler(
                template, template_vars, config.prompt_files
            )

            # Register the handler with the correct name and description
            mcp.prompt(name=metadata.name, description=metadata.description)(handler)
        else:
            # No template variables, register a simple prompt handler
            # Create a simple prompt handler for templates without variables
            async def prompt_handler() -> List[Message]:
                """Get a prompt with no variable substitution"""
                # Get the content sections
                content_sections = template.content_sections

                # Convert to MCP Message objects, handling resources properly
                return create_messages_with_resources(
                    content_sections, config.prompt_files
                )

            # Register the prompt handler
            mcp.prompt(name=metadata.name, description=metadata.description)(
                prompt_handler
            )

        # Register any referenced resources in the prompt
        for resource_path in metadata.resource_paths:
            if not resource_path.startswith(("http://", "https://")):
                # It's a local resource
                resource_file = file_path.parent / resource_path
                if resource_file.exists():
                    resource_id = f"resource://{resource_file.name}"

                    # Register the resource if not already registered
                    if resource_id not in exposed_resources:
                        exposed_resources[resource_id] = resource_file
                        mime_type = guess_mime_type(str(resource_file))

                        # Define a closure to capture the current resource_file
                        def create_resource_handler(resource_path):
                            async def get_resource() -> str:
                                with open(resource_path, "r", encoding="utf-8") as f:
                                    return f.read()

                            return get_resource

                        # Register with the correct resource ID
                        mcp.resource(
                            resource_id,
                            description=f"Resource from {file_path.name}",
                            mime_type=mime_type,
                        )(create_resource_handler(resource_file))

                        logger.info(
                            f"Registered resource: {resource_id} ({resource_file})"
                        )
    except Exception as e:
        logger.error(f"Error registering prompt {file_path}: {e}", exc_info=True)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FastMCP Prompt Server")
    parser.add_argument(
        "prompt_files", nargs="+", type=str, help="Prompt files to serve"
    )
    parser.add_argument(
        "--user-delimiter",
        type=str,
        default="---USER",
        help="Delimiter for user messages (default: ---USER)",
    )
    parser.add_argument(
        "--assistant-delimiter",
        type=str,
        default="---ASSISTANT",
        help="Delimiter for assistant messages (default: ---ASSISTANT)",
    )
    parser.add_argument(
        "--resource-delimiter",
        type=str,
        default="---RESOURCE",
        help="Delimiter for resource references (default: ---RESOURCE)",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="Timeout for HTTP requests in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--test", type=str, help="Test a specific prompt without starting the server"
    )

    return parser.parse_args()


async def async_main():
    """Run the FastMCP server (async version)"""
    global config

    # Parse command line arguments
    args = parse_args()

    # Resolve file paths
    prompt_files = []
    for file_path in args.prompt_files:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        prompt_files.append(path.resolve())

    if not prompt_files:
        logger.error("No valid prompt files specified")
        return 1

    # Initialize configuration
    config = PromptConfig(
        name="prompt_server",
        description="FastMCP Prompt Server",
        template_variables=set(),
        resource_paths=[],
        file_path=Path(__file__),
        prompt_files=prompt_files,
        user_delimiter=args.user_delimiter,
        assistant_delimiter=args.assistant_delimiter,
        resource_delimiter=args.resource_delimiter,
        http_timeout=args.http_timeout,
        transport=args.transport,
        port=args.port,
    )

    # Register general file resource handler
    @mcp.resource("file://{path}")
    async def get_file_resource(path: str) -> str:
        """Read a file from the given path."""
        try:
            # First check if it's a relative path from the prompt directory
            for prompt_file in config.prompt_files:
                potential_path = prompt_file.parent / path
                if potential_path.exists():
                    file_path = potential_path
                    break
            else:
                # If not found as relative path, try absolute path
                file_path = Path(path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Resource file not found: {path}")

            mime_type = guess_mime_type(str(file_path))

            # Check if it's a binary file based on mime type
            if mime_type.startswith("text/") or mime_type in [
                "application/json",
                "application/xml",
            ]:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                with open(file_path, "rb") as f:
                    return f.read()
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Error accessing resource at '{path}': {e}")
            raise

    # Register all prompts
    for file_path in config.prompt_files:
        register_prompt(file_path)

    # Print startup info
    logger.info("Starting prompt server")
    logger.info(f"Registered {len(prompt_registry)} prompts")
    logger.info(f"Registered {len(exposed_resources)} resources")
    logger.info(
        f"Using delimiters: {config.user_delimiter}, {config.assistant_delimiter}, {config.resource_delimiter}"
    )

    # If a test prompt was specified, print it and exit
    if args.test:
        if args.test not in prompt_registry:
            logger.error(f"Test prompt not found: {args.test}")
            return 1

        print(f"\nTesting prompt: {args.test}")
        metadata = prompt_registry[args.test]
        print(f"Description: {metadata.description}")
        print(f"Template variables: {', '.join(metadata.template_variables)}")

        # Load and print the template
        loader = PromptTemplateLoader(
            {
                config.user_delimiter: "user",
                config.assistant_delimiter: "assistant",
                config.resource_delimiter: "resource",
            }
        )
        template = loader.load_from_file(metadata.file_path)

        # Print each content section
        print("\nContent sections:")
        for i, section in enumerate(template.content_sections):
            print(f"\n[{i + 1}] Role: {section.role}")
            print(f"Content: {section.text}")
            if section.resources:
                print(f"Resources: {', '.join(section.resources)}")

        # If there are template variables, test with dummy values
        if metadata.template_variables:
            print("\nTemplate substitution test:")
            test_context = {var: f"[TEST-{var}]" for var in metadata.template_variables}
            applied = template.apply_substitutions(test_context)

            for i, section in enumerate(applied):
                print(f"\n[{i + 1}] Role: {section.role}")
                print(f"Content with substitutions: {section.text}")
                if section.resources:
                    print(
                        f"Resources with substitutions: {', '.join(section.resources)}"
                    )

        return 0

    # Start the server with the specified transport
    if config.transport == "stdio":
        await mcp.run_stdio_async()
    else:  # sse
        await mcp.run_sse_async(port=config.port)


def main():
    """Run the FastMCP server"""
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

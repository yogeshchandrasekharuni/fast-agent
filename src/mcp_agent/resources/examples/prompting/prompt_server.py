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


def prompt_content_to_message(content: PromptContent) -> Message:
    """Convert PromptContent to a Message object"""
    if content.role == "user":
        return UserMessage(content.text)
    else:
        return AssistantMessage(content.text)


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
            # Define a dynamic function with the correct signature
            param_str = ", ".join([f"{var}: str = None" for var in template_vars])

            # Define the function with proper typed parameters
            exec_globals = {
                "Path": Path,
                "List": List,
                "Message": Message,
                "template_vars": template_vars,
                "template": template,
                "prompt_content_to_message": prompt_content_to_message,
                "metadata": metadata,
            }

            exec_code = f"""
async def prompt_handler({param_str}) -> List[Message]:
    \"\"\"Prompt with template variables: {", ".join(template_vars)}\"\"\"
    # Build context from parameters
    context = {{}}
    for var in template_vars:
        value = locals().get(var)
        if value is not None:
            context[var] = value
    
    # Apply substitutions to the template
    content_sections = template.apply_substitutions(context)
    
    # Convert to MCP Message objects
    return [prompt_content_to_message(section) for section in content_sections]
"""
            # Execute the function definition
            exec(exec_code, exec_globals)

            # Register the prompt handler
            mcp.prompt(name=metadata.name, description=metadata.description)(
                exec_globals["prompt_handler"]
            )
        else:
            # No template variables, register a simple prompt handler
            @mcp.prompt(name=metadata.name, description=metadata.description)
            async def prompt_handler() -> List[Message]:
                """Get a prompt with no variable substitution"""
                # Get the content sections
                content_sections = template.content_sections

                # Convert to MCP Message objects
                return [
                    prompt_content_to_message(section) for section in content_sections
                ]

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

        # If there are template variables, test with dummy values
        if metadata.template_variables:
            print("\nTemplate substitution test:")
            test_context = {var: f"[TEST-{var}]" for var in metadata.template_variables}
            applied = template.apply_substitutions(test_context)

            for i, section in enumerate(applied):
                print(f"\n[{i + 1}] Role: {section.role}")
                print(f"Content with substitutions: {section.text}")

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

from typing import Optional, Union

from mcp.types import CallToolResult
from rich.panel import Panel
from rich.text import Text

from mcp_agent import console
from mcp_agent.mcp.mcp_aggregator import SEP

# Constants
HUMAN_INPUT_TOOL_NAME = "__human_input__"


class ConsoleDisplay:
    """
    Handles displaying formatted messages, tool calls, and results to the console.
    This centralizes the UI display logic used by LLM implementations.
    """

    def __init__(self, config=None):
        """
        Initialize the console display handler.

        Args:
            config: Configuration object containing display preferences
        """
        self.config = config

    def show_tool_result(self, result: CallToolResult):
        """Display a tool result in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        style = "red" if result.isError else "magenta"

        panel = Panel(
            Text(str(result.content), overflow="..."),
            title="[TOOL RESULT]",
            title_align="right",
            style=style,
            border_style="bold white",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(result.content)) > 360:
                panel.height = 8

        console.console.print(panel)
        console.console.print("\n")

    def show_oai_tool_result(self, result):
        """Display an OpenAI tool result in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        panel = Panel(
            Text(str(result), overflow="..."),
            title="[TOOL RESULT]",
            title_align="right",
            style="magenta",
            border_style="bold white",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(result)) > 360:
                panel.height = 8

        console.console.print(panel)
        console.console.print("\n")

    def show_tool_call(self, available_tools, tool_name, tool_args):
        """Display a tool call in a formatted panel."""
        if not self.config or not self.config.logger.show_tools:
            return

        display_tool_list = self._format_tool_list(available_tools, tool_name)

        panel = Panel(
            Text(str(tool_args), overflow="ellipsis"),
            title="[TOOL CALL]",
            title_align="left",
            style="magenta",
            border_style="bold white",
            subtitle=display_tool_list,
            subtitle_align="left",
            padding=(1, 2),
        )

        if self.config and self.config.logger.truncate_tools:
            if len(str(tool_args)) > 360:
                panel.height = 8

        console.console.print(panel)
        console.console.print("\n")

    def _format_tool_list(self, available_tools, selected_tool_name):
        """Format the list of available tools, highlighting the selected one."""
        display_tool_list = Text()
        for display_tool in available_tools:
            # Handle both OpenAI and Anthropic tool formats
            if isinstance(display_tool, dict):
                if "function" in display_tool:
                    # OpenAI format
                    tool_call_name = display_tool["function"]["name"]
                else:
                    # Anthropic format
                    tool_call_name = display_tool["name"]
            else:
                # Handle potential object format (e.g., Pydantic models)
                tool_call_name = (
                    display_tool.function.name
                    if hasattr(display_tool, "function")
                    else display_tool.name
                )

            parts = (
                tool_call_name.split(SEP)
                if SEP in tool_call_name
                else [tool_call_name, tool_call_name]
            )

            if selected_tool_name.split(SEP)[0] == parts[0]:
                style = (
                    "magenta" if tool_call_name == selected_tool_name else "dim white"
                )
                shortened_name = (
                    parts[1] if len(parts[1]) <= 12 else parts[1][:11] + "…"
                )
                display_tool_list.append(f"[{shortened_name}] ", style)

        return display_tool_list

    async def show_assistant_message(
        self,
        message_text: Union[str, Text],
        aggregator=None,
        highlight_namespaced_tool: str = "",
        title: str = "ASSISTANT",
        name: Optional[str] = None,
    ):
        """Display an assistant message in a formatted panel."""
        if not self.config or not self.config.logger.show_chat:
            return

        display_server_list = Text()

        if aggregator:
            # Add human input tool if available
            tools = await aggregator.list_tools()
            if any(tool.name == HUMAN_INPUT_TOOL_NAME for tool in tools.tools):
                style = (
                    "green"
                    if highlight_namespaced_tool == HUMAN_INPUT_TOOL_NAME
                    else "dim white"
                )
                display_server_list.append("[human] ", style)

            # Add all available servers
            mcp_server_name = (
                highlight_namespaced_tool.split(SEP)[0]
                if SEP in highlight_namespaced_tool
                else highlight_namespaced_tool
            )

            for server_name in await aggregator.list_servers():
                style = "green" if server_name == mcp_server_name else "dim white"
                display_server_list.append(f"[{server_name}] ", style)

        panel = Panel(
            message_text,
            title=f"[{title}]{f' ({name})' if name else ''}",
            title_align="left",
            style="green",
            border_style="bold white",
            padding=(1, 2),
            subtitle=display_server_list,
            subtitle_align="left",
        )
        console.console.print(panel)
        console.console.print("\n")

    def show_user_message(
        self, message, model: Optional[str], chat_turn: int, name: Optional[str] = None
    ):
        """Display a user message in a formatted panel."""
        if not self.config or not self.config.logger.show_chat:
            return

        panel = Panel(
            message,
            title=f"{f'({name}) [USER]' if name else '[USER]'}",
            title_align="right",
            style="blue",
            border_style="bold white",
            padding=(1, 2),
            subtitle=Text(f"{model or 'unknown'} turn {chat_turn}", style="dim white"),
            subtitle_align="left",
        )
        console.console.print(panel)
        console.console.print("\n")

    def show_prompt_loaded(
        self,
        prompt_name: str,
        description: Optional[str] = None,
        message_count: int = 0,
        server_name: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """
        Display information about a loaded prompt template.

        Args:
            prompt_name: The name of the prompt that was loaded
            description: Optional description of the prompt
            message_count: Number of messages added to the conversation history
            server_name: Name of the server the prompt was loaded from
            agent_name: Name of the agent using the prompt
        """
        if not self.config or not self.config.logger.show_tools:
            return

        # Build the subtitle string
        subtitle_parts = []
        if message_count:
            subtitle_parts.append(f"{message_count} messages")
        if server_name:
            subtitle_parts.append(f"from {server_name}")
        if agent_name:
            subtitle_parts.append(f"for {agent_name}")

        subtitle = " ".join(subtitle_parts)

        # Create content text with description if available
        content = Text()
        content.append("Loaded prompt template: ", style="cyan bold")
        content.append(prompt_name, style="cyan")

        if description:
            content.append("\n\n", style="default")
            content.append(description, style="dim white")

        # Create panel
        panel = Panel(
            content,
            title="[PROMPT LOADED]",
            title_align="right",
            style="cyan",
            border_style="bold white",
            padding=(1, 2),
            subtitle=Text(subtitle, style="dim white") if subtitle else None,
            subtitle_align="left",
        )

        console.console.print(panel)
        console.console.print("\n")

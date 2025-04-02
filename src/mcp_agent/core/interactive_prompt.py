"""
Interactive prompt functionality for agents.

This module provides interactive command-line functionality for agents,
extracted from the original AgentApp implementation to support the new DirectAgentApp.

Usage:
    prompt = InteractivePrompt()
    await prompt.prompt_loop(
        send_func=agent_app.send,
        default_agent="default_agent",
        available_agents=["agent1", "agent2"],
        apply_prompt_func=agent_app.apply_prompt
    )
"""

from typing import Dict, List, Optional

from rich import print as rich_print
from rich.console import Console
from rich.table import Table

from mcp_agent.core.enhanced_prompt import (
    get_argument_input,
    get_enhanced_input,
    get_selection_input,
    handle_special_commands,
)
from mcp_agent.progress_display import progress_display


class InteractivePrompt:
    """
    Provides interactive prompt functionality that works with any agent implementation.
    This is extracted from the original AgentApp implementation to support DirectAgentApp.
    """

    def __init__(self, agent_types: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize the interactive prompt.

        Args:
            agent_types: Dictionary mapping agent names to their types for display
        """
        self.agent_types = agent_types or {}

    async def prompt_loop(
        self,
        send_func,
        default_agent: str,
        available_agents: List[str],
        apply_prompt_func=None,
        list_prompts_func=None,
        default: str = "",
    ) -> str:
        """
        Start an interactive prompt session.

        Args:
            send_func: Function to send messages to agents (signature: async (message, agent_name))
            default_agent: Name of the default agent to use
            available_agents: List of available agent names
            apply_prompt_func: Optional function to apply prompts (signature: async (name, args, agent))
            list_prompts_func: Optional function to list available prompts (signature: async (agent_name))
            default: Default message to use when user presses enter

        Returns:
            The result of the interactive session
        """
        agent = default_agent
        if not agent:
            if available_agents:
                agent = available_agents[0]
            else:
                raise ValueError("No default agent available")

        if agent not in available_agents:
            raise ValueError(f"No agent named '{agent}'")

        # Create agent_types dictionary if not provided
        available_agents_set = set(available_agents)

        result = ""
        while True:
            with progress_display.paused():
                # Use the enhanced input method with advanced features
                user_input = await get_enhanced_input(
                    agent_name=agent,
                    default=default,
                    show_default=(default != ""),
                    show_stop_hint=True,
                    multiline=False,  # Default to single-line mode
                    available_agent_names=available_agents,
                    agent_types=self.agent_types,  # Pass agent types for display
                )

                # Handle special commands - pass "True" to enable agent switching
                command_result = await handle_special_commands(user_input, True)

                # Check if we should switch agents
                if isinstance(command_result, dict):
                    if "switch_agent" in command_result:
                        new_agent = command_result["switch_agent"]
                        if new_agent in available_agents_set:
                            agent = new_agent
                            continue
                        else:
                            rich_print(f"[red]Agent '{new_agent}' not found[/red]")
                            continue
                    elif "list_prompts" in command_result and list_prompts_func:
                        # Use the list_prompts_func directly
                        await self._list_prompts(list_prompts_func, agent)
                        continue
                    elif "select_prompt" in command_result and (list_prompts_func and apply_prompt_func):
                        # Handle prompt selection, using both list_prompts and apply_prompt
                        prompt_name = command_result.get("prompt_name")
                        await self._select_prompt(list_prompts_func, apply_prompt_func, agent, prompt_name)
                        continue

                # Skip further processing if command was handled
                if command_result:
                    continue

                if user_input.upper() == "STOP":
                    return result
                if user_input == "":
                    continue

            # Send the message to the agent
            result = await send_func(user_input, agent)

        return result

    async def _list_prompts(self, list_prompts_func, agent_name) -> None:
        """
        List available prompts for an agent.

        Args:
            list_prompts_func: Function to get available prompts
            agent_name: Name of the agent
        """
        from rich import print as rich_print

        try:
            # Directly call the list_prompts function for this agent
            rich_print(f"\n[bold]Fetching prompts for agent [cyan]{agent_name}[/cyan]...[/bold]")
            
            prompt_servers = await list_prompts_func(agent_name)
            
            # Process the returned prompt servers
            if prompt_servers:
                found_prompts = False
                for server_name, prompts_info in prompt_servers.items():
                    if prompts_info and hasattr(prompts_info, "prompts") and prompts_info.prompts:
                        rich_print(f"\n[bold cyan]{server_name}:[/bold cyan]")
                        for prompt in prompts_info.prompts:
                            rich_print(f"  {prompt.name}")
                        found_prompts = True
                    elif isinstance(prompts_info, list) and prompts_info:
                        rich_print(f"\n[bold cyan]{server_name}:[/bold cyan]")
                        for prompt in prompts_info:
                            if isinstance(prompt, dict) and "name" in prompt:
                                rich_print(f"  {prompt['name']}")
                            else:
                                rich_print(f"  {prompt}")
                        found_prompts = True

                if not found_prompts:
                    rich_print("[yellow]No prompts available[/yellow]")
            else:
                rich_print("[yellow]No prompts available[/yellow]")
        except Exception as e:
            import traceback
            rich_print(f"[red]Error listing prompts: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

    async def _select_prompt(self, list_prompts_func, apply_prompt_func, agent_name, requested_name=None) -> None:
        """
        Select and apply a prompt.

        Args:
            list_prompts_func: Function to get available prompts
            apply_prompt_func: Function to apply prompts
            agent_name: Name of the agent
            requested_name: Optional name of the prompt to apply
        """
        # We already imported these at the top
        from rich import print as rich_print

        console = Console()

        try:
            # Get all available prompts directly from the list_prompts function
            rich_print(f"\n[bold]Fetching prompts for agent [cyan]{agent_name}[/cyan]...[/bold]")
            prompt_servers = await list_prompts_func(agent_name)

            if not prompt_servers:
                rich_print("[yellow]No prompts available for this agent[/yellow]")
                return

            # Process fetched prompts
            all_prompts = []
            for server_name, prompts_info in prompt_servers.items():
                if not prompts_info:
                    continue

                # Extract prompts
                prompts = []
                if hasattr(prompts_info, "prompts"):
                    prompts = prompts_info.prompts
                elif isinstance(prompts_info, list):
                    prompts = prompts_info

                # Process each prompt
                for prompt in prompts:
                    # Get basic prompt info
                    prompt_name = getattr(prompt, "name", "Unknown")
                    description = getattr(prompt, "description", "No description")

                    # Extract argument information
                    arg_names = []
                    required_args = []
                    optional_args = []
                    arg_descriptions = {}

                    # Get arguments list
                    arguments = getattr(prompt, "arguments", None)
                    if arguments:
                        for arg in arguments:
                            name = getattr(arg, "name", None)
                            if name:
                                arg_names.append(name)

                                # Store description if available
                                description = getattr(arg, "description", None)
                                if description:
                                    arg_descriptions[name] = description

                                # Check if required
                                if getattr(arg, "required", False):
                                    required_args.append(name)
                                else:
                                    optional_args.append(name)

                    # Create namespaced version
                    namespaced_name = f"{server_name}-{prompt_name}"

                    # Add to collection
                    all_prompts.append(
                        {
                            "server": server_name,
                            "name": prompt_name,
                            "namespaced_name": namespaced_name,
                            "description": description,
                            "arg_count": len(arg_names),
                            "arg_names": arg_names,
                            "required_args": required_args,
                            "optional_args": optional_args,
                            "arg_descriptions": arg_descriptions,
                        }
                    )

            if not all_prompts:
                rich_print("[yellow]No prompts available for this agent[/yellow]")
                return

            # Sort prompts by server then name
            all_prompts.sort(key=lambda p: (p["server"], p["name"]))

            # Handle specifically requested prompt
            if requested_name:
                matching_prompts = [
                    p
                    for p in all_prompts
                    if p["name"] == requested_name or p["namespaced_name"] == requested_name
                ]

                if not matching_prompts:
                    rich_print(f"[red]Prompt '{requested_name}' not found[/red]")
                    rich_print("[yellow]Available prompts:[/yellow]")
                    for p in all_prompts:
                        rich_print(f"  {p['namespaced_name']}")
                    return

                # If exactly one match, use it
                if len(matching_prompts) == 1:
                    selected_prompt = matching_prompts[0]
                else:
                    # Handle multiple matches
                    rich_print(f"[yellow]Multiple prompts match '{requested_name}':[/yellow]")
                    for i, p in enumerate(matching_prompts):
                        rich_print(f"  {i + 1}. {p['namespaced_name']} - {p['description']}")

                    # Get user selection
                    selection = (
                        await get_selection_input("Enter prompt number to select: ", default="1")
                        or ""
                    )

                    try:
                        idx = int(selection) - 1
                        if 0 <= idx < len(matching_prompts):
                            selected_prompt = matching_prompts[idx]
                        else:
                            rich_print("[red]Invalid selection[/red]")
                            return
                    except ValueError:
                        rich_print("[red]Invalid input, please enter a number[/red]")
                        return
            else:
                # Show prompt selection UI
                table = Table(title="Available MCP Prompts")
                table.add_column("#", justify="right", style="cyan")
                table.add_column("Server", style="green")
                table.add_column("Prompt Name", style="bright_blue")
                table.add_column("Description")
                table.add_column("Args", justify="center")

                # Add prompts to table
                for i, prompt in enumerate(all_prompts):
                    required_args = prompt["required_args"]
                    optional_args = prompt["optional_args"]

                    # Format args column
                    if required_args and optional_args:
                        args_display = f"[bold]{len(required_args)}[/bold]+{len(optional_args)}"
                    elif required_args:
                        args_display = f"[bold]{len(required_args)}[/bold]"
                    elif optional_args:
                        args_display = f"{len(optional_args)} opt"
                    else:
                        args_display = "0"

                    table.add_row(
                        str(i + 1),
                        prompt["server"],
                        prompt["name"],
                        prompt["description"] or "No description",
                        args_display,
                    )

                console.print(table)
                prompt_names = [str(i + 1) for i in range(len(all_prompts))]

                # Get user selection
                selection = await get_selection_input(
                    "Enter prompt number to select (or press Enter to cancel): ",
                    options=prompt_names,
                    allow_cancel=True,
                )

                # Handle cancellation
                if not selection or selection.strip() == "":
                    rich_print("[yellow]Prompt selection cancelled[/yellow]")
                    return

                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(all_prompts):
                        selected_prompt = all_prompts[idx]
                    else:
                        rich_print("[red]Invalid selection[/red]")
                        return
                except ValueError:
                    rich_print("[red]Invalid input, please enter a number[/red]")
                    return

            # Get prompt arguments
            required_args = selected_prompt["required_args"]
            optional_args = selected_prompt["optional_args"]
            arg_descriptions = selected_prompt.get("arg_descriptions", {})
            arg_values = {}

            # Show argument info if any
            if required_args or optional_args:
                if required_args and optional_args:
                    rich_print(
                        f"\n[bold]Prompt [cyan]{selected_prompt['name']}[/cyan] requires {len(required_args)} arguments and has {len(optional_args)} optional arguments:[/bold]"
                    )
                elif required_args:
                    rich_print(
                        f"\n[bold]Prompt [cyan]{selected_prompt['name']}[/cyan] requires {len(required_args)} arguments:[/bold]"
                    )
                elif optional_args:
                    rich_print(
                        f"\n[bold]Prompt [cyan]{selected_prompt['name']}[/cyan] has {len(optional_args)} optional arguments:[/bold]"
                    )

                # Collect required arguments
                for arg_name in required_args:
                    description = arg_descriptions.get(arg_name, "")
                    arg_value = await get_argument_input(
                        arg_name=arg_name,
                        description=description,
                        required=True,
                    )
                    if arg_value is not None:
                        arg_values[arg_name] = arg_value

                # Collect optional arguments
                if optional_args:
                    for arg_name in optional_args:
                        description = arg_descriptions.get(arg_name, "")
                        arg_value = await get_argument_input(
                            arg_name=arg_name,
                            description=description,
                            required=False,
                        )
                        if arg_value:
                            arg_values[arg_name] = arg_value

            # Apply the prompt
            rich_print(
                f"\n[bold]Applying prompt [cyan]{selected_prompt['namespaced_name']}[/cyan]...[/bold]"
            )

            # Call apply_prompt function with the prompt name and arguments
            await apply_prompt_func(selected_prompt["namespaced_name"], arg_values, agent_name)

        except Exception as e:
            import traceback

            rich_print(f"[red]Error selecting or applying prompt: {e}[/red]")
            rich_print(f"[dim]{traceback.format_exc()}[/dim]")

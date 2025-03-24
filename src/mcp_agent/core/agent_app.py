"""
Main application wrapper for interacting with agents.
"""

from typing import Optional, Dict, Union, TYPE_CHECKING

from mcp_agent.app import MCPApp
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.progress_display import progress_display
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
)

# Import proxies directly - they handle their own circular imports
from mcp_agent.core.proxies import (
    BaseAgentProxy,
    LLMAgentProxy,
    RouterProxy,
    ChainProxy,
    WorkflowProxy,
)

# Handle possible circular imports with types
if TYPE_CHECKING:
    from mcp_agent.core.types import ProxyDict
else:
    ProxyDict = Dict[str, BaseAgentProxy]


class AgentApp:
    """Main application wrapper"""

    def __init__(self, app: MCPApp, agents: ProxyDict):
        self._app = app
        self._agents = agents
        # Optional: set default agent for direct calls
        self._default = next(iter(agents)) if agents else None

    async def send_prompt(
        self, prompt: PromptMessageMultipart, agent_name: Optional[str] = None
    ) -> str:
        """
        Send a PromptMessageMultipart to an agent

        Args:
            prompt: The PromptMessageMultipart to send
            agent_name: The name of the agent to send to (uses default if None)

        Returns:
            The agent's response as a string
        """
        target = agent_name or self._default
        if not target:
            raise ValueError("No default agent available")

        if target not in self._agents:
            raise ValueError(f"No agent named '{target}'")

        proxy = self._agents[target]
        return await proxy.send_prompt(prompt)

    async def send(
        self,
        message: Union[str, PromptMessageMultipart] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Send a message to the default agent or specified agent

        Args:
            message: Either a string message or a PromptMessageMultipart object
            agent_name: The name of the agent to send to (uses default if None)

        Returns:
            The agent's response as a string
        """
        target = agent_name or self._default
        if not target:
            raise ValueError("No default agent available")

        if target not in self._agents:
            raise ValueError(f"No agent named '{target}'")

        proxy = self._agents[target]
        return await proxy.send(message)

    async def apply_prompt(
        self,
        prompt_name: str,
        arguments: Optional[dict[str, str]] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Apply an MCP Server Prompt by name and return the assistant's response

        Args:
            prompt_name: The name of the prompt to apply
            arguments: Optional dictionary of string arguments to pass to the prompt template
            agent_name: The name of the agent to use (uses default if None)

        Returns:
            The assistant's response as a string
        """
        target = agent_name or self._default
        if not target:
            raise ValueError("No default agent available")

        if target not in self._agents:
            raise ValueError(f"No agent named '{target}'")

        proxy = self._agents[target]
        return await proxy.apply_prompt(prompt_name, arguments)

    async def with_resource(
        self,
        prompt_content: Union[str, PromptMessageMultipart],
        server_name: str,
        resource_name: str,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Create a prompt with the given content and resource, then send it to the agent.

        Args:
            prompt_content: Either a string message or an existing PromptMessageMultipart
            server_name: Name of the MCP server to retrieve the resource from
            resource_name: Name or URI of the resource to retrieve
            agent_name: The name of the agent to use (uses default if None)

        Returns:
            The agent's response as a string
        """
        target = agent_name or self._default
        if not target:
            raise ValueError("No default agent available")

        if target not in self._agents:
            raise ValueError(f"No agent named '{target}'")

        proxy = self._agents[target]
        return await proxy.with_resource(prompt_content, server_name, resource_name)

    async def prompt(self, agent_name: Optional[str] = None, default: str = "") -> str:
        """
        Interactive prompt for sending messages with advanced features.

        Args:
            agent_name: Optional target agent name (uses default if not specified)
            default: Default message to use when user presses enter
        """
        from mcp_agent.core.enhanced_prompt import (
            get_enhanced_input,
            handle_special_commands,
        )

        agent = agent_name or self._default

        if agent not in self._agents:
            raise ValueError(f"No agent named '{agent}'")

        # Pass all available agent names for auto-completion
        available_agents = list(self._agents.keys())

        # Create agent_types dictionary mapping agent names to their types
        agent_types = {}
        for name, proxy in self._agents.items():
            # Determine agent type based on the proxy type
            if isinstance(proxy, LLMAgentProxy):
                # Convert AgentType.BASIC.value ("agent") to "Agent"
                agent_types[name] = "Agent"
            elif isinstance(proxy, RouterProxy):
                agent_types[name] = "Router"
            elif isinstance(proxy, ChainProxy):
                agent_types[name] = "Chain"
            elif isinstance(proxy, WorkflowProxy):
                # For workflow proxies, check the workflow type
                workflow = proxy._workflow
                if isinstance(workflow, Orchestrator):
                    agent_types[name] = "Orchestrator"
                elif isinstance(workflow, ParallelLLM):
                    agent_types[name] = "Parallel"
                elif isinstance(workflow, EvaluatorOptimizerLLM):
                    agent_types[name] = "Evaluator"
                else:
                    agent_types[name] = "Workflow"

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
                    syntax=None,  # Can enable syntax highlighting for code input
                    agent_types=agent_types,  # Pass agent types for display
                )

                # Handle special commands
                command_result = await handle_special_commands(user_input, self)

                # Check if we should switch agents
                if isinstance(command_result, dict):
                    if "switch_agent" in command_result:
                        agent = command_result["switch_agent"]
                        continue
                    elif "list_prompts" in command_result:
                        # Handle listing of prompts
                        from rich import print as rich_print

                        try:
                            # Check if we have any agents with aggregator capabilities
                            found_prompts = False
                            for agent_name, agent_proxy in self._agents.items():
                                # Check if agent has an mcp_aggregator (agent instance)
                                if hasattr(agent_proxy, "_agent") and hasattr(
                                    agent_proxy._agent, "list_prompts"
                                ):
                                    rich_print(
                                        f"\n[bold]Fetching prompts for agent [cyan]{agent_name}[/cyan]...[/bold]"
                                    )
                                    prompt_servers = (
                                        await agent_proxy._agent.list_prompts()
                                    )

                                    if prompt_servers:
                                        found_prompts = True
                                        for (
                                            server_name,
                                            prompts_info,
                                        ) in prompt_servers.items():
                                            if (
                                                prompts_info
                                                and hasattr(prompts_info, "prompts")
                                                and prompts_info.prompts
                                            ):
                                                rich_print(
                                                    f"\n[bold cyan]{server_name}:[/bold cyan]"
                                                )
                                                for prompt in prompts_info.prompts:
                                                    rich_print(f"  {prompt.name}")
                                            elif (
                                                isinstance(prompts_info, list)
                                                and prompts_info
                                            ):
                                                rich_print(
                                                    f"\n[bold cyan]{server_name}:[/bold cyan]"
                                                )
                                                for prompt in prompts_info:
                                                    if (
                                                        isinstance(prompt, dict)
                                                        and "name" in prompt
                                                    ):
                                                        rich_print(
                                                            f"  {prompt['name']}"
                                                        )
                                                    else:
                                                        rich_print(f"  {prompt}")

                            if not found_prompts:
                                rich_print("[yellow]No prompts available[/yellow]")
                        except Exception as e:
                            rich_print(f"[red]Error listing prompts: {e}[/red]")
                        continue
                    elif "select_prompt" in command_result:
                        from rich import print as rich_print
                        from rich.table import Table
                        from rich.console import Console

                        console = Console()

                        # Get the current agent proxy
                        current_proxy = self._agents[agent]

                        # Check if the agent has prompt capabilities
                        if not hasattr(current_proxy, "_agent") or not hasattr(
                            current_proxy._agent, "list_prompts"
                        ):
                            rich_print(
                                f"[red]Current agent '{agent}' does not support prompts[/red]"
                            )
                            continue

                        try:
                            # Create a list to store prompt data for selection
                            all_prompts = []

                            # Get prompts from the current agent
                            rich_print(
                                f"\n[bold]Fetching prompts for agent [cyan]{agent}[/cyan]...[/bold]"
                            )
                            prompt_servers = await current_proxy._agent.list_prompts()

                            if not prompt_servers:
                                rich_print(
                                    "[yellow]No prompts available for this agent[/yellow]"
                                )
                                continue

                            # Process retrieved prompts
                            for server_name, prompts_info in prompt_servers.items():
                                # Skip servers with no prompts
                                if not prompts_info:
                                    continue

                                # Extract prompts from the response
                                prompts = []
                                if hasattr(prompts_info, "prompts"):
                                    prompts = prompts_info.prompts
                                elif isinstance(prompts_info, list):
                                    prompts = prompts_info

                                # Process each prompt
                                for prompt in prompts:
                                    # Basic prompt information
                                    prompt_name = getattr(prompt, "name", "Unknown")
                                    description = getattr(
                                        prompt, "description", "No description"
                                    )

                                    # Extract argument information
                                    arg_names = []
                                    required_args = []
                                    optional_args = []
                                    arg_descriptions = {}

                                    # Get arguments list from prompt (MCP SDK Prompt.arguments)
                                    arguments = getattr(prompt, "arguments", None)
                                    if arguments:
                                        for arg in arguments:
                                            # Each arg is a PromptArgument with name and required fields
                                            name = getattr(arg, "name", None)
                                            if name:
                                                arg_names.append(name)

                                                # Store description if available
                                                description = getattr(
                                                    arg, "description", None
                                                )
                                                if description:
                                                    arg_descriptions[name] = description

                                                # Check if required (default to False per MCP spec)
                                                if getattr(arg, "required", False):
                                                    required_args.append(name)
                                                else:
                                                    optional_args.append(name)

                                    # Create a namespaced version with the server
                                    namespaced_name = f"{server_name}-{prompt_name}"

                                    # Add to our collection
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

                            # If no prompts were found
                            if not all_prompts:
                                rich_print(
                                    "[yellow]No prompts available for this agent[/yellow]"
                                )
                                continue

                            # Sort prompts by server then name
                            all_prompts.sort(key=lambda p: (p["server"], p["name"]))

                            # Check if a specific prompt was requested
                            if (
                                "prompt_name" in command_result
                                and command_result["prompt_name"]
                            ):
                                requested_name = command_result["prompt_name"]
                                # Find the prompt in our list (either by name or namespaced name)
                                matching_prompts = [
                                    p
                                    for p in all_prompts
                                    if p["name"] == requested_name
                                    or p["namespaced_name"] == requested_name
                                ]

                                if not matching_prompts:
                                    rich_print(
                                        f"[red]Prompt '{requested_name}' not found[/red]"
                                    )
                                    rich_print("[yellow]Available prompts:[/yellow]")
                                    for p in all_prompts:
                                        rich_print(f"  {p['namespaced_name']}")
                                    continue

                                # If we found exactly one match, use it
                                if len(matching_prompts) == 1:
                                    selected_prompt = matching_prompts[0]
                                else:
                                    # If multiple matches, show them and ask user to be more specific
                                    rich_print(
                                        f"[yellow]Multiple prompts match '{requested_name}':[/yellow]"
                                    )
                                    for i, p in enumerate(matching_prompts):
                                        rich_print(
                                            f"  {i + 1}. {p['namespaced_name']} - {p['description']}"
                                        )

                                    # Ask user to select one
                                    from mcp_agent.core.enhanced_prompt import (
                                        get_selection_input,
                                    )

                                    selection = await get_selection_input(
                                        "Enter prompt number to select: ", default="1"
                                    )

                                    try:
                                        idx = int(selection) - 1
                                        if 0 <= idx < len(matching_prompts):
                                            selected_prompt = matching_prompts[idx]
                                        else:
                                            rich_print("[red]Invalid selection[/red]")
                                            continue
                                    except ValueError:
                                        rich_print(
                                            "[red]Invalid input, please enter a number[/red]"
                                        )
                                        continue
                            else:
                                # Display prompt selection UI
                                table = Table(title="Available MCP Prompts")
                                table.add_column("#", justify="right", style="cyan")
                                table.add_column("Server", style="green")
                                table.add_column("Prompt Name", style="bright_blue")
                                table.add_column("Description")
                                table.add_column("Args", justify="center")

                                # Add all prompts to the table
                                for i, prompt in enumerate(all_prompts):
                                    # Get argument counts
                                    required_args = prompt["required_args"]
                                    optional_args = prompt["optional_args"]

                                    # Format args column nicely
                                    if required_args and optional_args:
                                        args_display = f"[bold]{len(required_args)}[/bold]+{len(optional_args)}"
                                    elif required_args:
                                        args_display = (
                                            f"[bold]{len(required_args)}[/bold]"
                                        )
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
                                prompt_names = [
                                    str(i + 1) for i in range(len(all_prompts))
                                ]

                                # Ask user to select a prompt
                                from mcp_agent.core.enhanced_prompt import (
                                    get_selection_input,
                                )

                                selection = await get_selection_input(
                                    "Enter prompt number to select (or press Enter to cancel): ",
                                    options=prompt_names,
                                    allow_cancel=True,
                                )

                                # Make cancellation easier
                                if not selection or selection.strip() == "":
                                    rich_print(
                                        "[yellow]Prompt selection cancelled[/yellow]"
                                    )
                                    continue

                                try:
                                    idx = int(selection) - 1
                                    if 0 <= idx < len(all_prompts):
                                        selected_prompt = all_prompts[idx]
                                    else:
                                        rich_print("[red]Invalid selection[/red]")
                                        continue
                                except ValueError:
                                    rich_print(
                                        "[red]Invalid input, please enter a number[/red]"
                                    )
                                    continue

                            # Get our prompt arguments
                            required_args = selected_prompt["required_args"]
                            optional_args = selected_prompt["optional_args"]
                            arg_descriptions = selected_prompt.get(
                                "arg_descriptions", {}
                            )

                            # Always initialize arg_values
                            arg_values = {}

                            # Show argument info if we have any
                            if required_args or optional_args:
                                # Display information about the arguments
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
                                    # Get description if available
                                    description = arg_descriptions.get(arg_name, "")

                                    # Collect required argument value
                                    from mcp_agent.core.enhanced_prompt import (
                                        get_argument_input,
                                    )

                                    arg_value = await get_argument_input(
                                        arg_name=arg_name,
                                        description=description,
                                        required=True,
                                    )
                                    # Add to arg_values if a value was provided
                                    if arg_value is not None:
                                        arg_values[arg_name] = arg_value

                                # Only include non-empty values for optional arguments
                                if optional_args:
                                    # Collect optional arguments
                                    for arg_name in optional_args:
                                        # Get description if available
                                        description = arg_descriptions.get(arg_name, "")

                                        from mcp_agent.core.enhanced_prompt import (
                                            get_argument_input,
                                        )

                                        arg_value = await get_argument_input(
                                            arg_name=arg_name,
                                            description=description,
                                            required=False,
                                        )
                                        # Only include non-empty values for optional arguments
                                        if arg_value:
                                            arg_values[arg_name] = arg_value

                            # Apply the prompt with or without arguments
                            rich_print(
                                f"\n[bold]Applying prompt [cyan]{selected_prompt['namespaced_name']}[/cyan]...[/bold]"
                            )

                            # Call apply_prompt on the agent - always pass arg_values (empty dict if no args)
                            await current_proxy._agent.apply_prompt(
                                selected_prompt["namespaced_name"], arg_values
                            )

                        except Exception as e:
                            import traceback

                            rich_print(
                                f"[red]Error selecting or applying prompt: {e}[/red]"
                            )
                            rich_print(f"[dim]{traceback.format_exc()}[/dim]")
                        continue

                # Skip further processing if command was handled
                if command_result:
                    continue

                if user_input.upper() == "STOP":
                    return result
                if user_input == "":
                    continue

            result = await self.send(user_input, agent)

            # Check if current agent is a chain that should continue with final agent
            if agent_types.get(agent) == "Chain":
                proxy = self._agents[agent]
                if isinstance(proxy, ChainProxy) and proxy._continue_with_final:
                    # Get the last agent in the sequence
                    last_agent = proxy._sequence[-1]
                    # Switch to that agent for the next iteration
                    agent = last_agent

        return result

    def __getattr__(self, name: str) -> BaseAgentProxy:
        """Support: agent.researcher"""
        if name not in self._agents:
            raise AttributeError(f"No agent named '{name}'")
        return self._agents[name]

    def __getitem__(self, name: str) -> BaseAgentProxy:
        """Support: agent['researcher']"""
        if name not in self._agents:
            raise KeyError(f"No agent named '{name}'")
        return self._agents[name]

    async def __call__(
        self,
        message: Optional[Union[str, PromptMessageMultipart]] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        """
        Support: agent('message') or agent(Prompt.user('message'))

        Args:
            message: Either a string message or a PromptMessageMultipart object
            agent_name: The name of the agent to use (uses default if None)

        Returns:
            The agent's response as a string
        """
        target = agent_name or self._default
        if not target:
            raise ValueError("No default agent available")
        return await self.send(message, target)

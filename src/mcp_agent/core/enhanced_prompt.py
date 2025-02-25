"""
Enhanced prompt functionality with advanced prompt_toolkit features.
"""

from typing import List
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.filters import Condition
from pygments.lexers.python import PythonLexer
from rich import print as rich_print

# Map of agent names to their history
agent_histories = {}

# Store available agents for auto-completion
available_agents = set()

# Keep track of multi-line mode state
in_multiline_mode = False

# Track which agents have already shown welcome messages
agent_messages_shown = set()


class AgentCompleter(Completer):
    """Provide completion for agent names and common commands."""

    def __init__(self, agents: List[str], commands: List[str] = None, agent_types: dict = None):
        self.agents = agents
        self.commands = commands or ["help", "clear", "STOP"]
        self.agent_types = agent_types or {}

    def get_completions(self, document, complete_event):
        """Synchronous completions method - this is what prompt_toolkit expects by default"""
        text = document.text_before_cursor.lower()

        # Complete commands
        if text.startswith("/"):
            cmd = text[1:]
            for command in self.commands:
                if command.lower().startswith(cmd):
                    yield Completion(
                        command,
                        start_position=-len(cmd),
                        display=command,
                        display_meta="Command",
                    )

        # Complete agent names for agent-related commands
        elif text.startswith("@"):
            agent_name = text[1:]
            for agent in self.agents:
                if agent.lower().startswith(agent_name.lower()):
                    # Get agent type or default to "Agent"
                    agent_type = self.agent_types.get(agent, "Agent")
                    yield Completion(
                        agent,
                        start_position=-len(agent_name),
                        display=agent,
                        display_meta=agent_type,
                    )


def create_keybindings(on_toggle_multiline=None, app=None):
    """Create custom key bindings."""
    kb = KeyBindings()

    @kb.add("c-m", filter=Condition(lambda: not in_multiline_mode))
    def _(event):
        """Enter: accept input when not in multiline mode."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-m", filter=Condition(lambda: in_multiline_mode))
    def _(event):
        """Enter: insert newline when in multiline mode."""
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "enter")
    def _(event):
        """Alt+Enter: always submit even in multiline mode."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-t")
    def _(event):
        """Ctrl+T: Toggle multiline mode."""
        global in_multiline_mode
        in_multiline_mode = not in_multiline_mode

        # Force redraw the app to update toolbar
        if event.app:
            event.app.invalidate()
        elif app:
            app.invalidate()

        # Call the toggle callback if provided
        if on_toggle_multiline:
            on_toggle_multiline(in_multiline_mode)

        # Instead of printing, we'll just update the toolbar
        # The toolbar will show the current mode

    @kb.add("c-l")
    def _(event):
        """Ctrl+L: Clear input."""
        event.current_buffer.text = ""

    return kb


async def get_enhanced_input(
    agent_name: str,
    default: str = "",
    show_default: bool = False,
    show_stop_hint: bool = False,
    multiline: bool = False,
    available_agent_names: List[str] = None,
    syntax: str = None,
    agent_types: dict = None,
) -> str:
    """
    Enhanced input with advanced prompt_toolkit features.

    Args:
        agent_name: Name of the agent (used for prompt and history)
        default: Default value if user presses enter
        show_default: Whether to show the default value in the prompt
        show_stop_hint: Whether to show the STOP hint
        multiline: Start in multiline mode
        available_agent_names: List of agent names for auto-completion
        syntax: Syntax highlighting (e.g., 'python', 'sql')
        agent_types: Dictionary mapping agent names to their types for display

    Returns:
        User input string
    """
    global in_multiline_mode, available_agents

    # Update global state
    in_multiline_mode = multiline
    if available_agent_names:
        available_agents = set(available_agent_names)

    # Get or create history object for this agent
    if agent_name not in agent_histories:
        agent_histories[agent_name] = InMemoryHistory()

    # Define callback for multiline toggle
    def on_multiline_toggle(enabled):
        nonlocal session
        if hasattr(session, "app") and session.app:
            session.app.invalidate()

    # Define toolbar function that will update dynamically
    def get_toolbar():
        if in_multiline_mode:
            mode_style = "ansired"  # More noticeable for multiline mode
            mode_text = "MULTILINE"
            toggle_text = "Normal Editing"
        else:
            mode_style = "ansigreen"
            mode_text = "NORMAL"
            toggle_text = "Multiline Editing"

        shortcuts = [
            ("Ctrl+T", toggle_text),
            ("Alt+Enter", "Submit" if in_multiline_mode else ""),
            ("Ctrl+L", "Clear"),
            ("↑/↓", "History"),
        ]
        # Only show relevant shortcuts based on mode
        shortcuts = [(k, v) for k, v in shortcuts if v]

        shortcut_text = " | ".join(f"{key}:{action}" for key, action in shortcuts)
        return HTML(
            f" <b>Agent:</b> <ansiblue> {agent_name} </ansiblue> | <b>Mode:</b> <{mode_style}> {mode_text} </{mode_style}> | {shortcut_text}"
        )

    # Create session with history and completions
    session = PromptSession(
        history=agent_histories[agent_name],
        completer=AgentCompleter(
            agents=list(available_agents) if available_agents else [],
            agent_types=agent_types or {},
        ),
        complete_while_typing=True,
        lexer=PygmentsLexer(PythonLexer) if syntax == "python" else None,
        multiline=Condition(lambda: in_multiline_mode),
        complete_in_thread=True,
        mouse_support=True,
        bottom_toolbar=get_toolbar,  # Pass the function here
    )

    # Create key bindings with a reference to the app
    bindings = create_keybindings(
        on_toggle_multiline=on_multiline_toggle, app=session.app
    )
    session.app.key_bindings = bindings

    # Create formatted prompt text
    prompt_text = f"<ansicyan>{agent_name}</ansicyan> > "

    # Add default value display if requested
    if show_default and default and default != "STOP":
        prompt_text = f"{prompt_text} [<ansigreen>{default}</ansigreen>] "

    # Only show hints at startup if requested
    if show_stop_hint:
        if default == "STOP":
            rich_print("[yellow]Press <ENTER> to finish.[/yellow]")
        else:
            rich_print("Enter a prompt, or [red]STOP[/red] to finish")
            if default:
                rich_print(
                    f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]"
                )

    # Mention available features but only on first usage for this agent
    if agent_name not in agent_messages_shown:
        rich_print(
            "[dim]Tip: Type /help for commands, press F1 for keyboard shortcuts. Ctrl+T toggles multiline mode. @Agent to switch agent[/dim]"
        )
        agent_messages_shown.add(agent_name)

    # Process special commands
    def pre_process_input(text):
        # Command processing
        if text and text.startswith("/"):
            cmd = text[1:].strip().lower()
            if cmd == "help":
                return "HELP"
            elif cmd == "clear":
                return "CLEAR"
            elif cmd == "agents":
                return "LIST_AGENTS"

        # Agent switching
        if text and text.startswith("@"):
            return f"SWITCH:{text[1:].strip()}"

        return text

    # Get the input - using async version
    try:
        result = await session.prompt_async(HTML(prompt_text), default=default)
        return pre_process_input(result)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        return "STOP"
    except EOFError:
        # Handle Ctrl+D gracefully
        return "STOP"
    except Exception as e:
        # Log and gracefully handle other exceptions
        print(f"\nInput error: {type(e).__name__}: {e}")
        return "STOP"


async def handle_special_commands(command, agent_app=None):
    """Handle special input commands."""
    # Quick guard for empty or None commands
    if not command:
        return False

    # Check for special commands
    if command == "HELP":
        rich_print("\n[bold]Available Commands:[/bold]")
        rich_print("  /help          - Show this help")
        rich_print("  /clear         - Clear screen")
        rich_print("  /agents        - List available agents")
        rich_print("  @agent_name    - Switch to agent")
        rich_print("  STOP           - End session")
        rich_print("\n[bold]Keyboard Shortcuts:[/bold]")
        rich_print(
            "  Enter          - Submit (normal mode) / New line (multiline mode)"
        )
        rich_print("  Alt+Enter      - Always submit (even in multiline mode)")
        rich_print("  Ctrl+T         - Toggle multiline mode")
        rich_print("  Ctrl+L         - Clear input")
        rich_print("  Up/Down        - Navigate history")
        rich_print("  F1             - Show help")
        return True

    elif command == "CLEAR":
        # Clear screen (ANSI escape sequence)
        print("\033c", end="")
        return True

    elif command == "LIST_AGENTS":
        if available_agents:
            rich_print("\n[bold]Available Agents:[/bold]")
            for agent in sorted(available_agents):
                rich_print(f"  @{agent}")
        else:
            rich_print("[yellow]No agents available[/yellow]")
        return True

    elif isinstance(command, str) and command.startswith("SWITCH:"):
        agent_name = command.split(":", 1)[1]
        if agent_name in available_agents:
            if agent_app:
                rich_print(f"[green]Switching to agent: {agent_name}[/green]")
                return {"switch_agent": agent_name}
            else:
                rich_print(
                    "[yellow]Agent switching not available in this context[/yellow]"
                )
        else:
            rich_print(f"[red]Unknown agent: {agent_name}[/red]")
        return True

    return False

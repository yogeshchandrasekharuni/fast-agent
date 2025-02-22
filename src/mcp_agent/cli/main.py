"""Main CLI entry point for MCP Agent."""

import typer
from rich.console import Console
from rich.table import Table
from mcp_agent.cli.terminal import Application
from mcp_agent.cli.commands import setup, bootstrap

app = typer.Typer(
    help="MCP Agent CLI - Build effective agents using Model Context Protocol",
    add_completion=False,  # We'll add this later when we have more commands
)

# Subcommands
app.add_typer(setup.app, name="setup", help="Set up a new agent project")
app.add_typer(bootstrap.app, name="bootstrap", help="Create example applications")

# Shared application context
application = Application()
console = Console()


def show_welcome():
    """Show a welcome message with available commands."""
    console.print("\n[bold]Welcome to MCP Agent![/bold]")
    console.print("Build effective agents using Model Context Protocol (MCP)")

    # Create a table for commands
    table = Table(title="\nAvailable Commands")
    table.add_column("Command", style="green")
    table.add_column("Description")

    table.add_row("setup", "Set up a new agent project with configuration files")
    table.add_row(
        "bootstrap", "Create example applications (decorator, researcher, etc.)"
    )
    # table.add_row("config", "Manage agent configuration settings")

    console.print(table)

    console.print("\n[bold]Getting Started:[/bold]")
    console.print("1. Set up a new project:")
    console.print("   mcp-agent setup")
    console.print("\n2. Try an example:")
    console.print("   mcp-agent bootstrap create decorator")
    console.print("\nUse --help with any command for more information")
    console.print("Example: mcp-agent bootstrap --help")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable output"),
    color: bool = typer.Option(
        True, "--color/--no-color", help="Enable/disable color output"
    ),
):
    """MCP Agent CLI - Build effective agents using Model Context Protocol (MCP).

    Use --help with any command for detailed usage information.
    """
    application.verbosity = 1 if verbose else 0 if not quiet else -1
    application.console = application.console if color else None

    # Show welcome message if no command was invoked
    if ctx.invoked_subcommand is None:
        show_welcome()

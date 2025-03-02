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
    from importlib.metadata import version

    try:
        app_version = version("fast-agent-mcp")
    except:  # noqa: E722
        app_version = "unknown"

    console.print(f"\n[bold]fast-agent (fast-agent-mcp) {app_version}[/bold]")
    console.print("Build effective agents using Model Context Protocol (MCP)")

    # Create a table for commands
    table = Table(title="\nAvailable Commands")
    table.add_column("Command", style="green")
    table.add_column("Description")

    table.add_row("setup", "Set up a new agent project with configuration files")
    table.add_row(
        "bootstrap", "Create example applications (workflow, researcher, etc.)"
    )
    # table.add_row("config", "Manage agent configuration settings")

    console.print(table)

    console.print("\n[bold]Getting Started:[/bold]")
    console.print("1. Set up a new project:")
    console.print("   fastagent setup")
    console.print("\n2. Create Building Effective Agents  workflow examples:")
    console.print("   fastagent bootstrap workflow")
    console.print("\n3. Explore other examples:")
    console.print("   fastagent bootstrap")

    console.print("\nUse --help with any command for more information")
    console.print("Example: fastagent bootstrap --help")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable output"),
    color: bool = typer.Option(
        True, "--color/--no-color", help="Enable/disable color output"
    ),
):
    """FastAgent CLI - Build effective agents using Model Context Protocol (MCP).

    Use --help with any command for detailed usage information.
    """
    application.verbosity = 1 if verbose else 0 if not quiet else -1
    application.console = application.console if color else None

    # Show welcome message if no command was invoked
    if ctx.invoked_subcommand is None:
        show_welcome()

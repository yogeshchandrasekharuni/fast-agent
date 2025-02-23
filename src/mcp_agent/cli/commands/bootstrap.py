"""Bootstrap command to create example applications."""

import shutil
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    help="Create example applications",
    no_args_is_help=False,  # Allow showing our custom help instead
)
console = Console()

EXAMPLE_TYPES = {
    "workflow": {
        "description": "Example workflows, demonstrating each of the patterns in the 'Building Effective Agents' paper.",
        "details": """
        The workflow examples show how to:
        - Chain agents together
        - Parallelize Agents 
        - Orchestrate Workers
        - Evaluate and Optimize
        - Route Requests
        """,
        "files": [
            "chaining.py",
            "evaluator.py",
            "human_input.py",
            "orchestrator.py",
            "parallel.py",
            "router.py",
        ],
    },
    "researcher": {
        "description": "Research agent example with evaluation optimization. Uses Brave Search and Docker.",
        "details": """
        The researcher example demonstrates:
        - Building a research-focused agent
        - Implementing evaluation strategies
        - Optimizing agent responses
        - Handling complex research tasks
        """,
        "files": ["main.py", "main-evalopt.py"],
    },
}


def copy_example_files(
    example_type: str, target_dir: Path, force: bool = False
) -> list[str]:
    """Copy example files from resources to target directory."""
    created = []

    # Use the resources directory from the package
    source_dir = (
        Path(__file__).parent.parent.parent
        / "resources"
        / "examples"
        / ("workflows" if example_type == "workflow" else "mcp_researcher")
    )

    if not source_dir.exists():
        console.print(f"[red]Error: Source directory not found: {source_dir}[/red]")
        return created

    for filename in EXAMPLE_TYPES[example_type]["files"]:
        source = source_dir / filename
        target = target_dir / filename

        try:
            if not source.exists():
                console.print(f"[red]Error: Source file not found: {source}[/red]")
                continue

            if target.exists() and not force:
                console.print(f"[yellow]Skipping[/yellow] {filename} (already exists)")
                continue

            shutil.copy2(source, target)
            created.append(filename)
            console.print(f"[green]Created[/green] {filename}")

        except Exception as e:
            console.print(f"[red]Error copying {filename}: {str(e)}[/red]")

    return created


def show_overview():
    """Display an overview of available examples in a nicely formatted table."""
    console.print("\n[bold cyan]FastAgent Example Applications[/bold cyan]")
    console.print("Learn how to build effective agents through practical examples\n")

    # Create a table for better organization
    table = Table(
        show_header=True, header_style="bold magenta", box=None, padding=(0, 2)
    )
    table.add_column("Example")
    table.add_column("Description")
    table.add_column("Files")

    for name, info in EXAMPLE_TYPES.items():
        files_list = "\n".join(f"• {f}" for f in info["files"])
        table.add_row(f"[green]{name}[/green]", info["description"], files_list)

    console.print(table)

    # Show usage instructions in a panel
    usage_text = (
        "[bold]Commands:[/bold]\n"
        "  fastagent bootstrap workflow DIR      Create workflow examples in DIR\n"
        "  fastagent bootstrap researcher DIR    Create researcher example in DIR\n\n"
        "[bold]Options:[/bold]\n"
        "  --force            Overwrite existing files\n\n"
        "[bold]Examples:[/bold]\n"
        "  fastagent bootstrap workflow .              Create in current directory\n"
        "  fastagent bootstrap workflow ./my-workflows Create in specific directory\n"
        "  fastagent bootstrap researcher . --force    Force overwrite files"
    )
    console.print(Panel(usage_text, title="Usage", border_style="blue"))


@app.command()
def workflow(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where workflow examples will be created",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite existing files"
    ),
):
    """Create workflow pattern examples."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("workflow", target_dir, force)
    _show_completion_message("workflow", created)


@app.command()
def researcher(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where researcher examples will be created",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite existing files"
    ),
):
    """Create researcher pattern examples."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("researcher", target_dir, force)
    _show_completion_message("researcher", created)


def _show_completion_message(example_type: str, created: list[str]):
    """Show completion message and next steps."""
    if created:
        console.print("\n[green]Setup completed successfully![/green]")
        console.print("\nCreated files:")
        for f in created:
            console.print(f"  - {f}")

        console.print("\n[bold]Next Steps:[/bold]")
        if example_type == "workflow":
            console.print("1. Review chaining.py for the basic workflow example")
            console.print("2. Check other examples:")
            console.print("   - parallel.py: Run agents in parallel")
            console.print("   - router.py: Route requests between agents")
            console.print("   - evaluator.py: Add evaluation capabilities")
            console.print("   - human_input.py: Incorporate human feedback")
            console.print("3. Run an example with: python <example>.py")
        else:
            console.print("1. Review main.py for the basic researcher example")
            console.print(
                "2. Try main-evalopt.py for the evaluation optimization version"
            )
            console.print("3. Run with: python main.py or python main-evalopt.py")
    else:
        console.print("\n[yellow]No files were created.[/yellow]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Create example applications and learn FastAgent patterns."""
    if ctx.invoked_subcommand is None:
        show_overview()

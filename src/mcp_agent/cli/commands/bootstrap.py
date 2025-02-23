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
        "description": "Workflow patterns showing different agent composition approaches",
        "details": """
        The workflow examples show how to:
        - Compose agents using workflow patterns
        - Chain agents together for complex tasks
        - Run agents in parallel
        - Handle agent evaluation and routing
        - Incorporate human input
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
        "description": "Research agent example with evaluation optimization",
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
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("Example")
    table.add_column("Description")
    table.add_column("Files")

    for name, info in EXAMPLE_TYPES.items():
        files_list = "\n".join(f"• {f}" for f in info["files"])
        table.add_row(
            f"[green]{name}[/green]",
            info["description"],
            files_list
        )

    console.print(table)

    # Show usage instructions in a panel
    usage_text = (
        "[bold]Commands:[/bold]\n"
        "  fastagent bootstrap workflow           Create workflow examples\n"
        "  fastagent bootstrap researcher         Create researcher example\n\n"
        "[bold]Options:[/bold]\n"
        "  --directory PATH    Create files in specific directory\n"
        "  --force            Overwrite existing files\n\n"
        "[bold]Examples:[/bold]\n"
        "  fastagent bootstrap workflow --directory ./my-workflows\n"
        "  fastagent bootstrap workflow ./my-workflows  (using positional directory)\n"
        "  fastagent bootstrap researcher --force"
    )
    console.print(Panel(usage_text, title="Usage", border_style="blue"))


@app.callback(invoke_without_command=True)
def main(
    directory: str = typer.Option(
        ".", "--directory", "-d", help="Directory where example files will be created"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite existing files"
    ),
    example_type: str = typer.Argument(
        None, 
        help="Type of example to create (workflow or researcher)"
    ),
):
    """Create example applications and learn FastAgent patterns.
    
    Run without arguments to see available examples.
    """
    if example_type is None:
        show_overview()
        return

    example_type = example_type.lower()
    if example_type not in EXAMPLE_TYPES:
        console.print(f"[red]Error:[/red] Unknown example type '{example_type}'")
        console.print("\nAvailable types:")
        for name in EXAMPLE_TYPES:
            console.print(f"  - {name}")
        raise typer.Exit(1)

    target_dir = Path(directory).resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    # Show example details
    console.print(f"\n[bold]Creating {example_type} example[/bold]")
    console.print(
        Panel(
            EXAMPLE_TYPES[example_type]["details"].strip(),
            title="About this example",
            expand=False,
        )
    )

    if not force:
        # Check for existing files first
        existing = []
        for f in EXAMPLE_TYPES[example_type]["files"]:
            if (target_dir / f).exists():
                existing.append(f)

        if existing:
            console.print(
                "\n[yellow]Warning:[/yellow] The following files already exist:"
            )
            for f in existing:
                console.print(f"  - {f}")
            console.print("\nUse --force to overwrite existing files")
            raise typer.Exit(1)

    created = copy_example_files(example_type, target_dir, force)

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
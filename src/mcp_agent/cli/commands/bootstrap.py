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
        "description": "Example workflows, demonstrating each of the patterns in Anthropic's\n"
        "'Building Effective Agents' paper. Some agents use the 'fetch'\n"
        "and filesystem MCP Servers.",
        "files": [
            "agent_build.py",
            "chaining.py",
            "evaluator.py",
            "human_input.py",
            "orchestrator.py",
            "parallel.py",
            "router.py",
            "fastagent.config.yaml",
        ],
        "create_subdir": True,
    },
    "researcher": {
        "description": "Research agent example with additional evaluation/optimization\n"
        "example. Uses Brave Search and Docker MCP Servers.\n"
        "Creates examples in a 'researcher' subdirectory.",
        "files": ["researcher.py", "researcher-eval.py", "fastagent.config.yaml"],
        "create_subdir": True,
    },
    "data-analysis": {
        "description": "Data analysis agent examples that demonstrate working with\n"
        "datasets, performing statistical analysis, and generating visualizations.\n"
        "Creates examples in a 'data-analysis' subdirectory with mount-point for data.\n"
        "Uses MCP 'roots' feature for mapping",
        "files": ["analysis.py", "fastagent.config.yaml"],
        "mount_point_files": ["WA_Fn-UseC_-HR-Employee-Attrition.csv"],
        "create_subdir": True,
    },
}


def copy_example_files(
    example_type: str, target_dir: Path, force: bool = False
) -> list[str]:
    """Copy example files from resources to target directory."""
    created = []

    # Determine if we should create a subdirectory for this example type
    example_info = EXAMPLE_TYPES[example_type]
    if example_info["create_subdir"]:
        target_dir = target_dir / example_type
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
            console.print(f"Created subdirectory: {target_dir}")

    # Create mount-point directory if needed
    mount_point_files = example_info.get("mount_point_files", [])
    if mount_point_files:
        mount_point_dir = target_dir / "mount-point"
        if not mount_point_dir.exists():
            mount_point_dir.mkdir(parents=True)
            console.print(f"Created mount-point directory: {mount_point_dir}")

    # Use the resources directory from the package
    source_dir = (
        Path(__file__).parent.parent.parent
        / "resources"
        / "examples"
        / ("workflows" if example_type == "workflow" else f"{example_type}")
    )

    if not source_dir.exists():
        console.print(f"[red]Error: Source directory not found: {source_dir}[/red]")
        return created

    for filename in example_info["files"]:
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
            created.append(str(target.relative_to(target_dir.parent)))
            console.print(f"[green]Created[/green] {created[-1]}")

        except Exception as e:
            console.print(f"[red]Error copying {filename}: {str(e)}[/red]")

    # Copy mount-point files if any
    if mount_point_files:
        source_mount_point = source_dir / "mount-point"
        for filename in mount_point_files:
            source = source_mount_point / filename
            target = mount_point_dir / filename

            try:
                if not source.exists():
                    console.print(f"[red]Error: Source file not found: {source}[/red]")
                    continue

                if target.exists() and not force:
                    console.print(
                        f"[yellow]Skipping[/yellow] mount-point/{filename} (already exists)"
                    )
                    continue

                shutil.copy2(source, target)
                created.append(f"{example_type}/mount-point/{filename}")
                console.print(f"[green]Created[/green] mount-point/{filename}")

            except Exception as e:
                console.print(
                    f"[red]Error copying mount-point/{filename}: {str(e)}[/red]"
                )

    return created


def show_overview():
    """Display an overview of available examples in a nicely formatted table."""
    console.print("\n[bold cyan]fast-agent Example Applications[/bold cyan]")
    console.print("Build agents and compose workflows through practical examples\n")

    # Create a table for better organization
    table = Table(
        show_header=True, header_style="bold magenta", box=None, padding=(0, 2)
    )
    table.add_column("Example")
    table.add_column("Description")
    table.add_column("Files")

    for name, info in EXAMPLE_TYPES.items():
        files_list = "\n".join(f"• {f}" for f in info["files"])
        if "mount_point_files" in info:
            files_list += "\n[blue]mount-point:[/blue]\n" + "\n".join(
                f"• {f}" for f in info["mount_point_files"]
            )
        table.add_row(f"[green]{name}[/green]", info["description"], files_list)

    console.print(table)

    # Show usage instructions in a panel
    usage_text = (
        "[bold]Commands:[/bold]\n"
        "  fastagent bootstrap workflow DIR      Create workflow examples in DIR\n"
        "  fastagent bootstrap researcher DIR    Create researcher example in 'researcher' subdirectory\n"
        "  fastagent bootstrap data-analysis DIR Create data analysis examples in 'data-analysis' subdirectory\n\n"
        "[bold]Options:[/bold]\n"
        "  --force            Overwrite existing files\n\n"
        "[bold]Examples:[/bold]\n"
        "  fastagent bootstrap workflow .              Create in current directory\n"
        "  fastagent bootstrap researcher .            Create in researcher subdirectory\n"
        "  fastagent bootstrap data-analysis . --force Force overwrite files in data-analysis subdirectory"
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
        help="Directory where researcher examples will be created (in 'researcher' subdirectory)",
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


@app.command()
def data_analysis(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where data analysis examples will be created (creates 'data-analysis' subdirectory with mount-point)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite existing files"
    ),
):
    """Create data analysis examples with sample dataset."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("data-analysis", target_dir, force)
    _show_completion_message("data-analysis", created)


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
            console.print("3. Run an example with: uv run <example>.py")
            console.print(
                "4. Try a different model with --model=<model>, or update the agent config"
            )

        elif example_type == "researcher":
            console.print(
                "1. Set up the Brave MCP Server (get an API key from https://brave.com/search/api/)"
            )
            console.print("2. Try `uv run researcher.py` for the basic version")
            console.print(
                "3. Try `uv run researcher-eval.py` for the eval/optimize version"
            )
        elif example_type == "data-analysis":
            console.print(
                "1. Run uv `analysis.py` to perform data analysis and visualization"
            )
            console.print("2. The dataset is available in the mount-point directory:")
            console.print("   - mount-point/WA_Fn-UseC_-HR-Employee-Attrition.csv")
            console.print(
                "On Windows platforms, please edit the fastagent.config.yaml and adjust the volume mount point."
            )
    else:
        console.print("\n[yellow]No files were created.[/yellow]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Create example applications and learn FastAgent patterns."""
    if ctx.invoked_subcommand is None:
        show_overview()

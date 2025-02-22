"""Bootstrap command to create example applications."""
import os
import shutil
from pathlib import Path
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

EXAMPLE_TYPES = {
    "decorator": {
        "description": "Decorator pattern example showing different agent composition approaches",
        "files": [
            "main.py",
            "optimizer.py", 
            "orchestrator.py",
            "parallel.py",
            "router.py",
            "tiny.py"
        ]
    },
    "researcher": {
        "description": "Research agent example with evaluation optimization",
        "files": [
            "main.py",
            "main-evalopt.py"
        ]
    }
}

def copy_example_files(example_type: str, target_dir: Path, force: bool = False) -> list[str]:
    """Copy example files from resources to target directory."""
    created = []
    
    # During development, use the source directory
    source_dir = Path(__file__).parent.parent.parent / "resources" / "examples" / (
        "decorator" if example_type == "decorator" else "mcp_researcher"
    )
    
    if not source_dir.exists():
        console.print(f"[red]Error: Source directory not found: {source_dir}[/red]")
        return created
    
    for filename in EXAMPLE_TYPES[example_type]['files']:
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

@app.command()
def create(
    example_type: str = typer.Argument(
        None,
        help="Type of example to create (decorator or researcher)"
    ),
    directory: str = typer.Option(
        ".",
        "--directory", "-d",
        help="Directory where example files will be created"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force overwrite existing files"
    )
):
    """Create a new example application.
    
    This will create a set of example files based on the chosen type:
    
    \b
    - decorator: Decorator pattern example showing different agent composition approaches
    - researcher: Research agent example with evaluation optimization
    """
    if not example_type:
        console.print("\n[bold]Available Example Types:[/bold]")
        for name, info in EXAMPLE_TYPES.items():
            console.print(f"\n[green]{name}[/green]")
            console.print(f"  {info['description']}")
            console.print("  Files:")
            for f in info['files']:
                console.print(f"    - {f}")
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

    console.print(f"\n[bold]Creating {example_type} example[/bold]")
    console.print(f"Description: {EXAMPLE_TYPES[example_type]['description']}")
    
    if not force:
        # Check for existing files first
        existing = []
        for f in EXAMPLE_TYPES[example_type]['files']:
            if (target_dir / f).exists():
                existing.append(f)
        
        if existing:
            console.print("\n[yellow]Warning:[/yellow] The following files already exist:")
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
        
        console.print("\nTo get started:")
        if example_type == 'decorator':
            console.print("1. Review main.py for the basic example")
            console.print("2. Check other files for different composition patterns")
            console.print("3. Run with: python main.py")
        else:
            console.print("1. Review main.py for the basic researcher example")
            console.print("2. Try main-evalopt.py for the evaluation optimization version")
            console.print("3. Run with: python main.py or python main-evalopt.py")
    else:
        console.print("\n[yellow]No files were created.[/yellow]")
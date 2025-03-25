import typer
from typing import NoReturn

app = typer.Typer()


@app.command()
def show() -> NoReturn:
    """Show the configuration."""
    raise NotImplementedError("The show configuration command has not been implemented yet")

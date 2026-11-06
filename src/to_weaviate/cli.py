import typer

from to_weaviate import __version__

app = typer.Typer(help="Migrate your data to Weaviate from other vector databases.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"to-weaviate {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    pass

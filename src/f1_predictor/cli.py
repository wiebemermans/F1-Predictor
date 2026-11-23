"""Command line interface: `f1 fetch` en `f1 features`, later ook backtest/predict."""

import logging
from datetime import date
from typing import Annotated

import typer

from f1_predictor.data.fetch import fetch_seasons
from f1_predictor.features.build import write_feature_table

app = typer.Typer(help="F1 predictor", no_args_is_help=True)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Meer logging")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def fetch(
    start: Annotated[int, typer.Option(help="Eerste seizoen")] = 2014,
    end: Annotated[int, typer.Option(help="Laatste seizoen")] = date.today().year,
    force: Annotated[bool, typer.Option(help="Ook bestaande seizoenen opnieuw ophalen")] = False,
) -> None:
    """Haal seizoensdata op bij Jolpica en sla ze op als parquet in data/raw/."""
    if start > end:
        raise typer.BadParameter("start moet kleiner of gelijk zijn aan end")
    failed = fetch_seasons(start, end, force=force)
    if failed:
        typer.echo(f"Mislukte seizoenen: {failed}", err=True)
        raise typer.Exit(code=1)


@app.command()
def features() -> None:
    """Bouw de featuretabel uit data/raw/ en sla die op als data/processed/features.parquet."""
    try:
        write_feature_table()
    except FileNotFoundError as error:
        typer.echo(f"{error}. Haal eerst data op met `f1 fetch`.", err=True)
        raise typer.Exit(code=1) from error

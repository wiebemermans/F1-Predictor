"""Command line interface: van Jolpica-data tot een voorspelling na de kwalificatie.

Elk commando leest `config/config.yaml` (of het bestand achter `--config`). Commando's,
configuratievelden en uitvoerbestanden staan in `docs/usage.md`.
"""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import requests
import typer

from f1_predictor.config import ConfigError, ProjectConfig, load_config
from f1_predictor.data.fetch import QualifyingNotAvailableError, fetch_seasons
from f1_predictor.data.jolpica import JolpicaError
from f1_predictor.evaluation.backtest import run_baseline_backtest, run_feature_backtest
from f1_predictor.evaluation.model_selection import (
    run_model_backtest,
    run_model_tuning,
    select_final_model,
    select_tuned_models,
    summarize_periods,
    tuning_candidates,
    write_comparison_json,
)
from f1_predictor.features.build import write_feature_table
from f1_predictor.models.predict import (
    PredictionError,
    format_prediction,
    predict_race,
    write_prediction,
)
from f1_predictor.models.train import train_final_model
from f1_predictor.paths import CONFIG_PATH
from f1_predictor.runinfo import run_metadata

app = typer.Typer(help="F1-voorspeller", no_args_is_help=True)
log = logging.getLogger(__name__)

MISSING_DATA_HINT = "Haal eerst data op met `f1 fetch`."


class Suite(StrEnum):
    models = "models"
    features = "features"
    baselines = "baselines"


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Meer logging")] = False,
    config: Annotated[
        Path, typer.Option("--config", "-c", help="Configuratiebestand (YAML)")
    ] = CONFIG_PATH,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.obj = config


def _fail(message: str) -> NoReturn:
    log.error("%s", message)
    raise typer.Exit(code=1)


def _project(ctx: typer.Context) -> ProjectConfig:
    try:
        return load_config(ctx.obj)
    except ConfigError as error:
        _fail(str(error))


@app.command()
def fetch(
    ctx: typer.Context,
    start: Annotated[
        int | None, typer.Option(help="Eerste seizoen (standaard data.start_season)")
    ] = None,
    end: Annotated[
        int | None,
        typer.Option(help="Laatste seizoen (standaard data.end_season, anders het huidige jaar)"),
    ] = None,
    force: Annotated[bool, typer.Option(help="Ook bestaande seizoenen opnieuw ophalen")] = False,
) -> None:
    """Haal seizoensdata op bij Jolpica en sla ze op als parquet in data/raw/."""
    data = _project(ctx).data
    start = data.start_season if start is None else start
    end = data.resolved_end_season() if end is None else end
    if start > end:
        raise typer.BadParameter("start moet kleiner of gelijk zijn aan end")
    failed = fetch_seasons(start, end, force=force)
    if failed:
        _fail(f"Mislukte seizoenen: {failed}")


@app.command()
def features(ctx: typer.Context) -> None:
    """Bouw de featuretabel uit data/raw/ en sla die op als data/processed/features.parquet."""
    project = _project(ctx)
    try:
        write_feature_table(config=project.features)
    except FileNotFoundError as error:
        _fail(f"{error}. {MISSING_DATA_HINT}")


@app.command()
def backtest(
    ctx: typer.Context,
    suite: Annotated[
        Suite,
        typer.Option(
            help="models: baselines en de gekozen modellen; features: de featuregroepen "
            "(fase 6); baselines: alleen de baselines (fase 5)"
        ),
    ] = Suite.models,
) -> None:
    """Walk-forward-backtest; schrijft scores per race en samenvattingen naar output/."""
    project = _project(ctx)
    min_train_races = project.backtest.min_train_races
    try:
        if suite is Suite.baselines:
            run_baseline_backtest(min_train_races=min_train_races)
        elif suite is Suite.features:
            run_feature_backtest(min_train_races=min_train_races, config=project.features)
        else:
            models = project.model.tuned
            results = run_model_backtest(
                min_train_races=min_train_races, models=models, config=project.features
            )
            path = write_comparison_json(
                summarize_periods(results), models, project.final_spec.name, run_metadata(project)
            )
            log.info("Samenvatting per periode als JSON: %s", path)
    except FileNotFoundError as error:
        _fail(f"{error}. {MISSING_DATA_HINT}")


@app.command()
def tune(ctx: typer.Context) -> None:
    """Kies hyperparameters op de validatieraces (2015-2021); duurt ongeveer 45 minuten."""
    project = _project(ctx)
    try:
        summary = run_model_tuning(
            min_train_races=project.backtest.min_train_races, config=project.features
        )
    except FileNotFoundError as error:
        _fail(f"{error}. {MISSING_DATA_HINT}")
    chosen = select_tuned_models(summary, tuning_candidates())
    final = select_final_model(summary, chosen)
    configured = {family: spec.name for family, spec in project.model.tuned.items()}
    if chosen != configured or final != project.model.final:
        log.warning(
            "De selectieregels kiezen andere modellen dan %s: %s, eindmodel %s. Neem die keuze "
            "over in model.tuned en model.final als je ze wilt gebruiken.",
            ctx.obj,
            chosen,
            final,
        )
    else:
        log.info("De configuratie volgt de selectieregels")


@app.command()
def train(ctx: typer.Context) -> None:
    """Train het eindmodel op alle races; schrijft model, metadata en featurebelang."""
    project = _project(ctx)
    try:
        train_final_model(spec=project.final_spec, config=project.features)
    except FileNotFoundError as error:
        _fail(f"{error}. {MISSING_DATA_HINT}")


@app.command()
def predict(
    ctx: typer.Context,
    season: Annotated[int, typer.Option(help="Seizoen, bv. 2026")],
    round_number: Annotated[int, typer.Option("--round", help="Ronde in het seizoen", min=1)],
) -> None:
    """Voorspel één race na de kwalificatie; schrijft JSON naar output/predictions/."""
    project = _project(ctx)
    try:
        payload = predict_race(season, round_number, project)
    except FileNotFoundError as error:
        _fail(f"{error}. {MISSING_DATA_HINT}")
    except (PredictionError, QualifyingNotAvailableError) as error:
        _fail(str(error))
    except (JolpicaError, requests.RequestException) as error:
        _fail(f"Jolpica-API niet bereikbaar: {error}")
    path = write_prediction(payload)
    typer.echo(format_prediction(payload))
    log.info("Voorspelling opgeslagen: %s", path)

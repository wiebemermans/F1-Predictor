"""Voorspel één race na de kwalificatie en schrijf de voorspelling als JSON.

Het eindmodel uit de configuratie wordt bij elke voorspelling opnieuw getraind op alle races
vóór de gevraagde race. Zo is de voorspelling identiek aan die race in de walk-forward-
backtest, ook voor een race die al gereden is, en is er geen apart opgeslagen model nodig.
Het JSON-formaat staat in `docs/usage.md`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_predictor.config import ProjectConfig
from f1_predictor.data.fetch import fetch_race_entry
from f1_predictor.data.jolpica import JolpicaClient
from f1_predictor.evaluation.backtest import _load_raw_results
from f1_predictor.features.build import build_features
from f1_predictor.output import json_ready, write_atomic, write_json
from f1_predictor.paths import PREDICTIONS_DIR, RAW_DIR
from f1_predictor.runinfo import run_metadata, utc_now

from .train import effective_params, training_range

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PREDICTION_MOMENT = "na de kwalificatie, vóór de race (quali_position, zonder latere gridstraffen)"
# Labels van de race zelf. Ze worden gewist vóór de features gebouwd worden: bij een race die
# nog moet komen bestaan ze niet, bij een al gereden race mogen ze niet meetellen.
NUMERIC_LABELS = ("grid", "finish_position", "points", "laps", "sprint_points")
TEXT_LABELS = ("position_text", "status")
PREDICTION_COLUMNS = ["driver_id", "driver_code", "constructor_id", "quali_position"]
LATEST_NAME = "latest.json"
_PREDICTION_FILE = re.compile(r"(\d{4})-(\d{2,})\.json")


class PredictionError(ValueError):
    """Met de lokale data kan deze race (nog) niet voorspeld worden."""


def _races_before(
    history: pd.DataFrame, season: int, round_number: int, min_train_races: int
) -> pd.DataFrame:
    """De rijen van alle races vóór de gevraagde race; een duidelijke fout als data ontbreekt."""
    last_season = int(history["season"].max())
    if season > last_season + 1:
        raise PredictionError(
            f"De lokale data loopt tot seizoen {last_season}, te vroeg voor {season}; "
            "voer eerst `f1 fetch` uit"
        )
    rounds = set(history.loc[history["season"] == season, "round"])
    missing = [number for number in range(1, round_number) if number not in rounds]
    if missing:
        raise PredictionError(
            f"Ronde(s) {missing} van {season} ontbreken in de lokale data; "
            "voer eerst `f1 fetch` uit"
        )
    is_earlier = (history["season"] < season) | (
        (history["season"] == season) & (history["round"] < round_number)
    )
    earlier = history.loc[is_earlier]
    races = len(earlier[["season", "round"]].drop_duplicates())
    needed = max(min_train_races, 1)
    if races < needed:
        raise PredictionError(
            f"Te weinig eerdere races om op te trainen: {races}, minimaal {needed} nodig"
        )
    return earlier


def _without_labels(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for column in NUMERIC_LABELS:
        rows[column] = np.nan
    for column in TEXT_LABELS:
        rows[column] = None
    return rows


def predict_race(
    season: int,
    round_number: int,
    project: ProjectConfig,
    data_dir: Path = RAW_DIR,
    client: JolpicaClient | None = None,
) -> dict[str, Any]:
    """Voorspel de finishvolgorde van één race en geef de JSON-structuur terug.

    Staat de race al in de lokale data (`result_known`), dan worden haar labels gewist en
    telt alleen de historie ervóór: een retrospectieve voorspelling. Anders komen deelnemers,
    teams en kwalificatieposities live van Jolpica; vóór de kwalificatie lukt dat niet.
    """
    history = _load_raw_results(data_dir)
    earlier = _races_before(history, season, round_number, project.backtest.min_train_races)
    is_target = (history["season"] == season) & (history["round"] == round_number)
    result_known = bool(is_target.any())
    if result_known:
        log.warning(
            "%d ronde %d is al gereden: retrospectieve voorspelling met alleen eerdere races",
            season,
            round_number,
        )
        target = history.loc[is_target]
    else:
        target = fetch_race_entry(season, round_number, client)

    combined = pd.concat([earlier, _without_labels(target)], ignore_index=True)
    table = build_features(combined, project.features)
    is_race = (table["season"] == season) & (table["round"] == round_number)
    # Dezelfde volgorde als een fold in `run_backtest`.
    train = table.loc[~is_race].sort_values(["season", "round"], kind="stable")
    race = table.loc[is_race]

    spec = project.final_spec
    predictor = spec.predictor(project.features).fit(train)
    ordered = race.assign(predicted_position=predictor.predict(race)).sort_values(
        "predicted_position"
    )
    first = race.iloc[0]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "race": {
            "season": season,
            "round": round_number,
            "race_name": first["race_name"],
            "circuit_id": first["circuit_id"],
            "date": pd.Timestamp(first["date"]).date(),
        },
        "result_known": result_known,
        "prediction_moment": PREDICTION_MOMENT,
        "model": {
            "name": spec.name,
            "family": spec.family,
            "feature_set": spec.feature_set,
            "params": effective_params(spec, predictor),
            "training": training_range(train),
        },
        "predictions": ordered[["predicted_position", *PREDICTION_COLUMNS]].to_dict(
            orient="records"
        ),
        "run": run_metadata(project),
    }
    return json_ready(payload)


def prediction_path(season: int, round_number: int, output_dir: Path = PREDICTIONS_DIR) -> Path:
    return output_dir / f"{season}-{round_number:02d}.json"


def write_prediction(payload: Mapping[str, Any], output_dir: Path = PREDICTIONS_DIR) -> Path:
    """Schrijf de voorspelling naar `{seizoen}-{ronde}.json` en werk `latest.json` bij."""
    race = payload["race"]
    path = prediction_path(race["season"], race["round"], output_dir)
    write_json(path, payload)
    _update_latest(output_dir)
    return path


def _update_latest(output_dir: Path) -> None:
    """`latest.json` volgt de chronologisch laatste voorspelde race.

    Niet het laatst geschreven bestand: een test op een oude race mag de voorspelling voor
    de komende race niet vervangen.
    """
    saved = []
    for path in output_dir.glob("*.json"):
        match = _PREDICTION_FILE.fullmatch(path.name)
        if match:
            saved.append(((int(match[1]), int(match[2])), path))
    if not saved:
        return
    _, latest = max(saved)
    text = latest.read_text(encoding="utf-8")
    write_atomic(
        output_dir / LATEST_NAME, lambda temporary: temporary.write_text(text, encoding="utf-8")
    )


def format_prediction(payload: Mapping[str, Any]) -> str:
    """Leesbare tabel van een voorspelling voor de terminal."""
    race = payload["race"]
    lines = [
        f"{race['race_name']} {race['season']} (ronde {race['round']}), "
        f"model {payload['model']['name']}",
        f"{'Pos':>3}  {'Coureur':<8} {'Team':<16} {'Kwal.':>5}",
    ]
    for row in payload["predictions"]:
        quali = "-" if row["quali_position"] is None else str(row["quali_position"])
        driver = row["driver_code"] or row["driver_id"]
        lines.append(
            f"{row['predicted_position']:>3}  {driver:<8} {row['constructor_id']:<16} {quali:>5}"
        )
    if payload["result_known"]:
        lines.append("Retrospectief: deze race was al gereden; alleen eerdere races zijn gebruikt.")
    return "\n".join(lines)

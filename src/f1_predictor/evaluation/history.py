"""Voorspellingen van vóór de race tegen de echte uitslag (`f1 evaluate`).

De backtest scoort elke race achteraf. Deze historiek scoort de voorspellingen die echt vóór
de race gemaakt en bewaard zijn (`output/predictions/`, `result_known: false`), zodra hun
uitslag in de lokale data staat. Retrospectieve voorspellingen tellen niet mee: die staan al
in de backtest. Het JSON-formaat staat in `docs/usage.md`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from f1_predictor.models.baselines import QualifyingPositionBaseline
from f1_predictor.models.predict import saved_predictions
from f1_predictor.output import json_ready, read_json, write_json_if_changed
from f1_predictor.paths import OUTPUT_DIR, PREDICTIONS_DIR, RAW_DIR
from f1_predictor.runinfo import utc_now

from .backtest import _load_raw_results
from .metrics import score_race
from .model_selection import summarize_backtest

log = logging.getLogger(__name__)

HISTORY_SCHEMA_VERSION = 1
# Vast label voor de bewaarde voorspellingen, ook als het eindmodel ooit wisselt: de
# samenvatting vergelijkt dan alle voorspellingen vooraf met de kwalificatiebaseline.
PREDICTION_LABEL = "prediction"
REFERENCE = "qualifying_position"
DRIVER_COLUMNS = [
    "driver_id",
    "driver_code",
    "constructor_id",
    "quali_position",
    "predicted_position",
    "finish_position",
    "position_text",
]


def score_prediction(prediction: Mapping[str, Any], race: pd.DataFrame) -> dict[str, Any] | None:
    """Score één voorspelling vooraf tegen de uitslag van die race.

    Het veld van de voorspelling (de kwalificatie) en dat van de uitslag (inclusief
    niet-starters) kunnen verschillen. Alleen coureurs die in beide staan tellen; binnen die
    groep worden de voorspelde en de officiële volgorde allebei hernummerd tot 1..k. De
    kwalificatiebaseline krijgt dezelfde coureurs en de `quali_position` uit de voorspelling,
    dus precies de informatie van het voorspelmoment. Zonder gemeenschappelijke coureurs:
    `None`.
    """
    predicted = pd.DataFrame(prediction["predictions"])
    result = race[["driver_id", "finish_position", "position_text"]]
    not_in_result = sorted(set(predicted["driver_id"]) - set(result["driver_id"]))
    not_in_prediction = sorted(set(result["driver_id"]) - set(predicted["driver_id"]))
    merged = predicted.merge(result, on="driver_id", how="inner", validate="one_to_one")
    if merged.empty:
        log.warning("Geen enkele coureur staat zowel in de voorspelling als in de uitslag")
        return None

    merged = merged.sort_values(["predicted_position", "driver_id"], kind="stable")
    merged = merged.reset_index(drop=True)
    predicted_rank = pd.Series(range(1, len(merged) + 1), index=merged.index)
    official = merged.sort_values(["finish_position", "driver_id"], kind="stable")
    official_rank = pd.Series(range(1, len(merged) + 1), index=official.index)
    scored = merged.assign(finish_position=official_rank.reindex(merged.index))
    scores = {
        PREDICTION_LABEL: score_race(scored, predicted_rank),
        REFERENCE: score_race(scored, QualifyingPositionBaseline().predict(scored)),
    }
    return {
        "drivers_scored": len(merged),
        "not_in_result": not_in_result,
        "not_in_prediction": not_in_prediction,
        "scores": scores,
        "drivers": merged[DRIVER_COLUMNS].to_dict(orient="records"),
    }


def evaluate_predictions(
    predictions_dir: Path = PREDICTIONS_DIR,
    data_dir: Path = RAW_DIR,
    *,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    """De historiek van alle voorspellingen vooraf waarvan de uitslag nu bekend is.

    Per race blijven de modelnaam en het tijdstip uit het voorspellingsbestand staan: een
    latere wijziging van de configuratie herschrijft de historiek niet. De samenvatting is
    `summarize_backtest` met de kwalificatiebaseline als referentie.
    """
    results = _load_raw_results(data_dir)
    races: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for (season, round_number), path in saved_predictions(predictions_dir):
        prediction = read_json(path)
        if prediction is None:
            continue
        if prediction.get("result_known") is not False:
            log.debug("%s is retrospectief en telt niet mee", path.name)
            continue
        race = results.loc[(results["season"] == season) & (results["round"] == round_number)]
        if race.empty:
            log.info("%d ronde %d: nog geen uitslag in de lokale data", season, round_number)
            continue
        scored = score_prediction(prediction, race)
        if scored is None:
            continue
        races.append(
            {
                "race": prediction["race"],
                "prediction_generated_at": prediction.get("generated_at"),
                "model": prediction["model"]["name"],
                **scored,
            }
        )
        for label, scores in scored["scores"].items():
            records.append({"model": label, "season": season, "round": round_number, **scores})

    summary = (
        summarize_backtest(pd.DataFrame(records), reference=REFERENCE).to_dict(orient="records")
        if records
        else []
    )
    return json_ready(
        {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "reference": REFERENCE,
            "summary": summary,
            "races": races,
            "run": dict(run),
        }
    )


def write_prediction_history(
    payload: Mapping[str, Any], output_path: Path = OUTPUT_DIR / "prediction_history.json"
) -> Path:
    """Schrijf de historiek, ook zonder races; alleen bij een inhoudelijke wijziging."""
    if write_json_if_changed(output_path, payload):
        log.info("Historiek opgeslagen: %s", output_path)
    else:
        log.info("Historiek ongewijzigd, niet herschreven: %s", output_path)
    return output_path


def format_history(payload: Mapping[str, Any]) -> str:
    """Leesbare tabel van de historiek voor de terminal."""
    races = payload["races"]
    if not races:
        return "Nog geen voorspellingen van vóór de race met een bekende uitslag."
    lines = [
        f"Voorspellingen vooraf tegen de uitslag ({len(races)} races)",
        f"{'Race':<34} {'MAE':>5} {'Kwal.':>6} {'Winnaar':>8} {'Podium':>7}",
    ]
    for entry in races:
        race = entry["race"]
        model = entry["scores"][PREDICTION_LABEL]
        reference = entry["scores"][REFERENCE]
        name = f"{race['season']} R{race['round']} {race['race_name']}"
        winner = "goed" if model["winner_hit"] else "fout"
        lines.append(
            f"{name[:34]:<34} {model['mae_positions']:>5.2f} {reference['mae_positions']:>6.2f} "
            f"{winner:>8} {model['podium_overlap']:>5}/3"
        )
    rows = {row["model"]: row for row in payload["summary"]}
    model, reference = rows[PREDICTION_LABEL], rows[REFERENCE]
    lines.append(
        f"Gemiddeld: MAE {model['mae_positions']:.2f} tegen {reference['mae_positions']:.2f} "
        f"voor de kwalificatievolgorde (ΔMAE {model['delta_mae']:+.2f})"
    )
    return "\n".join(lines)

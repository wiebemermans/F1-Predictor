"""Chronologische walk-forward-evaluatie van racevoorspellers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from f1_predictor.models.baselines import (
    ChampionshipPointsBaseline,
    QualifyingPositionBaseline,
    RacePredictor,
)
from f1_predictor.paths import OUTPUT_DIR, RAW_DIR

from .metrics import score_race

BACKTEST_COLUMNS = [
    "model",
    "season",
    "round",
    "race_name",
    "mae_positions",
    "spearman",
    "winner_hit",
    "podium_overlap",
    "top10_overlap",
]

BASELINE_FACTORIES: dict[str, Callable[[], RacePredictor]] = {
    "qualifying_position": QualifyingPositionBaseline,
    "championship_points": ChampionshipPointsBaseline,
}


def run_backtest(
    df: pd.DataFrame,
    predictor_factories: Mapping[str, Callable[[], RacePredictor]],
    min_train_races: int = 20,
) -> pd.DataFrame:
    """Evalueer elke predictor op dezelfde races, met alleen eerdere races als training."""
    required = {
        "season",
        "round",
        "race_name",
        "driver_id",
        "quali_position",
        "finish_position",
        "points",
        "sprint_points",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Data mist vereiste kolommen: {sorted(missing)}")
    if (
        isinstance(min_train_races, bool)
        or not isinstance(min_train_races, int)
        or min_train_races < 0
    ):
        raise ValueError("min_train_races moet een niet-negatief geheel getal zijn")
    if not predictor_factories or any(not name for name in predictor_factories):
        raise ValueError("Geef minstens één predictor met een niet-lege naam op")
    if df.empty:
        raise ValueError("Kan geen backtest uitvoeren op lege data")
    if df[["season", "round", "driver_id"]].isna().any().any():
        raise ValueError("season, round en driver_id moeten volledig zijn")
    if df.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("(season, round, driver_id) moet uniek zijn")

    ordered = df.reset_index(drop=True).sort_values(["season", "round"], kind="stable")
    race_keys = ordered[["season", "round"]].drop_duplicates().itertuples(index=False, name=None)
    races = list(race_keys)
    if len(races) <= min_train_races:
        raise ValueError(
            f"Onvoldoende races: {len(races)} beschikbaar voor minimaal "
            f"{min_train_races} trainingsraces en één testrace"
        )
    records: list[dict[str, object]] = []
    for race_index in range(min_train_races, len(races)):
        season, round_number = races[race_index]
        is_train = (ordered["season"] < season) | (
            (ordered["season"] == season) & (ordered["round"] < round_number)
        )
        is_test = (ordered["season"] == season) & (ordered["round"] == round_number)
        train = ordered.loc[is_train]
        race = ordered.loc[is_test]

        for name, factory in predictor_factories.items():
            predictor = factory()
            predicted_rank = predictor.fit(train.copy()).predict(race.copy())
            records.append(
                {
                    "model": name,
                    "season": int(season),
                    "round": int(round_number),
                    "race_name": str(race["race_name"].iloc[0]),
                    **score_race(race, predicted_rank),
                }
            )
    return pd.DataFrame.from_records(records, columns=BACKTEST_COLUMNS)


def run_baseline_backtest(
    data_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_DIR / "backtest_baselines.csv",
    min_train_races: int = 20,
) -> pd.DataFrame:
    """Laad de seizoensparquets, voer beide baselines uit en schrijf race-scores weg."""
    data_files = sorted(data_dir.glob("results_*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"Geen results_*.parquet-bestanden gevonden in {data_dir}")
    data = pd.concat((pd.read_parquet(path) for path in data_files), ignore_index=True)
    results = run_backtest(data, BASELINE_FACTORIES, min_train_races=min_train_races)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    results.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return results

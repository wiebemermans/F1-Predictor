"""Bouw per seizoen één tabel (results + qualifying + sprint) en sla op als parquet."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from f1_predictor.data.jolpica import JolpicaClient
from f1_predictor.data.parse import flatten_qualifying, flatten_results, flatten_sprint
from f1_predictor.paths import CACHE_DIR, RAW_DIR

log = logging.getLogger(__name__)

KEYS = ["season", "round", "driver_id"]


def _collect(
    client: JolpicaClient,
    season: int,
    endpoint: str,
    flatten: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for race in client.races(f"{season}/{endpoint}"):
        rows.extend(flatten(race))
    return pd.DataFrame(rows)


def build_season_table(season: int, client: JolpicaClient) -> pd.DataFrame:
    df = _collect(client, season, "results", flatten_results)
    if df.empty:
        raise ValueError(f"Geen resultaten gevonden voor {season}")

    quali = _collect(client, season, "qualifying", flatten_qualifying)
    if quali.empty:
        df["quali_position"] = pd.NA
    else:
        df = df.merge(quali, on=KEYS, how="left", validate="one_to_one")

    sprint = _collect(client, season, "sprint", flatten_sprint)
    if sprint.empty:
        df["sprint_points"] = 0.0
    else:
        df = df.merge(sprint, on=KEYS, how="left", validate="one_to_one")
        df["sprint_points"] = df["sprint_points"].fillna(0.0)

    df["date"] = pd.to_datetime(df["date"])
    df["quali_position"] = df["quali_position"].astype("Int64")
    df = df.sort_values(["round", "finish_position"]).reset_index(drop=True)

    if df["round"].nunique() != df["round"].max():
        log.warning("%d: gaten in de rondenummers, controleer de data", season)
    return df


def season_path(season: int) -> Path:
    return RAW_DIR / f"results_{season}.parquet"


def save_season(df: pd.DataFrame, season: int) -> Path:
    path = season_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # eerst tijdelijk schrijven: geen half bestand bij een crash
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def fetch_seasons(
    start: int,
    end: int,
    force: bool = False,
    client: JolpicaClient | None = None,
) -> list[int]:
    """Haal seizoenen op; geeft de lijst mislukte seizoenen terug."""
    client = client or JolpicaClient(CACHE_DIR / "jolpica")
    this_year = date.today().year
    failed: list[int] = []
    for season in range(start, end + 1):
        # Afgelopen seizoenen veranderen niet meer; het lopende seizoen altijd verversen.
        if season_path(season).exists() and not force and season < this_year:
            log.info("%d: bestaat al, overgeslagen", season)
            continue
        try:
            df = build_season_table(season, client)
            save_season(df, season)
            log.info(
                "%d: %d rijen, %d races opgeslagen",
                season,
                len(df),
                df["round"].nunique(),
            )
        except Exception:
            log.exception("%d: mislukt", season)
            failed.append(season)
    return failed

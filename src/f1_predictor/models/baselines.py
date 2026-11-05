"""Eenvoudige racevoorspellers voor de evaluatiebaseline."""

from __future__ import annotations

from typing import Protocol, Self

import pandas as pd


class RacePredictor(Protocol):
    def fit(self, train: pd.DataFrame) -> Self: ...

    def predict(self, race: pd.DataFrame) -> pd.Series: ...


def _validate_drivers(race: pd.DataFrame) -> None:
    if "driver_id" not in race:
        raise ValueError("Race mist vereiste kolom: driver_id")
    if race.empty:
        raise ValueError("Kan geen voorspelling maken voor een lege race")
    if race["driver_id"].isna().any() or race["driver_id"].duplicated().any():
        raise ValueError("driver_id moet per race uniek en niet-leeg zijn")


def _rank_by(
    race: pd.DataFrame, sort_columns: list[str], ascending: list[bool] | None = None
) -> pd.Series:
    order = race.sort_values(
        sort_columns,
        ascending=True if ascending is None else ascending,
        na_position="last",
        kind="stable",
    )
    return pd.Series(range(1, len(order) + 1), index=order.index, dtype="int64").reindex(race.index)


class QualifyingPositionBaseline:
    """Gebruik de kwalificatiepositie; ontbrekende posities komen achteraan."""

    def fit(self, train: pd.DataFrame) -> Self:
        return self

    def predict(self, race: pd.DataFrame) -> pd.Series:
        _validate_drivers(race)
        if "quali_position" not in race:
            raise ValueError("Race mist vereiste kolom: quali_position")
        qualifying_position = pd.to_numeric(race["quali_position"], errors="raise")
        return _rank_by(
            race.assign(_quali_position=qualifying_position),
            ["_quali_position", "driver_id"],
        )


class ChampionshipPointsBaseline:
    """Rangschik op race- en sprintpunten die vóór de race zijn behaald."""

    def __init__(self) -> None:
        self._points: dict[tuple[object, object], float] = {}

    def fit(self, train: pd.DataFrame) -> Self:
        required = {"season", "driver_id", "points", "sprint_points"}
        missing = required.difference(train.columns)
        if missing:
            raise ValueError(f"Trainingsdata mist vereiste kolommen: {sorted(missing)}")
        if train[["season", "driver_id"]].isna().any().any():
            raise ValueError("season en driver_id moeten in trainingsdata niet-leeg zijn")

        race_points = pd.to_numeric(train["points"], errors="raise")
        sprint_points = pd.to_numeric(train["sprint_points"], errors="raise")
        if race_points.isna().any() or sprint_points.isna().any():
            raise ValueError("Race- en sprintpunten moeten in trainingsdata volledig zijn")

        history = train[["season", "driver_id"]].copy()
        history["_points"] = race_points + sprint_points
        totals = history.groupby(["season", "driver_id"], sort=False)["_points"].sum()
        self._points = totals.to_dict()
        return self

    def predict(self, race: pd.DataFrame) -> pd.Series:
        _validate_drivers(race)
        if "season" not in race or race["season"].isna().any():
            raise ValueError("Race mist een geldige kolom: season")
        seasons = race["season"].unique()
        if len(seasons) != 1:
            raise ValueError("Een voorspelling moet precies één seizoen bevatten")
        season = seasons[0]
        points = race["driver_id"].map(lambda driver_id: self._points.get((season, driver_id), 0.0))
        return _rank_by(
            race.assign(_points=points),
            ["_points", "driver_id"],
            ascending=[False, True],
        )

"""Featuregroepen voor een voorspelling na de kwalificatie en vóór de race.

Elke groepsfunctie krijgt de volledige racetabel met `race_index` en geeft alleen haar eigen
featurekolommen terug, met dezelfde index als de invoer. Historische features gebruiken
uitsluitend races vóór race N: per reeks eerst `shift(1)`, pas daarna rolling/expanding.
Van race N zelf worden alleen gegevens gebruikt die na de kwalificatie bekend zijn
(`quali_position`, `constructor_id`, `circuit_id` en de deelnemerslijst).
De betekenis per feature staat in `docs/features.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .build import FeatureConfig


def retirement_mask(status: pd.Series) -> pd.Series:
    """True voor uitvallers (DNF) en niet-starters (DNS).

    Jolpica gebruikt twee statusvocabulaires: oudere seizoenen "+1 Lap", "+2 Laps", ...;
    recentere seizoenen "Lapped". Beide gelden als gefinisht. Diskwalificaties tellen
    niet als uitval: de auto reed de race uit en het zegt niets over betrouwbaarheid.
    """
    status = status.fillna("").astype(str).str.strip().str.lower()
    finished_like = (
        status.eq("finished")
        | status.eq("lapped")
        | status.str.fullmatch(r"\+\d+\s*laps?", na=False)
    )
    return ~(finished_like | status.eq("disqualified"))


def _chronological(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("race_index", kind="stable")


def _prior_rolling_mean(
    values: pd.Series, keys: pd.Series | list[pd.Series], window: int
) -> pd.Series:
    """Gemiddelde van de vorige `window` waarden per sleutel; de huidige rij telt niet mee.

    `values` moet chronologisch gesorteerd zijn. Ontbrekende waarden in het venster tellen
    niet mee in het gemiddelde; zonder eerdere waarden is het resultaat NaN.
    """
    return values.groupby(keys, sort=False).transform(
        lambda series: series.shift(1).rolling(window, min_periods=1).mean()
    )


def _prior_expanding_mean(values: pd.Series, keys: pd.Series | list[pd.Series]) -> pd.Series:
    """Gemiddelde van alle vorige waarden per sleutel; de huidige rij telt niet mee."""
    return values.groupby(keys, sort=False).transform(
        lambda series: series.shift(1).expanding().mean()
    )


def _as_float(column: pd.Series) -> pd.Series:
    """Numerieke kolom als float64; ontbrekende waarden (ook `pd.NA`) worden NaN."""
    return pd.to_numeric(column, errors="raise").astype("float64")


def starting_position(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Kwalificatiepositie van race N; ontbrekend telt als laatste plaats van het veld.

    Bewust `quali_position` en niet `grid`: het voorspelmoment ligt na de kwalificatie,
    vóór eventuele gridstraffen (zie `docs/phase5_baselines.md`).
    """
    field_size = df.groupby("race_index")["driver_id"].transform("size").astype("float64")
    quali = _as_float(df["quali_position"])
    missing = quali.isna()
    return pd.DataFrame(
        {
            "start_position": quali.where(~missing, field_size),
            "start_quali_missing": missing.astype("int64"),
        },
        index=df.index,
    )


def driver_form(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Recente vorm van de coureur over zijn eigen vorige races (over seizoenen heen)."""
    ordered = _chronological(df)
    drivers = ordered["driver_id"]
    finish = _as_float(ordered["finish_position"])
    points = _as_float(ordered["points"])
    retired = retirement_mask(ordered["status"]).astype("float64")

    columns: dict[str, pd.Series] = {}
    for window in config.form_windows:
        columns[f"driver_finish_mean_{window}"] = _prior_rolling_mean(finish, drivers, window)
    for window in config.form_windows:
        columns[f"driver_points_mean_{window}"] = _prior_rolling_mean(points, drivers, window)
    columns[f"driver_retirement_rate_{config.rate_window}"] = _prior_rolling_mean(
        retired, drivers, config.rate_window
    )
    return pd.DataFrame(columns, index=ordered.index).reindex(df.index)


def team_form(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Recente vorm van het team, als maat voor de sterkte van de auto.

    Eerst per (team, race) samenvatten en pas daarna verschuiven: een `shift(1)` op
    coureursrijen zou het resultaat van de teamgenoot in dezelfde race laten meetellen.
    Een nieuwe `constructor_id` (ook na een naamswijziging) begint zonder historie.
    """
    per_team_race = (
        df.assign(_finish=_as_float(df["finish_position"]), _points=_as_float(df["points"]))
        .groupby(["constructor_id", "race_index"], sort=True)
        .agg(_finish=("_finish", "mean"), _points=("_points", "sum"))
        .reset_index()
    )
    teams = per_team_race["constructor_id"]
    columns: dict[str, pd.Series] = {}
    for window in config.form_windows:
        columns[f"team_finish_mean_{window}"] = _prior_rolling_mean(
            per_team_race["_finish"], teams, window
        )
    for window in config.form_windows:
        columns[f"team_points_mean_{window}"] = _prior_rolling_mean(
            per_team_race["_points"], teams, window
        )
    team_features = pd.DataFrame(columns).set_index(
        pd.MultiIndex.from_frame(per_team_race[["constructor_id", "race_index"]])
    )
    joined = df[["constructor_id", "race_index"]].join(
        team_features, on=["constructor_id", "race_index"]
    )
    return joined.drop(columns=["constructor_id", "race_index"])


def championship(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Kampioenschapsstand vóór race N binnen hetzelfde seizoen.

    Race- plus sprintpunten van eerdere ronden; de sprint van race N telt niet mee, zoals
    in `ChampionshipPointsBaseline`. De positie is de rang onder de deelnemers van race N
    (meeste punten = 1, gelijke punten delen de beste rang).
    """
    ordered = _chronological(df)
    weekend_points = _as_float(ordered["points"]) + _as_float(ordered["sprint_points"])
    points_before = weekend_points.groupby(
        [ordered["season"], ordered["driver_id"]], sort=False
    ).transform(lambda series: series.shift(1, fill_value=0.0).cumsum())
    position_before = points_before.groupby(ordered["race_index"]).rank(
        method="min", ascending=False
    )
    return pd.DataFrame(
        {"champ_points_before": points_before, "champ_position_before": position_before},
        index=ordered.index,
    ).reindex(df.index)


def circuit_history(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Afwijking van de coureur op dit circuit ten opzichte van zijn eigen gemiddelde.

    `circuit_delta_shrunk = n / (n + k) * (gem. finish op circuit - gem. finish overall)`,
    beide over races vóór race N, met n eerdere races op dit circuit en k de shrinkage.
    Negatief betekent beter dan normaal; zonder circuithistorie is de waarde 0.
    """
    ordered = _chronological(df)
    finish = _as_float(ordered["finish_position"])
    driver_circuit = [ordered["driver_id"], ordered["circuit_id"]]
    prior_races = ordered.groupby(["driver_id", "circuit_id"], sort=False).cumcount()
    circuit_mean = _prior_expanding_mean(finish, driver_circuit)
    driver_mean = _prior_expanding_mean(finish, ordered["driver_id"])
    weight = prior_races / (prior_races + config.circuit_shrinkage)
    delta = (weight * (circuit_mean - driver_mean)).where(prior_races > 0, 0.0)
    return pd.DataFrame(
        {
            "circuit_prior_races": prior_races.astype("int64"),
            "circuit_delta_shrunk": delta.astype("float64"),
        },
        index=ordered.index,
    ).reindex(df.index)


def teammate(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Vergelijking met de teamgenoot(en) in hetzelfde team en dezelfde race.

    `teammate_quali_gap`: eigen kwalificatiepositie min het gemiddelde van de teamgenoten
    in race N (negatief = sneller). `teammate_beat_rate_{n}`: aandeel verslagen teamgenoten
    over de vorige n races van de coureur; races zonder teamgenoot tellen niet mee.
    """
    team_race = [df["race_index"], df["constructor_id"]]
    team_size = df.groupby(["race_index", "constructor_id"])["driver_id"].transform("size")

    quali = _as_float(df["quali_position"])
    quali_group = quali.groupby(team_race)
    other_quali_count = quali_group.transform("count") - quali.notna().astype("int64")
    other_quali_mean = (quali_group.transform("sum") - quali.fillna(0.0)) / other_quali_count
    quali_gap = (quali - other_quali_mean).where(quali.notna() & (other_quali_count > 0))

    finish = _as_float(df["finish_position"])
    # Aantal teamgenoten met een slechtere finish: teamgrootte min (#coureurs met finish <= eigen).
    beaten = team_size - finish.groupby(team_race).rank(method="max")
    beat_share = (beaten / (team_size - 1)).where(team_size > 1)

    ordered_index = _chronological(df).index
    beat_rate = _prior_rolling_mean(
        beat_share.loc[ordered_index], df.loc[ordered_index, "driver_id"], config.rate_window
    )
    return pd.DataFrame(
        {
            "teammate_quali_gap": quali_gap.astype("float64"),
            f"teammate_beat_rate_{config.rate_window}": beat_rate.reindex(df.index),
        },
        index=df.index,
    )

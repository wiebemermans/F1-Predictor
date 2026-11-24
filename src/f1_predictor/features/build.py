"""Bouw de featuretabel: ruwe kolommen, `race_index` en de ingeschakelde featuregroepen."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1_predictor.evaluation.explore import load_all_results
from f1_predictor.paths import PROCESSED_DIR, RAW_DIR

from .groups import (
    championship,
    circuit_history,
    driver_form,
    starting_position,
    team_form,
    teammate,
)

log = logging.getLogger(__name__)

# Vaste volgorde: bepaalt ook de kolomvolgorde in de featuretabel.
FEATURE_GROUPS: dict[str, Callable[[pd.DataFrame, FeatureConfig], pd.DataFrame]] = {
    "starting_position": starting_position,
    "driver_form": driver_form,
    "team_form": team_form,
    "championship": championship,
    "circuit_history": circuit_history,
    "teammate": teammate,
}

REQUIRED_COLUMNS = frozenset(
    {
        "season",
        "round",
        "driver_id",
        "constructor_id",
        "circuit_id",
        "quali_position",
        "finish_position",
        "status",
        "points",
        "sprint_points",
    }
)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class FeatureConfig:
    """Welke featuregroepen aanstaan en met welke vensters ze worden berekend.

    De CLI leest deze waarden uit de sectie `features` van `config/config.yaml`; de defaults
    hier zijn dezelfde als in dat bestand.
    """

    groups: tuple[str, ...] = tuple(FEATURE_GROUPS)
    form_windows: tuple[int, ...] = (3, 5)
    rate_window: int = 10
    circuit_shrinkage: float = 3.0

    def __post_init__(self) -> None:
        # Lijsten (bv. uit YAML) worden tuples, zodat de config onveranderlijk blijft.
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "form_windows", tuple(self.form_windows))

        if not self.groups:
            raise ValueError("Kies minstens één featuregroep")
        unknown = [group for group in self.groups if group not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"Onbekende featuregroepen: {unknown}; kies uit {list(FEATURE_GROUPS)}"
            )
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("Elke featuregroep mag maar één keer voorkomen")
        if not self.form_windows or not all(map(_is_positive_int, self.form_windows)):
            raise ValueError("form_windows moet positieve gehele getallen bevatten")
        if len(set(self.form_windows)) != len(self.form_windows):
            raise ValueError("form_windows mag geen dubbele vensters bevatten")
        if not _is_positive_int(self.rate_window):
            raise ValueError("rate_window moet een positief geheel getal zijn")
        shrinkage = self.circuit_shrinkage
        if (
            isinstance(shrinkage, bool)
            or not isinstance(shrinkage, int | float)
            or not math.isfinite(shrinkage)
            or shrinkage < 0
        ):
            raise ValueError("circuit_shrinkage moet een eindig, niet-negatief getal zijn")

    @property
    def enabled_groups(self) -> list[str]:
        """Ingeschakelde groepen in de vaste volgorde van `FEATURE_GROUPS`."""
        return [group for group in FEATURE_GROUPS if group in self.groups]


def group_columns(group: str, config: FeatureConfig) -> list[str]:
    """Namen van de featurekolommen die één groep met deze config oplevert."""
    windows = config.form_windows
    rate = config.rate_window
    columns = {
        "starting_position": ["start_position", "start_quali_missing"],
        "driver_form": [
            *(f"driver_finish_mean_{window}" for window in windows),
            *(f"driver_points_mean_{window}" for window in windows),
            f"driver_retirement_rate_{rate}",
        ],
        "team_form": [
            *(f"team_finish_mean_{window}" for window in windows),
            *(f"team_points_mean_{window}" for window in windows),
        ],
        "championship": ["champ_points_before", "champ_position_before"],
        "circuit_history": ["circuit_prior_races", "circuit_delta_shrunk"],
        "teammate": ["teammate_quali_gap", f"teammate_beat_rate_{rate}"],
    }
    if group not in columns:
        raise ValueError(f"Onbekende featuregroep: {group}")
    return columns[group]


def feature_columns(config: FeatureConfig | None = None) -> list[str]:
    """Alle featurekolommen voor de config; modellen lezen alleen deze kolommen."""
    config = config or FeatureConfig()
    return [column for group in config.enabled_groups for column in group_columns(group, config)]


def add_race_index(df: pd.DataFrame) -> pd.DataFrame:
    """Voeg `race_index` toe: chronologisch racenummer 0..n-1 op (`season`, `round`)."""
    race_index = df.groupby(["season", "round"], sort=True).ngroup().astype("int64")
    return df.assign(race_index=race_index)


def _validate_input(df: pd.DataFrame, config: FeatureConfig) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Data mist vereiste kolommen: {sorted(missing)}")
    if df.empty:
        raise ValueError("Kan geen features bouwen op lege data")
    if not df.index.is_unique:
        raise ValueError("De index van de data moet uniek zijn")
    if df[["season", "round", "driver_id"]].isna().any().any():
        raise ValueError("season, round en driver_id moeten volledig zijn")
    if df.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("(season, round, driver_id) moet uniek zijn")
    clashes = {"race_index", *feature_columns(config)}.intersection(df.columns)
    if clashes:
        raise ValueError(f"Data bevat al featurekolommen: {sorted(clashes)}")


def build_features(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Geef een nieuwe tabel terug: ruwe kolommen + `race_index` + featurekolommen.

    Index en rijvolgorde blijven gelijk aan de invoer; de ruwe kolommen worden niet
    aangepast. Labels zoals `finish_position` blijven staan voor training, maar mogen
    nooit als feature worden gebruikt; gebruik daarvoor `feature_columns(config)`.
    """
    config = config or FeatureConfig()
    _validate_input(df, config)
    table = add_race_index(df)
    parts = [FEATURE_GROUPS[group](table, config) for group in config.enabled_groups]
    return pd.concat([table, *parts], axis=1)


def write_feature_table(
    raw_dir: Path = RAW_DIR,
    output_path: Path = PROCESSED_DIR / "features.parquet",
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Bouw de featuretabel uit alle seizoensparquets en schrijf die atomisch weg."""
    config = config or FeatureConfig()
    table = build_features(load_all_results(raw_dir), config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    table.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    log.info(
        "Featuretabel: %d rijen, %d races, %d features -> %s",
        len(table),
        table["race_index"].nunique(),
        len(feature_columns(config)),
        output_path,
    )
    return table

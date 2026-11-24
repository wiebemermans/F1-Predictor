"""Gedeelde invoercontroles voor voorspellers die op featurekolommen trainen."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def check_feature_columns(feature_columns: Sequence[str]) -> list[str]:
    """Geef de featurekolommen als lijst terug; een lege of dubbele lijst is ongeldig."""
    columns = list(feature_columns)
    if not columns:
        raise ValueError("Geef minstens één featurekolom op")
    if len(set(columns)) != len(columns):
        raise ValueError("Featurekolommen moeten uniek zijn")
    return columns


def feature_matrix(data: pd.DataFrame, feature_columns: list[str], label: str) -> pd.DataFrame:
    """Selecteer de featurekolommen als float64; ontbrekende waarden blijven NaN."""
    missing = set(feature_columns).difference(data.columns)
    if missing:
        raise ValueError(f"{label} mist featurekolommen: {sorted(missing)}")
    return data[feature_columns].astype("float64")


def finish_target(train: pd.DataFrame) -> pd.Series:
    """De officiële finishpositie als trainingslabel; die moet voor elke rij bekend zijn."""
    if "finish_position" not in train:
        raise ValueError("Trainingsdata mist vereiste kolom: finish_position")
    if train.empty:
        raise ValueError("Kan niet trainen op lege data")
    target = pd.to_numeric(train["finish_position"], errors="raise")
    if target.isna().any():
        raise ValueError("finish_position moet in trainingsdata volledig zijn")
    return target

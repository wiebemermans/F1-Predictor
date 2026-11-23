"""Lineaire voorspeller als meetinstrument voor featuregroepen (fase 6)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import _rank_by, _validate_drivers


class RidgeFeaturePredictor:
    """Ridge-regressie op `finish_position`; een lagere voorspelde finish geeft een betere rang.

    Ontbrekende featurewaarden worden aangevuld met de mediaan van de trainingsdata van
    dezelfde fold, zodat de testrace nooit invloed heeft op de imputatie of schaling.
    Gelijke voorspellingen worden op `driver_id` alfabetisch gesorteerd.
    """

    def __init__(self, feature_columns: Sequence[str], alpha: float = 1.0) -> None:
        columns = list(feature_columns)
        if not columns:
            raise ValueError("Geef minstens één featurekolom op")
        if len(set(columns)) != len(columns):
            raise ValueError("Featurekolommen moeten uniek zijn")
        self.feature_columns = columns
        self.alpha = alpha
        self._pipeline: Pipeline | None = None

    def _features(self, data: pd.DataFrame, label: str) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(data.columns)
        if missing:
            raise ValueError(f"{label} mist featurekolommen: {sorted(missing)}")
        return data[self.feature_columns].astype("float64")

    def fit(self, train: pd.DataFrame) -> Self:
        if "finish_position" not in train:
            raise ValueError("Trainingsdata mist vereiste kolom: finish_position")
        if train.empty:
            raise ValueError("Kan niet trainen op lege data")
        target = pd.to_numeric(train["finish_position"], errors="raise")
        if target.isna().any():
            raise ValueError("finish_position moet in trainingsdata volledig zijn")
        self._pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=self.alpha),
        ).fit(self._features(train, "Trainingsdata"), target)
        return self

    def predict(self, race: pd.DataFrame) -> pd.Series:
        _validate_drivers(race)
        if self._pipeline is None:
            raise RuntimeError("Roep fit aan vóór predict")
        predicted_finish = self._pipeline.predict(self._features(race, "Race"))
        return _rank_by(
            race.assign(_predicted_finish=predicted_finish), ["_predicted_finish", "driver_id"]
        )

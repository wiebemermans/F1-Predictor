"""Lineaire voorspeller: meetinstrument voor featuregroepen (fase 6) en eerste model (fase 7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import _rank_by, _validate_drivers
from .common import check_feature_columns, feature_matrix, finish_target


class RidgeFeaturePredictor:
    """Ridge-regressie op `finish_position`; een lagere voorspelde finish geeft een betere rang.

    Ontbrekende featurewaarden worden aangevuld met de mediaan van de trainingsdata van
    dezelfde fold, zodat de testrace nooit invloed heeft op de imputatie of schaling.
    Gelijke voorspellingen worden op `driver_id` alfabetisch gesorteerd.
    """

    def __init__(self, feature_columns: Sequence[str], alpha: float = 1.0) -> None:
        self.feature_columns = check_feature_columns(feature_columns)
        self.alpha = alpha
        self._pipeline: Pipeline | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        target = finish_target(train)
        self._pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=self.alpha),
        ).fit(feature_matrix(train, self.feature_columns, "Trainingsdata"), target)
        return self

    def _fitted_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise RuntimeError("Roep fit aan vóór predict")
        return self._pipeline

    def predict(self, race: pd.DataFrame) -> pd.Series:
        _validate_drivers(race)
        pipeline = self._fitted_pipeline()
        predicted_finish = pipeline.predict(feature_matrix(race, self.feature_columns, "Race"))
        return _rank_by(
            race.assign(_predicted_finish=predicted_finish), ["_predicted_finish", "driver_id"]
        )

    def feature_importance(self, data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Gestandaardiseerde coëfficiënten (voorspelde finish per standaardafwijking).

        Positief betekent dat een hogere featurewaarde een slechtere (hogere) finish voorspelt.
        `data` wordt niet gebruikt; de parameter bestaat voor dezelfde aanroep als bij XGBoost.
        """
        coefficients = self._fitted_pipeline()[-1].coef_
        return pd.DataFrame({"feature": self.feature_columns, "coefficient": coefficients})

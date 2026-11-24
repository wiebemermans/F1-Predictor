"""Gradient boosting (XGBoost) achter dezelfde `fit`/`predict`-interface als de baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import pandas as pd
import xgboost as xgb

from .baselines import _rank_by, _validate_drivers
from .common import check_feature_columns, feature_matrix, finish_target

# Conservatieve standaardinstellingen. Bewust geen `subsample`: rijsampling hangt in XGBoost af
# van het aantal threads, waardoor dezelfde code per machine andere voorspellingen geeft.
# Kolomsampling (`colsample_bytree`) heeft dat probleem niet.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 0,
}


class _XGBoostPredictor:
    """Gemeenschappelijke basis: featurekolommen, parameters en interpretatie.

    Ontbrekende featurewaarden (NaN) worden niet aangevuld; XGBoost leert per split zelf
    naar welke kant ontbrekende waarden gaan.
    """

    objective: str

    def __init__(self, feature_columns: Sequence[str], **params: Any) -> None:
        self.feature_columns = check_feature_columns(feature_columns)
        if "objective" in params:
            raise ValueError(f"De objective ligt vast op {self.objective}")
        self.params: dict[str, Any] = {**DEFAULT_XGB_PARAMS, **params}
        self._model: xgb.XGBModel | None = None

    def _fitted_model(self) -> xgb.XGBModel:
        if self._model is None:
            raise RuntimeError("Roep fit aan vóór predict")
        return self._model

    def _race_scores(self, race: pd.DataFrame) -> np.ndarray:
        _validate_drivers(race)
        model = self._fitted_model()
        return model.predict(feature_matrix(race, self.feature_columns, "Race"))

    def feature_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aandeel in de totale gain en gemiddelde absolute SHAP-waarde per feature over `data`.

        SHAP-waarden komen uit XGBoost zelf (`pred_contribs`, exacte TreeSHAP) en staan op de
        schaal van de modeluitvoer: voorspelde finish bij de regressor, rangscore bij de ranker.
        """
        booster = self._fitted_model().get_booster()
        gain = booster.get_score(importance_type="total_gain")
        total_gain = sum(gain.values())
        features = feature_matrix(data, self.feature_columns, "Data")
        contributions = booster.predict(xgb.DMatrix(features), pred_contribs=True)[:, :-1]
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "gain_share": [
                    gain.get(column, 0.0) / total_gain if total_gain > 0 else 0.0
                    for column in self.feature_columns
                ],
                "mean_abs_shap": np.abs(contributions).mean(axis=0),
            }
        )


class XGBRegressorPredictor(_XGBoostPredictor):
    """XGBoost-regressie op `finish_position`; een lagere voorspelde finish geeft een betere rang.

    Gelijke voorspellingen worden op `driver_id` alfabetisch gesorteerd.
    """

    objective = "reg:squarederror"

    def fit(self, train: pd.DataFrame) -> Self:
        target = finish_target(train)
        features = feature_matrix(train, self.feature_columns, "Trainingsdata")
        self._model = xgb.XGBRegressor(objective=self.objective, **self.params).fit(
            features, target
        )
        return self

    def predict(self, race: pd.DataFrame) -> pd.Series:
        predicted_finish = self._race_scores(race)
        return _rank_by(
            race.assign(_predicted_finish=predicted_finish), ["_predicted_finish", "driver_id"]
        )


class XGBRankerPredictor(_XGBoostPredictor):
    """XGBoost learning-to-rank (`rank:pairwise`) met één groep per race.

    Label per race: aantal deelnemers min `finish_position` (de winnaar krijgt het hoogste
    label, de laatste 0). Een hogere score geeft een betere rang; gelijke scores worden op
    `driver_id` alfabetisch gesorteerd.
    """

    objective = "rank:pairwise"

    def fit(self, train: pd.DataFrame) -> Self:
        target = finish_target(train)
        missing = {"season", "round", "driver_id"}.difference(train.columns)
        if missing:
            raise ValueError(f"Trainingsdata mist vereiste kolommen: {sorted(missing)}")
        if train[["season", "round", "driver_id"]].isna().any().any():
            raise ValueError("season, round en driver_id moeten in trainingsdata volledig zijn")

        race_id = train.groupby(["season", "round"], sort=True).ngroup()
        label = race_id.map(race_id.value_counts()) - target
        if (label < 0).any():
            raise ValueError("finish_position mag niet groter zijn dan het aantal deelnemers")
        # XGBoost eist rijen gesorteerd per groep; binnen een race op driver_id, zodat de
        # volgorde van de invoerrijen het model niet beïnvloedt.
        order = np.lexsort((train["driver_id"].astype(str).to_numpy(), race_id.to_numpy()))
        features = feature_matrix(train.iloc[order], self.feature_columns, "Trainingsdata")
        self._model = xgb.XGBRanker(objective=self.objective, **self.params).fit(
            features, label.iloc[order].astype("int64"), qid=race_id.iloc[order].to_numpy()
        )
        return self

    def predict(self, race: pd.DataFrame) -> pd.Series:
        scores = self._race_scores(race)
        return _rank_by(
            race.assign(_score=scores), ["_score", "driver_id"], ascending=[False, True]
        )


class XGBNDCGRankerPredictor(XGBRankerPredictor):
    """Als `XGBRankerPredictor`, maar met `rank:ndcg`: fouten bovenaan het veld wegen zwaarder.

    XGBoost gebruikt standaard een exponentiële gain (2^label − 1), dus de winnaar (label
    n − 1) telt veel zwaarder dan de achterhoede. Labels moeten kleiner zijn dan 32; met
    hoogstens 22 deelnemers per race is dat ruim voldaan.
    """

    objective = "rank:ndcg"

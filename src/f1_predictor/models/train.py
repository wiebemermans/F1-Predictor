"""Modelspecificaties en het trainen van het eindmodel.

Welke configuratie per familie gekozen is en welk model het eindmodel is, staat in
`config/config.yaml` (zie `f1_predictor.config`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from functools import cache, partial
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from f1_predictor.evaluation.explore import load_all_results
from f1_predictor.features.build import FEATURE_GROUPS, FeatureConfig, build_features
from f1_predictor.features.build import feature_columns as config_feature_columns
from f1_predictor.output import write_atomic, write_json
from f1_predictor.paths import OUTPUT_DIR, RAW_DIR
from f1_predictor.runinfo import git_revision, library_versions, utc_now

from .baselines import RacePredictor
from .boosting import XGBNDCGRankerPredictor, XGBRankerPredictor, XGBRegressorPredictor
from .linear import RidgeFeaturePredictor

log = logging.getLogger(__name__)

MODEL_FAMILIES: dict[str, Callable[..., RacePredictor]] = {
    "ridge": RidgeFeaturePredictor,
    "xgb_regressor": XGBRegressorPredictor,
    "xgb_ranker": XGBRankerPredictor,
    "xgb_ranker_ndcg": XGBNDCGRankerPredictor,
}

# Featuresets die in de tuning worden vergeleken: alle groepen, of de groepen die in fase 6
# de Ridge-backtest verbeterden.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": tuple(FEATURE_GROUPS),
    "upto_championship": ("starting_position", "driver_form", "team_form", "championship"),
}

_NAME_ABBREVIATIONS = {"alpha": "a", "max_depth": "d", "n_estimators": "n"}

# Parameters die bewust niet instelbaar zijn, met de reden voor de foutmelding.
_FORBIDDEN_XGB_PARAMS = {
    "objective": "de objective ligt per modelfamilie vast",
    "subsample": "rijsampling hangt af van het aantal threads en is daardoor niet reproduceerbaar",
}


@cache
def _xgb_parameter_names() -> frozenset[str]:
    """Alle parameters die XGBoost kent (regressor en ranker hebben dezelfde set)."""
    return frozenset(xgb.XGBRegressor().get_params()) | frozenset(xgb.XGBRanker().get_params())


def _check_params(family: str, params: Mapping[str, Any]) -> None:
    """Weiger onbekende parameters (bv. tikfouten), zodat ze niet stil genegeerd worden."""
    for name, value in params.items():
        if family == "ridge":
            if name != "alpha":
                raise ValueError(f"Onbekende parameter voor ridge: {name}; alleen alpha kan")
        elif name in _FORBIDDEN_XGB_PARAMS:
            raise ValueError(f"Parameter {name} is niet toegestaan: {_FORBIDDEN_XGB_PARAMS[name]}")
        elif name not in _xgb_parameter_names():
            raise ValueError(f"Onbekende XGBoost-parameter voor {family}: {name}")
        if not isinstance(value, bool | int | float | str):
            raise ValueError(f"Parameter {name} moet een enkel getal of woord zijn")


@dataclass(frozen=True)
class ModelSpec:
    """Een modelfamilie met parameters en featureset; levert verse voorspellers per fold."""

    family: str
    feature_set: str = "all"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise ValueError(
                f"Onbekende modelfamilie: {self.family}; kies uit {list(MODEL_FAMILIES)}"
            )
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Onbekende featureset: {self.feature_set}; kies uit {list(FEATURE_SETS)}"
            )
        object.__setattr__(self, "params", dict(self.params))
        _check_params(self.family, self.params)

    @property
    def name(self) -> str:
        """Leesbare, unieke naam, bv. `xgb_ranker_d3_n300_all`."""
        settings = [
            f"{_NAME_ABBREVIATIONS.get(key, key)}{value:g}"
            if isinstance(value, int | float)
            else f"{key}{value}"
            for key, value in self.params.items()
        ]
        return "_".join([self.family, *settings, self.feature_set])

    def feature_columns(self, config: FeatureConfig) -> list[str]:
        return config_feature_columns(replace(config, groups=FEATURE_SETS[self.feature_set]))

    def predictor(self, config: FeatureConfig | None = None) -> RacePredictor:
        columns = self.feature_columns(config or FeatureConfig())
        return MODEL_FAMILIES[self.family](columns, **self.params)

    def factory(self, config: FeatureConfig | None = None) -> Callable[[], RacePredictor]:
        return partial(self.predictor, config or FeatureConfig())


def effective_params(spec: ModelSpec, predictor: RacePredictor) -> dict[str, Any]:
    """De parameters waarmee het model werkelijk traint, inclusief de XGBoost-defaults."""
    return dict(getattr(predictor, "params", spec.params))


def training_range(train: pd.DataFrame) -> dict[str, Any]:
    """Eerste en laatste trainingsrace, aantal races en aantal rijen (chronologische data)."""
    races = train[["season", "round"]].drop_duplicates()
    first, last = races.iloc[0], races.iloc[-1]
    return {
        "first_race": {"season": int(first["season"]), "round": int(first["round"])},
        "last_race": {"season": int(last["season"]), "round": int(last["round"])},
        "races": len(races),
        "rows": len(train),
    }


def _model_metadata(
    spec: ModelSpec, predictor: RacePredictor, train: pd.DataFrame, config: FeatureConfig
) -> dict[str, Any]:
    return {
        "model": spec.name,
        "family": spec.family,
        "class": type(predictor).__name__,
        "params": effective_params(spec, predictor),
        "feature_set": spec.feature_set,
        "feature_columns": spec.feature_columns(config),
        "feature_config": asdict(config),
        "training": training_range(train),
        "versions": library_versions(),
        "git": git_revision(),
        "created": utc_now(),
    }


def train_final_model(
    data_dir: Path = RAW_DIR,
    model_dir: Path = OUTPUT_DIR / "models",
    importance_path: Path = OUTPUT_DIR / "model_feature_importance.csv",
    *,
    spec: ModelSpec,
    config: FeatureConfig | None = None,
) -> RacePredictor:
    """Train het eindmodel `spec` op alle races met een uitslag; sla model en interpretatie op.

    Schrijft `<naam>.joblib` (het getrainde voorspellerobject) en `<naam>.json` (metadata met
    git-commit) in `model_dir`, plus het featurebelang op de trainingsdata naar
    `importance_path`. De CLI (`f1 train`) haalt `spec` uit `config/config.yaml`.
    """
    config = config or FeatureConfig()
    table = build_features(load_all_results(data_dir), config)
    train = table.dropna(subset=["finish_position"]).sort_values(["season", "round"], kind="stable")
    predictor = spec.predictor(config).fit(train)

    model_path = model_dir / f"{spec.name}.joblib"
    write_atomic(model_path, partial(joblib.dump, predictor))
    write_json(model_path.with_suffix(".json"), _model_metadata(spec, predictor, train, config))

    importance = predictor.feature_importance(train)
    importance.insert(0, "model", spec.name)
    write_atomic(importance_path, partial(importance.to_csv, index=False))
    log.info("Eindmodel %s getraind op %d rijen -> %s", spec.name, len(train), model_path)
    return predictor

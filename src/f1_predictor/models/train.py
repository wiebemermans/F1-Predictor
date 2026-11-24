"""Modelspecificaties, de gekozen modellen uit fase 7 en het trainen van het eindmodel."""

from __future__ import annotations

import json
import logging
import platform
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from f1_predictor.evaluation.explore import load_all_results
from f1_predictor.features.build import FEATURE_GROUPS, FeatureConfig, build_features
from f1_predictor.features.build import feature_columns as config_feature_columns
from f1_predictor.paths import OUTPUT_DIR, RAW_DIR

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


# Gekozen per familie met de selectieregel uit docs/model.md: laagste gemiddelde MAE op de
# validatieraces (2015-2021), zie output/model_tuning.csv. In fase 8 verhuist dit naar
# config/config.yaml.
TUNED_MODELS: dict[str, ModelSpec] = {
    "ridge": ModelSpec("ridge", "upto_championship", {"alpha": 10.0}),
    "xgb_regressor": ModelSpec("xgb_regressor", "all", {"max_depth": 2, "n_estimators": 100}),
    "xgb_ranker": ModelSpec("xgb_ranker", "all", {"max_depth": 2, "n_estimators": 50}),
    "xgb_ranker_ndcg": ModelSpec("xgb_ranker_ndcg", "all", {"max_depth": 1, "n_estimators": 300}),
}
# Eindmodel volgens `select_final_model`: hoogste podiumoverlap op de validatieraces (2.086,
# gelijk met Ridge), bij gelijkstand het hoogste aandeel juiste winnaars (53.6% tegen 49.3%).
# Dit model wordt opgeslagen en geïnterpreteerd.
FINAL_MODEL = "xgb_ranker"


def _write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    write(temporary_path)
    temporary_path.replace(path)


def _model_metadata(
    spec: ModelSpec, predictor: RacePredictor, train: pd.DataFrame, config: FeatureConfig
) -> dict[str, Any]:
    races = train[["season", "round"]].drop_duplicates()
    first, last = races.iloc[0], races.iloc[-1]
    return {
        "model": spec.name,
        "family": spec.family,
        "class": type(predictor).__name__,
        "params": getattr(predictor, "params", dict(spec.params)),
        "feature_set": spec.feature_set,
        "feature_columns": spec.feature_columns(config),
        "feature_config": asdict(config),
        "training": {
            "first_race": {"season": int(first["season"]), "round": int(first["round"])},
            "last_race": {"season": int(last["season"]), "round": int(last["round"])},
            "races": len(races),
            "rows": len(train),
        },
        "versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("xgboost", "scikit-learn", "pandas")},
        },
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def train_final_model(
    data_dir: Path = RAW_DIR,
    model_dir: Path = OUTPUT_DIR / "models",
    importance_path: Path = OUTPUT_DIR / "model_feature_importance.csv",
    spec: ModelSpec | None = None,
    config: FeatureConfig | None = None,
) -> RacePredictor:
    """Train het eindmodel op alle races met een uitslag en sla model en interpretatie op.

    Schrijft `<naam>.joblib` (het getrainde voorspellerobject) en `<naam>.json` (metadata) in
    `model_dir`, plus het featurebelang op de trainingsdata naar `importance_path`.
    """
    config = config or FeatureConfig()
    spec = spec or TUNED_MODELS[FINAL_MODEL]
    table = build_features(load_all_results(data_dir), config)
    train = table.dropna(subset=["finish_position"]).sort_values(["season", "round"], kind="stable")
    predictor = spec.predictor(config).fit(train)

    model_path = model_dir / f"{spec.name}.joblib"
    _write_atomic(model_path, partial(joblib.dump, predictor))
    metadata = json.dumps(_model_metadata(spec, predictor, train, config), indent=2)
    _write_atomic(
        model_path.with_suffix(".json"),
        lambda path: path.write_text(metadata + "\n", encoding="utf-8"),
    )

    importance = predictor.feature_importance(train)
    importance.insert(0, "model", spec.name)
    _write_atomic(importance_path, partial(importance.to_csv, index=False))
    log.info("Eindmodel %s getraind op %d rijen -> %s", spec.name, len(train), model_path)
    return predictor

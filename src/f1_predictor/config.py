"""Projectconfiguratie uit `config/config.yaml`: data, features, modellen en backtest.

De CLI leest dit bestand en geeft de instellingen als expliciete argumenten door aan de
bibliotheekfuncties. Velden, defaults en toegestane waarden staan in `docs/usage.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from f1_predictor.features.build import FeatureConfig
from f1_predictor.models.train import FEATURE_SETS, MODEL_FAMILIES, ModelSpec
from f1_predictor.paths import CONFIG_PATH

FIRST_F1_SEASON = 1950


class ConfigError(ValueError):
    """Het configuratiebestand ontbreekt of bevat een ongeldige waarde."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class DataConfig:
    """Welke seizoenen `f1 fetch` standaard ophaalt; `end_season=None` is het huidige jaar."""

    start_season: int = 2014
    end_season: int | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.start_season) or self.start_season < FIRST_F1_SEASON:
            raise ValueError(f"start_season moet een geheel getal vanaf {FIRST_F1_SEASON} zijn")
        if self.end_season is not None and (
            not _is_int(self.end_season) or self.end_season < self.start_season
        ):
            raise ValueError("end_season moet leeg zijn of een geheel getal ≥ start_season")

    def resolved_end_season(self, today: date | None = None) -> int:
        """Het laatste seizoen; zonder `end_season` het kalenderjaar van vandaag."""
        if self.end_season is not None:
            return self.end_season
        return (today or date.today()).year


@dataclass(frozen=True)
class BacktestConfig:
    """Instellingen van de walk-forward-backtest en het minimum aan historie bij `predict`."""

    min_train_races: int = 20

    def __post_init__(self) -> None:
        if not _is_int(self.min_train_races) or self.min_train_races < 0:
            raise ValueError("min_train_races moet een niet-negatief geheel getal zijn")


@dataclass(frozen=True)
class ModelConfig:
    """De gekozen configuratie per modelfamilie (`tuned`) en welke daarvan het eindmodel is."""

    final: str
    tuned: Mapping[str, ModelSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tuned", dict(self.tuned))
        if not self.tuned:
            raise ValueError("tuned moet minstens één model bevatten")
        for family, spec in self.tuned.items():
            if spec.family != family:
                raise ValueError(f"tuned.{family} bevat een model van familie {spec.family}")
        if self.final not in self.tuned:
            raise ValueError(f"final ({self.final}) moet een van de modellen in tuned zijn")

    @property
    def final_spec(self) -> ModelSpec:
        return self.tuned[self.final]


@dataclass(frozen=True)
class ProjectConfig:
    """De volledige configuratie; elke sectie behalve `model` heeft defaults."""

    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        # Een model kan alleen trainen op featuregroepen die ook gebouwd worden.
        enabled = set(self.features.groups)
        for family, spec in self.model.tuned.items():
            missing = [group for group in FEATURE_SETS[spec.feature_set] if group not in enabled]
            if missing:
                raise ValueError(
                    f"model.tuned.{family} gebruikt featureset {spec.feature_set}, maar "
                    f"features.groups mist {missing}"
                )

    @property
    def final_spec(self) -> ModelSpec:
        """Het eindmodel: wordt getraind door `f1 train` en gebruikt door `f1 predict`."""
        return self.model.final_spec

    def to_dict(self) -> dict[str, Any]:
        """Dezelfde structuur als `config.yaml`, zodat `parse_config` het terug kan inlezen."""
        return {
            "data": {
                "start_season": self.data.start_season,
                "end_season": self.data.end_season,
            },
            "features": {
                "groups": list(self.features.groups),
                "form_windows": list(self.features.form_windows),
                "rate_window": self.features.rate_window,
                "circuit_shrinkage": self.features.circuit_shrinkage,
            },
            "model": {
                "final": self.model.final,
                "tuned": {
                    family: {"feature_set": spec.feature_set, "params": dict(spec.params)}
                    for family, spec in self.model.tuned.items()
                },
            },
            "backtest": {"min_train_races": self.backtest.min_train_races},
        }


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} moet een sectie met sleutels zijn")
    return value


def _check_keys(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(str(key) for key in section if key not in allowed)
    if unknown:
        raise ConfigError(f"Onbekende sleutel(s) in {where}: {unknown}; kies uit {sorted(allowed)}")


def _section(raw: object, cls: type, where: str) -> Any:
    """Bouw een dataclass uit een sectie; een ontbrekende sectie geeft de defaults."""
    if raw is None:
        return cls()
    section = _mapping(raw, where)
    _check_keys(section, {item.name for item in fields(cls)}, where)
    try:
        return cls(**section)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where}: {error}") from error


def _model_section(raw: object) -> ModelConfig:
    section = _mapping(raw, "model")
    _check_keys(section, {"final", "tuned"}, "model")
    for key in ("final", "tuned"):
        if key not in section:
            raise ConfigError(f"model mist verplichte sleutel: {key}")
    tuned: dict[str, ModelSpec] = {}
    for family, raw_spec in _mapping(section["tuned"], "model.tuned").items():
        where = f"model.tuned.{family}"
        if family not in MODEL_FAMILIES:
            raise ConfigError(
                f"Onbekende modelfamilie in model.tuned: {family}; kies uit {list(MODEL_FAMILIES)}"
            )
        spec_section = _mapping(raw_spec, where)
        _check_keys(spec_section, {"feature_set", "params"}, where)
        params = spec_section.get("params") or {}
        try:
            tuned[family] = ModelSpec(
                family,
                spec_section.get("feature_set", "all"),
                _mapping(params, f"{where}.params"),
            )
        except ValueError as error:
            raise ConfigError(f"{where}: {error}") from error
    try:
        return ModelConfig(final=section["final"], tuned=tuned)
    except ValueError as error:
        raise ConfigError(f"model: {error}") from error


def parse_config(raw: object) -> ProjectConfig:
    """Valideer de ingelezen YAML-structuur en bouw de configuratie."""
    if raw is None:
        raise ConfigError("De configuratie is leeg; de sectie model is verplicht")
    root = _mapping(raw, "De configuratie")
    _check_keys(root, {"data", "features", "model", "backtest"}, "de configuratie")
    if "model" not in root:
        raise ConfigError("De configuratie mist de verplichte sectie model")
    model = _model_section(root["model"])
    data = _section(root.get("data"), DataConfig, "data")
    features = _section(root.get("features"), FeatureConfig, "features")
    backtest = _section(root.get("backtest"), BacktestConfig, "backtest")
    try:
        return ProjectConfig(model=model, data=data, features=features, backtest=backtest)
    except ValueError as error:
        raise ConfigError(str(error)) from error


def load_config(path: Path = CONFIG_PATH) -> ProjectConfig:
    """Lees en valideer een configuratiebestand; fouten worden een `ConfigError` met het pad."""
    if not path.is_file():
        raise ConfigError(f"Configuratiebestand niet gevonden: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is geen geldige YAML: {error}") from error
    try:
        return parse_config(raw)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error

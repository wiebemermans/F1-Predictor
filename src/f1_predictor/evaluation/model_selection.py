"""Fase 7: hyperparameters kiezen op validatieraces en modellen vergelijken op de holdout.

De validatieperiode zijn de testraces tot en met `VALIDATION_LAST_SEASON`; daarop worden
hyperparameters, featureset en het eindmodel gekozen. De holdout (vanaf `HOLDOUT_FIRST_SEASON`)
wordt pas daarna gescoord en levert de eerlijke vergelijking met de baselines.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from f1_predictor.features.build import FeatureConfig, build_features
from f1_predictor.models.baselines import RacePredictor
from f1_predictor.models.train import FEATURE_SETS, ModelSpec
from f1_predictor.output import write_json_if_changed
from f1_predictor.paths import OUTPUT_DIR, RAW_DIR
from f1_predictor.runinfo import utc_now

from .backtest import BASELINE_FACTORIES, _load_raw_results, _write_csv, run_backtest

log = logging.getLogger(__name__)

VALIDATION_LAST_SEASON = 2021
HOLDOUT_FIRST_SEASON = 2022

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
# Diepte 1 en 50 bomen zijn toegevoegd nadat de eerste tuningrun de kleinste instellingen
# (diepte 2, 100 bomen) op de rand van het rooster als beste gaf; zie docs/model.md.
XGB_MAX_DEPTHS = (1, 2, 3, 4)
XGB_N_ESTIMATORS = (50, 100, 300)

METRIC_COLUMNS = ["mae_positions", "spearman", "winner_hit", "podium_overlap", "top10_overlap"]
SUMMARY_COLUMNS = [
    "model",
    "races",
    *METRIC_COLUMNS,
    "delta_mae",
    "delta_mae_se",
    "better_share",
    "worse_share",
]


def summarize_backtest(
    results: pd.DataFrame, reference: str = "qualifying_position"
) -> pd.DataFrame:
    """Gemiddelde scores per model, plus het gepaarde MAE-verschil met de referentie.

    `delta_mae` is het gemiddelde per race van (MAE model − MAE referentie); negatief is beter.
    `delta_mae_se` is de standaardfout van dat gemiddelde. `better_share` en `worse_share` zijn
    de aandelen races waarin het model een lagere of hogere MAE heeft dan de referentie.
    Elk model moet op exact dezelfde races gescoord zijn als de referentie.
    """
    missing = {"model", "season", "round", *METRIC_COLUMNS}.difference(results.columns)
    if missing:
        raise ValueError(f"Backtestresultaten missen kolommen: {sorted(missing)}")
    if reference not in set(results["model"]):
        raise ValueError(f"Referentiemodel ontbreekt in de resultaten: {reference}")

    keys = ["season", "round"]
    reference_mae = results.loc[results["model"] == reference].set_index(keys)["mae_positions"]
    rows = []
    for model, scores in results.groupby("model", sort=False):
        mae = scores.set_index(keys)["mae_positions"]
        if not mae.index.is_unique or set(mae.index) != set(reference_mae.index):
            raise ValueError(f"{model} is niet op exact dezelfde races gescoord als {reference}")
        difference = mae - reference_mae.reindex(mae.index)
        count = len(difference)
        rows.append(
            {
                "model": model,
                "races": count,
                **scores[METRIC_COLUMNS].astype("float64").mean().to_dict(),
                "delta_mae": difference.mean(),
                "delta_mae_se": difference.std(ddof=1) / math.sqrt(count) if count > 1 else None,
                "better_share": (difference < 0).mean(),
                "worse_share": (difference > 0).mean(),
            }
        )
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def tuning_candidates() -> list[ModelSpec]:
    """Het tuningrooster: elke familie met elke featureset en conservatieve parameters."""
    candidates: list[ModelSpec] = []
    for family in ("ridge", "xgb_regressor", "xgb_ranker", "xgb_ranker_ndcg"):
        for feature_set in FEATURE_SETS:
            if family == "ridge":
                grid = [{"alpha": alpha} for alpha in RIDGE_ALPHAS]
            else:
                grid = [
                    {"max_depth": depth, "n_estimators": trees}
                    for depth in XGB_MAX_DEPTHS
                    for trees in XGB_N_ESTIMATORS
                ]
            candidates.extend(ModelSpec(family, feature_set, params) for params in grid)
    return candidates


def _predictors(
    specs: Iterable[ModelSpec], config: FeatureConfig
) -> dict[str, Callable[[], RacePredictor]]:
    specs = list(specs)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names) or set(names) & set(BASELINE_FACTORIES):
        raise ValueError("Modelnamen moeten uniek zijn en mogen niet met een baseline botsen")
    return {**BASELINE_FACTORIES, **{spec.name: spec.factory(config) for spec in specs}}


def select_tuned_models(summary: pd.DataFrame, specs: Iterable[ModelSpec]) -> dict[str, str]:
    """Selectieregel: per familie de laagste gemiddelde MAE; bij gelijkspel het eenvoudigste.

    "Eenvoudiger" is: minder features, dan ondiepere bomen, minder bomen en sterkere
    Ridge-regularisatie. Geeft per familie de naam van de gekozen configuratie terug.
    """
    mae = summary.set_index("model")["mae_positions"]
    chosen: dict[str, tuple[tuple[float, ...], str]] = {}
    for spec in specs:
        params = spec.params
        key = (
            float(mae[spec.name]),
            float(len(FEATURE_SETS[spec.feature_set])),
            float(params.get("max_depth", 0)),
            float(params.get("n_estimators", 0)),
            -float(params.get("alpha", 0)),
        )
        if spec.family not in chosen or key < chosen[spec.family][0]:
            chosen[spec.family] = (key, spec.name)
    return {family: name for family, (_, name) in chosen.items()}


def select_final_model(summary: pd.DataFrame, tuned: Mapping[str, str]) -> str:
    """Kies het eindmodel uit de gekozen configuratie per familie (`tuned`: familie -> naam).

    Regel: de hoogste gemiddelde podiumoverlap in `summary` (de validatieraces). Bij gelijkstand
    beslist het hoogste aandeel juiste winnaars, daarna de laagste MAE. Geeft de familie terug.
    """
    if not tuned:
        raise ValueError("Geef minstens één gekozen model op")
    scores = summary.set_index("model")
    missing = set(tuned.values()).difference(scores.index)
    if missing:
        raise ValueError(f"Samenvatting mist gekozen modellen: {sorted(missing)}")

    def key(family: str) -> tuple[float, float, float]:
        row = scores.loc[tuned[family]]
        # Afronden, zodat gelijke gemiddelden met andere afrondingsfouten ook gelijk tellen.
        return (
            -round(float(row["podium_overlap"]), 9),
            -round(float(row["winner_hit"]), 9),
            round(float(row["mae_positions"]), 9),
        )

    return min(tuned, key=key)


def run_model_tuning(
    data_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_DIR / "model_tuning.csv",
    min_train_races: int = 20,
    candidates: Iterable[ModelSpec] | None = None,
    last_season: int = VALIDATION_LAST_SEASON,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Score het tuningrooster en de baselines op de validatieraces en schrijf de samenvatting.

    De backtest krijgt alleen seizoenen t/m `last_season`. Dat geeft dezelfde folds als op alle
    data, omdat features alleen eerdere races gebruiken en elke fold alleen op eerdere races
    traint; de holdout blijft zo buiten de keuze.
    """
    config = config or FeatureConfig()
    specs = list(candidates or tuning_candidates())
    data = build_features(_load_raw_results(data_dir), config)
    validation = data.loc[data["season"] <= last_season]
    results = run_backtest(validation, _predictors(specs, config), min_train_races)

    summary = summarize_backtest(results)
    details = {
        spec.name: (spec.family, spec.feature_set, json.dumps(spec.params)) for spec in specs
    }
    summary[["family", "feature_set", "params"]] = [
        details.get(model, ("baseline", "", "")) for model in summary["model"]
    ]
    _write_csv(summary, output_path)
    tuned = select_tuned_models(summary, specs)
    for family, name in tuned.items():
        log.info("Beste %s op de validatieraces: %s", family, name)
    log.info("Eindmodel volgens de podiumregel: %s", tuned[select_final_model(summary, tuned)])
    return summary


def summarize_periods(
    results: pd.DataFrame,
    validation_last_season: int = VALIDATION_LAST_SEASON,
    holdout_first_season: int = HOLDOUT_FIRST_SEASON,
) -> pd.DataFrame:
    """Samenvatting per periode: validatie, holdout en alle testraces."""
    periods = {
        "validation": results["season"] <= validation_last_season,
        "holdout": results["season"] >= holdout_first_season,
        "all": pd.Series(True, index=results.index),
    }
    frames = [
        summarize_backtest(results.loc[mask]).assign(period=period)
        for period, mask in periods.items()
        if mask.any()
    ]
    summary = pd.concat(frames, ignore_index=True)
    return summary[["period", *SUMMARY_COLUMNS]]


def run_model_backtest(
    data_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_DIR / "backtest_models.csv",
    summary_path: Path = OUTPUT_DIR / "model_comparison.csv",
    min_train_races: int = 20,
    *,
    models: Mapping[str, ModelSpec],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Score de baselines en de gekozen modellen op exact dezelfde races (alle testraces).

    Schrijft de scores per race naar `output_path` en de samenvatting per periode
    (validatie, holdout, alles) naar `summary_path`. De CLI (`f1 backtest`) haalt `models`
    uit `config/config.yaml`.
    """
    config = config or FeatureConfig()
    data = build_features(_load_raw_results(data_dir), config)
    results = run_backtest(data, _predictors(models.values(), config), min_train_races)
    _write_csv(results, output_path)
    _write_csv(summarize_periods(results), summary_path)
    return results


COMPARISON_SCHEMA_VERSION = 1


def comparison_payload(
    summary: pd.DataFrame,
    models: Mapping[str, ModelSpec],
    final_model: str,
    run: Mapping[str, Any],
    reference: str = "qualifying_position",
) -> dict[str, Any]:
    """De samenvatting per periode (`summarize_periods`) als JSON-structuur voor een website.

    Elke modelrij krijgt een `family` (`baseline` voor de baselines); `final_model` is de
    naam van het eindmodel, zodat een website dat kan uitlichten.
    """
    families = {spec.name: spec.family for spec in models.values()}
    periods: dict[str, list[dict[str, Any]]] = {}
    for period, rows in summary.groupby("period", sort=False):
        records = rows.drop(columns="period").to_dict(orient="records")
        periods[str(period)] = [
            {"model": row["model"], "family": families.get(row["model"], "baseline"), **row}
            for row in records
        ]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "final_model": final_model,
        "reference": reference,
        "periods": periods,
        "run": dict(run),
    }


def write_comparison_json(
    summary: pd.DataFrame,
    models: Mapping[str, ModelSpec],
    final_model: str,
    run: Mapping[str, Any],
    output_path: Path = OUTPUT_DIR / "model_comparison.json",
) -> Path:
    """Schrijf `comparison_payload` naar `output_path` (standaard naast de CSV-samenvatting).

    Alleen bij een inhoudelijke wijziging: zonder nieuwe races blijft het bestand ongemoeid.
    """
    if not write_json_if_changed(
        output_path, comparison_payload(summary, models, final_model, run)
    ):
        log.info("%s ongewijzigd, niet herschreven", output_path)
    return output_path

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from f1_predictor.evaluation.model_selection import (
    SUMMARY_COLUMNS,
    run_model_backtest,
    run_model_tuning,
    select_final_model,
    select_tuned_models,
    summarize_backtest,
    tuning_candidates,
)
from f1_predictor.features.build import FeatureConfig, build_features, feature_columns
from f1_predictor.models.train import (
    FEATURE_SETS,
    FINAL_MODEL,
    TUNED_MODELS,
    ModelSpec,
    train_final_model,
)

POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0}
SMALL_XGB = {"max_depth": 2, "n_estimators": 5}
SMALL_SPECS = [
    ModelSpec("ridge", "upto_championship", {"alpha": 1.0}),
    ModelSpec("xgb_regressor", "all", SMALL_XGB),
    ModelSpec("xgb_ranker", "all", SMALL_XGB),
]


def _raw_dir(tmp_path, seasons=range(2020, 2024), rounds: int = 3, n_drivers: int = 6):
    """Ruwe seizoensbestanden: finish = kwalificatie met telkens twee verwisselde coureurs."""
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for season in seasons:
        frames = []
        for round_ in range(1, rounds + 1):
            quali = rng.permutation(n_drivers) + 1
            finish = quali.copy()
            first, second = rng.choice(n_drivers, 2, replace=False)
            finish[[first, second]] = finish[[second, first]]
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "round": round_,
                        "race_name": f"Race {season}-{round_}",
                        "circuit_id": f"circuit_{round_}",
                        "driver_id": [f"d{index}" for index in range(n_drivers)],
                        "constructor_id": [f"team_{index // 2}" for index in range(n_drivers)],
                        "grid": quali,
                        "quali_position": pd.array(quali, dtype="Int64"),
                        "finish_position": finish,
                        "position_text": [str(position) for position in finish],
                        "status": "Finished",
                        "points": [POINTS.get(position, 0.0) for position in finish],
                        "laps": 50,
                        "sprint_points": 0.0,
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_parquet(
            raw_dir / f"results_{season}.parquet", index=False
        )
    return raw_dir


def _results(model: str, maes: list[float], winners: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model,
            "season": 2022,
            "round": range(1, len(maes) + 1),
            "mae_positions": maes,
            "spearman": [0.5] * len(maes),
            "winner_hit": winners,
            "podium_overlap": 2,
            "top10_overlap": 8,
        }
    )


# --- Samenvatting ---


def test_summarize_backtest_reports_paired_difference_with_reference():
    results = pd.concat(
        [
            _results("qualifying_position", [2.0, 3.0], [1, 0]),
            _results("model", [1.0, 3.5], [1, 1]),
        ],
        ignore_index=True,
    )

    summary = summarize_backtest(results).set_index("model")

    assert summary.columns.tolist() == SUMMARY_COLUMNS[1:]
    model = summary.loc["model"]
    assert model["races"] == 2
    assert model["mae_positions"] == pytest.approx(2.25)
    assert model["winner_hit"] == pytest.approx(1.0)
    # Verschillen per race: -1.0 en +0.5.
    assert model["delta_mae"] == pytest.approx(-0.25)
    assert model["delta_mae_se"] == pytest.approx(np.std([-1.0, 0.5], ddof=1) / np.sqrt(2))
    assert model["better_share"] == pytest.approx(0.5)
    assert model["worse_share"] == pytest.approx(0.5)
    reference = summary.loc["qualifying_position"]
    assert reference["delta_mae"] == 0
    assert reference["better_share"] == reference["worse_share"] == 0


def test_summarize_backtest_requires_identical_races():
    results = pd.concat(
        [
            _results("qualifying_position", [2.0, 3.0], [1, 0]),
            _results("model", [1.0], [1]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="dezelfde races"):
        summarize_backtest(results)
    with pytest.raises(ValueError, match="Referentiemodel"):
        summarize_backtest(results, reference="championship_points")


# --- Modelspecificaties en tuningrooster ---


def test_model_spec_builds_named_predictor_with_feature_set():
    spec = ModelSpec("xgb_ranker", "upto_championship", {"max_depth": 3, "n_estimators": 300})

    predictor = spec.predictor()

    assert spec.name == "xgb_ranker_d3_n300_upto_championship"
    assert ModelSpec("ridge", "all", {"alpha": 0.1}).name == "ridge_a0.1_all"
    expected = feature_columns(FeatureConfig(groups=FEATURE_SETS["upto_championship"]))
    assert predictor.feature_columns == expected
    assert predictor.params["max_depth"] == 3


def test_model_spec_rejects_unknown_family_and_feature_set():
    with pytest.raises(ValueError, match="modelfamilie"):
        ModelSpec("random_forest")
    with pytest.raises(ValueError, match="featureset"):
        ModelSpec("ridge", "everything")


def test_tuning_candidates_cover_the_grid_once():
    candidates = tuning_candidates()
    names = [spec.name for spec in candidates]

    assert len(candidates) == 8 + 3 * 2 * 4 * 3
    assert len(set(names)) == len(names)
    assert sum(spec.family == "ridge" for spec in candidates) == 8
    assert {
        "ridge_a100_upto_championship",
        "xgb_ranker_d4_n300_all",
        "xgb_ranker_ndcg_d1_n50_upto_championship",
    }.issubset(names)


def test_select_tuned_models_prefers_lowest_mae_then_simplest():
    specs = [
        ModelSpec("xgb_ranker", "all", {"max_depth": 4, "n_estimators": 300}),
        ModelSpec("xgb_ranker", "all", {"max_depth": 2, "n_estimators": 300}),
        ModelSpec("xgb_ranker", "upto_championship", {"max_depth": 3, "n_estimators": 100}),
        ModelSpec("ridge", "all", {"alpha": 1.0}),
        ModelSpec("ridge", "all", {"alpha": 10.0}),
    ]
    summary = pd.DataFrame(
        {"model": [spec.name for spec in specs], "mae_positions": [3.4, 3.4, 3.5, 3.3, 3.3]}
    )

    chosen = select_tuned_models(summary, specs)

    assert chosen == {"xgb_ranker": "xgb_ranker_d2_n300_all", "ridge": "ridge_a10_all"}


def test_select_final_model_uses_podium_then_winner_then_mae():
    summary = pd.DataFrame(
        {
            "model": ["ridge_x", "ranker_x", "regressor_x", "ndcg_x"],
            "podium_overlap": [2.1, 2.1, 2.0, 2.1],
            "winner_hit": [0.5, 0.6, 0.9, 0.6],
            "mae_positions": [3.3, 3.5, 3.0, 3.4],
        }
    )
    tuned = {"ridge": "ridge_x", "xgb_ranker": "ranker_x", "xgb_regressor": "regressor_x"}

    # Podium gelijk tussen ridge en ranker; de ranker heeft meer juiste winnaars.
    assert select_final_model(summary, tuned) == "xgb_ranker"
    # Podium en winnaar gelijk tussen ranker en ndcg; de lagere MAE beslist.
    assert select_final_model(summary, {**tuned, "xgb_ranker_ndcg": "ndcg_x"}) == "xgb_ranker_ndcg"
    with pytest.raises(ValueError, match="mist gekozen modellen"):
        select_final_model(summary, {"ridge": "unknown"})


def test_current_final_model_follows_from_the_selection_rules():
    # De vastgelegde keuze in train.py moet volgen uit de regels; scores uit de tuningrun van
    # 2026-11-24 (validatieraces 2015-2021) voor de vier gekozen configuraties.
    summary = pd.DataFrame(
        {
            "model": [spec.name for spec in TUNED_MODELS.values()],
            "podium_overlap": [292 / 140, 284 / 140, 292 / 140, 290 / 140],
            "winner_hit": [69 / 140, 71 / 140, 75 / 140, 74 / 140],
            "mae_positions": [3.573506, 3.562403, 3.533377, 3.576818],
        }
    )
    tuned = {family: spec.name for family, spec in TUNED_MODELS.items()}

    assert list(TUNED_MODELS) == ["ridge", "xgb_regressor", "xgb_ranker", "xgb_ranker_ndcg"]
    assert select_final_model(summary, tuned) == FINAL_MODEL


# --- Tuning, modelbacktest en eindmodel ---


def test_run_model_tuning_scores_only_validation_seasons(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    output_path = tmp_path / "output" / "model_tuning.csv"

    summary = run_model_tuning(
        raw_dir, output_path, min_train_races=2, candidates=SMALL_SPECS, last_season=2021
    )

    saved = pd.read_csv(output_path)
    assert len(saved) == len(summary) == 2 + len(SMALL_SPECS)
    # Seizoenen 2020-2021 hebben 6 races; na 2 trainingsraces blijven er 4 testraces over.
    assert (saved["races"] == 4).all()
    families = saved.set_index("model")["family"]
    assert families["qualifying_position"] == "baseline"
    assert families["xgb_ranker_d2_n5_all"] == "xgb_ranker"


def test_run_model_backtest_matches_tuning_folds_and_reports_periods(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    models = {spec.family: spec for spec in SMALL_SPECS}
    tuning = run_model_tuning(
        raw_dir, tmp_path / "tuning.csv", 2, candidates=SMALL_SPECS, last_season=2021
    )

    results = run_model_backtest(
        raw_dir, tmp_path / "models.csv", tmp_path / "comparison.csv", 2, models=models
    )

    races_per_model = results.groupby("model")[["season", "round"]].apply(
        lambda frame: tuple(map(tuple, frame.to_numpy()))
    )
    assert races_per_model.nunique() == 1
    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert comparison.groupby("period")["races"].first().to_dict() == {
        "validation": 4,
        "holdout": 6,
        "all": 10,
    }
    # Validatiescores uit de volledige backtest zijn gelijk aan die uit de tuning.
    validation = comparison[comparison["period"] == "validation"].set_index("model")
    pd.testing.assert_series_equal(
        validation["mae_positions"],
        tuning.set_index("model")["mae_positions"].loc[validation.index],
    )


def test_train_final_model_saves_model_metadata_and_importance(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    model_dir = tmp_path / "models"
    importance_path = tmp_path / "importance.csv"
    spec = ModelSpec("xgb_ranker", "all", SMALL_XGB)

    predictor = train_final_model(raw_dir, model_dir, importance_path, spec=spec)

    loaded = joblib.load(model_dir / f"{spec.name}.joblib")
    race = pd.read_parquet(raw_dir / "results_2023.parquet")
    race_features = build_features(race).query("round == 3")
    pd.testing.assert_series_equal(loaded.predict(race_features), predictor.predict(race_features))

    metadata = json.loads((model_dir / f"{spec.name}.json").read_text(encoding="utf-8"))
    assert metadata["model"] == spec.name
    assert metadata["feature_columns"] == feature_columns()
    assert metadata["params"]["n_estimators"] == 5
    assert metadata["training"]["races"] == 12
    assert metadata["training"]["last_race"] == {"season": 2023, "round": 3}

    importance = pd.read_csv(importance_path)
    assert importance["feature"].tolist() == feature_columns()
    assert set(importance["model"]) == {spec.name}

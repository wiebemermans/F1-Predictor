import pandas as pd
import pytest

from f1_predictor.evaluation.backtest import run_backtest, run_baseline_backtest
from f1_predictor.evaluation.metrics import score_race
from f1_predictor.models.baselines import (
    ChampionshipPointsBaseline,
    QualifyingPositionBaseline,
)
from f1_predictor.models.linear import RidgeFeaturePredictor


def _race(season: int, round_: int, positions: list[int]) -> pd.DataFrame:
    drivers = [f"driver_{index}" for index in range(len(positions))]
    return pd.DataFrame(
        {
            "season": season,
            "round": round_,
            "race_name": f"Race {round_}",
            "driver_id": drivers,
            "quali_position": positions,
            "finish_position": positions,
            "points": [25.0 if position == 1 else 0.0 for position in positions],
            "sprint_points": 0.0,
        }
    )


def test_score_race_returns_expected_metrics():
    race = pd.DataFrame({"driver_id": ["a", "b", "c", "d"], "finish_position": [1, 2, 3, 4]})
    predicted = pd.Series([1, 3, 2, 4], index=race.index)

    scores = score_race(race, predicted)

    assert scores["mae_positions"] == pytest.approx(0.5)
    assert scores["spearman"] == pytest.approx(0.8)
    assert scores["winner_hit"] == 1
    assert scores["podium_overlap"] == 3
    assert scores["top10_overlap"] == 4


def test_score_race_handles_ties_in_actual_positions_deterministically():
    race = pd.DataFrame({"driver_id": ["b", "a", "c"], "finish_position": [1, 1, 3]})
    predicted = pd.Series([2, 1, 3], index=race.index)

    scores = score_race(race, predicted)

    assert scores["winner_hit"] == 1
    assert scores["mae_positions"] == pytest.approx(1 / 3)


def test_score_race_rejects_non_permutation_predictions():
    race = pd.DataFrame({"driver_id": ["a", "b"], "finish_position": [1, 2]})

    with pytest.raises(ValueError, match="unieke plaatsen"):
        score_race(race, pd.Series([1, 1], index=race.index))


def test_score_race_returns_no_spearman_for_single_driver():
    race = pd.DataFrame({"driver_id": ["a"], "finish_position": [1]})

    scores = score_race(race, pd.Series([1], index=race.index))

    assert scores["spearman"] is None
    assert scores["top10_overlap"] == 1


def _score(finish: list[int], predicted: list[int]) -> dict:
    race = pd.DataFrame(
        {"driver_id": [f"d{index:02d}" for index in range(len(finish))], "finish_position": finish}
    )
    return score_race(race, pd.Series(predicted, index=race.index))


def test_score_race_perfect_and_reversed_prediction():
    perfect = _score([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    reversed_ = _score([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])

    assert perfect == {
        "mae_positions": 0.0,
        "spearman": pytest.approx(1.0),
        "winner_hit": 1,
        "podium_overlap": 3,
        "top10_overlap": 5,
    }
    # Afwijkingen 4, 2, 0, 2, 4; alleen d02 staat in beide top-3's.
    assert reversed_ == {
        "mae_positions": pytest.approx(2.4),
        "spearman": pytest.approx(-1.0),
        "winner_hit": 0,
        "podium_overlap": 1,
        "top10_overlap": 5,
    }


def test_score_race_top10_overlap_uses_only_the_first_ten_places():
    finish = list(range(1, 21))
    predicted = finish.copy()
    predicted[2], predicted[3] = 4, 3  # P3 en P4 verwisseld: podium mist één coureur
    predicted[9], predicted[10] = 11, 10  # P10 en P11 verwisseld: top 10 mist één coureur

    scores = _score(finish, predicted)

    assert scores["mae_positions"] == pytest.approx(4 / 20)
    assert scores["winner_hit"] == 1
    assert scores["podium_overlap"] == 2
    assert scores["top10_overlap"] == 9


VALID_RACE = pd.DataFrame({"driver_id": ["a", "b"], "finish_position": [1, 2]})


@pytest.mark.parametrize(
    ("race", "predicted", "message"),
    [
        (VALID_RACE.drop(columns="finish_position"), [1, 2], "vereiste kolommen"),
        (VALID_RACE.iloc[0:0], [], "lege race"),
        (VALID_RACE.assign(driver_id=["a", "a"]), [1, 2], "uniek en niet-leeg"),
        (VALID_RACE.assign(driver_id=["a", None]), [1, 2], "uniek en niet-leeg"),
        (VALID_RACE.assign(finish_position=[1, None]), [1, 2], "eindige waarde"),
        (VALID_RACE.assign(finish_position=[1, float("inf")]), [1, 2], "eindige waarde"),
        (VALID_RACE.assign(finish_position=[0, 2]), [1, 2], "positief"),
        (VALID_RACE, [1, None], "Voorspelde rangen moeten eindige"),
    ],
)
def test_score_race_rejects_invalid_input(race, predicted, message):
    with pytest.raises(ValueError, match=message):
        score_race(race, pd.Series(predicted, index=race.index, dtype="float64"))


def test_score_race_requires_prediction_with_the_race_index():
    with pytest.raises(ValueError, match="dezelfde index"):
        score_race(VALID_RACE, pd.Series([1, 2], index=[5, 6]))


def test_qualifying_baseline_puts_missing_positions_last():
    race = pd.DataFrame({"driver_id": ["b", "a", "c"], "quali_position": [1, 1, pd.NA]})

    predicted = QualifyingPositionBaseline().fit(pd.DataFrame()).predict(race)

    assert predicted.tolist() == [2, 1, 3]


def test_championship_baseline_uses_prior_race_and_sprint_points():
    history = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "driver_id": ["a", "b", "a", "b"],
            "points": [8.0, 10.0, 0.0, 0.0],
            "sprint_points": [0.0, 0.0, 5.0, 0.0],
        }
    )
    race = pd.DataFrame({"season": [2023] * 3, "driver_id": ["a", "b", "rookie"]})

    predicted = ChampionshipPointsBaseline().fit(history).predict(race)

    assert predicted.tolist() == [1, 2, 3]


def test_championship_baseline_resets_points_for_new_season():
    history = pd.DataFrame(
        {
            "season": [2022],
            "driver_id": ["a"],
            "points": [100.0],
            "sprint_points": [10.0],
        }
    )
    race = pd.DataFrame({"season": [2023, 2023], "driver_id": ["b", "a"]})

    predicted = ChampionshipPointsBaseline().fit(history).predict(race)

    assert predicted.tolist() == [2, 1]


def test_championship_baseline_uses_driver_id_for_points_ties():
    history = pd.DataFrame(
        {
            "season": [2023, 2023],
            "driver_id": ["b", "a"],
            "points": [5.0, 5.0],
            "sprint_points": [0.0, 0.0],
        }
    )
    race = pd.DataFrame({"season": [2023, 2023], "driver_id": ["b", "a"]})

    predicted = ChampionshipPointsBaseline().fit(history).predict(race)

    assert predicted.tolist() == [2, 1]


@pytest.mark.parametrize(
    ("race", "message"),
    [
        (pd.DataFrame({"quali_position": [1]}), "driver_id"),
        (pd.DataFrame({"driver_id": [], "quali_position": []}), "lege race"),
        (pd.DataFrame({"driver_id": ["a", "a"], "quali_position": [1, 2]}), "uniek"),
        (pd.DataFrame({"driver_id": ["a", None], "quali_position": [1, 2]}), "niet-leeg"),
        (pd.DataFrame({"driver_id": ["a"]}), "quali_position"),
    ],
)
def test_qualifying_baseline_rejects_invalid_race(race, message):
    with pytest.raises(ValueError, match=message):
        QualifyingPositionBaseline().predict(race)


HISTORY = pd.DataFrame(
    {"season": [2023], "driver_id": ["a"], "points": [10.0], "sprint_points": [0.0]}
)


@pytest.mark.parametrize(
    ("train", "message"),
    [
        (HISTORY.drop(columns="sprint_points"), "vereiste kolommen"),
        (HISTORY.assign(season=[None]), "niet-leeg"),
        (HISTORY.assign(driver_id=[None]), "niet-leeg"),
        (HISTORY.assign(points=[None]), "volledig"),
        (HISTORY.assign(sprint_points=[None]), "volledig"),
    ],
)
def test_championship_baseline_rejects_incomplete_history(train, message):
    with pytest.raises(ValueError, match=message):
        ChampionshipPointsBaseline().fit(train)


@pytest.mark.parametrize(
    ("race", "message"),
    [
        (pd.DataFrame({"driver_id": ["a"]}), "season"),
        (pd.DataFrame({"season": [None], "driver_id": ["a"]}), "season"),
        (pd.DataFrame({"season": [2023, 2024], "driver_id": ["a", "b"]}), "één seizoen"),
    ],
)
def test_championship_baseline_rejects_race_without_one_season(race, message):
    predictor = ChampionshipPointsBaseline().fit(HISTORY)

    with pytest.raises(ValueError, match=message):
        predictor.predict(race)


def test_backtest_uses_same_chronological_races_for_each_baseline():
    data = pd.concat([_race(2023, round_, [1, 2, 3]) for round_ in range(1, 4)])

    results = run_backtest(
        data,
        {
            "quali": QualifyingPositionBaseline,
            "points": ChampionshipPointsBaseline,
        },
        min_train_races=1,
    )

    assert results[["season", "round"]].drop_duplicates().values.tolist() == [
        [2023, 2],
        [2023, 3],
    ]
    assert results.groupby(["season", "round"])["model"].nunique().tolist() == [2, 2]
    assert results["spearman"].notna().all()


def test_backtest_rejects_duplicate_driver_in_race():
    data = pd.concat([_race(2023, 1, [1, 2]), _race(2023, 2, [1, 2])])
    data.loc[data.index[-1], "round"] = 1

    with pytest.raises(ValueError, match="moet uniek zijn"):
        run_backtest(data, {"quali": QualifyingPositionBaseline}, min_train_races=0)


def test_backtest_orders_races_across_seasons():
    data = pd.concat(
        [
            _race(2023, 1, [1, 2]),
            _race(2023, 2, [1, 2]),
            _race(2024, 1, [1, 2]),
        ],
        ignore_index=True,
    )

    results = run_backtest(
        data,
        {"quali": QualifyingPositionBaseline},
        min_train_races=2,
    )

    assert results[["season", "round"]].values.tolist() == [[2024, 1]]


def test_backtest_fits_only_on_strictly_prior_races():
    data = pd.concat([_race(2023, round_, [1, 2]) for round_ in range(1, 4)])
    training_races = []

    class RecordingPredictor:
        def fit(self, train):
            training_races.append(set(zip(train["season"], train["round"], strict=True)))
            return self

        def predict(self, race):
            return pd.Series(range(1, len(race) + 1), index=race.index)

    run_backtest(data, {"recording": RecordingPredictor}, min_train_races=1)

    assert training_races == [
        {(2023, 1)},
        {(2023, 1), (2023, 2)},
    ]


def test_backtest_requires_one_test_race_after_training_history():
    with pytest.raises(ValueError, match="Onvoldoende races"):
        run_backtest(
            _race(2023, 1, [1, 2]),
            {"quali": QualifyingPositionBaseline},
            min_train_races=1,
        )


QUALI_ONLY = {"quali": QualifyingPositionBaseline}


@pytest.mark.parametrize(
    ("data", "min_train_races", "predictors", "message"),
    [
        (_race(2023, 1, [1, 2]).drop(columns="points"), 0, QUALI_ONLY, "vereiste kolommen"),
        (_race(2023, 1, [1, 2]), -1, QUALI_ONLY, "niet-negatief geheel getal"),
        (_race(2023, 1, [1, 2]), True, QUALI_ONLY, "niet-negatief geheel getal"),
        (_race(2023, 1, [1, 2]), 1.0, QUALI_ONLY, "niet-negatief geheel getal"),
        (_race(2023, 1, [1, 2]), 0, {}, "minstens één predictor"),
        (_race(2023, 1, [1, 2]), 0, {"": QualifyingPositionBaseline}, "minstens één predictor"),
        (_race(2023, 1, [1, 2]).iloc[0:0], 0, QUALI_ONLY, "lege data"),
        (_race(2023, 1, [1, 2]).assign(round=[1, None]), 0, QUALI_ONLY, "volledig"),
    ],
)
def test_backtest_rejects_invalid_input(data, min_train_races, predictors, message):
    with pytest.raises(ValueError, match=message):
        run_backtest(data, predictors, min_train_races=min_train_races)


def test_run_baseline_backtest_writes_csv(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    pd.concat(
        [_race(2023, round_, [1, 2, 3]) for round_ in range(1, 3)],
        ignore_index=True,
    ).to_parquet(data_dir / "results_2023.parquet", index=False)
    output_path = tmp_path / "output" / "backtest_baselines.csv"

    results = run_baseline_backtest(data_dir, output_path, min_train_races=1)

    assert len(results) == 2
    saved = pd.read_csv(output_path)
    assert saved["model"].tolist() == ["qualifying_position", "championship_points"]
    assert saved["round"].tolist() == [2, 2]


def test_ridge_predictor_ranks_lowest_predicted_finish_first():
    train = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], "finish_position": [1, 2, 3, 4] * 2}
    )
    race = pd.DataFrame({"driver_id": ["a", "b", "c"], "x": [3.0, 1.0, 2.0]}, index=[10, 11, 12])

    predicted = RidgeFeaturePredictor(["x"]).fit(train).predict(race)

    assert predicted.index.equals(race.index)
    assert predicted.tolist() == [3, 1, 2]


def test_ridge_predictor_imputes_missing_features_from_training_data():
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0],
            "empty": [None] * 4,
            "finish_position": [1, 2, 3, 4],
        }
    )
    race = pd.DataFrame({"driver_id": ["a", "b"], "x": [None, 1.0], "empty": [None, None]})

    predicted = RidgeFeaturePredictor(["x", "empty"]).fit(train).predict(race)

    assert sorted(predicted.tolist()) == [1, 2]


def test_ridge_predictor_requires_fit_and_feature_columns():
    race = pd.DataFrame({"driver_id": ["a"], "x": [1.0]})

    with pytest.raises(RuntimeError, match="fit"):
        RidgeFeaturePredictor(["x"]).predict(race)
    with pytest.raises(ValueError, match="featurekolommen"):
        RidgeFeaturePredictor(["y"]).fit(pd.DataFrame({"x": [1.0], "finish_position": [1]}))
    with pytest.raises(ValueError, match="minstens één"):
        RidgeFeaturePredictor([])

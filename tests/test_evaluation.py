import pandas as pd
import pytest

from f1_predictor.evaluation.backtest import run_backtest, run_baseline_backtest
from f1_predictor.evaluation.metrics import score_race
from f1_predictor.models.baselines import (
    ChampionshipPointsBaseline,
    QualifyingPositionBaseline,
)


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

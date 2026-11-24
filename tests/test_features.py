import math

import pandas as pd
import pytest
from typer.testing import CliRunner

from f1_predictor import cli
from f1_predictor.evaluation.backtest import feature_experiments, run_feature_backtest
from f1_predictor.features.build import (
    FEATURE_GROUPS,
    FeatureConfig,
    add_race_index,
    build_features,
    feature_columns,
    group_columns,
    write_feature_table,
)
from f1_predictor.features.groups import retirement_mask

POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0}

# Uitslagen van race N; deze kolommen zijn labels en mogen features van race N niet beïnvloeden.
LABEL_COLUMNS = ["finish_position", "position_text", "status", "points", "laps", "sprint_points"]


def _race(
    season: int,
    round_: int,
    finishes: list[int],
    *,
    circuit: str = "ring",
    drivers: list[str] | None = None,
    teams: list[str] | None = None,
    quali: list[int | None] | None = None,
    status: list[str] | None = None,
    sprint: list[float] | None = None,
) -> pd.DataFrame:
    size = len(finishes)
    return pd.DataFrame(
        {
            "season": season,
            "round": round_,
            "race_name": f"Race {season}-{round_}",
            "circuit_id": circuit,
            "driver_id": drivers or [f"d{index}" for index in range(size)],
            "constructor_id": teams or [f"team_{index // 2}" for index in range(size)],
            "grid": finishes,
            "quali_position": pd.array(finishes if quali is None else quali, dtype="Int64"),
            "finish_position": finishes,
            "position_text": [str(position) for position in finishes],
            "status": status or ["Finished"] * size,
            "points": [POINTS.get(position, 0.0) for position in finishes],
            "laps": 50,
            "sprint_points": sprint or [0.0] * size,
        }
    )


def _data(*races: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(races, ignore_index=True)


def _value(features: pd.DataFrame, season: int, round_: int, driver: str, column: str):
    row = features[
        (features["season"] == season)
        & (features["round"] == round_)
        & (features["driver_id"] == driver)
    ]
    assert len(row) == 1
    return row[column].iloc[0]


def _sample_data() -> pd.DataFrame:
    """Twee seizoenen, twee circuits, wisselende statussen en een sprint in ronde 2."""
    finishes = [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [1, 3, 2, 4], [4, 2, 3, 1], [2, 4, 1, 3]]
    statuses = ["Finished", "+1 Lap", "Lapped", "Engine", "Retired"]
    races = []
    for number, race_finishes in enumerate(finishes):
        round_ = number % 3 + 1
        races.append(
            _race(
                2023 + number // 3,
                round_,
                race_finishes,
                circuit="a" if number % 2 == 0 else "b",
                quali=race_finishes[1:] + race_finishes[:1],
                status=[statuses[(number + index) % len(statuses)] for index in range(4)],
                sprint=[3.0, 2.0, 1.0, 0.0] if round_ == 2 else None,
            )
        )
    return _data(*races)


# --- Basis en config ---


def test_retirement_mask_handles_old_and_new_status_vocabularies():
    status = pd.Series(
        [
            "Finished",
            "+1 Lap",
            "+2 Laps",
            "Lapped",
            "Disqualified",
            "Engine",
            "Retired",
            "Did not start",
        ]
    )

    assert retirement_mask(status).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_race_index_is_chronological_across_seasons():
    data = _data(_race(2024, 1, [1, 2]), _race(2023, 2, [1, 2]), _race(2023, 1, [1, 2]))

    assert add_race_index(data)["race_index"].tolist() == [2, 2, 1, 1, 0, 0]


def test_build_features_keeps_raw_data_and_index():
    data = _sample_data()
    data.index = data.index + 100
    original = data.copy()

    features = build_features(data)

    pd.testing.assert_frame_equal(data, original)
    assert features.index.equals(data.index)
    pd.testing.assert_frame_equal(features[data.columns], data)
    assert features.columns.tolist() == [*data.columns, "race_index", *feature_columns()]


@pytest.mark.parametrize("group", list(FEATURE_GROUPS))
def test_each_group_produces_exactly_its_columns(group):
    data = _sample_data()
    config = FeatureConfig(groups=(group,))

    features = build_features(data, config)

    new_columns = [column for column in features.columns if column not in data.columns]
    assert new_columns == ["race_index", *group_columns(group, config)]
    assert feature_columns(config) == group_columns(group, config)


def test_feature_config_accepts_lists_and_uses_fixed_group_order():
    config = FeatureConfig(groups=["teammate", "driver_form"], form_windows=[2])

    assert config.groups == ("teammate", "driver_form")
    assert config.form_windows == (2,)
    assert config.enabled_groups == ["driver_form", "teammate"]
    assert feature_columns(config) == [
        "driver_finish_mean_2",
        "driver_points_mean_2",
        "driver_retirement_rate_10",
        "teammate_quali_gap",
        "teammate_beat_rate_10",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": ()},
        {"groups": ("weather",)},
        {"groups": ("driver_form", "driver_form")},
        {"form_windows": ()},
        {"form_windows": (0,)},
        {"form_windows": (True,)},
        {"form_windows": (3, 3)},
        {"rate_window": 0},
        {"circuit_shrinkage": -1.0},
        {"circuit_shrinkage": math.nan},
    ],
)
def test_feature_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)


def test_build_features_rejects_invalid_input():
    data = _sample_data()

    with pytest.raises(ValueError, match="vereiste kolommen"):
        build_features(data.drop(columns="status"))
    with pytest.raises(ValueError, match="moet uniek zijn"):
        build_features(pd.concat([data, data.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="al featurekolommen"):
        build_features(data.assign(start_position=1))


# --- Featuregroepen: exacte waarden op mini-data ---


def test_starting_position_puts_missing_qualifying_last():
    data = _data(_race(2023, 1, [1, 2, 3], quali=[2, None, 1]))

    features = build_features(data, FeatureConfig(groups=("starting_position",)))

    assert features["start_position"].tolist() == [2.0, 3.0, 1.0]
    assert features["start_quali_missing"].tolist() == [0, 1, 0]


def test_driver_form_uses_only_previous_races():
    finished = ["Finished"] * 3
    data = _data(
        _race(2023, 1, [1, 2, 3, 4]),
        _race(2023, 2, [2, 1, 3, 4], status=["Engine", *finished]),
        _race(2023, 3, [3, 1, 2, 4], status=["+1 Lap", *finished]),
        _race(2024, 1, [4, 1, 2, 3], status=["Lapped", *finished]),
    )

    features = build_features(data, FeatureConfig(groups=("driver_form",), form_windows=(2, 3)))

    assert math.isnan(_value(features, 2023, 1, "d0", "driver_finish_mean_3"))
    assert _value(features, 2023, 2, "d0", "driver_finish_mean_3") == 1.0
    assert _value(features, 2024, 1, "d0", "driver_finish_mean_3") == 2.0
    assert _value(features, 2024, 1, "d0", "driver_finish_mean_2") == 2.5
    assert _value(features, 2024, 1, "d0", "driver_points_mean_3") == pytest.approx(58 / 3)
    assert _value(features, 2024, 1, "d0", "driver_retirement_rate_10") == pytest.approx(1 / 3)


def test_team_form_summarises_previous_team_races():
    data = _data(_race(2023, 1, [1, 4, 2, 3]), _race(2023, 2, [2, 3, 1, 4]))

    features = build_features(data, FeatureConfig(groups=("team_form",)))

    assert math.isnan(_value(features, 2023, 1, "d0", "team_finish_mean_3"))
    for driver in ["d0", "d1"]:
        assert _value(features, 2023, 2, driver, "team_finish_mean_3") == 2.5
        assert _value(features, 2023, 2, driver, "team_points_mean_3") == 37.0
    assert _value(features, 2023, 2, "d2", "team_points_mean_3") == 33.0


def test_team_form_ignores_teammate_result_in_same_race():
    data = _data(_race(2023, 1, [1, 4, 2, 3]), _race(2023, 2, [2, 3, 1, 4]))
    changed = data.copy()
    is_second_race = changed["round"] == 2
    # Elke coureur krijgt een andere uitslag, zodat ook een rij-shift binnen het team opvalt.
    changed.loc[is_second_race, "finish_position"] = [3, 2, 4, 1]
    changed.loc[is_second_race, "points"] = [15.0, 18.0, 12.0, 25.0]
    config = FeatureConfig(groups=("team_form",))
    columns = feature_columns(config)

    original = build_features(data, config).loc[is_second_race, columns]
    modified = build_features(changed, config).loc[is_second_race, columns]

    pd.testing.assert_frame_equal(original, modified)


def test_championship_uses_earlier_rounds_of_same_season_only():
    data = _data(
        _race(2023, 1, [1, 2, 3, 4], sprint=[8.0, 7.0, 0.0, 0.0]),
        _race(2023, 2, [2, 1, 3, 4], sprint=[0.0, 0.0, 8.0, 0.0]),
        _race(2023, 3, [1, 2, 3, 4]),
        _race(2024, 1, [1, 2, 3, 4]),
    )

    features = build_features(data, FeatureConfig(groups=("championship",)))

    round_two = features[features["round"].eq(2)]
    assert round_two["champ_points_before"].tolist() == [33.0, 25.0, 15.0, 12.0]
    assert round_two["champ_position_before"].tolist() == [1.0, 2.0, 3.0, 4.0]
    # De sprint van ronde 2 telt pas vanaf ronde 3 mee.
    assert _value(features, 2023, 3, "d2", "champ_points_before") == 38.0
    new_season = features[features["season"].eq(2024)]
    assert new_season["champ_points_before"].tolist() == [0.0] * 4
    assert new_season["champ_position_before"].tolist() == [1.0] * 4


@pytest.mark.parametrize(("shrinkage", "expected"), [(3.0, -0.375), (0.0, -1.5)])
def test_circuit_history_shrinks_delta_towards_zero(shrinkage, expected):
    data = _data(
        _race(2023, 1, [1, 2, 3, 4], circuit="a"),
        _race(2023, 2, [4, 1, 2, 3], circuit="b"),
        _race(2023, 3, [2, 1, 3, 4], circuit="a"),
    )
    config = FeatureConfig(groups=("circuit_history",), circuit_shrinkage=shrinkage)

    features = build_features(data, config)

    assert _value(features, 2023, 1, "d0", "circuit_delta_shrunk") == 0.0
    assert _value(features, 2023, 2, "d0", "circuit_delta_shrunk") == 0.0
    assert _value(features, 2023, 3, "d0", "circuit_prior_races") == 1
    assert _value(features, 2023, 3, "d0", "circuit_delta_shrunk") == pytest.approx(expected)


def test_teammate_features_compare_within_team():
    drivers = ["d0", "d1", "d2", "d3", "solo"]
    teams = ["t0", "t0", "t1", "t1", "t2"]
    data = _data(
        _race(2023, 1, [1, 3, 2, 4, 5], drivers=drivers, teams=teams, quali=[1, 4, 2, None, 5]),
        _race(2023, 2, [2, 1, 3, 4, 5], drivers=drivers, teams=teams),
    )

    features = build_features(data, FeatureConfig(groups=("teammate",)))

    first = features[features["round"].eq(1)].set_index("driver_id")["teammate_quali_gap"]
    assert first["d0"] == -3.0
    assert first["d1"] == 3.0
    assert math.isnan(first["d2"])  # teamgenoot zonder kwalificatiepositie
    assert math.isnan(first["d3"])  # eigen kwalificatiepositie ontbreekt
    assert math.isnan(first["solo"])  # geen teamgenoot
    second = features[features["round"].eq(2)].set_index("driver_id")["teammate_beat_rate_10"]
    assert second[["d0", "d1", "d2", "d3"]].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert math.isnan(second["solo"])
    first_rate = features[features["round"].eq(1)]["teammate_beat_rate_10"]
    assert first_rate.isna().all()


# --- Leakage ---


@pytest.mark.parametrize("race_number", range(6))
def test_changing_race_result_does_not_change_features_of_that_race(race_number):
    data = _sample_data()
    is_race = add_race_index(data)["race_index"] == race_number
    perturbed = data.copy()
    perturbed.loc[is_race, "finish_position"] = data.loc[is_race, "finish_position"].to_numpy()[
        ::-1
    ]
    perturbed.loc[is_race, "position_text"] = "R"
    perturbed.loc[is_race, "status"] = "Engine"
    perturbed.loc[is_race, "points"] = 99.0
    perturbed.loc[is_race, "laps"] = 1
    perturbed.loc[is_race, "sprint_points"] = 42.0
    columns = feature_columns()

    original = build_features(data)
    changed = build_features(perturbed)

    pd.testing.assert_frame_equal(original.loc[is_race, columns], changed.loc[is_race, columns])
    # Controle dat de test iets meet: latere races zien de gewijzigde uitslag wel.
    later = original["race_index"] > race_number
    if later.any():
        assert not original.loc[later, columns].equals(changed.loc[later, columns])


@pytest.mark.parametrize("race_number", range(6))
def test_future_races_do_not_change_features(race_number):
    data = _sample_data()
    full = build_features(data)
    keep = full["race_index"] <= race_number

    truncated = build_features(data.loc[keep])

    pd.testing.assert_frame_equal(full.loc[keep], truncated)


# --- Wegschrijven ---


def test_write_feature_table_writes_parquet(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    data = _sample_data()
    for season, part in data.groupby("season"):
        part.to_parquet(raw_dir / f"results_{season}.parquet", index=False)
    output_path = tmp_path / "processed" / "features.parquet"

    table = write_feature_table(raw_dir, output_path)

    saved = pd.read_parquet(output_path)
    assert len(saved) == len(table) == len(data)
    assert set(feature_columns()).issubset(saved.columns)
    assert not output_path.with_suffix(".parquet.tmp").exists()


def test_cli_features_builds_table_with_configured_features(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "write_feature_table", lambda **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(cli.app, ["features"])

    assert result.exit_code == 0
    # De vastgelegde config/config.yaml gebruikt de standaardfeatures.
    assert calls == [{"config": FeatureConfig()}]


def test_cli_features_reports_missing_raw_data(monkeypatch, caplog):
    def missing_data(**kwargs):
        raise FileNotFoundError("Geen parquet-bestanden gevonden in data/raw")

    monkeypatch.setattr(cli, "write_feature_table", missing_data)

    result = CliRunner().invoke(cli.app, ["features"])

    assert result.exit_code == 1
    assert "f1 fetch" in caplog.text


# --- Meetinstrument: feature-backtest ---


def test_feature_experiments_add_groups_cumulatively():
    experiments = feature_experiments()

    assert list(experiments) == [f"ridge_upto_{group}" for group in FEATURE_GROUPS]
    first, *_, last = experiments.values()
    assert first().feature_columns == ["start_position", "start_quali_missing"]
    assert last().feature_columns == feature_columns()


def test_run_feature_backtest_scores_all_models_on_same_races(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _sample_data().to_parquet(raw_dir / "results_2023.parquet", index=False)
    output_path = tmp_path / "output" / "backtest_features.csv"

    results = run_feature_backtest(raw_dir, output_path, min_train_races=2)

    saved = pd.read_csv(output_path)
    assert len(saved) == len(results) == 4 * (2 + len(FEATURE_GROUPS))
    races_per_model = saved.groupby("model")[["season", "round"]].apply(
        lambda frame: tuple(map(tuple, frame.to_numpy()))
    )
    assert races_per_model.nunique() == 1
    assert {"qualifying_position", "championship_points"}.issubset(saved["model"])

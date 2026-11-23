import pandas as pd
import pytest

from f1_predictor.evaluation.explore import (
    build_circuit_effects,
    build_driver_circuit_bias,
    build_quality_summary,
    build_retirement_by_season,
    build_retirement_stats,
    write_phase4_outputs,
)


def _race(season: int, round_: int, circuit: str, grids: list[int], statuses: list[str]):
    size = len(grids)
    return pd.DataFrame(
        {
            "season": season,
            "round": round_,
            "race_name": f"Race {round_}",
            "circuit_id": circuit,
            "driver_id": [f"driver_{index}" for index in range(size)],
            "constructor_id": ["team_a", "team_b"] * (size // 2),
            "grid": grids,
            "quali_position": pd.array(range(1, size + 1), dtype="Int64"),
            "finish_position": list(range(1, size + 1)),
            "status": statuses,
            "points": [25.0] + [0.0] * (size - 1),
            "sprint_points": 0.0,
            "laps": 50,
        }
    )


def test_quality_summary_reports_ranges_and_domain_violations():
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "round": [1, 1],
            "driver_id": ["a", "b"],
            "grid": [0, 3],
            "finish_position": [1, 3],
            "points": [25.0, -1.0],
            "sprint_points": [0.0, 0.0],
            "laps": [0, 57],
        }
    )

    summary = build_quality_summary(df)

    assert summary["numeric_ranges"]["grid"] == (0.0, 3.0)
    assert summary["invalid_domain_values"] == {
        "negative_grid": 0,
        "finish_position_outside_race_field": 1,
        "negative_points": 1,
        "negative_sprint_points": 0,
        "negative_laps": 0,
    }


def test_quality_summary_treats_pitlane_start_as_last_on_grid():
    # Pitlane-starter eindigt laatst: met grid 0 als "pole" zou de correlatie dalen.
    df = pd.DataFrame(
        {
            "season": 2024,
            "round": 1,
            "driver_id": ["a", "b", "c"],
            "grid": [1, 2, 0],
            "finish_position": [1, 2, 3],
            "points": 0.0,
            "sprint_points": 0.0,
            "laps": 50,
        }
    )

    assert build_quality_summary(df)["grid_finish_spearman"] == pytest.approx(1.0)


def test_retirement_stats_per_team_and_season():
    df = pd.concat(
        [
            _race(2023, 1, "x", [1, 2], ["Finished", "+1 Lap"]),
            _race(2024, 1, "x", [1, 2], ["Lapped", "Engine"]),
        ]
    )

    by_team = build_retirement_stats(df).set_index("constructor_id")
    by_season = build_retirement_by_season(df).set_index("season")

    assert by_team.loc["team_b", "retirement_rate"] == 50.0
    assert by_team.loc["team_a", "retirement_rate"] == 0.0
    assert by_season["retirement_rate"].tolist() == [0.0, 50.0]
    assert by_season["starts"].tolist() == [2, 2]


def test_circuit_effects_measure_predictability_not_average_finish():
    finished = ["Finished"] * 4
    df = pd.concat(
        [
            _race(2023, 1, "orderly", [1, 2, 3, 4], finished),
            _race(2023, 2, "chaotic", [4, 3, 2, 1], ["Finished", "Finished", "Engine", "Engine"]),
        ]
    )

    effects = build_circuit_effects(df).set_index("circuit_id")

    assert effects.index.tolist() == ["chaotic", "orderly"]
    assert effects.loc["orderly", "start_finish_spearman"] == pytest.approx(1.0)
    assert effects.loc["chaotic", "start_finish_spearman"] == pytest.approx(-1.0)
    assert effects.loc["orderly", "mean_abs_position_change"] == 0.0
    assert effects.loc["chaotic", "retirement_rate"] == 50.0
    assert effects.loc["chaotic", "races"] == 1


def test_driver_circuit_bias_requires_minimum_races():
    finished = ["Finished"] * 2
    df = pd.concat([_race(2020 + index, 1, "x", [1, 2], finished) for index in range(3)])
    df = pd.concat([df, _race(2023, 2, "y", [1, 2], finished)])

    bias = build_driver_circuit_bias(df, min_races=3)

    assert set(bias["circuit_id"]) == {"x"}
    assert (bias["races"] >= 3).all()


def test_write_phase4_outputs_uses_given_paths(tmp_path):
    finished = ["Finished"] * 2
    df = pd.concat([_race(2023, round_, "x", [1, 2], finished) for round_ in range(1, 4)])
    report_path = tmp_path / "docs" / "report.md"

    outputs = write_phase4_outputs(df, output_dir=tmp_path / "out", report_path=report_path)

    assert outputs["report"] == report_path
    assert "Uitvalpercentage per seizoen" in report_path.read_text(encoding="utf-8")
    assert all(path.exists() for path in outputs.values())

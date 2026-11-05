import pandas as pd

from f1_predictor.evaluation.explore import build_quality_summary


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

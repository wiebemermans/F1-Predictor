"""Metrics voor voorspelde raceklasseringen."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_race(race: pd.DataFrame, predicted_rank: pd.Series) -> dict[str, float | int | None]:
    """Score één race; `predicted_rank` bevat de unieke voorspelde plaatsen 1..n."""
    required = {"driver_id", "finish_position"}
    missing = required.difference(race.columns)
    if missing:
        raise ValueError(f"Race mist vereiste kolommen: {sorted(missing)}")
    if race.empty:
        raise ValueError("Kan geen lege race scoren")
    if race["driver_id"].isna().any() or race["driver_id"].duplicated().any():
        raise ValueError("driver_id moet per race uniek en niet-leeg zijn")
    if not predicted_rank.index.equals(race.index):
        raise ValueError("Voorspelde rangen moeten dezelfde index hebben als de race")

    actual_rank = pd.to_numeric(race["finish_position"], errors="raise")
    prediction = pd.to_numeric(predicted_rank, errors="raise")
    if actual_rank.isna().any() or not np.isfinite(actual_rank.to_numpy(dtype=float)).all():
        raise ValueError("finish_position moet voor elke coureur een eindige waarde hebben")
    if (actual_rank <= 0).any():
        raise ValueError("finish_position moet positief zijn")
    if prediction.isna().any() or not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise ValueError("Voorspelde rangen moeten eindige waarden zijn")
    actual_values = actual_rank.to_numpy(dtype=float)
    prediction_values = prediction.to_numpy(dtype=float)
    expected_ranks = np.arange(1, len(race) + 1, dtype=float)
    if not np.array_equal(np.sort(prediction_values), expected_ranks):
        raise ValueError("Voorspelde rangen moeten exact de unieke plaatsen 1..n bevatten")

    predicted_order = (
        pd.DataFrame({"driver_id": race["driver_id"].to_numpy(), "rank": prediction_values})
        .sort_values(["rank", "driver_id"], kind="stable")["driver_id"]
        .tolist()
    )
    actual_order = (
        pd.DataFrame({"driver_id": race["driver_id"].to_numpy(), "rank": actual_values})
        .sort_values(["rank", "driver_id"], kind="stable")["driver_id"]
        .tolist()
    )
    spearman = pd.Series(prediction_values).corr(pd.Series(actual_values), method="spearman")

    return {
        "mae_positions": float(np.abs(prediction_values - actual_values).mean()),
        "spearman": None if pd.isna(spearman) else float(spearman),
        "winner_hit": int(predicted_order[0] == actual_order[0]),
        "podium_overlap": len(set(predicted_order[:3]) & set(actual_order[:3])),
        "top10_overlap": len(set(predicted_order[:10]) & set(actual_order[:10])),
    }

"""Zet race-blokken uit de API om naar platte rijen."""

from __future__ import annotations

from typing import Any

Race = dict[str, Any]


def flatten_results(race: Race) -> list[dict[str, Any]]:
    info = {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_name": race["raceName"],
        "circuit_id": race["Circuit"]["circuitId"],
        "date": race["date"],
    }
    return [
        {
            **info,
            "driver_id": r["Driver"]["driverId"],
            "driver_code": r["Driver"].get("code"),
            "constructor_id": r["Constructor"]["constructorId"],
            "grid": int(r["grid"]),
            "finish_position": int(r["position"]),
            "position_text": r["positionText"],
            "status": r["status"],
            "points": float(r["points"]),
            "laps": int(r["laps"]),
        }
        for r in race.get("Results", [])
    ]


def flatten_qualifying(race: Race) -> list[dict[str, Any]]:
    return [
        {
            "season": int(race["season"]),
            "round": int(race["round"]),
            "driver_id": r["Driver"]["driverId"],
            "quali_position": int(r["position"]),
        }
        for r in race.get("QualifyingResults", [])
    ]


def flatten_sprint(race: Race) -> list[dict[str, Any]]:
    return [
        {
            "season": int(race["season"]),
            "round": int(race["round"]),
            "driver_id": r["Driver"]["driverId"],
            "sprint_points": float(r["points"]),
        }
        for r in race.get("SprintResults", [])
    ]

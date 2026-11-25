"""Zet race-blokken uit de API om naar platte rijen."""

from __future__ import annotations

from typing import Any

Race = dict[str, Any]


def _race_info(race: Race) -> dict[str, Any]:
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_name": race["raceName"],
        "circuit_id": race["Circuit"]["circuitId"],
        "date": race["date"],
    }


def flatten_results(race: Race) -> list[dict[str, Any]]:
    info = _race_info(race)
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


def _session_start(day: str, time: str | None) -> str:
    """Begin van een sessie als ISO-tekst in UTC; zonder tijd middernacht UTC van die dag."""
    clock = (time or "00:00:00Z").removesuffix("Z")
    return f"{day}T{clock}+00:00"


def flatten_schedule(race: Race) -> dict[str, Any]:
    """Eén regel van de racekalender: race-info plus de start van race en kwalificatie (UTC).

    Zonder racetijd telt middernacht UTC van de racedag: een race geldt dan eerder als
    begonnen, zodat er nooit na de start nog voorspeld wordt. Zonder kwalificatieblok is
    `qualifying_start` leeg.
    """
    qualifying = race.get("Qualifying")
    return {
        **_race_info(race),
        "race_start": _session_start(race["date"], race.get("time")),
        "qualifying_start": (
            None
            if qualifying is None
            else _session_start(qualifying["date"], qualifying.get("time"))
        ),
    }


def flatten_race_entry(race: Race) -> list[dict[str, Any]]:
    """Deelnemerslijst van een race uit het kwalificatieblok, met race-info en team.

    Bruikbaar voor een race die nog niet gereden is: na de kwalificatie staan coureur, team
    en kwalificatiepositie al vast, de uitslag nog niet.
    """
    info = _race_info(race)
    return [
        {
            **info,
            "driver_id": r["Driver"]["driverId"],
            "driver_code": r["Driver"].get("code"),
            "constructor_id": r["Constructor"]["constructorId"],
            "quali_position": int(r["position"]),
        }
        for r in race.get("QualifyingResults", [])
    ]

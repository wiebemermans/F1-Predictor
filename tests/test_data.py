import pandas as pd

from f1_predictor.data import fetch as fetch_module
from f1_predictor.data.fetch import build_season_table, fetch_seasons
from f1_predictor.data.jolpica import PAGE_LIMIT, JolpicaClient, JolpicaError
from f1_predictor.data.parse import flatten_results


def _race(round_: str, results: list[dict]) -> dict:
    return {
        "season": "2023",
        "round": round_,
        "raceName": "Test Grand Prix",
        "Circuit": {"circuitId": "testring"},
        "date": "2023-03-05",
        "Results": results,
    }


def _result(driver: str, position: str, grid: str, position_text: str, status: str) -> dict:
    return {
        "position": position,
        "positionText": position_text,
        "points": "25" if position == "1" else "0",
        "grid": grid,
        "laps": "57",
        "status": status,
        "Driver": {"driverId": driver, "code": driver[:3].upper()},
        "Constructor": {"constructorId": "team_x"},
    }


RESULTS = [
    _result("driver_a", "1", "1", "1", "Finished"),
    _result("driver_b", "2", "0", "R", "Engine"),  # pitlane-start + uitgevallen
]


class FakeClient(JolpicaClient):
    """Client zonder netwerk: levert vooraf bepaalde races per endpoint."""

    def __init__(self, data: dict[str, list[dict]], fail_on: str | None = None) -> None:
        self._data = data
        self._fail_on = fail_on

    def races(self, path: str):
        if self._fail_on and path.startswith(self._fail_on):
            raise JolpicaError("boem")
        return iter(self._data.get(path.split("/")[-1], []))


class PagedClient(JolpicaClient):
    """Levert vooraf bepaalde pagina's op basis van de offset."""

    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages

    def get(self, path: str, **params):
        return self.pages[params["offset"] // PAGE_LIMIT]


def test_flatten_results_types_and_special_cases():
    rows = flatten_results(_race("1", RESULTS))
    assert len(rows) == 2
    assert rows[0]["season"] == 2023 and isinstance(rows[0]["points"], float)
    assert rows[1]["grid"] == 0 and rows[1]["position_text"] == "R"


def test_flatten_results_without_results_key():
    race = _race("1", RESULTS)
    del race["Results"]
    assert flatten_results(race) == []


def test_races_follows_pagination_and_keeps_split_race():
    part1, part2 = _race("1", RESULTS[:1]), _race("1", RESULTS[1:])
    pages = [
        {"total": "150", "RaceTable": {"Races": [part1]}},
        {"total": "150", "RaceTable": {"Races": [part2]}},
    ]
    races = list(PagedClient(pages).races("2023/results"))
    assert len(races) == 2  # dezelfde race, over twee pagina's
    assert sum(len(flatten_results(r)) for r in races) == 2  # geen rijen verloren


def _fake_data() -> dict[str, list[dict]]:
    quali = {
        "season": "2023",
        "round": "1",
        "QualifyingResults": [{"position": "1", "Driver": {"driverId": "driver_a"}}],
    }
    return {"results": [_race("1", RESULTS)], "qualifying": [quali], "sprint": []}


def test_build_season_table_merges_and_handles_missing():
    df = build_season_table(2023, FakeClient(_fake_data()))
    assert len(df) == 2
    assert df.loc[df.driver_id == "driver_a", "quali_position"].iloc[0] == 1
    assert df.loc[df.driver_id == "driver_b", "quali_position"].isna().all()
    assert (df["sprint_points"] == 0.0).all()
    assert str(df["quali_position"].dtype) == "Int64"


def test_fetch_seasons_continues_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_module, "RAW_DIR", tmp_path)
    client = FakeClient(_fake_data(), fail_on="2022")
    failed = fetch_seasons(2022, 2023, client=client)
    assert failed == [2022]
    assert not (tmp_path / "results_2022.parquet").exists()
    saved = pd.read_parquet(tmp_path / "results_2023.parquet")
    assert len(saved) == 2

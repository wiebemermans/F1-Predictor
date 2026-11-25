from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
import requests_cache

from f1_predictor.data import fetch as fetch_module
from f1_predictor.data import jolpica as jolpica_module
from f1_predictor.data.fetch import build_season_table, fetch_schedule, fetch_seasons
from f1_predictor.data.jolpica import (
    BASE_URL,
    MAX_RETRIES,
    PAGE_LIMIT,
    PAUSE_SECONDS,
    JolpicaClient,
    JolpicaError,
)
from f1_predictor.data.parse import flatten_results, flatten_schedule


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


# --- Requests, retries en cache van de client ---

MRDATA = {"total": "0", "RaceTable": {"Races": []}}


class FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None, from_cache: bool = False):
        self.status_code = status_code
        self.headers = headers or {}
        self.from_cache = from_cache

    def json(self) -> dict:
        return {"MRData": MRDATA}

    def raise_for_status(self) -> None:
        raise requests.HTTPError(f"HTTP {self.status_code}")


class ScriptedSession:
    """Sessie zonder netwerk: geeft de responses in volgorde terug en onthoudt de requests."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls.append((url, params))
        return self.responses.pop(0)


def _scripted_client(responses: list[FakeResponse]) -> JolpicaClient:
    client = JolpicaClient()
    client.session = ScriptedSession(responses)
    return client


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Vervang `time.sleep` in de client: de test wacht niet echt, maar ziet elke pauze."""
    recorded: list[float] = []
    monkeypatch.setattr(jolpica_module, "time", SimpleNamespace(sleep=recorded.append))
    return recorded


def test_get_returns_mrdata_and_pauses_only_after_network_calls(waits):
    client = _scripted_client([FakeResponse(200), FakeResponse(200, from_cache=True)])

    assert client.get("/2023/results/", limit=100, offset=0) == MRDATA
    assert client.get("2023/results") == MRDATA

    assert client.session.calls[0] == (f"{BASE_URL}/2023/results/", {"limit": 100, "offset": 0})
    assert waits == [PAUSE_SECONDS]  # geen pauze na een antwoord uit de cache


@pytest.mark.parametrize(
    ("status", "headers", "expected_wait"),
    [
        (429, {"Retry-After": "3"}, 3),
        (429, {"Retry-After": "later"}, 2),  # onleesbare header: exponentiële wachttijd
        (503, {}, 2),
    ],
)
def test_get_retries_rate_limits_and_server_errors(waits, status, headers, expected_wait):
    client = _scripted_client([FakeResponse(status, headers), FakeResponse(200)])

    assert client.get("2023/results") == MRDATA
    assert waits == [expected_wait, PAUSE_SECONDS]


def test_get_backs_off_exponentially_and_gives_up(waits):
    client = _scripted_client([FakeResponse(500)] * MAX_RETRIES)

    with pytest.raises(JolpicaError, match=f"{MAX_RETRIES} pogingen"):
        client.get("2023/results")

    assert waits == [2**attempt for attempt in range(1, MAX_RETRIES + 1)]


def test_get_fails_immediately_on_other_client_errors(waits):
    client = _scripted_client([FakeResponse(404), FakeResponse(200)])

    with pytest.raises(requests.HTTPError, match="404"):
        client.get("2023/unknown")

    assert len(client.session.calls) == 1
    assert waits == []


def test_client_with_cache_path_uses_a_sqlite_cache(tmp_path):
    client = JolpicaClient(tmp_path / "cache" / "jolpica")
    try:
        assert isinstance(client.session, requests_cache.CachedSession)
        assert (tmp_path / "cache").is_dir()
    finally:
        client.session.close()
    assert not isinstance(JolpicaClient().session, requests_cache.CachedSession)


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


def test_fetch_seasons_skips_current_season_without_results(tmp_path, monkeypatch):
    # Vóór de eerste race van het jaar heeft het lopende seizoen nog geen uitslagen.
    monkeypatch.setattr(fetch_module, "RAW_DIR", tmp_path)
    this_year = date.today().year
    client = FakeClient({})

    assert fetch_seasons(this_year, this_year, client=client) == []
    # Een afgelopen seizoen zonder uitslagen blijft een fout.
    assert fetch_seasons(this_year - 1, this_year - 1, client=client) == [this_year - 1]
    assert not list(tmp_path.iterdir())


def test_fetch_seasons_continues_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_module, "RAW_DIR", tmp_path)
    client = FakeClient(_fake_data(), fail_on="2022")
    failed = fetch_seasons(2022, 2023, client=client)
    assert failed == [2022]
    assert not (tmp_path / "results_2022.parquet").exists()
    saved = pd.read_parquet(tmp_path / "results_2023.parquet")
    assert len(saved) == 2


# --- Racekalender ---


def _calendar_race(round_: str, date_: str, time: str | None, qualifying: dict | None) -> dict:
    race = {
        "season": "2026",
        "round": round_,
        "raceName": f"Race {round_}",
        "Circuit": {"circuitId": f"circuit_{round_}"},
        "date": date_,
    }
    if time is not None:
        race["time"] = time
    if qualifying is not None:
        race["Qualifying"] = qualifying
    return race


def test_flatten_schedule_builds_utc_start_times():
    race = _calendar_race(
        "22", "2026-11-29", "16:00:00Z", {"date": "2026-11-28", "time": "18:00:00Z"}
    )

    assert flatten_schedule(race) == {
        "season": 2026,
        "round": 22,
        "race_name": "Race 22",
        "circuit_id": "circuit_22",
        "date": "2026-11-29",
        "race_start": "2026-11-29T16:00:00+00:00",
        "qualifying_start": "2026-11-28T18:00:00+00:00",
    }


def test_flatten_schedule_without_times_or_qualifying():
    # Zonder racetijd telt middernacht UTC: liever te vroeg als gestart dan na de start voorspellen.
    without_time = flatten_schedule(_calendar_race("1", "2026-03-08", None, {"date": "2026-03-07"}))
    without_qualifying = flatten_schedule(_calendar_race("2", "2026-03-15", "07:00:00Z", None))

    assert without_time["race_start"] == "2026-03-08T00:00:00+00:00"
    assert without_time["qualifying_start"] == "2026-03-07T00:00:00+00:00"
    assert without_qualifying["qualifying_start"] is None


class CalendarClient(JolpicaClient):
    """Client zonder netwerk die de kalender teruggeeft en de opgevraagde paden onthoudt."""

    def __init__(self, races: list[dict]) -> None:
        self._races = races
        self.paths: list[str] = []

    def races(self, path: str):
        self.paths.append(path)
        return iter(self._races)


def test_fetch_schedule_returns_the_season_calendar_in_utc():
    client = CalendarClient(
        [
            _calendar_race("2", "2026-03-15", "07:00:00Z", None),
            _calendar_race(
                "1", "2026-03-08", "04:00:00Z", {"date": "2026-03-07", "time": "05:00:00Z"}
            ),
        ]
    )

    schedule = fetch_schedule(2026, client)

    assert client.paths == ["2026"]
    assert schedule["round"].tolist() == [1, 2]
    assert schedule.loc[0, "race_start"] == pd.Timestamp("2026-03-08 04:00", tz="UTC")
    assert schedule.loc[0, "qualifying_start"] == pd.Timestamp("2026-03-07 05:00", tz="UTC")
    assert pd.isna(schedule.loc[1, "qualifying_start"])


def test_fetch_schedule_of_an_unpublished_season_is_empty_and_uncached(monkeypatch):
    created = []

    class RecordingClient(CalendarClient):
        def __init__(self, *args) -> None:
            created.append(args)
            super().__init__([])

    monkeypatch.setattr(fetch_module, "JolpicaClient", RecordingClient)

    schedule = fetch_schedule(2027)

    # Zonder cachepad: een verouderde kalender mag niet uit de cache terugkomen.
    assert created == [()]
    assert schedule.empty
    assert str(schedule["race_start"].dt.tz) == "UTC"

import json
import logging
import math
import subprocess
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
import requests
from typer.testing import CliRunner

from f1_predictor import cli, runinfo
from f1_predictor.config import BacktestConfig, ModelConfig, ProjectConfig, load_config
from f1_predictor.data.fetch import QualifyingNotAvailableError, fetch_race_entry
from f1_predictor.data.jolpica import JolpicaClient, JolpicaError
from f1_predictor.data.parse import flatten_race_entry
from f1_predictor.evaluation.backtest import run_backtest
from f1_predictor.evaluation.metrics import score_race
from f1_predictor.features.build import build_features
from f1_predictor.models.predict import (
    PredictionError,
    format_prediction,
    next_race,
    predict_race,
    race_to_predict,
    write_prediction,
)
from f1_predictor.models.train import ModelSpec
from f1_predictor.output import json_ready, read_json, write_json_if_changed

POINTS = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0}
SMALL_PARAMS = {
    "ridge": {"alpha": 1.0},
    "xgb_regressor": {"max_depth": 2, "n_estimators": 5},
    "xgb_ranker": {"max_depth": 2, "n_estimators": 5},
}
METRICS = ["mae_positions", "spearman", "winner_hit", "podium_overlap", "top10_overlap"]


def _project(family: str = "xgb_ranker", min_train_races: int = 2) -> ProjectConfig:
    spec = ModelSpec(family, "all", SMALL_PARAMS[family])
    return ProjectConfig(
        model=ModelConfig(final=family, tuned={family: spec}),
        backtest=BacktestConfig(min_train_races=min_train_races),
    )


def _raw_dir(tmp_path, seasons=(2022, 2023), rounds: int = 3, n_drivers: int = 6):
    """Ruwe seizoensbestanden zoals `f1 fetch` ze schrijft, met synthetische uitslagen."""
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for season in seasons:
        frames = []
        for round_ in range(1, rounds + 1):
            quali = rng.permutation(n_drivers) + 1
            finish = quali.copy()
            first, second = rng.choice(n_drivers, 2, replace=False)
            finish[[first, second]] = finish[[second, first]]
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "round": round_,
                        "race_name": f"Race {season}-{round_}",
                        "circuit_id": f"circuit_{round_}",
                        "date": pd.Timestamp(f"{season}-0{round_}-15"),
                        "driver_id": [f"d{index}" for index in range(n_drivers)],
                        "driver_code": [f"D{index}" for index in range(n_drivers)],
                        "constructor_id": [f"team_{index // 2}" for index in range(n_drivers)],
                        "grid": quali,
                        "finish_position": finish,
                        "position_text": [str(position) for position in finish],
                        "status": "Finished",
                        "points": [POINTS.get(position, 0.0) for position in finish],
                        "laps": 50,
                        "quali_position": pd.array(quali, dtype="Int64"),
                        "sprint_points": 0.0,
                    }
                )
            )
        pd.concat(frames, ignore_index=True).to_parquet(
            raw_dir / f"results_{season}.parquet", index=False
        )
    return raw_dir


def _qualifying_race(season: int, round_: int, drivers: list[str]) -> dict:
    return {
        "season": str(season),
        "round": str(round_),
        "raceName": "Volgende Grand Prix",
        "Circuit": {"circuitId": "circuit_1"},
        "date": f"{season}-03-10",
        "QualifyingResults": [
            {
                "position": str(position),
                "Driver": {"driverId": driver, "code": driver.upper()},
                "Constructor": {"constructorId": f"team_{index // 2}"},
            }
            for index, (position, driver) in enumerate(enumerate(drivers, start=1))
        ],
    }


class QualifyingClient(JolpicaClient):
    """Client zonder netwerk die een vaste kwalificatie teruggeeft en de paden onthoudt."""

    def __init__(self, races: list[dict]) -> None:
        self._races = races
        self.paths: list[str] = []

    def races(self, path: str):
        self.paths.append(path)
        return iter(self._races)


class UnusedClient(JolpicaClient):
    def __init__(self) -> None:
        pass

    def races(self, path: str):
        raise AssertionError(f"Onverwachte API-call: {path}")


def _ranks(payload: dict) -> dict[str, int]:
    return {row["driver_id"]: row["predicted_position"] for row in payload["predictions"]}


# --- Deelnemerslijst uit de kwalificatie ---


def test_flatten_race_entry_reads_driver_team_and_qualifying_position():
    rows = flatten_race_entry(_qualifying_race(2024, 1, ["a", "b"]))

    assert rows[1] == {
        "season": 2024,
        "round": 1,
        "race_name": "Volgende Grand Prix",
        "circuit_id": "circuit_1",
        "date": "2024-03-10",
        "driver_id": "b",
        "driver_code": "B",
        "constructor_id": "team_0",
        "quali_position": 2,
    }


def test_fetch_race_entry_asks_one_race_and_requires_qualifying():
    client = QualifyingClient([_qualifying_race(2024, 2, ["a", "b"])])

    entry = fetch_race_entry(2024, 2, client)

    assert client.paths == ["2024/2/qualifying"]
    assert entry["driver_id"].tolist() == ["a", "b"]
    assert str(entry["quali_position"].dtype) == "Int64"
    with pytest.raises(QualifyingNotAvailableError, match="Nog geen kwalificatie"):
        fetch_race_entry(2024, 3, QualifyingClient([]))


# --- Volgende race volgens de kalender ---


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": [22, 23],
            "race_name": ["Qatar Grand Prix", "Abu Dhabi Grand Prix"],
            "race_start": pd.to_datetime(["2026-11-29 16:00", "2026-12-06 13:00"], utc=True),
            "qualifying_start": pd.to_datetime(["2026-11-28 18:00", "2026-12-05 14:00"], utc=True),
        }
    )


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "expected_round"),
    [
        ("2026-11-25 12:00", 22),  # vóór de kwalificatie
        ("2026-11-28 20:00", 22),  # tussen kwalificatie en race
        ("2026-11-29 16:00", 23),  # exact op de starttijd telt de race als begonnen
        ("2026-12-06 13:00", None),  # na de start van de laatste race
    ],
)
def test_next_race_is_the_first_race_that_has_not_started(now, expected_round):
    race = next_race(_schedule(), _utc(now))

    assert (None if race is None else race["round"]) == expected_round


def test_next_race_needs_a_timezone_and_handles_an_empty_calendar():
    with pytest.raises(ValueError, match="tijdzone"):
        next_race(_schedule(), datetime(2026, 11, 25))
    assert next_race(_schedule().iloc[0:0], _utc("2026-11-25 12:00")) is None


def _calendar_race(round_: int, race_start: str, qualifying_start: str) -> dict:
    race_day, race_time = race_start.split(" ")
    quali_day, quali_time = qualifying_start.split(" ")
    return {
        "season": "2026",
        "round": str(round_),
        "raceName": f"Race {round_}",
        "Circuit": {"circuitId": f"circuit_{round_}"},
        "date": race_day,
        "time": f"{race_time}:00Z",
        "Qualifying": {"date": quali_day, "time": f"{quali_time}:00Z"},
    }


CALENDAR = [
    _calendar_race(22, "2026-11-29 16:00", "2026-11-28 18:00"),
    _calendar_race(23, "2026-12-06 13:00", "2026-12-05 14:00"),
]


@pytest.mark.parametrize(
    ("now", "expected", "message"),
    [
        ("2026-11-25 12:00", None, "de kwalificatie begint pas op 2026-11-28T18:00:00+00:00"),
        ("2026-11-28 18:00", (2026, 22), "Volgende race: Race 22 (2026 ronde 22)"),
        ("2026-11-29 20:00", None, "de kwalificatie begint pas op 2026-12-05"),
        ("2026-12-07 08:00", None, "Geen komende race in de kalender van 2026"),
    ],
)
def test_race_to_predict_waits_until_qualifying_has_started(now, expected, message, caplog):
    caplog.set_level(logging.INFO)
    client = QualifyingClient(CALENDAR)

    assert race_to_predict(_utc(now), client) == expected
    assert client.paths == ["2026"]
    assert message in caplog.text


# --- Voorspellen ---


@pytest.mark.parametrize("family", ["xgb_ranker", "xgb_regressor", "ridge"])
def test_retrospective_prediction_equals_the_backtest_fold(tmp_path, family):
    raw_dir = _raw_dir(tmp_path)
    project = _project(family)

    payload = predict_race(2023, 3, project, raw_dir, client=UnusedClient())

    # Dezelfde race in de walk-forward-backtest, met features op alle data.
    data = build_features(
        pd.concat(
            [pd.read_parquet(path) for path in sorted(raw_dir.glob("*.parquet"))],
            ignore_index=True,
        )
    )
    ordered = data.sort_values(["season", "round"], kind="stable")
    is_race = (ordered["season"] == 2023) & (ordered["round"] == 3)
    before = (ordered["season"] < 2023) | ((ordered["season"] == 2023) & (ordered["round"] < 3))
    race = ordered.loc[is_race]
    fold = project.final_spec.predictor(project.features).fit(ordered.loc[before]).predict(race)
    assert _ranks(payload) == dict(zip(race["driver_id"], fold.tolist(), strict=True))

    backtest = run_backtest(data, {"model": project.final_spec.factory(project.features)}, 2)
    expected = backtest.set_index(["season", "round"]).loc[(2023, 3), METRICS]
    scores = score_race(race, race["driver_id"].map(_ranks(payload)))
    assert scores == pytest.approx(expected.to_dict())

    assert payload["result_known"] is True
    assert payload["model"]["training"]["last_race"] == {"season": 2023, "round": 2}
    assert payload["model"]["training"]["races"] == 5


def test_prediction_ignores_the_result_of_the_predicted_race(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    original = predict_race(2023, 3, _project(), raw_dir)

    path = raw_dir / "results_2023.parquet"
    raw = pd.read_parquet(path)
    race = raw["round"] == 3
    raw.loc[race, "finish_position"] = raw.loc[race, "finish_position"].to_numpy()[::-1]
    raw.loc[race, "points"] = 99.0
    raw.loc[race, "status"] = "Engine"
    raw.to_parquet(path, index=False)

    changed = predict_race(2023, 3, _project(), raw_dir)

    assert changed["predictions"] == original["predictions"]


def test_upcoming_race_uses_live_qualifying_including_new_drivers(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    drivers = ["d3", "rookie", "d0", "d1", "d2", "d4"]
    client = QualifyingClient([_qualifying_race(2024, 1, drivers)])

    payload = predict_race(2024, 1, _project(), raw_dir, client)

    assert client.paths == ["2024/1/qualifying"]
    assert payload["result_known"] is False
    assert payload["race"] == {
        "season": 2024,
        "round": 1,
        "race_name": "Volgende Grand Prix",
        "circuit_id": "circuit_1",
        "date": "2024-03-10",
    }
    rows = payload["predictions"]
    assert [row["predicted_position"] for row in rows] == list(range(1, 7))
    assert {row["driver_id"] for row in rows} == set(drivers)
    rookie = next(row for row in rows if row["driver_id"] == "rookie")
    assert rookie == {
        "predicted_position": rookie["predicted_position"],
        "driver_id": "rookie",
        "driver_code": "ROOKIE",
        "constructor_id": "team_0",
        "quali_position": 2,
    }
    assert payload["model"]["training"]["last_race"] == {"season": 2023, "round": 3}


@pytest.mark.parametrize(
    ("season", "round_", "min_train_races", "message"),
    [
        (2023, 5, 2, r"Ronde\(s\) \[4\] van 2023 ontbreken"),
        (2025, 1, 2, "loopt tot seizoen 2023"),
        (2023, 1, 4, "Te weinig eerdere races om op te trainen: 3, minimaal 4"),
    ],
)
def test_prediction_reports_missing_history_before_calling_the_api(
    tmp_path, season, round_, min_train_races, message
):
    raw_dir = _raw_dir(tmp_path)

    with pytest.raises(PredictionError, match=message):
        predict_race(
            season, round_, _project(min_train_races=min_train_races), raw_dir, UnusedClient()
        )


def test_upcoming_race_without_qualifying_cannot_be_predicted(tmp_path):
    raw_dir = _raw_dir(tmp_path)

    with pytest.raises(QualifyingNotAvailableError, match="voorspel na de kwalificatie"):
        predict_race(2024, 1, _project(), raw_dir, QualifyingClient([]))


# --- JSON-uitvoer ---


def _strict_json(text: str) -> dict:
    def reject(constant: str) -> None:
        raise AssertionError(f"Ongeldige JSON-constante: {constant}")

    return json.loads(text, parse_constant=reject)


def test_written_prediction_is_strict_json_with_the_contract_keys(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    payload = predict_race(2023, 3, _project(), raw_dir)

    path = write_prediction(payload, tmp_path / "predictions")

    assert path.name == "2023-03.json"
    saved = _strict_json(path.read_text(encoding="utf-8"))
    assert saved == payload
    assert saved["schema_version"] == 1
    assert set(saved) == {
        "schema_version",
        "generated_at",
        "race",
        "result_known",
        "prediction_moment",
        "model",
        "predictions",
        "run",
    }
    assert set(saved["run"]) == {"git_commit", "git_dirty", "package_version", "versions", "config"}
    assert saved["run"]["config"]["model"]["final"] == "xgb_ranker"


def test_latest_json_follows_the_chronologically_latest_race(tmp_path):
    raw_dir = _raw_dir(tmp_path)
    output_dir = tmp_path / "predictions"
    later = predict_race(2023, 3, _project(), raw_dir)
    earlier = predict_race(2023, 2, _project(), raw_dir)

    write_prediction(later, output_dir)
    write_prediction(earlier, output_dir)

    latest = _strict_json((output_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["race"]["round"] == 3
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "2023-02.json",
        "2023-03.json",
        "latest.json",
    ]


def test_unchanged_prediction_is_not_rewritten(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    payload = predict_race(2023, 3, _project(), _raw_dir(tmp_path))
    output_dir = tmp_path / "predictions"
    path = write_prediction(payload, output_dir)
    files = (path, output_dir / "latest.json")
    before = [file.stat().st_mtime_ns for file in files]

    # Een latere run: alleen tijdstip en herkomst verschillen.
    later = {**payload, "generated_at": "2099-01-01T00:00:00+00:00", "run": {"git_commit": "x"}}
    write_prediction(later, output_dir)

    assert [file.stat().st_mtime_ns for file in files] == before
    assert _strict_json(path.read_text(encoding="utf-8"))["generated_at"] == payload["generated_at"]
    assert "Voorspelling ongewijzigd" in caplog.text


@pytest.mark.parametrize(
    ("existing_known", "new_known", "rewritten"),
    [
        (False, True, False),  # een retrospectieve voorspelling vervangt nooit die van vooraf
        (False, False, True),  # bv. een gecorrigeerde kwalificatie vóór de race
        (True, True, True),
        (True, False, True),
    ],
)
def test_prediction_overwrite_rules(tmp_path, caplog, existing_known, new_known, rewritten):
    payload = predict_race(2023, 3, _project(), _raw_dir(tmp_path))
    output_dir = tmp_path / "predictions"
    write_prediction({**payload, "result_known": existing_known}, output_dir)
    changed = {**payload, "result_known": new_known, "predictions": payload["predictions"][::-1]}

    path = write_prediction(changed, output_dir)

    saved = _strict_json(path.read_text(encoding="utf-8"))
    assert (saved["predictions"] == changed["predictions"]) is rewritten
    assert saved["result_known"] is (new_known if rewritten else existing_known)
    assert _strict_json((output_dir / "latest.json").read_text(encoding="utf-8")) == saved
    assert ("blijft staan" in caplog.text) is not rewritten


def test_write_json_if_changed_ignores_generated_at_and_run(tmp_path):
    path = tmp_path / "uitvoer.json"
    first = {"schema_version": 1, "generated_at": "2026-11-28T18:17:00+00:00", "value": 1.5}
    first["run"] = {"git_commit": "a"}

    assert write_json_if_changed(path, first) is True
    text = path.read_text(encoding="utf-8")
    later = {**first, "generated_at": "2026-11-28T21:17:00+00:00", "run": {"git_commit": "b"}}
    assert write_json_if_changed(path, later) is False
    assert path.read_text(encoding="utf-8") == text
    assert write_json_if_changed(path, {**later, "value": float("nan")}) is True
    assert read_json(path)["value"] is None
    assert read_json(path)["run"] == {"git_commit": "b"}


def test_write_json_if_changed_overwrites_an_unreadable_file(tmp_path, caplog):
    path = tmp_path / "uitvoer.json"
    path.write_text("{kapot", encoding="utf-8")

    assert write_json_if_changed(path, {"value": 1}) is True
    assert read_json(path) == {"value": 1}
    assert "geen geldige JSON" in caplog.text
    path.write_text("[1, 2]", encoding="utf-8")
    assert read_json(path) is None
    assert read_json(tmp_path / "ontbreekt.json") is None


def test_json_ready_turns_missing_values_into_null():
    value = {
        "nan": float("nan"),
        "inf": math.inf,
        "na": pd.NA,
        "nat": pd.NaT,
        "numpy": [np.int64(3), np.float64(0.5), np.bool_(True)],
        "date": date(2026, 11, 29),
        "tuple": (1, "a"),
    }

    assert json_ready(value) == {
        "nan": None,
        "inf": None,
        "na": None,
        "nat": None,
        "numpy": [3, 0.5, True],
        "date": "2026-11-29",
        "tuple": [1, "a"],
    }
    with pytest.raises(TypeError, match="object"):
        json_ready(object())


def test_format_prediction_lists_drivers_in_predicted_order(tmp_path):
    payload = predict_race(2023, 3, _project(), _raw_dir(tmp_path))

    lines = format_prediction(payload).splitlines()

    assert lines[0].startswith("Race 2023-3 2023 (ronde 3)")
    first = payload["predictions"][0]
    assert lines[2].split() == [
        "1",
        first["driver_code"],
        first["constructor_id"],
        str(first["quali_position"]),
    ]
    assert lines[-1].startswith("Retrospectief")


# --- Herkomst ---


def test_run_metadata_records_config_and_survives_missing_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)

    metadata = runinfo.run_metadata(load_config())

    assert metadata["git_commit"] is None
    assert metadata["git_dirty"] is None
    assert metadata["config"] == load_config().to_dict()
    assert set(metadata["versions"]) == {"python", "xgboost", "scikit-learn", "pandas"}


# --- CLI ---


def test_cli_predict_writes_json_and_prints_table(tmp_path, monkeypatch):
    payload = predict_race(2023, 3, _project(), _raw_dir(tmp_path))
    calls = []

    def fake_predict(season, round_number, project):
        calls.append((season, round_number, project.final_spec.name))
        return payload

    monkeypatch.setattr(cli, "predict_race", fake_predict)
    monkeypatch.setattr(
        cli, "write_prediction", lambda data: write_prediction(data, tmp_path / "out")
    )

    result = CliRunner().invoke(cli.app, ["predict", "--season", "2023", "--round", "3"])

    assert result.exit_code == 0, result.output
    assert calls == [(2023, 3, "xgb_ranker_d2_n50_all")]
    assert result.stdout.splitlines()[0].startswith("Race 2023-3")
    assert (tmp_path / "out" / "2023-03.json").exists()


def test_cli_predict_reports_missing_qualifying(monkeypatch, caplog):
    def no_qualifying(season, round_number, project):
        raise QualifyingNotAvailableError("Nog geen kwalificatie-uitslag voor 2026 ronde 22")

    monkeypatch.setattr(cli, "predict_race", no_qualifying)

    result = CliRunner().invoke(cli.app, ["predict", "--season", "2026", "--round", "22"])

    assert result.exit_code == 1
    assert "Nog geen kwalificatie" in caplog.text


@pytest.mark.parametrize(
    "error",
    [JolpicaError("Mislukt na 5 pogingen"), requests.ConnectionError("geen netwerk")],
)
def test_cli_predict_reports_unreachable_api(monkeypatch, caplog, error):
    def unreachable(season, round_number, project):
        raise error

    monkeypatch.setattr(cli, "predict_race", unreachable)

    result = CliRunner().invoke(cli.app, ["predict", "--season", "2026", "--round", "22"])

    assert result.exit_code == 1
    assert "Jolpica-API niet bereikbaar" in caplog.text
    assert str(error) in caplog.text


def test_cli_predict_requires_season_and_round():
    result = CliRunner().invoke(cli.app, ["predict", "--season", "2026"])

    assert result.exit_code != 0


def _fake_prediction(payload: dict, calls: list):
    def fake_predict(season, round_number, project):
        calls.append((season, round_number))
        return payload

    return fake_predict


def test_cli_predict_next_predicts_the_race_from_the_calendar(tmp_path, monkeypatch):
    payload = predict_race(2023, 3, _project(), _raw_dir(tmp_path))
    calls = []
    monkeypatch.setattr(cli, "race_to_predict", lambda: (2023, 3))
    monkeypatch.setattr(cli, "predict_race", _fake_prediction(payload, calls))
    monkeypatch.setattr(
        cli, "write_prediction", lambda data: write_prediction(data, tmp_path / "out")
    )

    result = CliRunner().invoke(cli.app, ["predict", "--next"])

    assert result.exit_code == 0, result.output
    assert calls == [(2023, 3)]
    assert (tmp_path / "out" / "2023-03.json").exists()


def test_cli_predict_next_without_a_race_to_predict_does_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "race_to_predict", lambda: None)
    monkeypatch.setattr(cli, "predict_race", _fake_prediction({}, calls))

    result = CliRunner().invoke(cli.app, ["predict", "--next"])

    assert result.exit_code == 0, result.output
    assert calls == []
    assert result.stdout == ""


def test_cli_predict_next_waits_while_jolpica_has_no_qualifying(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    def no_qualifying(season, round_number, project):
        raise QualifyingNotAvailableError("Nog geen kwalificatie-uitslag voor 2026 ronde 22")

    monkeypatch.setattr(cli, "race_to_predict", lambda: (2026, 22))
    monkeypatch.setattr(cli, "predict_race", no_qualifying)

    result = CliRunner().invoke(cli.app, ["predict", "--next"])

    # Bij een geplande run is dat geen fout: een latere run probeert het opnieuw.
    assert result.exit_code == 0, result.output
    assert "2026 ronde 22 staat nog niet in Jolpica" in caplog.text


def test_cli_predict_next_reports_an_unreachable_calendar(monkeypatch, caplog):
    def unreachable():
        raise JolpicaError("Mislukt na 5 pogingen")

    monkeypatch.setattr(cli, "race_to_predict", unreachable)

    result = CliRunner().invoke(cli.app, ["predict", "--next"])

    assert result.exit_code == 1
    assert "Jolpica-API niet bereikbaar" in caplog.text


@pytest.mark.parametrize(
    "args",
    [
        ["predict"],
        ["predict", "--round", "3"],
        ["predict", "--next", "--season", "2026"],
        ["predict", "--next", "--season", "2026", "--round", "22"],
    ],
)
def test_cli_predict_takes_either_next_or_season_and_round(args, monkeypatch):
    monkeypatch.setattr(cli, "race_to_predict", lambda: pytest.fail("geen kalender verwacht"))

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 2

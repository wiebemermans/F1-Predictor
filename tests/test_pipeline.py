"""Integratietest: de volledige CLI-pipeline op een mini-dataset van drie gereden races.

`f1 fetch` -> `features` -> `backtest` (alle suites) -> `train` -> `predict`, zoals een nieuwe
gebruiker ze uitvoert (zie docs/usage.md). Jolpica wordt vervangen door de fixture
`tests/fixtures/jolpica_mini.json` en alle paden wijzen naar een tijdelijke map; verder draait
de echte code. De pipeline loopt één keer per module, de tests controleren elk een stap.
"""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from f1_predictor import cli
from f1_predictor.config import ProjectConfig, load_config
from f1_predictor.data import fetch as fetch_module
from f1_predictor.data.fetch import fetch_seasons
from f1_predictor.data.jolpica import PAGE_LIMIT, JolpicaClient
from f1_predictor.evaluation.backtest import run_baseline_backtest, run_feature_backtest
from f1_predictor.evaluation.model_selection import run_model_backtest, write_comparison_json
from f1_predictor.features.build import feature_columns, write_feature_table
from f1_predictor.models.predict import predict_race, write_prediction
from f1_predictor.models.train import train_final_model
from f1_predictor.paths import ROOT

FIXTURE = Path(__file__).parent / "fixtures" / "jolpica_mini.json"
CONFIG = {
    "data": {"start_season": 2023, "end_season": 2024},
    "model": {
        "final": "xgb_ranker",
        "tuned": {
            "ridge": {"feature_set": "upto_championship", "params": {"alpha": 1.0}},
            "xgb_ranker": {"feature_set": "all", "params": {"max_depth": 2, "n_estimators": 5}},
        },
    },
    "backtest": {"min_train_races": 1},
}
COMMANDS = [
    ["fetch"],
    ["features"],
    ["backtest", "--suite", "baselines"],
    ["backtest", "--suite", "features"],
    ["backtest"],
    ["train"],
    ["predict", "--season", "2024", "--round", "1"],  # al gereden: retrospectieve voorspelling
    ["predict", "--season", "2024", "--round", "2"],  # na de kwalificatie, vóór de race
]
# Met `min_train_races: 1` is 2023-1 alleen trainingshistorie.
TEST_RACES = [(2023, 2), (2024, 1)]
CONTRACT_KEYS = {
    "schema_version",
    "generated_at",
    "race",
    "result_known",
    "prediction_moment",
    "model",
    "predictions",
    "run",
}


class FixtureClient(JolpicaClient):
    """Jolpica zonder netwerk: `get` levert pagina's uit de fixture, `races` is de echte code."""

    def __init__(self, races_per_path: dict[str, list[dict]]) -> None:
        self._races = races_per_path
        self.paths: list[str] = []

    def get(self, path: str, **params) -> dict:
        self.paths.append(path)
        races = self._races.get(path, [])
        offset = params.get("offset", 0)
        limit = params.get("limit", PAGE_LIMIT)
        return {"total": str(len(races)), "RaceTable": {"Races": races[offset : offset + limit]}}


@dataclass
class PipelineRun:
    raw_dir: Path
    processed_dir: Path
    output_dir: Path
    project: ProjectConfig
    client: FixtureClient
    stdout: dict[str, str]
    repo_before: dict[str, tuple[int, int]]
    repo_after: dict[str, tuple[int, int]]


def _snapshot(*directories: Path) -> dict[str, tuple[int, int]]:
    """Grootte en wijzigingstijd van elk bestand, zodat elke schrijfactie opvalt."""
    files = {}
    for directory in directories:
        for path in sorted(directory.rglob("*")) if directory.exists() else []:
            if path.is_file():
                stat = path.stat()
                files[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return files


def _strict_json(path: Path) -> dict:
    def reject(constant: str) -> None:
        raise AssertionError(f"Ongeldige JSON-constante in {path.name}: {constant}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> PipelineRun:
    root = tmp_path_factory.mktemp("project")
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    output_dir = root / "output"
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    client = FixtureClient(json.loads(FIXTURE.read_text(encoding="utf-8")))

    # De CLI roept deze functies aan zonder paden; hier krijgen ze de tijdelijke map mee.
    redirects = {
        "fetch_seasons": partial(fetch_seasons, client=client),
        "write_feature_table": partial(
            write_feature_table, raw_dir=raw_dir, output_path=processed_dir / "features.parquet"
        ),
        "run_baseline_backtest": partial(
            run_baseline_backtest,
            data_dir=raw_dir,
            output_path=output_dir / "backtest_baselines.csv",
        ),
        "run_feature_backtest": partial(
            run_feature_backtest, data_dir=raw_dir, output_path=output_dir / "backtest_features.csv"
        ),
        "run_model_backtest": partial(
            run_model_backtest,
            data_dir=raw_dir,
            output_path=output_dir / "backtest_models.csv",
            summary_path=output_dir / "model_comparison.csv",
        ),
        "write_comparison_json": partial(
            write_comparison_json, output_path=output_dir / "model_comparison.json"
        ),
        "train_final_model": partial(
            train_final_model,
            data_dir=raw_dir,
            model_dir=output_dir / "models",
            importance_path=output_dir / "model_feature_importance.csv",
        ),
        "predict_race": partial(predict_race, data_dir=raw_dir, client=client),
        "write_prediction": partial(write_prediction, output_dir=output_dir / "predictions"),
    }
    # Vangnet: mist er een omleiding, dan schrijft de pipeline in de repo en faalt de test.
    repo_dirs = (ROOT / "data", ROOT / "output")
    repo_before = _snapshot(*repo_dirs)
    stdout = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(fetch_module, "RAW_DIR", raw_dir)
        for name, function in redirects.items():
            monkeypatch.setattr(cli, name, function)
        runner = CliRunner()
        for args in COMMANDS:
            result = runner.invoke(cli.app, ["--config", str(config_path), *args])
            assert result.exit_code == 0, (
                f"`f1 {' '.join(args)}` faalde:\n{result.output}\n{result.exception!r}"
            )
            stdout[" ".join(args)] = result.stdout

    return PipelineRun(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=output_dir,
        project=load_config(config_path),
        client=client,
        stdout=stdout,
        repo_before=repo_before,
        repo_after=_snapshot(*repo_dirs),
    )


def test_pipeline_leaves_repository_data_and_output_untouched(pipeline):
    assert pipeline.repo_after == pipeline.repo_before


def test_fetch_stores_raw_results_with_special_cases_untouched(pipeline):
    assert sorted(path.name for path in pipeline.raw_dir.iterdir()) == [
        "results_2023.parquet",
        "results_2024.parquet",
    ]
    raw = pd.concat(map(pd.read_parquet, sorted(pipeline.raw_dir.iterdir())), ignore_index=True)
    rows = raw.set_index(["season", "round", "driver_id"])

    # Alleen gereden races: de kwalificatie van 2024-2 levert nog geen rijen op.
    assert sorted(set(zip(raw["season"], raw["round"], strict=True))) == [
        (2023, 1),
        (2023, 2),
        (2024, 1),
    ]
    assert len(raw) == 18
    # Pitlane-start: grid 0 blijft staan, de kwalificatiepositie wordt apart bewaard.
    assert rows.loc[(2023, 1, "fay"), "grid"] == 0
    assert rows.loc[(2023, 1, "fay"), "quali_position"] == 6
    assert pd.isna(rows.loc[(2023, 2, "dex"), "quali_position"])
    assert rows.loc[(2023, 1, "dex"), ["position_text", "status"]].tolist() == ["R", "Engine"]
    # Beide statusvocabulaires blijven zoals de API ze levert.
    assert set(raw.loc[raw["season"] == 2023, "status"]) == {"Finished", "+1 Lap", "Engine"}
    assert set(raw.loc[raw["season"] == 2024, "status"]) == {"Finished", "Lapped", "Retired"}
    assert rows.loc[(2023, 2, "ben"), "sprint_points"] == 8.0
    assert (rows.drop(index=(2023, 2))["sprint_points"] == 0.0).all()


def test_features_use_qualifying_position_and_earlier_races(pipeline):
    features = pd.read_parquet(pipeline.processed_dir / "features.parquet")
    rows = features.set_index(["season", "round", "driver_id"])

    assert len(features) == 18
    assert set(feature_columns(pipeline.project.features)) <= set(features.columns)
    # Startpositie uit de kwalificatie (P6), niet de definitieve grid (pitlane, 0).
    assert rows.loc[(2023, 1, "fay"), "start_position"] == 6.0
    # Zonder kwalificatie: achteraan het veld van zes, met een vlag.
    assert rows.loc[(2023, 2, "dex"), ["start_position", "start_quali_missing"]].tolist() == [6, 1]
    # Circuithistoriek over de seizoensgrens heen: alpha_ring is één keer eerder gereden.
    assert rows.loc[(2023, 1, "ada"), "circuit_prior_races"] == 0
    assert rows.loc[(2024, 1, "ada"), "circuit_prior_races"] == 1


def test_backtests_score_every_model_on_the_same_races(pipeline):
    for name in ("backtest_baselines.csv", "backtest_features.csv", "backtest_models.csv"):
        results = pd.read_csv(pipeline.output_dir / name)
        for model, rows in results.groupby("model"):
            races = list(zip(rows["season"], rows["round"], strict=True))
            assert races == TEST_RACES, f"{name}: {model}"

    tuned = {spec.name for spec in pipeline.project.model.tuned.values()}
    expected_models = {"qualifying_position", "championship_points", *tuned}
    models = pd.read_csv(pipeline.output_dir / "backtest_models.csv")
    assert set(models["model"]) == expected_models

    comparison = _strict_json(pipeline.output_dir / "model_comparison.json")
    assert comparison["final_model"] == pipeline.project.final_spec.name
    # Alle testraces liggen na 2021: geen validatieperiode, wel holdout en alles.
    assert set(comparison["periods"]) == {"holdout", "all"}
    assert {row["model"] for row in comparison["periods"]["all"]} == expected_models


def test_train_writes_model_metadata_and_feature_importance(pipeline):
    spec = pipeline.project.final_spec
    model_dir = pipeline.output_dir / "models"

    assert sorted(path.name for path in model_dir.iterdir()) == [
        f"{spec.name}.joblib",
        f"{spec.name}.json",
    ]
    metadata = _strict_json(model_dir / f"{spec.name}.json")
    assert metadata["training"]["races"] == 3
    assert metadata["training"]["rows"] == 18
    importance = pd.read_csv(pipeline.output_dir / "model_feature_importance.csv")
    assert set(importance["model"]) == {spec.name}
    assert importance["feature"].tolist() == metadata["feature_columns"]


def test_predict_writes_contract_json_for_a_past_and_an_upcoming_race(pipeline):
    predictions_dir = pipeline.output_dir / "predictions"
    assert sorted(path.name for path in predictions_dir.iterdir()) == [
        "2024-01.json",
        "2024-02.json",
        "latest.json",
    ]
    past = _strict_json(predictions_dir / "2024-01.json")
    upcoming = _strict_json(predictions_dir / "2024-02.json")

    for payload in (past, upcoming):
        assert set(payload) == CONTRACT_KEYS
        assert payload["model"]["name"] == pipeline.project.final_spec.name
        ranks = [row["predicted_position"] for row in payload["predictions"]]
        assert ranks == list(range(1, 7))

    # Retrospectief: getraind op de twee races ervóór, niet op de race zelf.
    assert past["result_known"] is True
    assert past["model"]["training"]["last_race"] == {"season": 2023, "round": 2}
    assert past["model"]["training"]["races"] == 2
    # Komende race: deelnemers uit de live kwalificatie, inclusief rookie gus zonder historie.
    assert upcoming["result_known"] is False
    assert "2024/2/qualifying" in pipeline.client.paths
    assert upcoming["race"] == {
        "season": 2024,
        "round": 2,
        "race_name": "Bravo Grand Prix",
        "circuit_id": "bravo_park",
        "date": "2024-03-17",
    }
    drivers = {row["driver_id"] for row in upcoming["predictions"]}
    assert drivers == {"ada", "ben", "cora", "dex", "eli", "gus"}
    assert upcoming["model"]["training"]["races"] == 3

    assert _strict_json(predictions_dir / "latest.json") == upcoming
    table = pipeline.stdout["predict --season 2024 --round 2"]
    assert table.splitlines()[0].startswith("Bravo Grand Prix 2024 (ronde 2)")
    assert "GUS" in table

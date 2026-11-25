import copy
import json
from datetime import date

import pytest
import yaml
from typer.testing import CliRunner

from f1_predictor import cli
from f1_predictor.config import (
    BacktestConfig,
    ConfigError,
    DataConfig,
    load_config,
    parse_config,
)
from f1_predictor.features.build import FeatureConfig

MODEL = {
    "final": "xgb_ranker",
    "tuned": {
        "ridge": {"feature_set": "upto_championship", "params": {"alpha": 10.0}},
        "xgb_ranker": {"feature_set": "all", "params": {"max_depth": 2, "n_estimators": 50}},
    },
}


def _raw(**sections) -> dict:
    return {"model": copy.deepcopy(MODEL), **sections}


def _write(tmp_path, raw) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


# --- Inlezen en defaults ---


def test_committed_config_loads_with_phase7_choices():
    config = load_config()

    assert config.final_spec.name == "xgb_ranker_d2_n50_all"
    assert list(config.model.tuned) == ["ridge", "xgb_regressor", "xgb_ranker", "xgb_ranker_ndcg"]
    assert config.features == FeatureConfig()
    assert config.backtest.min_train_races == 20
    assert config.data == DataConfig(start_season=2014, end_season=None)


def test_missing_sections_get_defaults_and_lists_become_tuples():
    config = parse_config(_raw(features={"form_windows": [2, 4]}))

    assert config.data == DataConfig()
    assert config.backtest == BacktestConfig()
    assert config.features.form_windows == (2, 4)
    assert config.features.groups == FeatureConfig().groups
    assert config.model.tuned["ridge"].params == {"alpha": 10.0}


def test_to_dict_is_json_and_reads_back_to_the_same_config():
    config = load_config()

    text = json.dumps(config.to_dict())

    assert parse_config(json.loads(text)) == config


def test_end_season_defaults_to_current_year():
    assert DataConfig().resolved_end_season(today=date(2027, 2, 1)) == 2027
    assert DataConfig(end_season=2025).resolved_end_season(today=date(2027, 2, 1)) == 2025


# --- Validatie ---


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (None, "leeg"),
        ({"data": {}}, "verplichte sectie model"),
        (_raw(extra={}), "Onbekende sleutel"),
        (_raw(data={"start": 2014}), "Onbekende sleutel.*data"),
        (_raw(data=5), "data moet een sectie"),
        (_raw(data={"start_season": 2020, "end_season": 2019}), "end_season"),
        (_raw(data={"start_season": 1900}), "start_season"),
        (_raw(backtest={"min_train_races": -1}), "min_train_races"),
        (_raw(features={"groups": ["onbekend"]}), "Onbekende featuregroepen"),
        (_raw(features={"groups": ["starting_position"]}), "features.groups mist"),
        ({"model": {"tuned": MODEL["tuned"]}}, "verplichte sleutel: final"),
        ({"model": {"final": "lasso", "tuned": MODEL["tuned"]}}, "final .*lasso"),
        ({"model": {"final": "x", "tuned": {"lasso": {}}}}, "Onbekende modelfamilie.*lasso"),
        (
            {"model": {"final": "xgb_ranker", "tuned": {"xgb_ranker": {"set": "all"}}}},
            "Onbekende sleutel.*model.tuned.xgb_ranker",
        ),
    ],
)
def test_invalid_config_is_rejected_with_the_offending_key(raw, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


@pytest.mark.parametrize(
    ("family", "params", "message"),
    [
        ("xgb_ranker", {"max_dept": 2}, "Onbekende XGBoost-parameter.*max_dept"),
        ("xgb_ranker", {"subsample": 0.8}, "subsample.*niet reproduceerbaar"),
        ("xgb_regressor", {"objective": "reg:absoluteerror"}, "objective"),
        ("ridge", {"max_depth": 2}, "ridge: max_depth"),
        ("xgb_ranker", {"max_depth": [2, 3]}, "enkel getal"),
    ],
)
def test_invalid_model_parameters_are_rejected(family, params, message):
    raw = {"model": {"final": family, "tuned": {family: {"feature_set": "all", "params": params}}}}

    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_load_config_reports_missing_file_and_invalid_yaml(tmp_path):
    with pytest.raises(ConfigError, match="niet gevonden"):
        load_config(tmp_path / "ontbreekt.yaml")

    broken = tmp_path / "kapot.yaml"
    broken.write_text("model: [niet afgesloten", encoding="utf-8")
    with pytest.raises(ConfigError, match="geen geldige YAML"):
        load_config(broken)


# --- CLI met --config ---


def test_cli_stops_with_readable_error_for_invalid_config(tmp_path, caplog):
    path = _write(tmp_path, _raw(backtest={"min_train_race": 20}))

    result = CliRunner().invoke(cli.app, ["--config", path, "features"])

    assert result.exit_code == 1
    assert "min_train_race" in caplog.text


def test_cli_stops_when_config_file_is_missing(tmp_path, caplog):
    result = CliRunner().invoke(cli.app, ["--config", str(tmp_path / "x.yaml"), "features"])

    assert result.exit_code == 1
    assert "niet gevonden" in caplog.text


def test_cli_fetch_uses_configured_seasons(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "fetch_seasons", lambda start, end, force: calls.append((start, end, force)) or []
    )
    path = _write(tmp_path, _raw(data={"start_season": 2020, "end_season": 2021}))

    default = CliRunner().invoke(cli.app, ["--config", path, "fetch"])
    override = CliRunner().invoke(cli.app, ["--config", path, "fetch", "--end", "2020"])

    assert default.exit_code == override.exit_code == 0
    assert calls == [(2020, 2021, False), (2020, 2020, False)]


def test_cli_fetch_fails_when_a_season_fails(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cli, "fetch_seasons", lambda start, end, force: [2021])
    path = _write(tmp_path, _raw(data={"start_season": 2020, "end_season": 2021}))

    result = CliRunner().invoke(cli.app, ["--config", path, "fetch"])

    assert result.exit_code == 1
    assert "2021" in caplog.text


def test_cli_fetch_rejects_start_after_end(monkeypatch):
    monkeypatch.setattr(cli, "fetch_seasons", lambda *args, **kwargs: pytest.fail("geen fetch"))

    result = CliRunner().invoke(cli.app, ["fetch", "--start", "2025", "--end", "2024"])

    assert result.exit_code == 2  # ongeldige parameter, zoals bij een typfout in de opties


@pytest.mark.parametrize(
    ("args", "function"),
    [
        (["backtest", "--suite", "baselines"], "run_baseline_backtest"),
        (["backtest", "--suite", "features"], "run_feature_backtest"),
        (["backtest"], "run_model_backtest"),
        (["tune"], "run_model_tuning"),
        (["train"], "train_final_model"),
        (["predict", "--season", "2026", "--round", "1"], "predict_race"),
        (["evaluate"], "evaluate_predictions"),
    ],
)
def test_cli_points_to_fetch_when_raw_data_is_missing(args, function, monkeypatch, caplog):
    def missing_data(*_args, **_kwargs):
        raise FileNotFoundError("Geen results_*.parquet-bestanden gevonden in data/raw")

    monkeypatch.setattr(cli, function, missing_data)

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 1
    assert cli.MISSING_DATA_HINT in caplog.text

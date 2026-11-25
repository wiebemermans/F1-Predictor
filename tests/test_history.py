"""Voorspellingen van vóór de race tegen de uitslag: `f1 evaluate` en prediction_history.json."""

import json
import logging
from functools import partial

import pandas as pd
import pytest
from typer.testing import CliRunner

from f1_predictor import cli
from f1_predictor.evaluation.history import (
    evaluate_predictions,
    format_history,
    score_prediction,
    write_prediction_history,
)
from f1_predictor.output import write_json

RUN = {"git_commit": "abc"}


def _prediction(
    season: int,
    round_: int,
    order: list[str],
    quali: dict[str, int | None],
    result_known: bool = False,
    model: str = "xgb_ranker_d2_n50_all",
) -> dict:
    """Een voorspelling zoals `f1 predict` die bewaart (alleen de velden die tellen)."""
    return {
        "schema_version": 1,
        "generated_at": f"{season}-03-0{round_}T18:17:00+00:00",
        "race": {"season": season, "round": round_, "race_name": f"Race {round_}"},
        "result_known": result_known,
        "model": {"name": model},
        "predictions": [
            {
                "predicted_position": position,
                "driver_id": driver,
                "driver_code": driver.upper(),
                "constructor_id": "team",
                "quali_position": quali[driver],
            }
            for position, driver in enumerate(order, start=1)
        ],
        "run": RUN,
    }


def _result(season: int, round_: int, order: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": season,
            "round": round_,
            "driver_id": order,
            "finish_position": range(1, len(order) + 1),
            "position_text": [str(position) for position in range(1, len(order) + 1)],
        }
    )


def _raw_dir(tmp_path, results: list[pd.DataFrame]):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.concat(results, ignore_index=True).to_parquet(raw_dir / "results_2026.parquet")
    return raw_dir


def _save(predictions_dir, payload: dict) -> None:
    race = payload["race"]
    write_json(predictions_dir / f"{race['season']}-{race['round']:02d}.json", payload)


QUALI = {"a": 1, "b": 2, "c": 3, "d": 4}


def test_score_prediction_uses_only_drivers_in_prediction_and_result():
    # Voorspeld b, a, c, d (kwalificatie a, b, c, d); uitslag b, a, e, c. d viel weg vóór de
    # race, e reed zonder kwalificatieresultaat: alleen a, b en c tellen, hernummerd tot 1..3.
    prediction = _prediction(2026, 1, ["b", "a", "c", "d"], QUALI)

    scored = score_prediction(prediction, _result(2026, 1, ["b", "a", "e", "c"]))

    assert scored["drivers_scored"] == 3
    assert scored["not_in_result"] == ["d"]
    assert scored["not_in_prediction"] == ["e"]
    model = scored["scores"]["prediction"]
    assert model["mae_positions"] == pytest.approx(0.0)
    assert model["spearman"] == pytest.approx(1.0)
    assert (model["winner_hit"], model["podium_overlap"]) == (1, 3)
    # Kwalificatievolgorde a, b, c tegen officieel b, a, c (c was P4, maar e telt niet).
    reference = scored["scores"]["qualifying_position"]
    assert reference["mae_positions"] == pytest.approx(2 / 3)
    assert reference["spearman"] == pytest.approx(0.5)
    assert (reference["winner_hit"], reference["podium_overlap"]) == (0, 3)
    # De rijen tonen de echte posities, niet de hernummerde.
    rows = {row["driver_id"]: row for row in scored["drivers"]}
    assert rows["c"]["finish_position"] == 4
    assert [row["driver_id"] for row in scored["drivers"]] == ["b", "a", "c"]


def test_score_prediction_without_common_drivers_returns_none(caplog):
    prediction = _prediction(2026, 1, ["a", "b"], QUALI)

    assert score_prediction(prediction, _result(2026, 1, ["x", "y"])) is None
    assert "Geen enkele coureur" in caplog.text


def test_evaluate_predictions_scores_only_pre_race_predictions_with_a_result(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    raw_dir = _raw_dir(tmp_path, [_result(2026, 1, ["a", "b", "c"]), _result(2026, 2, ["c"])])
    predictions_dir = tmp_path / "predictions"
    _save(predictions_dir, _prediction(2026, 1, ["b", "a", "c"], QUALI))
    _save(predictions_dir, _prediction(2026, 2, ["c"], QUALI, result_known=True))
    _save(predictions_dir, _prediction(2026, 3, ["a", "b"], QUALI))  # nog niet gereden
    (predictions_dir / "2026-04.json").write_text("{kapot", encoding="utf-8")
    write_json(predictions_dir / "latest.json", _prediction(2026, 3, ["a", "b"], QUALI))

    payload = evaluate_predictions(predictions_dir, raw_dir, run=RUN)

    assert [entry["race"]["round"] for entry in payload["races"]] == [1]
    entry = payload["races"][0]
    assert entry["model"] == "xgb_ranker_d2_n50_all"
    assert entry["prediction_generated_at"] == "2026-03-01T18:17:00+00:00"
    assert payload["reference"] == "qualifying_position"
    summary = {row["model"]: row for row in payload["summary"]}
    assert summary["prediction"]["races"] == 1
    # Voorspeld b, a, c; de kwalificatievolgorde a, b, c was hier exact juist.
    assert summary["prediction"]["delta_mae"] == pytest.approx(2 / 3)
    assert payload["run"] == RUN
    assert "2026 ronde 3: nog geen uitslag" in caplog.text
    assert "geen geldige JSON" in caplog.text


def test_evaluate_predictions_keeps_each_prediction_model_name(tmp_path):
    raw_dir = _raw_dir(tmp_path, [_result(2026, 1, ["a", "b"]), _result(2026, 2, ["b", "a"])])
    predictions_dir = tmp_path / "predictions"
    _save(predictions_dir, _prediction(2026, 1, ["a", "b"], QUALI, model="oud_model"))
    _save(predictions_dir, _prediction(2026, 2, ["a", "b"], QUALI, model="nieuw_model"))

    payload = evaluate_predictions(predictions_dir, raw_dir, run=RUN)

    assert [entry["model"] for entry in payload["races"]] == ["oud_model", "nieuw_model"]
    summary = {row["model"]: row for row in payload["summary"]}
    assert summary["prediction"]["races"] == 2
    assert summary["prediction"]["winner_hit"] == pytest.approx(0.5)


def test_history_without_evaluated_races_is_still_written(tmp_path):
    raw_dir = _raw_dir(tmp_path, [_result(2026, 1, ["a", "b"])])
    output_path = tmp_path / "out" / "prediction_history.json"

    payload = evaluate_predictions(tmp_path / "geen_voorspellingen", raw_dir, run=RUN)
    write_prediction_history(payload, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["races"] == []
    assert saved["summary"] == []
    assert format_history(saved).startswith("Nog geen voorspellingen")


def test_unchanged_history_is_not_rewritten(tmp_path):
    raw_dir = _raw_dir(tmp_path, [_result(2026, 1, ["a", "b"])])
    predictions_dir = tmp_path / "predictions"
    _save(predictions_dir, _prediction(2026, 1, ["a", "b"], QUALI))
    output_path = tmp_path / "prediction_history.json"
    write_prediction_history(evaluate_predictions(predictions_dir, raw_dir, run=RUN), output_path)
    before = output_path.stat().st_mtime_ns

    later = evaluate_predictions(predictions_dir, raw_dir, run={"git_commit": "def"})
    write_prediction_history({**later, "generated_at": "2099-01-01T00:00:00+00:00"}, output_path)

    assert output_path.stat().st_mtime_ns == before


def test_format_history_lists_each_race_and_the_average(tmp_path):
    raw_dir = _raw_dir(tmp_path, [_result(2026, 1, ["b", "a", "c"])])
    predictions_dir = tmp_path / "predictions"
    _save(predictions_dir, _prediction(2026, 1, ["b", "a", "c"], QUALI))

    lines = format_history(evaluate_predictions(predictions_dir, raw_dir, run=RUN)).splitlines()

    assert lines[0] == "Voorspellingen vooraf tegen de uitslag (1 races)"
    assert lines[2].split() == ["2026", "R1", "Race", "1", "0.00", "0.67", "goed", "3/3"]
    assert lines[-1].startswith("Gemiddeld: MAE 0.00 tegen 0.67")


def test_cli_evaluate_writes_history_and_prints_table(tmp_path, monkeypatch):
    raw_dir = _raw_dir(tmp_path, [_result(2026, 1, ["a", "b"])])
    predictions_dir = tmp_path / "predictions"
    _save(predictions_dir, _prediction(2026, 1, ["a", "b"], QUALI))
    output_path = tmp_path / "prediction_history.json"
    monkeypatch.setattr(
        cli,
        "evaluate_predictions",
        partial(evaluate_predictions, predictions_dir=predictions_dir, data_dir=raw_dir),
    )
    monkeypatch.setattr(
        cli, "write_prediction_history", partial(write_prediction_history, output_path=output_path)
    )

    result = CliRunner().invoke(cli.app, ["evaluate"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("Voorspellingen vooraf tegen de uitslag (1 races)")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run"]["config"]["model"]["final"] == "xgb_ranker"

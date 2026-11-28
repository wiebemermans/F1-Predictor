"""De top 3 van de laatste voorspelling vooraf in README.md (`f1 readme`)."""

import logging

import pytest
from typer.testing import CliRunner

from f1_predictor import cli
from f1_predictor.output import write_json
from f1_predictor.paths import ROOT
from f1_predictor.readme import (
    README_END,
    README_START,
    ReadmeError,
    latest_pre_race_prediction,
    update_readme,
)

PLACEHOLDER = "Nog geen voorspelling."
DRIVERS = [  # (code, team, kwalificatie) in voorspelde volgorde
    ("NOR", "mclaren", 2),
    ("VER", "red_bull", 1),
    ("PIA", "mclaren", None),
    ("LEC", "ferrari", 3),
]


def _readme(tmp_path, newline: str = "\n"):
    path = tmp_path / "README.md"
    lines = ["# F1 Predictor", "", README_START, PLACEHOLDER, README_END, "", "## Verder", ""]
    path.write_text(newline.join(lines), encoding="utf-8", newline="")
    return path


def _prediction(season=2026, round_=22, known=False, drivers=DRIVERS, name="Qatar Grand Prix"):
    return {
        "schema_version": 1,
        "generated_at": "2026-11-28T19:47:00+00:00",
        "race": {
            "season": season,
            "round": round_,
            "race_name": name,
            "circuit_id": "losail",
            "date": "2026-11-29",
        },
        "result_known": known,
        "model": {"name": "xgb_ranker"},
        # Bewust niet op volgorde: de top 3 volgt `predicted_position`.
        "predictions": [
            {
                "predicted_position": position,
                "driver_id": code.lower(),
                "driver_code": code,
                "constructor_id": team,
                "quali_position": quali,
            }
            for position, (code, team, quali) in reversed(list(enumerate(drivers, start=1)))
        ],
        "run": {"git_commit": "abc"},
    }


def _save(predictions_dir, payload):
    race = payload["race"]
    write_json(predictions_dir / f"{race['season']}-{race['round']:02d}.json", payload)


@pytest.fixture
def predictions_dir(tmp_path):
    return tmp_path / "output" / "predictions"


def test_readme_shows_the_top_three_with_a_link_to_the_json(tmp_path, predictions_dir):
    readme = _readme(tmp_path)
    _save(predictions_dir, _prediction())

    assert update_readme(readme, predictions_dir) is True

    text = readme.read_text(encoding="utf-8")
    block = text[text.index(README_START) : text.index(README_END)]
    assert "**Qatar Grand Prix 2026** (ronde 22, race op 2026-11-29)" in block
    assert "`xgb_ranker`" in block
    assert "| 1 | NOR | mclaren | P2 |" in block
    assert "| 2 | VER | red_bull | P1 |" in block
    assert "| 3 | PIA | mclaren | - |" in block  # zonder kwalificatieresultaat
    assert "LEC" not in block
    assert "[Volledige voorspelling](output/predictions/2026-22.json)" in block
    assert PLACEHOLDER not in text
    # De rest van de README blijft staan.
    assert text.startswith("# F1 Predictor\n\n")
    assert text.endswith(f"{README_END}\n\n## Verder\n")


def test_readme_is_not_rewritten_without_a_new_top_three(tmp_path, predictions_dir, caplog):
    caplog.set_level(logging.INFO)
    readme = _readme(tmp_path)
    _save(predictions_dir, _prediction())
    update_readme(readme, predictions_dir)
    before = readme.stat().st_mtime_ns

    # Een latere run met een herschreven voorspelling: alleen tijdstip, herkomst en
    # posities buiten de top 3 verschillen.
    later = _prediction(drivers=[*DRIVERS[:3], ("HAM", "ferrari", 4)])
    later["generated_at"] = "2026-11-28T21:17:00+00:00"
    _save(predictions_dir, later)

    assert update_readme(readme, predictions_dir) is False
    assert readme.stat().st_mtime_ns == before
    assert "niet herschreven" in caplog.text


def test_a_new_grand_prix_replaces_the_top_three(tmp_path, predictions_dir):
    readme = _readme(tmp_path)
    _save(predictions_dir, _prediction())
    update_readme(readme, predictions_dir)
    _save(predictions_dir, _prediction(round_=23, name="Abu Dhabi Grand Prix"))

    assert update_readme(readme, predictions_dir) is True

    text = readme.read_text(encoding="utf-8")
    assert "Abu Dhabi Grand Prix 2026" in text
    assert "Qatar" not in text
    assert text.count(README_START) == text.count(README_END) == 1


def test_a_corrected_pre_race_prediction_with_another_top_three_is_shown(tmp_path, predictions_dir):
    readme = _readme(tmp_path)
    _save(predictions_dir, _prediction())
    update_readme(readme, predictions_dir)
    # Bv. een diskwalificatie in de kwalificatie, nog vóór de race.
    _save(predictions_dir, _prediction(drivers=[DRIVERS[1], DRIVERS[0], DRIVERS[3]]))

    assert update_readme(readme, predictions_dir) is True
    assert "| 1 | VER | red_bull | P1 |" in readme.read_text(encoding="utf-8")


def test_retrospective_predictions_are_skipped(tmp_path, predictions_dir):
    _save(predictions_dir, _prediction(round_=21, name="Las Vegas Grand Prix"))
    _save(predictions_dir, _prediction(known=True))

    path, payload = latest_pre_race_prediction(predictions_dir)

    assert path.name == "2026-21.json"
    assert payload["race"]["race_name"] == "Las Vegas Grand Prix"


def test_without_a_pre_race_prediction_the_readme_is_untouched(tmp_path, predictions_dir):
    readme = _readme(tmp_path)
    before = readme.read_bytes()
    _save(predictions_dir, _prediction(known=True))

    assert update_readme(readme, predictions_dir) is False
    assert update_readme(readme, tmp_path / "bestaat_niet") is False
    assert readme.read_bytes() == before


def test_readme_keeps_windows_line_endings(tmp_path, predictions_dir):
    readme = _readme(tmp_path, newline="\r\n")
    _save(predictions_dir, _prediction())

    update_readme(readme, predictions_dir)

    content = readme.read_bytes()
    assert b"| 1 | NOR | mclaren | P2 |\r\n" in content
    assert content.count(b"\n") == content.count(b"\r\n")


@pytest.mark.parametrize(
    "lines",
    [
        ["# Zonder blok"],
        [README_START, README_END, README_START, README_END],
        [README_END, README_START],
    ],
)
def test_readme_without_exactly_one_block_is_an_error(tmp_path, predictions_dir, lines):
    readme = tmp_path / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    _save(predictions_dir, _prediction())

    with pytest.raises(ReadmeError, match="precies één blok"):
        update_readme(readme, predictions_dir)


def test_project_readme_has_one_prediction_block():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert text.count(README_START) == 1
    assert text.count(README_END) == 1
    assert text.index(README_START) < text.index(README_END)


def test_cli_readme_reports_a_missing_block(tmp_path, monkeypatch, caplog):
    readme = tmp_path / "README.md"
    readme.write_text("# Zonder blok\n", encoding="utf-8")
    monkeypatch.setattr(cli, "update_readme", lambda: update_readme(readme, tmp_path))

    result = CliRunner().invoke(cli.app, ["readme"])

    assert result.exit_code == 1
    assert "precies één blok" in caplog.text

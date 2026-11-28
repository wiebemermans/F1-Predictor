"""De voorspelde top 3 van de laatste race in README.md.

`f1 readme` vervangt het blok tussen `README_START` en `README_END` door de top 3 van de
chronologisch laatste voorspelling van vóór de race (`result_known: false`). Retrospectieve
voorspellingen tellen niet mee. Het bestand wordt alleen herschreven als het blok verandert:
bij de voorspelling van een nieuwe GP, of als een voorspelling vooraf een andere top 3 krijgt
(een gecorrigeerde kwalificatie). Een geplande run zonder nieuws maakt zo geen commit.
"""

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from f1_predictor.models.predict import saved_predictions
from f1_predictor.output import read_json, write_atomic
from f1_predictor.paths import PREDICTIONS_DIR, README_PATH

log = logging.getLogger(__name__)

README_START = "<!-- voorspelling:start -->"
README_END = "<!-- voorspelling:end -->"
TOP = 3
_BLOCK = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.DOTALL)


class ReadmeError(ValueError):
    """README.md heeft geen bruikbaar blok voor de voorspelling."""


def latest_pre_race_prediction(
    predictions_dir: Path = PREDICTIONS_DIR,
) -> tuple[Path, dict[str, Any]] | None:
    """Pad en inhoud van de chronologisch laatste voorspelling van vóór de race, of `None`."""
    for _, path in reversed(saved_predictions(predictions_dir)):
        payload = read_json(path)
        if payload is not None and payload.get("result_known") is False:
            return path, payload
    return None


def format_top(payload: Mapping[str, Any], link: str) -> list[str]:
    """De Markdown-regels voor de top 3 van één voorspelling, met een link naar de JSON."""
    race = payload["race"]
    rows = sorted(payload["predictions"], key=lambda row: row["predicted_position"])[:TOP]
    lines = [
        f"**{race['race_name']} {race['season']}** (ronde {race['round']}, race op "
        f"{race['date']}), voorspeld na de kwalificatie met model `{payload['model']['name']}`:",
        "",
        "| Pos | Coureur | Team | Kwalificatie |",
        "|---:|---|---|---:|",
    ]
    for row in rows:
        quali = "-" if row["quali_position"] is None else f"P{row['quali_position']}"
        driver = row["driver_code"] or row["driver_id"]
        lines.append(
            f"| {row['predicted_position']} | {driver} | {row['constructor_id']} | {quali} |"
        )
    lines += ["", f"[Volledige voorspelling]({link})"]
    return lines


def update_readme(readme_path: Path = README_PATH, predictions_dir: Path = PREDICTIONS_DIR) -> bool:
    """Zet de top 3 van de laatste voorspelling vooraf in README.md; `True` als er geschreven is.

    Zonder voorspelling vooraf blijft het bestand ongemoeid. De regeleinden van het bestand
    blijven behouden (CRLF in een Windows-checkout, LF in de workflow).
    """
    text = readme_path.read_text(encoding="utf-8", newline="")
    if text.count(README_START) != 1 or text.count(README_END) != 1 or not _BLOCK.search(text):
        raise ReadmeError(
            f"{readme_path} moet precies één blok tussen {README_START} en {README_END} bevatten"
        )
    latest = latest_pre_race_prediction(predictions_dir)
    if latest is None:
        log.info("Nog geen voorspelling van vóór een race; %s blijft ongewijzigd", readme_path)
        return False
    path, payload = latest
    newline = "\r\n" if "\r\n" in text else "\n"
    link = path.relative_to(readme_path.parent, walk_up=True).as_posix()
    block = newline.join([README_START, *format_top(payload, link), README_END])
    updated = _BLOCK.sub(lambda _: block, text)
    if updated == text:
        log.info("Top 3 in %s is al die van %s; niet herschreven", readme_path, path.name)
        return False
    write_atomic(
        readme_path,
        lambda temporary: temporary.write_text(updated, encoding="utf-8", newline=""),
    )
    log.info("Top 3 van %s in %s gezet", path.name, readme_path)
    return True

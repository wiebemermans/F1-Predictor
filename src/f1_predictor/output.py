"""Atomisch wegschrijven van uitvoerbestanden, ook als JSON die een website kan inlezen."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Tijdstip en herkomst van een run: ze veranderen bij elke run, ook als de inhoud gelijk blijft.
VOLATILE_KEYS = ("generated_at", "run")


def write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    """Schrijf eerst naar een tijdelijk bestand en hernoem dan: nooit een half bestand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    write(temporary_path)
    temporary_path.replace(path)


def json_ready(value: Any) -> Any:
    """Zet een waarde om naar gewone JSON-types; ontbrekende waarden (NaN, NA, NaT) worden `None`.

    Browsers kunnen `NaN` niet parsen, dus ontbrekende en oneindige getallen worden `null`.
    Datums worden ISO-tekst, numpy-getallen gewone Python-getallen.
    """
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    # Vóór de datumcontrole: `pd.NaT` is technisch ook een `datetime`.
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Waarde van type {type(value).__name__} kan niet naar JSON")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Schrijf `payload` als UTF-8-JSON met inspringing; `allow_nan=False` bewaakt geldige JSON."""
    text = json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False)
    write_atomic(path, lambda temporary: temporary.write_text(text + "\n", encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any] | None:
    """Een eerder geschreven JSON-object, of `None` als het bestand ontbreekt of onleesbaar is."""
    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        log.warning("%s is geen geldige JSON: %s", path, error)
        return None
    if not isinstance(content, dict):
        log.warning("%s bevat geen JSON-object", path)
        return None
    return content


def _content(payload: Mapping[str, Any], ignore: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in ignore}


def write_json_if_changed(
    path: Path, payload: Mapping[str, Any], ignore: tuple[str, ...] = VOLATILE_KEYS
) -> bool:
    """Schrijf `payload` alleen als de inhoud verschilt van het bestaande bestand.

    De sleutels in `ignore` (standaard tijdstip en herkomst) tellen niet mee. Een geplande run
    zonder nieuwe data laat het bestand dus ongemoeid en maakt geen lege commit; `generated_at`
    blijft dan het moment waarop deze inhoud voor het eerst ontstond. De vergelijking gebeurt
    op de ingelezen JSON, niet op de tekst (regeleinden verschillen tussen Windows en Linux).
    Een onleesbaar bestand telt als gewijzigd. Geeft `True` als er geschreven is.
    """
    new = json.loads(json.dumps(json_ready(payload), allow_nan=False))
    existing = read_json(path)
    if existing is not None and _content(existing, ignore) == _content(new, ignore):
        return False
    write_json(path, payload)
    return True

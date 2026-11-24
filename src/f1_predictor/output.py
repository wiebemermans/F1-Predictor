"""Atomisch wegschrijven van uitvoerbestanden, ook als JSON die een website kan inlezen."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

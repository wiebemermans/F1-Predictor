"""Centrale padconstanten."""

from pathlib import Path

# src/f1_predictor/paths.py -> repo-root (werkt met de editable install van uv)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"

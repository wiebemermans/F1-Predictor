"""Centrale padconstanten."""

from pathlib import Path

# src/f1_predictor/paths.py -> repo-root (werkt met de editable install van uv)
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

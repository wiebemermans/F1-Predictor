"""Herkomst van een run: tijdstip, git-commit, pakketversies en de gebruikte configuratie."""

from __future__ import annotations

import logging
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from f1_predictor.paths import ROOT

if TYPE_CHECKING:
    from f1_predictor.config import ProjectConfig

log = logging.getLogger(__name__)

LIBRARIES = ("xgboost", "scikit-learn", "pandas")


def utc_now() -> str:
    """Huidig tijdstip in UTC als ISO-tekst, op de seconde."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_revision(cwd: Path = ROOT) -> dict[str, Any] | None:
    """Huidige commit en of er niet-gecommitte wijzigingen zijn; `None` zonder git of repo.

    `dirty` telt ook niet-gevolgde bestanden mee: een nieuw, niet-gecommit bestand kan de
    uitkomst net zo goed beïnvloeden als een gewijzigd bestand.
    """
    try:
        commit = _git(["rev-parse", "HEAD"], cwd).strip()
        status = _git(["status", "--porcelain"], cwd)
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("Git-commit onbekend, wordt als null opgeslagen: %s", error)
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def _git(arguments: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
    )
    return completed.stdout


def library_versions() -> dict[str, str]:
    """Versies van Python en de bibliotheken die de uitkomst bepalen."""
    return {"python": platform.python_version(), **{name: version(name) for name in LIBRARIES}}


def package_version() -> str | None:
    try:
        return version("f1-predictor")
    except PackageNotFoundError:
        return None


def run_metadata(config: ProjectConfig) -> dict[str, Any]:
    """Herkomstblok voor JSON-uitvoer: commit, versies en de volledige configuratie."""
    git = git_revision()
    return {
        "git_commit": git["commit"] if git else None,
        "git_dirty": git["dirty"] if git else None,
        "package_version": package_version(),
        "versions": library_versions(),
        "config": config.to_dict(),
    }

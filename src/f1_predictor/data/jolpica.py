"""Dunne client voor de Jolpica-F1 API (Ergast-compatibel)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import requests
import requests_cache

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100  # maximum van de API
PAUSE_SECONDS = 0.3  # ruim onder de burst-limiet van 4 requests per seconde
MAX_RETRIES = 5

log = logging.getLogger(__name__)


class JolpicaError(RuntimeError):
    """De API bleef falen na alle herpogingen."""


class JolpicaClient:
    def __init__(
        self,
        cache_path: Path | None = None,
        expire_after: timedelta = timedelta(hours=1),
    ) -> None:
        if cache_path is None:
            self.session: requests.Session = requests.Session()
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.session = requests_cache.CachedSession(
                str(cache_path), backend="sqlite", expire_after=expire_after
            )

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        """Eén request met pauze, retry bij 429/5xx. Geeft het `MRData`-blok terug."""
        url = f"{BASE_URL}/{path.strip('/')}/"
        for attempt in range(1, MAX_RETRIES + 1):
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                if not getattr(response, "from_cache", False):
                    time.sleep(PAUSE_SECONDS)  # alleen wachten na echte netwerkcalls
                return response.json()["MRData"]
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else 2**attempt
                log.warning(
                    "HTTP %s op %s (poging %d/%d), wacht %ds",
                    response.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()  # andere 4xx: direct falen
        raise JolpicaError(f"Mislukt na {MAX_RETRIES} pogingen: {url}")

    def races(self, path: str) -> Iterator[dict[str, Any]]:
        """Loop alle pagina's af en geef de race-blokken terug.

        Een race kan over twee pagina's gesplitst zijn; de aanroeper moet
        daarom per race-blok rijen uitlezen en niet op race-niveau samenvoegen.
        """
        offset = 0
        while True:
            data = self.get(path, limit=PAGE_LIMIT, offset=offset)
            yield from data["RaceTable"]["Races"]
            offset += PAGE_LIMIT
            if offset >= int(data["total"]):
                return

"""Cached downloads. Every raw file goes through `fetch` so the manifest records
where each byte came from and when."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .. import config

MANIFEST = config.PROCESSED / "raw_manifest.json"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; research-data-fetch)"}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def fetch(url: str, name: str, *, refresh: bool = False, retries: int = 4, **kwargs) -> Path:
    """Download `url` to data/raw/`name` unless cached; return the local path."""
    path = config.RAW / name
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_UA, timeout=120, **kwargs)
            r.raise_for_status()
            break
        except requests.RequestException as e:  # retry with backoff
            last = e
            time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError(f"could not fetch {url}: {last}")
    path.write_bytes(r.content)
    manifest = _load_manifest()
    manifest[name] = {
        "url": url,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(r.content).hexdigest(),
        "bytes": len(r.content),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def fred(series_id: str, *, refresh: bool = False) -> Path:
    """FRED series as CSV (latest revised values; no API key needed)."""
    return fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", f"fred/{series_id}.csv", refresh=refresh)

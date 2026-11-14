"""Cached downloads with provenance. Every raw file the return-side predictor
track downloads goes through ``fetch`` so that ``data/processed/raw_manifest.json``
records where each byte came from and when.

Ported unchanged from ``assetpred.data.fetch`` (paths from ``assetforecast.paths``).
Cache names are relative to ``data/raw`` and are the ones assetpred used, so an
existing cache is reused as is.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..paths import PROCESSED, RAW

MANIFEST = PROCESSED / "raw_manifest.json"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; research-data-fetch)"}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def fetch(url: str, name: str, *, refresh: bool = False, retries: int = 4, **kwargs) -> Path:
    """Download `url` to data/raw/`name` unless cached; return the local path."""
    path = RAW / name
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
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(r.content)
    tmp.replace(path)  # atomic, so a concurrent reader never sees a partial file
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST.with_suffix(".lock"), "w") as lock:  # several builders may fetch at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _load_manifest()
        manifest[name] = {
            "url": url,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(r.content).hexdigest(),
            "bytes": len(r.content),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def fred(series_id: str, *, refresh: bool = False) -> Path:
    """FRED series as CSV (latest revised values; no API key needed)."""
    return fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", f"fred/{series_id}.csv", refresh=refresh)

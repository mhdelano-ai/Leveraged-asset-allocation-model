"""On-disk parquet cache with a provenance manifest.

Every fetched series is written once and re-read thereafter. The manifest records
URL, sha256 of the raw payload, fetch timestamp and row count so that a run is
reproducible and so that silent upstream changes are detectable.

Two upstream behaviours make this mandatory rather than an optimisation:

* Yahoo revises ``adjclose`` retroactively, so a backtest re-fetched next month is
  not the same backtest unless the payload is pinned.
* FRED throttles bursts of requests hard (observed: timeouts after ~40 rapid
  requests). Re-fetching what we already have will get the build blocked.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"


@dataclass(frozen=True)
class CacheEntry:
    key: str
    url: str
    sha256: str
    fetched_at: str
    nrows: int
    start: str
    end: str


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.parquet"


def has(key: str) -> bool:
    return cache_path(key).exists()


def load(key: str) -> pd.Series | pd.DataFrame:
    df = pd.read_parquet(cache_path(key))
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return df


def store(key: str, obj: pd.Series | pd.DataFrame, *, url: str, raw: bytes) -> None:
    """Persist a fetched series plus its provenance record."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    frame.to_parquet(cache_path(key))

    manifest = _load_manifest()
    manifest[key] = {
        "url": url,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nrows": int(len(frame)),
        "start": str(frame.index.min().date()) if len(frame) else None,
        "end": str(frame.index.max().date()) if len(frame) else None,
    }
    _save_manifest(manifest)


def manifest_entry(key: str) -> dict | None:
    return _load_manifest().get(key)

"""``python -m assetforecast.dashboard [--out DIR] [--manifest PATH]``: write the dashboard's data files."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.dashboard", description=__doc__)
    ap.add_argument("--out", type=Path, default=data.paths.DASHBOARD_DATA, help="directory for the CSV files")
    ap.add_argument("--manifest", type=Path, default=data.MANIFEST, help="path of datasets.json")
    a = ap.parse_args(argv)
    written = data.write(data.build_all(), a.out, a.manifest)
    print(f"wrote {len(written)} datasets to {a.out} and the manifest {a.manifest}")


if __name__ == "__main__":
    main()

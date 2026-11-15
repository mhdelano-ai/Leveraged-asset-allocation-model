"""``python -m assetforecast.dashboard [--out DIR] [--manifest PATH]``: write the dashboard's data files,
then rebuild ``dashboard/index.html`` with that data inlined and the one-page summary
``dashboard/summary.html`` (both skipped when ``--out`` or ``--manifest`` is moved)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import data, page, summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.dashboard", description=__doc__)
    ap.add_argument("--out", type=Path, default=data.paths.DASHBOARD_DATA, help="directory for the CSV files")
    ap.add_argument("--manifest", type=Path, default=data.MANIFEST, help="path of datasets.json")
    a = ap.parse_args(argv)
    written = data.write(data.build_all(), a.out, a.manifest)
    print(f"wrote {len(written)} datasets to {a.out} and the manifest {a.manifest}")
    if a.out == data.paths.DASHBOARD_DATA and a.manifest == data.MANIFEST:
        p = page.build()
        print(f"rebuilt {p} ({p.stat().st_size / 1e6:.2f} MB, data inlined)")
        p = summary.build()
        print(f"rebuilt {p} ({p.stat().st_size / 1e3:.0f} kB, the one-page summary)")


if __name__ == "__main__":
    main()

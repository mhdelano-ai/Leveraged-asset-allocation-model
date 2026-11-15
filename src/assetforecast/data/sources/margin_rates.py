"""Robinhood's margin interest schedule: what the efficient frontier pays to borrow.

Run ``python -m assetforecast.data.sources.margin_rates [--refresh]``.

Downloads Robinhood's published margin rates page (cached in ``data/raw/margin/``; ``--refresh``
downloads it again), reads the table of rates by margin balance, the date the rates are stated
as of and the day count of the interest example, and writes ``data/processed/margin_rates.json``:

    lender, source     who lends, and the page
    as_of              "The margin interest rates shown are as of ..." (ISO date)
    retrieved          the day the page was downloaded (UTC)
    day_count          daily interest = balance x rate / day_count
    tiers              [{"from": dollars, "below": dollars or null, "rate": annual rate}, ...];
                       the rate of a tier is charged on the whole balance

A page that cannot be read into a complete, contiguous schedule is an error and leaves the
committed file as it was. ``assetforecast.frontier`` reads the file (``margin_schedule``).
"""
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime

from ...paths import PROCESSED
from ..fetch import MANIFEST, fetch

URL = "https://robinhood.com/us/en/support/articles/margin-rates/"
RAW_NAME = "margin/robinhood_margin_rates.html"
OUT = PROCESSED / "margin_rates.json"
LENDER = "Robinhood Financial (margin investing)"

_UNIT = {None: 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9}
_MONEY = r"\$\s*([\d,]+(?:\.\d+)?)(\+?)\s*(thousand|million|billion)?"
_RATE = r"(\d+(?:\.\d+)?)\s*%"


def _text(page: str) -> str:
    """The page's visible text, one element per line."""
    page = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", page)
    lines = (re.sub(r"\s+", " ", html.unescape(s)).strip() for s in re.sub(r"<[^>]+>", "\n", page).split("\n"))
    return "\n".join(s for s in lines if s)


def _dollars(m: tuple[str, str, str]) -> float:
    return float(m[0].replace(",", "")) * _UNIT[m[2].lower() or None]


def parse(page: str) -> dict:
    """``{"as_of", "day_count", "tiers"}`` from the page's HTML; ``ValueError`` if it does not hold a schedule."""
    text = _text(page)
    start = re.search(r"(?i)margin balance\s*\n\s*margin interest rate", text)
    if not start:
        raise ValueError("no 'Margin balance / Margin interest rate' table on the page")
    tiers = []
    lines = text[start.end():].strip().split("\n")
    for label, rate in zip(lines[0::2], lines[1::2]):
        r = re.fullmatch(_RATE, rate)
        money = re.findall(_MONEY, label, flags=re.I)
        if not r or not money:
            break
        if re.match(r"(?i)up to", label) and len(money) == 1:
            lo, hi = 0.0, _dollars(money[0])
        elif len(money) == 2 and re.search(r"(?i)up to", label):
            lo, hi = _dollars(money[0]), _dollars(money[1])
        elif len(money) == 1 and money[0][1] == "+":
            lo, hi = _dollars(money[0]), None
        else:
            raise ValueError(f"cannot read the balance tier {label!r}")
        tiers.append({"from": lo, "below": hi, "rate": round(float(r.group(1)) / 100, 6)})
        if hi is None:
            break
    if (len(tiers) < 2 or tiers[0]["from"] != 0 or tiers[-1]["below"] is not None
            or any(a["below"] != b["from"] or a["below"] <= a["from"] for a, b in zip(tiers, tiers[1:]))
            or not all(0 < t["rate"] < 0.30 for t in tiers)):
        raise ValueError(f"the balance tiers do not form a complete schedule: {tiers}")
    as_of = re.search(r"(?i)margin interest rates shown are as of ([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    if not as_of:
        raise ValueError("the page does not say what date its rates are as of")
    days = set(re.findall(r"%\s*/\s*(\d{3})\s*\)", text))
    if len(days) != 1:
        raise ValueError(f"the interest example does not give one day count: {sorted(days)}")
    return {"as_of": datetime.strptime(as_of.group(1), "%B %d, %Y").date().isoformat(),
            "day_count": int(days.pop()), "tiers": tiers}


def build(refresh: bool = False) -> dict:
    path = fetch(URL, RAW_NAME, refresh=refresh)
    parsed = parse(path.read_text(encoding="utf-8", errors="replace"))
    retrieved = json.loads(MANIFEST.read_text())[RAW_NAME]["fetched_utc"][:10]
    out = {"lender": LENDER, "source": URL, "as_of": parsed["as_of"], "retrieved": retrieved,
           "day_count": parsed["day_count"], "tiers": parsed["tiers"]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m assetforecast.data.sources.margin_rates",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="download the page again even if it is cached")
    a = ap.parse_args(argv)
    old = json.loads(OUT.read_text()) if OUT.exists() else None
    new = build(a.refresh)
    same = old is not None and all(old[k] == new[k] for k in ("as_of", "day_count", "tiers"))
    print(f"{OUT}: rates as of {new['as_of']}, retrieved {new['retrieved']}"
          f" ({'unchanged' if same else 'CHANGED' if old else 'new'})")
    for t in new["tiers"]:
        print(f"  from ${t['from']:>13,.0f}  {t['rate']:.2%}")


if __name__ == "__main__":
    main()

"""The asset universe. Ids here are the column names used everywhere."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    id: str
    name: str
    group: str          # equity | rates | credit | real
    etf: str            # liquid investable proxy for the recent era
    async_close: bool   # trades mostly outside US hours (stale vs US close)


ASSETS = [
    Asset("us_large", "US large-cap equity", "equity", "SPY", False),
    Asset("us_small", "US small-cap equity", "equity", "IWM", False),
    Asset("intl_dev", "International developed equity", "equity", "EFA", True),
    Asset("em", "Emerging-market equity", "equity", "EEM", True),
    Asset("ust_int", "Intermediate Treasuries (~7y)", "rates", "IEF", False),
    Asset("ust_long", "Long Treasuries (~20y+)", "rates", "TLT", False),
    Asset("tips", "TIPS (broad)", "rates", "TIP", False),
    Asset("ig_credit", "Investment-grade corporates", "credit", "LQD", False),
    Asset("hy_credit", "High-yield corporates", "credit", "HYG", False),
    Asset("reits", "US REITs", "real", "VNQ", False),
    Asset("gold", "Gold", "real", "GLD", False),
    Asset("commodities", "Broad commodities (futures TR)", "real", "DBC", False),
]
ASSET_IDS = [a.id for a in ASSETS]
BY_ID = {a.id: a for a in ASSETS}

# Data-quality tiers for each (asset, period) segment.
TIERS = {
    "A": "daily price of an exchange-traded fund (market close), total return",
    "B": "daily open-end mutual-fund NAV, index, or reconstructed total return",
    "C": "monthly, index total return",
    "D": "monthly, model-based reconstruction (e.g. bond returns from yields, month-average prices)",
}

TRADING_DAYS = 252
HORIZON_MONTHS = 12

"""Risk and return metrics toolkit for asset return series."""

from .data import fetch_prices
from .metrics import (
    annualized_return,
    annualized_vol,
    cvar_historic,
    drawdown,
    kurtosis,
    max_drawdown,
    returns_from_prices,
    semideviation,
    sharpe_ratio,
    skewness,
    summary_stats,
    var_cornish_fisher,
    var_historic,
)

__version__ = "1.0.0"
__all__ = [
    "fetch_prices",
    "returns_from_prices",
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "drawdown",
    "max_drawdown",
    "skewness",
    "kurtosis",
    "semideviation",
    "var_historic",
    "cvar_historic",
    "var_cornish_fisher",
    "summary_stats",
]

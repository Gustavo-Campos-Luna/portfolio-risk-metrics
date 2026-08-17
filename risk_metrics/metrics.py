"""Risk and return metrics for asset/portfolio return series.

All functions operate on periodic simple returns (e.g. daily or monthly
percentage change), not price levels. Formulas follow standard risk
management and quantitative finance literature (Bodie/Kane/Marcus;
Cornish & Fisher, 1938 for the modified VaR expansion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple periodic returns from a price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed price levels.

    Returns
    -------
    pd.DataFrame
        Percentage change between consecutive periods, first row dropped.
    """
    return prices.pct_change().dropna(how="all")


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate implied by a return series.

    CAGR = (prod(1 + r_t))^(periods_per_year / n) - 1

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly).

    Returns
    -------
    float
        Annualized return as a decimal.
    """
    n_periods = returns.shape[0]
    if n_periods == 0:
        return 0.0
    compounded_growth = (1 + returns).prod()
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized volatility (standard deviation scaled by sqrt(time)).

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized volatility as a decimal.
    """
    return returns.std() * (periods_per_year ** 0.5)


def sharpe_ratio(
    returns: pd.Series,
    riskfree_rate: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio: excess return per unit of volatility.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    riskfree_rate : float
        Annual risk-free rate as a decimal (e.g. 0.03 = 3%).
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if volatility is zero.
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    ann_excess_return = annualized_return(excess_returns, periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)
    return ann_excess_return / ann_vol if ann_vol > 0 else 0.0


def drawdown(returns: pd.Series, start_value: float = 1000.0) -> pd.DataFrame:
    """Wealth index, running maximum, and drawdown series.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    start_value : float
        Initial portfolio value.

    Returns
    -------
    pd.DataFrame
        Columns: Wealth, Peaks, Drawdown (drawdown as a negative decimal).
    """
    wealth_index = start_value * (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns,
    })


def max_drawdown(returns: pd.Series) -> float:
    """Deepest peak-to-trough decline over the return series.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g. -0.20 = -20%).
        Returns 0.0 for an empty series.
    """
    if returns.empty:
        return 0.0
    return drawdown(returns)["Drawdown"].min()


def skewness(returns: pd.Series) -> float:
    """Sample skewness (third standardized moment) of a return series."""
    demeaned = returns - returns.mean()
    sigma = returns.std(ddof=0)
    if sigma == 0:
        return 0.0
    return (demeaned ** 3).mean() / sigma ** 3


def kurtosis(returns: pd.Series) -> float:
    """Sample kurtosis (fourth standardized moment) of a return series.

    Not excess kurtosis - a normal distribution has kurtosis 3.0.
    """
    demeaned = returns - returns.mean()
    sigma = returns.std(ddof=0)
    if sigma == 0:
        return 0.0
    return (demeaned ** 4).mean() / sigma ** 4


def semideviation(returns: pd.Series) -> float:
    """Standard deviation of returns below the mean (downside risk only)."""
    below_mean = returns[returns < returns.mean()]
    if below_mean.empty:
        return 0.0
    return below_mean.std(ddof=0)


def var_historic(returns: pd.Series, level: float = 5.0) -> float:
    """Historic Value at Risk at the given confidence level.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    level : float
        Percentile threshold (5.0 = worst 5% of outcomes).

    Returns
    -------
    float
        VaR as a positive decimal (a loss magnitude).
    """
    return -np.percentile(returns, level)


def cvar_historic(returns: pd.Series, level: float = 5.0) -> float:
    """Conditional VaR (expected shortfall): average loss beyond the VaR threshold.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    level : float
        Percentile threshold (5.0 = worst 5% of outcomes).

    Returns
    -------
    float
        CVaR as a positive decimal.
    """
    is_beyond = returns <= -var_historic(returns, level=level)
    if not is_beyond.any():
        return 0.0
    return -returns[is_beyond].mean()


def var_cornish_fisher(returns: pd.Series, level: float = 5.0) -> float:
    """Modified (Cornish-Fisher) VaR, adjusting the Gaussian VaR for the
    observed skewness and kurtosis of the return distribution.

    Cornish & Fisher (1938) expansion:
        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*(K-3)/24 - (2z^3 - 5z)*S^2/36

    where z is the standard normal quantile, S is skewness, K is kurtosis.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    level : float
        Percentile threshold (5.0 = worst 5% of outcomes).

    Returns
    -------
    float
        Modified VaR as a positive decimal.
    """
    z = norm.ppf(level / 100)
    s = skewness(returns)
    k = kurtosis(returns)
    z_cf = (
        z
        + (z ** 2 - 1) * s / 6
        + (z ** 3 - 3 * z) * (k - 3) / 24
        - (2 * z ** 3 - 5 * z) * (s ** 2) / 36
    )
    return -(returns.mean() + z_cf * returns.std(ddof=0))


def summary_stats(
    returns: pd.DataFrame,
    riskfree_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute the full metrics suite for each column of a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        One column of periodic returns per asset.
    riskfree_rate : float
        Annual risk-free rate as a decimal, used for the Sharpe ratio.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    pd.DataFrame
        One row per asset with annualized return, volatility, Sharpe
        ratio, skewness, kurtosis, historic VaR/CVaR, Cornish-Fisher VaR,
        and max drawdown.
    """
    return pd.DataFrame({
        "Annualized Return": returns.aggregate(
            annualized_return, periods_per_year=periods_per_year
        ),
        "Annualized Vol": returns.aggregate(annualized_vol, periods_per_year=periods_per_year),
        "Sharpe Ratio": returns.aggregate(
            sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year
        ),
        "Skewness": returns.aggregate(skewness),
        "Kurtosis": returns.aggregate(kurtosis),
        "VaR (5%)": returns.aggregate(var_historic),
        "CVaR (5%)": returns.aggregate(cvar_historic),
        "Cornish-Fisher VaR (5%)": returns.aggregate(var_cornish_fisher),
        "Max Drawdown": returns.aggregate(max_drawdown),
    })

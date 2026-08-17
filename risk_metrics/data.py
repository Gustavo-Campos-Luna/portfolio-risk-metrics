"""Market data acquisition layer.

Isolates the yfinance dependency so the metrics module works on any
price series, regardless of source.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted closing prices for one or more tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols (e.g. ["AAPL", "MSFT"]).
    start : str
        Start date, 'YYYY-MM-DD'.
    end : str, optional
        End date, 'YYYY-MM-DD'. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Date-indexed adjusted closing prices, one column per ticker.

    Raises
    ------
    ValueError
        If no data was returned for the requested tickers/date range.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"No se encontraron datos para {tickers} entre {start} y {end}."
        )

    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    return close.dropna(how="all")

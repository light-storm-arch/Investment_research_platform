"""
Data fetching layer for the investment research platform.
Pulls price data from Yahoo Finance (via direct API) and economic data from FRED.
"""

import datetime
import io
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})


def fetch_price_history(
    ticker: str,
    start: str = "1990-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a single ticker from Yahoo Finance."""
    if end is None:
        end = datetime.date.today().isoformat()

    logger.info("Fetching %s from %s to %s", ticker, start, end)

    # Convert dates to Unix timestamps
    start_ts = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.datetime.strptime(end, "%Y-%m-%d").timestamp()) + 86400

    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )

    for attempt in range(3):
        try:
            resp = _SESSION.get(url, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"Failed to fetch {ticker}: {e}") from e

    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Ensure expected columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in data for {ticker}")

    # Drop rows with null prices
    df = df.dropna(subset=["Close"])

    # Convert to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Adjust for splits/dividends if Adj Close is present
    if "Adj Close" in df.columns:
        adj = pd.to_numeric(df["Adj Close"], errors="coerce")
        ratio = adj / df["Close"]
        ratio = ratio.fillna(1.0)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col] * ratio
        df.drop(columns=["Adj Close"], inplace=True, errors="ignore")

    return df


def fetch_multiple(
    tickers: list[str],
    start: str = "1990-01-01",
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers, returned as a dict."""
    results = {}
    for t in tickers:
        try:
            results[t] = fetch_price_history(t, start, end)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", t, e)
    return results


def compute_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Compute daily simple returns from a price column."""
    return df[column].pct_change().dropna()


def compute_rolling_return(df: pd.DataFrame, window: int = 252, column: str = "Close") -> pd.Series:
    """Compute rolling cumulative return over *window* trading days."""
    return df[column].pct_change(window).dropna()


def compute_drawdown(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Compute drawdown from the running peak."""
    cummax = df[column].cummax()
    return (df[column] - cummax) / cummax


def fetch_fred_series(
    series_id: str,
    api_key: Optional[str] = None,
    start: str = "1990-01-01",
) -> pd.Series:
    """Fetch a FRED economic data series. Requires fredapi + API key."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    if api_key is None:
        import os
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise ValueError(
                "Provide a FRED API key via the api_key parameter "
                "or set the FRED_API_KEY environment variable."
            )
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start)
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data


def resample_to_monthly(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Resample daily prices to month-end."""
    return df[column].resample("ME").last().dropna()


def align_series(*series: pd.Series) -> pd.DataFrame:
    """Align multiple time series on their common dates (forward-fill gaps)."""
    combined = pd.concat(series, axis=1)
    combined = combined.ffill().dropna()
    return combined

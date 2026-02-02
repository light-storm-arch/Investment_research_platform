"""
Pair trend analysis utilities.

Calculates price ratios, moving averages, rolling return z-scores,
volatility metrics, and return dispersion for paired asset classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_fetcher import fetch_price_history

# ---------------------------------------------------------------------------
# Pair configuration
# ---------------------------------------------------------------------------

PAIR_CONFIG: list[dict] = [
    {"label": "US Value vs US Growth", "ticker_a": "VTV", "ticker_b": "VUG"},
    {"label": "International vs US Equities", "ticker_a": "VEA", "ticker_b": "VTI"},
    {"label": "US Large Cap vs US Small Cap", "ticker_a": "SPY", "ticker_b": "IWM"},
    {"label": "US Corporate vs US Treasury Bonds", "ticker_a": "LQD", "ticker_b": "IEF"},
    {"label": "International vs US Bonds", "ticker_a": "BNDX", "ticker_b": "BND"},
]

# Rolling return windows in trading days
ROLLING_WINDOWS = {
    "3-Month": 63,
    "6-Month": 126,
    "12-Month": 252,
    "36-Month": 756,
}

# SMA periods
SMA_PERIODS = [50, 100, 200]

# Volatility windows
VOL_SHORT = 20
VOL_LONG = 60

# Dispersion windows (subset of rolling windows)
DISPERSION_WINDOWS = {
    "3-Month": 63,
    "6-Month": 126,
    "12-Month": 252,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PairData:
    """Container for a fetched pair's price data and derived ratio."""

    label: str
    ticker_a: str
    ticker_b: str
    price_a: pd.Series
    price_b: pd.Series
    ratio: pd.Series
    fetched_at: str  # ISO timestamp


@dataclass
class ZScoreRow:
    """Single row for the z-score table."""

    period: str
    current_return: float
    hist_mean: float
    hist_std: float
    z_score: float
    lookback_obs: int = 0  # number of observations in historical distribution


@dataclass
class DispersionRow:
    """Single row for the return dispersion table."""

    period: str
    current_dispersion: float
    hist_mean: float
    hist_std: float
    z_score: float
    lookback_obs: int = 0  # number of observations in historical distribution


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_pair(ticker_a: str, ticker_b: str, label: str, start: str = "2005-01-01") -> PairData:
    """Fetch adjusted close prices for both tickers and compute the price ratio.

    Uses a start date far enough back to support 36-month rolling returns
    with a meaningful historical distribution for z-score calculation.
    """
    df_a = fetch_price_history(ticker_a, start=start)
    df_b = fetch_price_history(ticker_b, start=start)

    close_a = df_a["Close"]
    close_b = df_b["Close"]

    # Align on common dates
    combined = pd.concat([close_a.rename("A"), close_b.rename("B")], axis=1)
    combined = combined.ffill().dropna()

    ratio = np.log(combined["A"] / combined["B"])

    fetched_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    return PairData(
        label=label,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        price_a=combined["A"],
        price_b=combined["B"],
        ratio=ratio,
        fetched_at=fetched_at,
    )


# ---------------------------------------------------------------------------
# Section 1: Trend Analysis (SMA)
# ---------------------------------------------------------------------------

def compute_sma(ratio: pd.Series, periods: list[int] | None = None) -> pd.DataFrame:
    """Return a DataFrame with the log ratio and its SMAs."""
    if periods is None:
        periods = SMA_PERIODS
    data = {"Log Ratio": ratio}
    for p in periods:
        data[f"SMA {p}"] = ratio.rolling(p).mean()
    return pd.DataFrame(data)


def current_sma_readings(sma_df: pd.DataFrame, slope_window: int = 5) -> dict:
    """Return the latest log ratio / SMA values, above/below status, and slope.

    Slope is the average daily change of each SMA over the last
    *slope_window* days, expressed per day.
    """
    last = sma_df.dropna().iloc[-1]
    ratio_val = last["Log Ratio"]
    readings = {"Log Ratio": ratio_val}
    for col in sma_df.columns:
        if col.startswith("SMA"):
            readings[col] = last[col]
            readings[f"{col} Signal"] = "Above" if ratio_val >= last[col] else "Below"
            sma_series = sma_df[col].dropna()
            if len(sma_series) >= slope_window + 1:
                recent = sma_series.iloc[-(slope_window + 1):]
                readings[f"{col} Slope"] = (recent.iloc[-1] - recent.iloc[0]) / slope_window
            else:
                readings[f"{col} Slope"] = 0.0
    return readings


# ---------------------------------------------------------------------------
# Section 2: Z-Scores of Rolling Returns
# ---------------------------------------------------------------------------

def compute_rolling_ratio_returns(ratio: pd.Series) -> dict[str, pd.Series]:
    """Compute rolling log-return changes of the ratio for each window."""
    result = {}
    for label, window in ROLLING_WINDOWS.items():
        result[label] = ratio.diff(window)
    return result


def compute_zscore_table(ratio: pd.Series) -> list[ZScoreRow]:
    """For each rolling window, compute the z-score of the current log-return
    against the expanding historical distribution."""
    rows = []
    for label, window in ROLLING_WINDOWS.items():
        rolling_ret = ratio.diff(window).dropna()
        if len(rolling_ret) < 2:
            continue

        current = rolling_ret.iloc[-1]
        # Observations used for the distribution (all except the current one)
        n_obs = len(rolling_ret) - 1
        hist_mean = rolling_ret.expanding().mean().iloc[-2]  # exclude current
        hist_std = rolling_ret.expanding().std().iloc[-2]

        if hist_std == 0 or np.isnan(hist_std):
            z = 0.0
        else:
            z = (current - hist_mean) / hist_std

        rows.append(ZScoreRow(
            period=label,
            current_return=current,
            hist_mean=hist_mean,
            hist_std=hist_std,
            z_score=z,
            lookback_obs=n_obs,
        ))
    return rows


# ---------------------------------------------------------------------------
# Section 3: Volatility of the Ratio
# ---------------------------------------------------------------------------

def compute_ratio_volatility(ratio: pd.Series) -> pd.DataFrame:
    """Compute short-term and long-term rolling volatility of daily log-ratio changes."""
    daily_ret = ratio.diff().dropna()
    vol_df = pd.DataFrame({
        f"{VOL_SHORT}-Day Vol": daily_ret.rolling(VOL_SHORT).std() * np.sqrt(252),
        f"{VOL_LONG}-Day Vol": daily_ret.rolling(VOL_LONG).std() * np.sqrt(252),
    })
    return vol_df.dropna()


def current_vol_readings(vol_df: pd.DataFrame) -> dict:
    """Latest volatility values and whether short-term vol is elevated."""
    last = vol_df.iloc[-1]
    short_col = f"{VOL_SHORT}-Day Vol"
    long_col = f"{VOL_LONG}-Day Vol"
    short_val = last[short_col]
    long_val = last[long_col]
    elevated = short_val > 1.5 * long_val if long_val > 0 else False
    return {
        short_col: short_val,
        long_col: long_val,
        "Short-term elevated": elevated,
    }


# ---------------------------------------------------------------------------
# Section 4: Return Dispersion
# ---------------------------------------------------------------------------

def compute_dispersion_table(price_a: pd.Series, price_b: pd.Series) -> list[DispersionRow]:
    """Compute the absolute log-return spread between two assets for each
    window and compare to the historical distribution."""
    log_a = np.log(price_a)
    log_b = np.log(price_b)
    rows = []
    for label, window in DISPERSION_WINDOWS.items():
        ret_a = log_a.diff(window).dropna()
        ret_b = log_b.diff(window).dropna()

        # Align
        common = ret_a.index.intersection(ret_b.index)
        if len(common) < 2:
            continue
        ret_a = ret_a.loc[common]
        ret_b = ret_b.loc[common]

        spread = (ret_a - ret_b).abs()
        current = spread.iloc[-1]
        n_obs = len(spread) - 1

        hist_mean = spread.expanding().mean().iloc[-2]
        hist_std = spread.expanding().std().iloc[-2]

        if hist_std == 0 or np.isnan(hist_std):
            z = 0.0
        else:
            z = (current - hist_mean) / hist_std

        rows.append(DispersionRow(
            period=label,
            current_dispersion=current,
            hist_mean=hist_mean,
            hist_std=hist_std,
            z_score=z,
            lookback_obs=n_obs,
        ))
    return rows

"""
Module 1: Historical Pattern Analysis Engine

Answers questions like:
- "What happens after the S&P 500 is up 20% in one year?"
- "How does the S&P 500 perform after the VIX exceeds 20?"
- "What happens after the yield curve inverts?"

Supports arbitrary user-defined conditions on price or indicator data.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from data_fetcher import (
    compute_returns,
    compute_rolling_return,
    fetch_price_history,
)

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Container for the results of a historical pattern study."""
    description: str
    trigger_dates: list[pd.Timestamp]
    forward_returns: dict[str, pd.Series]  # horizon label -> series of returns
    summary: Optional[pd.DataFrame] = None

    def compute_summary(self) -> pd.DataFrame:
        rows = []
        for label, rets in self.forward_returns.items():
            clean = rets.dropna()
            if len(clean) == 0:
                continue
            rows.append({
                "Horizon": label,
                "Observations": len(clean),
                "Mean Return (%)": round(clean.mean() * 100, 2),
                "Median Return (%)": round(clean.median() * 100, 2),
                "Std Dev (%)": round(clean.std() * 100, 2),
                "Win Rate (%)": round((clean > 0).mean() * 100, 1),
                "Best (%)": round(clean.max() * 100, 2),
                "Worst (%)": round(clean.min() * 100, 2),
            })
        self.summary = pd.DataFrame(rows)
        return self.summary


# ---------------------------------------------------------------------------
# Pre-built condition functions
# ---------------------------------------------------------------------------

def condition_annual_return_exceeds(threshold: float) -> Callable:
    """Return True on dates where the trailing 252-day return exceeds *threshold*.

    Example: condition_annual_return_exceeds(0.20) triggers after a 20% annual gain.
    """
    def _cond(df: pd.DataFrame) -> pd.Series:
        rolling = compute_rolling_return(df, window=252)
        return rolling > threshold
    return _cond


def condition_annual_return_below(threshold: float) -> Callable:
    """Trigger when trailing 1-year return is below threshold."""
    def _cond(df: pd.DataFrame) -> pd.Series:
        rolling = compute_rolling_return(df, window=252)
        return rolling < threshold
    return _cond


def condition_price_above_level(level: float) -> Callable:
    """Trigger when Close crosses above a fixed level."""
    def _cond(df: pd.DataFrame) -> pd.Series:
        above = df["Close"] > level
        crossed = above & (~above).shift(1, fill_value=False)
        return crossed
    return _cond


def condition_price_crosses_sma(window: int = 200, direction: str = "above") -> Callable:
    """Trigger when price crosses its SMA."""
    def _cond(df: pd.DataFrame) -> pd.Series:
        sma = df["Close"].rolling(window).mean()
        if direction == "above":
            cross = (df["Close"] > sma) & (df["Close"].shift(1) <= sma.shift(1))
        else:
            cross = (df["Close"] < sma) & (df["Close"].shift(1) >= sma.shift(1))
        return cross.fillna(False)
    return _cond


def condition_drawdown_exceeds(threshold: float) -> Callable:
    """Trigger when drawdown from peak exceeds *threshold* (e.g. -0.10 for 10% drawdown)."""
    def _cond(df: pd.DataFrame) -> pd.Series:
        cummax = df["Close"].cummax()
        dd = (df["Close"] - cummax) / cummax
        exceeded = dd < threshold  # threshold is negative
        # Only trigger on the first day of each episode
        trigger = exceeded & (~exceeded).shift(1, fill_value=True)
        return trigger.fillna(False)
    return _cond


def condition_vix_above(threshold: float = 20.0) -> Callable:
    """Trigger on the first day VIX crosses above *threshold*.

    NOTE: This condition requires the input DataFrame to contain VIX data,
    so pass a VIX dataframe directly.
    """
    def _cond(df: pd.DataFrame) -> pd.Series:
        above = df["Close"] > threshold
        crossed = above & (~above).shift(1, fill_value=False)
        return crossed.fillna(False)
    return _cond


def condition_consecutive_down_days(n: int = 5) -> Callable:
    """Trigger after *n* consecutive down days."""
    def _cond(df: pd.DataFrame) -> pd.Series:
        daily = df["Close"].pct_change()
        down = (daily < 0).astype(int)
        rolling_sum = down.rolling(n).sum()
        return rolling_sum == n
    return _cond


# ---------------------------------------------------------------------------
# Core analysis engine
# ---------------------------------------------------------------------------

FORWARD_HORIZONS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


def run_pattern_study(
    price_df: pd.DataFrame,
    condition_fn: Callable[[pd.DataFrame], pd.Series],
    description: str = "",
    forward_ticker_df: Optional[pd.DataFrame] = None,
    horizons: Optional[dict[str, int]] = None,
    min_gap_days: int = 21,
) -> PatternResult:
    """Run a historical pattern study.

    Parameters
    ----------
    price_df : DataFrame
        The data on which the condition is evaluated (e.g. VIX data).
    condition_fn : callable
        A function(df) -> boolean Series indicating trigger dates.
    description : str
        Human-readable description of the study.
    forward_ticker_df : DataFrame, optional
        The data on which to measure forward returns. Defaults to *price_df*.
        Useful when the condition is on VIX but you want S&P 500 forward returns.
    horizons : dict, optional
        Mapping of horizon label to number of trading days.
    min_gap_days : int
        Minimum gap between trigger dates to avoid overlapping signals.
    """
    if horizons is None:
        horizons = FORWARD_HORIZONS
    if forward_ticker_df is None:
        forward_ticker_df = price_df

    mask = condition_fn(price_df)
    if mask.dtype != bool:
        mask = mask.astype(bool)
    trigger_dates_all = mask[mask].index.tolist()

    # Thin out overlapping triggers
    trigger_dates: list[pd.Timestamp] = []
    last_date = None
    for d in trigger_dates_all:
        if last_date is None or (d - last_date).days >= min_gap_days:
            trigger_dates.append(d)
            last_date = d

    logger.info("Pattern '%s': %d trigger dates found", description, len(trigger_dates))

    # Compute forward returns for each horizon
    close = forward_ticker_df["Close"]
    forward_returns: dict[str, pd.Series] = {}
    for label, days in horizons.items():
        rets = []
        for d in trigger_dates:
            loc = close.index.get_indexer([d], method="nearest")[0]
            future_loc = loc + days
            if future_loc < len(close):
                ret = (close.iloc[future_loc] - close.iloc[loc]) / close.iloc[loc]
                rets.append(ret)
            else:
                rets.append(np.nan)
        forward_returns[label] = pd.Series(rets, index=trigger_dates[:len(rets)])

    result = PatternResult(
        description=description,
        trigger_dates=trigger_dates,
        forward_returns=forward_returns,
    )
    result.compute_summary()
    return result


# ---------------------------------------------------------------------------
# Convenience wrappers for common studies
# ---------------------------------------------------------------------------

def study_after_annual_gain(
    ticker: str = "^GSPC",
    threshold: float = 0.20,
    start: str = "1990-01-01",
) -> PatternResult:
    """What happens after *ticker* gains more than *threshold* over trailing 1 year?"""
    df = fetch_price_history(ticker, start=start)
    cond = condition_annual_return_exceeds(threshold)
    return run_pattern_study(
        df, cond,
        description=f"{ticker} trailing 1Y return > {threshold:.0%}",
        min_gap_days=126,
    )


def study_after_annual_loss(
    ticker: str = "^GSPC",
    threshold: float = -0.20,
    start: str = "1990-01-01",
) -> PatternResult:
    """What happens after *ticker* loses more than *threshold* over trailing 1 year?"""
    df = fetch_price_history(ticker, start=start)
    cond = condition_annual_return_below(threshold)
    return run_pattern_study(
        df, cond,
        description=f"{ticker} trailing 1Y return < {threshold:.0%}",
        min_gap_days=126,
    )


def study_vix_spike(
    vix_threshold: float = 30.0,
    forward_ticker: str = "^GSPC",
    start: str = "1990-01-01",
) -> PatternResult:
    """What happens to *forward_ticker* after VIX crosses above *vix_threshold*?"""
    vix_df = fetch_price_history("^VIX", start=start)
    fwd_df = fetch_price_history(forward_ticker, start=start)
    cond = condition_vix_above(vix_threshold)
    return run_pattern_study(
        vix_df, cond,
        description=f"VIX crosses above {vix_threshold}",
        forward_ticker_df=fwd_df,
        min_gap_days=21,
    )


def study_drawdown_recovery(
    ticker: str = "^GSPC",
    drawdown_threshold: float = -0.10,
    start: str = "1990-01-01",
) -> PatternResult:
    """What happens after *ticker* experiences a drawdown exceeding *drawdown_threshold*?"""
    df = fetch_price_history(ticker, start=start)
    cond = condition_drawdown_exceeds(drawdown_threshold)
    return run_pattern_study(
        df, cond,
        description=f"{ticker} drawdown exceeds {drawdown_threshold:.0%}",
        min_gap_days=63,
    )


def study_sma_cross(
    ticker: str = "^GSPC",
    sma_window: int = 200,
    direction: str = "above",
    start: str = "1990-01-01",
) -> PatternResult:
    """What happens after price crosses SMA?"""
    df = fetch_price_history(ticker, start=start)
    cond = condition_price_crosses_sma(sma_window, direction)
    return run_pattern_study(
        df, cond,
        description=f"{ticker} crosses {'above' if direction == 'above' else 'below'} {sma_window}-day SMA",
        min_gap_days=21,
    )

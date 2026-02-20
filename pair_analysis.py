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

# Rolling return windows in trading days (legacy, kept for reference)
ROLLING_WINDOWS = {
    "3-Month": 63,
    "6-Month": 126,
    "12-Month": 252,
    "36-Month": 756,
}

# Non-overlapping return configuration
# Approximate trading days per period
TRADING_DAYS_PER_QUARTER = 63
TRADING_DAYS_PER_HALF_YEAR = 126
TRADING_DAYS_PER_YEAR = 252

# Signal thresholds
ZSCORE_THRESHOLD = 1.5
DRAWDOWN_THRESHOLD = 0.10  # 10%

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


@dataclass
class NonOverlappingReturn:
    """Single non-overlapping return observation."""

    date: pd.Timestamp
    period_type: str  # "quarterly", "semi_annual", "annual"
    spread_return: float  # log(A) - log(B) change over period


@dataclass
class Layer1State:
    """Layer 1: Single quarter z-score state."""

    current_return: float
    hist_mean: float
    hist_std: float
    z_score: float
    is_active: bool  # |z| >= threshold
    lookback_obs: int


@dataclass
class Layer2State:
    """Layer 2: Trailing 4-quarter cumulative sum z-score state."""

    trailing_4q_sum: float
    hist_mean: float
    hist_std: float
    z_score: float
    is_active: bool  # |z| >= threshold
    lookback_obs: int
    quarters_included: list  # dates of quarters in the sum


@dataclass
class Layer3State:
    """Layer 3: Drawdown/rally from peak/trough state."""

    cumulative_spread: float
    trailing_high: float
    trailing_low: float
    drawdown_from_high: float  # as decimal (negative when below peak)
    rally_from_low: float  # as decimal (positive when above trough)
    is_active: bool  # drawdown or rally exceeds threshold
    direction: str  # "drawdown" or "rally" or "neutral"


@dataclass
class SignalState:
    """Combined signal state from all three layers."""

    layer1: Layer1State
    layer2: Layer2State
    layer3: Layer3State
    signal_active: bool
    direction: str  # "FAVOR_VALUE", "FAVOR_GROWTH", "NEUTRAL"
    signal_date: pd.Timestamp
    ticker_a: str
    ticker_b: str


@dataclass
class ForwardReturnRow:
    """Single horizon row for the forward return backtest table."""

    period: str  # e.g. "1M", "3M", "1Y"
    end_date: str | None  # actual end date used, or None if unavailable
    return_a: float | None  # simple % return for ticker A
    return_b: float | None  # simple % return for ticker B
    spread: float | None  # return_a - return_b
    annualized_a: float | None  # annualized return (1Y+ only)
    annualized_b: float | None
    annualized_spread: float | None


@dataclass
class ForwardReturnResult:
    """Full result container for the forward return backtest."""

    entry_date: pd.Timestamp
    rebased_a: pd.Series  # price_a rebased to 100 from entry_date
    rebased_b: pd.Series  # price_b rebased to 100 from entry_date
    rows: list[ForwardReturnRow]


# Forward return horizons: label -> (trading days, calendar years for annualizing)
FORWARD_HORIZONS: dict[str, tuple[int, float | None]] = {
    "1M": (21, None),
    "3M": (63, None),
    "6M": (126, None),
    "1Y": (252, 1.0),
    "3Y": (756, 3.0),
    "5Y": (1260, 5.0),
    "10Y": (2520, 10.0),
}


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


def compute_sma_slopes(sma_df: pd.DataFrame, slope_window: int = 10) -> pd.DataFrame:
    """Compute the rolling annualised slope for each SMA column.

    Slope at each date is the change over the prior *slope_window* days,
    annualised by multiplying by 252.
    """
    slope_data = {}
    for col in sma_df.columns:
        if col.startswith("SMA"):
            slope_data[f"{col} Slope"] = sma_df[col].diff(slope_window) / slope_window * 252
    return pd.DataFrame(slope_data, index=sma_df.index)


def current_sma_readings(sma_df: pd.DataFrame, slope_window: int = 10) -> dict:
    """Return the latest log ratio / SMA values, above/below status, and slope.

    Slope is the average daily change of each SMA over the last
    *slope_window* days, annualised in the display layer (×252).
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
# Section 1b: RSI of the Ratio
# ---------------------------------------------------------------------------

RSI_DEFAULT_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Bollinger Bands defaults
BB_DEFAULT_WINDOW = 20
BB_DEFAULT_NUM_STD = 2.0


def compute_ratio_rsi(
    ratio: pd.Series,
    window: int = RSI_DEFAULT_WINDOW,
    frequency: str = "daily",
) -> pd.Series:
    """Compute RSI on the log price ratio series.

    Args:
        ratio: Log price ratio series (log(A/B)).
        window: RSI lookback period (number of bars). Default 14.
        frequency: ``"daily"`` uses daily closes; ``"weekly"`` resamples
            to Friday closes before computing RSI.

    Returns:
        RSI series indexed by date.
    """
    series = ratio.copy()
    if frequency == "weekly":
        series = series.resample("W-FRI").last().dropna()

    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Section 1c: Bollinger Bands of the Ratio
# ---------------------------------------------------------------------------

def compute_ratio_bollinger(
    ratio: pd.Series,
    window: int = BB_DEFAULT_WINDOW,
    num_std: float = BB_DEFAULT_NUM_STD,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Compute Bollinger Bands on the log price ratio series.

    Args:
        ratio: Log price ratio series (log(A/B)).
        window: Moving average lookback period (number of bars). Default 20.
        num_std: Number of standard deviations for band width. Default 2.0.
        frequency: ``"daily"`` uses daily closes; ``"weekly"`` resamples
            to Friday closes before computing bands.

    Returns:
        DataFrame with columns: Log Ratio, Middle Band, Upper Band,
        Lower Band, Percent B.
    """
    series = ratio.copy()
    if frequency == "weekly":
        series = series.resample("W-FRI").last().dropna()

    middle = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std

    band_width = upper - lower
    percent_b = (series - lower) / band_width

    return pd.DataFrame({
        "Log Ratio": series,
        "Middle Band": middle,
        "Upper Band": upper,
        "Lower Band": lower,
        "Percent B": percent_b,
    })


# ---------------------------------------------------------------------------
# Section 2: Forward Return Backtest
# ---------------------------------------------------------------------------

def _snap_to_trading_day(
    target: pd.Timestamp,
    index: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Return the nearest trading day in *index* on or after *target*.

    Falls back to the last available date if *target* is beyond the range.
    Returns ``None`` if the index is empty.
    """
    if index.empty:
        return None
    on_or_after = index[index >= target]
    if not on_or_after.empty:
        return on_or_after[0]
    return index[-1]


def compute_forward_returns(
    price_a: pd.Series,
    price_b: pd.Series,
    entry_date: pd.Timestamp,
) -> ForwardReturnResult | None:
    """Compute forward performance for both assets from an entry date.

    Args:
        price_a: Adjusted close prices for ticker A.
        price_b: Adjusted close prices for ticker B.
        entry_date: The user-selected entry date (will snap to nearest
            available trading day on or after this date).

    Returns:
        ForwardReturnResult with rebased price series and a row per
        forward horizon, or ``None`` if the entry date falls outside the
        available data range.
    """
    common_idx = price_a.index.intersection(price_b.index)
    if common_idx.empty:
        return None

    snapped = _snap_to_trading_day(entry_date, common_idx)
    if snapped is None:
        return None

    fwd_a = price_a.loc[snapped:]
    fwd_b = price_b.loc[snapped:]
    if fwd_a.empty or fwd_b.empty:
        return None

    base_a = fwd_a.iloc[0]
    base_b = fwd_b.iloc[0]
    rebased_a = (fwd_a / base_a) * 100
    rebased_b = (fwd_b / base_b) * 100

    rows: list[ForwardReturnRow] = []
    for label, (tdays, years) in FORWARD_HORIZONS.items():
        if len(fwd_a) <= tdays:
            # Not enough data for this horizon — record as unavailable
            rows.append(ForwardReturnRow(
                period=label,
                end_date=None,
                return_a=None,
                return_b=None,
                spread=None,
                annualized_a=None,
                annualized_b=None,
                annualized_spread=None,
            ))
            continue

        end_a = fwd_a.iloc[tdays]
        end_b = fwd_b.iloc[tdays]
        end_date = fwd_a.index[tdays].strftime("%Y-%m-%d")

        ret_a = end_a / base_a - 1
        ret_b = end_b / base_b - 1
        spread = ret_a - ret_b

        # Annualize for horizons >= 1Y
        if years is not None and years >= 1.0:
            ann_a = (1 + ret_a) ** (1 / years) - 1
            ann_b = (1 + ret_b) ** (1 / years) - 1
            ann_spread = ann_a - ann_b
        else:
            ann_a = None
            ann_b = None
            ann_spread = None

        rows.append(ForwardReturnRow(
            period=label,
            end_date=end_date,
            return_a=ret_a,
            return_b=ret_b,
            spread=spread,
            annualized_a=ann_a,
            annualized_b=ann_b,
            annualized_spread=ann_spread,
        ))

    return ForwardReturnResult(
        entry_date=snapped,
        rebased_a=rebased_a,
        rebased_b=rebased_b,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Section 3: Z-Scores of Rolling Returns
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


# ---------------------------------------------------------------------------
# Section 5: Non-Overlapping Returns
# ---------------------------------------------------------------------------

def _get_quarter_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Get actual trading dates closest to quarter ends (Mar, Jun, Sep, Dec).

    Returns the last trading day of each quarter that exists in the index.
    """
    # Resample to quarter end and get last valid date in each quarter
    temp_series = pd.Series(1, index=index)
    quarter_ends = temp_series.resample("QE").last().dropna().index

    # Map each quarter end to the closest actual trading date in our index
    actual_dates = []
    for qe in quarter_ends:
        # Find dates on or before the quarter end
        mask = index <= qe
        if mask.any():
            actual_dates.append(index[mask][-1])

    return pd.DatetimeIndex(actual_dates)


def _get_semi_annual_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Get actual trading dates closest to semi-annual ends (Jun, Dec)."""
    quarter_ends = _get_quarter_end_dates(index)
    # Filter to only Jun (Q2) and Dec (Q4)
    semi_annual = quarter_ends[quarter_ends.month.isin([6, 12])]
    return semi_annual


def _get_annual_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Get actual trading dates closest to year ends (Dec only)."""
    quarter_ends = _get_quarter_end_dates(index)
    # Filter to only Dec (Q4)
    annual = quarter_ends[quarter_ends.month == 12]
    return annual


def compute_non_overlapping_returns(ratio: pd.Series) -> dict[str, pd.DataFrame]:
    """Compute non-overlapping spread returns for quarterly, semi-annual, and annual periods.

    Args:
        ratio: Log price ratio series (log(A) - log(B))

    Returns:
        Dictionary with keys 'quarterly', 'semi_annual', 'annual', each containing
        a DataFrame with columns ['date', 'spread_return'].
    """
    result = {}

    # Quarterly returns (end of Mar, Jun, Sep, Dec)
    q_dates = _get_quarter_end_dates(ratio.index)
    if len(q_dates) >= 2:
        q_values = ratio.loc[q_dates]
        q_returns = q_values.diff().dropna()
        result["quarterly"] = pd.DataFrame({
            "date": q_returns.index,
            "spread_return": q_returns.values,
        }).reset_index(drop=True)
    else:
        result["quarterly"] = pd.DataFrame(columns=["date", "spread_return"])

    # Semi-annual returns (end of Jun, Dec)
    sa_dates = _get_semi_annual_end_dates(ratio.index)
    if len(sa_dates) >= 2:
        sa_values = ratio.loc[sa_dates]
        sa_returns = sa_values.diff().dropna()
        result["semi_annual"] = pd.DataFrame({
            "date": sa_returns.index,
            "spread_return": sa_returns.values,
        }).reset_index(drop=True)
    else:
        result["semi_annual"] = pd.DataFrame(columns=["date", "spread_return"])

    # Annual returns (end of Dec only)
    a_dates = _get_annual_end_dates(ratio.index)
    if len(a_dates) >= 2:
        a_values = ratio.loc[a_dates]
        a_returns = a_values.diff().dropna()
        result["annual"] = pd.DataFrame({
            "date": a_returns.index,
            "spread_return": a_returns.values,
        }).reset_index(drop=True)
    else:
        result["annual"] = pd.DataFrame(columns=["date", "spread_return"])

    return result


# ---------------------------------------------------------------------------
# Section 6: Three-Layer Signal Framework
# ---------------------------------------------------------------------------

def compute_layer1_single_quarter(
    quarterly_returns: pd.DataFrame,
    threshold: float = ZSCORE_THRESHOLD,
) -> Layer1State | None:
    """Layer 1: Z-score the current quarter's spread return against historical distribution.

    Args:
        quarterly_returns: DataFrame with 'date' and 'spread_return' columns
        threshold: Z-score threshold for activation (default 1.5)

    Returns:
        Layer1State or None if insufficient data
    """
    if len(quarterly_returns) < 3:  # Need at least 2 historical + 1 current
        return None

    returns = quarterly_returns["spread_return"].values
    current = returns[-1]
    historical = returns[:-1]

    hist_mean = np.mean(historical)
    hist_std = np.std(historical, ddof=1)  # Sample std

    if hist_std == 0 or np.isnan(hist_std):
        z_score = 0.0
    else:
        z_score = (current - hist_mean) / hist_std

    return Layer1State(
        current_return=current,
        hist_mean=hist_mean,
        hist_std=hist_std,
        z_score=z_score,
        is_active=abs(z_score) >= threshold,
        lookback_obs=len(historical),
    )


def compute_layer2_trailing_4q(
    quarterly_returns: pd.DataFrame,
    threshold: float = ZSCORE_THRESHOLD,
) -> Layer2State | None:
    """Layer 2: Z-score the trailing 4-quarter cumulative sum against historical distribution.

    Args:
        quarterly_returns: DataFrame with 'date' and 'spread_return' columns
        threshold: Z-score threshold for activation (default 1.5)

    Returns:
        Layer2State or None if insufficient data
    """
    if len(quarterly_returns) < 5:  # Need at least 4 quarters + 1 historical 4Q sum
        return None

    returns = quarterly_returns["spread_return"].values
    dates = quarterly_returns["date"].values

    # Compute all possible 4-quarter sums
    four_q_sums = []
    four_q_end_dates = []
    for i in range(3, len(returns)):
        four_q_sum = returns[i-3:i+1].sum()
        four_q_sums.append(four_q_sum)
        four_q_end_dates.append(dates[i])

    if len(four_q_sums) < 2:
        return None

    four_q_sums = np.array(four_q_sums)
    current_sum = four_q_sums[-1]
    historical_sums = four_q_sums[:-1]

    hist_mean = np.mean(historical_sums)
    hist_std = np.std(historical_sums, ddof=1)

    if hist_std == 0 or np.isnan(hist_std):
        z_score = 0.0
    else:
        z_score = (current_sum - hist_mean) / hist_std

    # Get the dates of quarters included in the current sum
    quarters_included = list(dates[-4:])

    return Layer2State(
        trailing_4q_sum=current_sum,
        hist_mean=hist_mean,
        hist_std=hist_std,
        z_score=z_score,
        is_active=abs(z_score) >= threshold,
        lookback_obs=len(historical_sums),
        quarters_included=quarters_included,
    )


def compute_layer3_drawdown(
    ratio: pd.Series,
    threshold: float = DRAWDOWN_THRESHOLD,
) -> Layer3State:
    """Layer 3: Track drawdown from peak or rally from trough of cumulative log spread.

    Args:
        ratio: Log price ratio series (log(A) - log(B)), which is the cumulative spread
        threshold: Drawdown/rally threshold for activation (default 0.10 = 10%)

    Returns:
        Layer3State
    """
    current = ratio.iloc[-1]
    trailing_high = ratio.max()
    trailing_low = ratio.min()

    # Drawdown from high (negative when below peak)
    drawdown_from_high = current - trailing_high  # Will be <= 0

    # Rally from low (positive when above trough)
    rally_from_low = current - trailing_low  # Will be >= 0

    # Calculate percentages relative to peak/trough
    # Using absolute value of the spread level as denominator, or just use raw difference
    # Since log ratios can be negative, we use the absolute difference directly
    drawdown_pct = drawdown_from_high  # Already in log terms
    rally_pct = rally_from_low

    # Determine direction and activity
    # Active if |drawdown| > threshold OR |rally| > threshold
    is_drawdown_active = abs(drawdown_pct) >= threshold
    is_rally_active = rally_pct >= threshold

    if is_drawdown_active and abs(drawdown_pct) > rally_pct:
        direction = "drawdown"
        is_active = True
    elif is_rally_active:
        direction = "rally"
        is_active = True
    else:
        direction = "neutral"
        is_active = False

    return Layer3State(
        cumulative_spread=current,
        trailing_high=trailing_high,
        trailing_low=trailing_low,
        drawdown_from_high=drawdown_pct,
        rally_from_low=rally_pct,
        is_active=is_active,
        direction=direction,
    )


def compute_combined_signal(
    pair_data: PairData,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    drawdown_threshold: float = DRAWDOWN_THRESHOLD,
) -> SignalState | None:
    """Compute the combined signal from all three layers.

    Signal is active when:
    - Layer 3 (drawdown) is active (>10% from peak/trough) AND
    - Layer 2 (trailing 4Q z-score) exceeds ±1.5 AND
    - Layer 1 (single Q z-score) exceeds ±1.5

    Direction:
    - FAVOR_VALUE: Value (ticker_a) is underperforming, expect mean reversion
      (negative z-scores + drawdown from high)
    - FAVOR_GROWTH: Growth (ticker_b) is underperforming, expect mean reversion
      (positive z-scores + rally from low)

    Args:
        pair_data: PairData object with ratio series
        zscore_threshold: Threshold for z-score layers (default 1.5)
        drawdown_threshold: Threshold for drawdown layer (default 0.10)

    Returns:
        SignalState or None if insufficient data
    """
    # Compute non-overlapping returns
    non_overlap = compute_non_overlapping_returns(pair_data.ratio)
    quarterly = non_overlap["quarterly"]

    if quarterly.empty:
        return None

    # Compute each layer
    layer1 = compute_layer1_single_quarter(quarterly, zscore_threshold)
    layer2 = compute_layer2_trailing_4q(quarterly, zscore_threshold)
    layer3 = compute_layer3_drawdown(pair_data.ratio, drawdown_threshold)

    if layer1 is None or layer2 is None:
        return None

    # Combined signal logic
    signal_active = layer1.is_active and layer2.is_active and layer3.is_active

    # Determine direction
    if signal_active:
        # Check z-score signs for direction
        if layer1.z_score < 0 and layer2.z_score < 0 and layer3.direction == "drawdown":
            # Negative spread returns + drawdown = Value underperforming
            # Expect mean reversion toward value
            direction = "FAVOR_VALUE"
        elif layer1.z_score > 0 and layer2.z_score > 0 and layer3.direction == "rally":
            # Positive spread returns + rally from low = Growth underperforming
            # Expect mean reversion toward growth
            direction = "FAVOR_GROWTH"
        else:
            # Mixed signals - still active but direction unclear
            direction = "MIXED"
    else:
        direction = "NEUTRAL"

    return SignalState(
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        signal_active=signal_active,
        direction=direction,
        signal_date=quarterly["date"].iloc[-1],
        ticker_a=pair_data.ticker_a,
        ticker_b=pair_data.ticker_b,
    )


def compute_signal_history(
    pair_data: PairData,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    drawdown_threshold: float = DRAWDOWN_THRESHOLD,
) -> pd.DataFrame:
    """Compute historical signal states for backtesting/visualization.

    Returns a DataFrame with signal state at each quarter end.
    """
    non_overlap = compute_non_overlapping_returns(pair_data.ratio)
    quarterly = non_overlap["quarterly"]

    if len(quarterly) < 5:
        return pd.DataFrame()

    records = []

    # Need at least 4 quarters for Layer 2, and 1 more for historical distribution
    for i in range(5, len(quarterly) + 1):
        subset = quarterly.iloc[:i].copy()

        # Get ratio up to this quarter's date
        q_date = subset["date"].iloc[-1]
        ratio_subset = pair_data.ratio.loc[:q_date]

        layer1 = compute_layer1_single_quarter(subset, zscore_threshold)
        layer2 = compute_layer2_trailing_4q(subset, zscore_threshold)
        layer3 = compute_layer3_drawdown(ratio_subset, drawdown_threshold)

        if layer1 is None or layer2 is None:
            continue

        signal_active = layer1.is_active and layer2.is_active and layer3.is_active

        if signal_active:
            if layer1.z_score < 0 and layer2.z_score < 0 and layer3.direction == "drawdown":
                direction = "FAVOR_VALUE"
            elif layer1.z_score > 0 and layer2.z_score > 0 and layer3.direction == "rally":
                direction = "FAVOR_GROWTH"
            else:
                direction = "MIXED"
        else:
            direction = "NEUTRAL"

        records.append({
            "date": q_date,
            "spread_return": layer1.current_return,
            "l1_zscore": layer1.z_score,
            "l1_active": layer1.is_active,
            "trailing_4q_sum": layer2.trailing_4q_sum,
            "l2_zscore": layer2.z_score,
            "l2_active": layer2.is_active,
            "cumulative_spread": layer3.cumulative_spread,
            "drawdown": layer3.drawdown_from_high,
            "rally": layer3.rally_from_low,
            "l3_direction": layer3.direction,
            "l3_active": layer3.is_active,
            "signal_active": signal_active,
            "direction": direction,
        })

    return pd.DataFrame(records)


def get_non_overlapping_zscore_table(pair_data: PairData) -> list[ZScoreRow]:
    """Compute z-scores using non-overlapping returns for all periods.

    This replaces the old compute_zscore_table with non-overlapping samples.
    """
    non_overlap = compute_non_overlapping_returns(pair_data.ratio)
    rows = []

    period_map = {
        "quarterly": "3-Month (Quarterly)",
        "semi_annual": "6-Month (Semi-Annual)",
        "annual": "12-Month (Annual)",
    }

    for period_key, label in period_map.items():
        df = non_overlap[period_key]
        if len(df) < 3:
            continue

        returns = df["spread_return"].values
        current = returns[-1]
        historical = returns[:-1]

        hist_mean = np.mean(historical)
        hist_std = np.std(historical, ddof=1)

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
            lookback_obs=len(historical),
        ))

    return rows

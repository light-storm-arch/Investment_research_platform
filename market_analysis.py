"""
Market Analysis module: Sector Rotation and Factor Analysis.

Computes relative strength, momentum rankings, breadth proxies,
percentile context, trend direction, regime classification, and
pairwise factor comparisons.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data_fetcher import fetch_price_history

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------

SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

FACTOR_ETFS: dict[str, str] = {
    "IWM": "Small Cap",
    "SPY": "Large Cap",
    "IWD": "Value",
    "IWF": "Growth",
    "VTI": "US Total Market",
    "VEA": "Intl Developed",
    "VWO": "Emerging Markets",
    "QUAL": "Quality",
    "MTUM": "Momentum",
    "USMV": "Low Volatility",
    "VYM": "Dividend Yield",
}

BENCHMARK = "SPY"

RS_WINDOWS = [20, 60, 252]

# Tickers whose outperformance signals risk-on behaviour
RISK_ON_TICKERS = {"IWM", "VWO", "XLY", "XLF", "XLK", "MTUM", "IWF"}
# Tickers whose outperformance signals risk-off behaviour
RISK_OFF_TICKERS = {"XLU", "XLP", "USMV", "VYM", "XLV", "IWD", "XLRE"}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_universe(
    tickers: list[str],
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """Fetch adjusted close prices for *tickers* and return a single DataFrame.

    Columns are ticker symbols, index is DatetimeIndex.
    """
    frames: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            df = fetch_price_history(t, start=start)
            frames[t] = df["Close"]
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", t, e)
    if not frames:
        raise ValueError("Could not fetch data for any ticker")
    prices = pd.DataFrame(frames)
    prices = prices.ffill().dropna()
    return prices


# ---------------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------------

def relative_strength(
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling relative strength (excess return) vs *benchmark*.

    Returns a MultiIndex DataFrame: (date) x (ticker, window).
    """
    if windows is None:
        windows = RS_WINDOWS
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not in price data")

    results = {}
    bench = prices[benchmark]
    for ticker in prices.columns:
        if ticker == benchmark:
            continue
        for w in windows:
            etf_ret = prices[ticker].pct_change(w)
            bench_ret = bench.pct_change(w)
            rs = etf_ret - bench_ret
            results[(ticker, f"{w}d")] = rs

    df = pd.DataFrame(results)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "window"])
    return df.dropna(how="all")


# ---------------------------------------------------------------------------
# Momentum ranking
# ---------------------------------------------------------------------------

def momentum_ranking(
    rs_df: pd.DataFrame,
    window: str = "60d",
    prior_offset: int = 20,
) -> pd.DataFrame:
    """Rank tickers by relative strength for *window*.

    Returns DataFrame with columns: ticker, name, rs_value, rank, prior_rank,
    rank_change.
    """
    tickers = rs_df.columns.get_level_values("ticker").unique()
    current = rs_df.xs(window, level="window", axis=1).iloc[-1]
    prior = rs_df.xs(window, level="window", axis=1).iloc[-min(prior_offset, len(rs_df))]

    current_rank = current.rank(ascending=False).astype(int)
    prior_rank = prior.rank(ascending=False).astype(int)

    rows = []
    for t in tickers:
        rows.append({
            "ticker": t,
            "rs_value": round(current[t], 4) if not np.isnan(current.get(t, np.nan)) else np.nan,
            "rank": current_rank.get(t, np.nan),
            "prior_rank": prior_rank.get(t, np.nan),
            "rank_change": int(prior_rank.get(t, 0) - current_rank.get(t, 0)),
        })
    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Breadth proxy
# ---------------------------------------------------------------------------

def breadth_proxy(prices: pd.DataFrame) -> pd.DataFrame:
    """For each ETF compute whether price is above its 50- and 200-day SMA.

    Returns DataFrame with columns: ticker, above_sma50, above_sma200.
    True/False per ticker for the most recent date.
    """
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 200:
            continue
        sma50 = series.rolling(50).mean()
        sma200 = series.rolling(200).mean()
        rows.append({
            "ticker": ticker,
            "above_sma50": bool(series.iloc[-1] > sma50.iloc[-1]),
            "above_sma200": bool(series.iloc[-1] > sma200.iloc[-1]),
            "pct_from_sma50": round((series.iloc[-1] / sma50.iloc[-1] - 1) * 100, 2),
            "pct_from_sma200": round((series.iloc[-1] / sma200.iloc[-1] - 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def breadth_timeseries(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily fraction of tickers above their 50- and 200-day SMAs.

    Returns DataFrame with columns: above_sma50_pct, above_sma200_pct.
    """
    above50 = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    above200 = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        sma50 = prices[ticker].rolling(50).mean()
        sma200 = prices[ticker].rolling(200).mean()
        above50[ticker] = (prices[ticker] > sma50).astype(float)
        above200[ticker] = (prices[ticker] > sma200).astype(float)
    result = pd.DataFrame({
        "above_sma50_pct": above50.mean(axis=1) * 100,
        "above_sma200_pct": above200.mean(axis=1) * 100,
    })
    return result.dropna()


# ---------------------------------------------------------------------------
# Constituent breadth (per-ETF drill-down)
# ---------------------------------------------------------------------------

def fetch_etf_holdings(ticker: str, max_holdings: int = 25) -> list[str]:
    """Fetch top holdings of an ETF using yfinance funds_data.

    Returns a list of up to *max_holdings* ticker symbols.  Falls back to
    an empty list if the data is unavailable.
    """
    import yfinance as yf

    try:
        etf = yf.Ticker(ticker)
        fd = etf.funds_data
        holdings_df = fd.top_holdings
        if holdings_df is None or holdings_df.empty:
            return []
        symbols = holdings_df.index.tolist()[:max_holdings]
        # Clean up any index-level names (yfinance returns ticker as index)
        return [str(s) for s in symbols if isinstance(s, str) and s]
    except Exception as e:
        logger.warning("Could not fetch holdings for %s: %s", ticker, e)
        return []


def constituent_breadth(
    etf_ticker: str,
    start: str = "2019-01-01",
    max_holdings: int = 25,
) -> pd.DataFrame | None:
    """Compute breadth stats for the individual holdings of an ETF.

    Returns a DataFrame with columns: ticker, name, close, sma50, sma200,
    above_sma50, above_sma200, pct_from_sma50, pct_from_sma200.
    Returns None if holdings cannot be retrieved.
    """
    import yfinance as yf

    holdings = fetch_etf_holdings(etf_ticker, max_holdings=max_holdings)
    if not holdings:
        return None

    prices = fetch_universe(holdings, start=start)
    if prices.empty:
        return None

    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 50:
            continue
        sma50 = series.rolling(50).mean().iloc[-1]
        sma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else np.nan
        last = series.iloc[-1]

        # Try to get company name
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName", ticker)
        except Exception:
            name = ticker

        rows.append({
            "ticker": ticker,
            "name": name,
            "close": round(last, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2) if not np.isnan(sma200) else np.nan,
            "above_sma50": bool(last > sma50),
            "above_sma200": bool(last > sma200) if not np.isnan(sma200) else None,
            "pct_from_sma50": round((last / sma50 - 1) * 100, 2),
            "pct_from_sma200": round((last / sma200 - 1) * 100, 2) if not np.isnan(sma200) else np.nan,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    above_50_pct = df["above_sma50"].sum() / len(df) * 100
    above_200_count = df["above_sma200"].dropna()
    above_200_pct = above_200_count.sum() / len(above_200_count) * 100 if len(above_200_count) > 0 else np.nan

    # Attach summary stats as DataFrame attributes for easy access
    df.attrs["above_sma50_pct"] = round(above_50_pct, 1)
    df.attrs["above_sma200_pct"] = round(above_200_pct, 1) if not np.isnan(above_200_pct) else None
    df.attrs["etf_ticker"] = etf_ticker
    df.attrs["n_holdings"] = len(df)

    return df


# ---------------------------------------------------------------------------
# Percentile context
# ---------------------------------------------------------------------------

def percentile_ranks(
    rs_df: pd.DataFrame,
    lookbacks: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute percentile rank of the latest relative strength reading
    within its trailing history.

    Returns DataFrame: ticker x (window, lookback_label).
    """
    if lookbacks is None:
        lookbacks = {"1Y": 252, "3Y": 756}

    rows = []
    for ticker in rs_df.columns.get_level_values("ticker").unique():
        row: dict = {"ticker": ticker}
        for w in rs_df.columns.get_level_values("window").unique():
            series = rs_df[(ticker, w)].dropna()
            if len(series) < 2:
                continue
            current = series.iloc[-1]
            for lb_label, lb_days in lookbacks.items():
                history = series.iloc[-lb_days:] if len(series) >= lb_days else series
                pctile = (history < current).mean() * 100
                row[f"rs_{w}_{lb_label}_pctile"] = round(pctile, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trend direction
# ---------------------------------------------------------------------------

def trend_direction(
    rs_df: pd.DataFrame,
    trend_window: int = 20,
    flat_threshold: float = 0.0001,
) -> pd.DataFrame:
    """Classify recent trend of relative strength as increasing / decreasing / flat.

    Uses OLS slope of the last *trend_window* observations.
    Returns DataFrame: ticker, window, slope, trend.
    """
    rows = []
    for ticker in rs_df.columns.get_level_values("ticker").unique():
        for w in rs_df.columns.get_level_values("window").unique():
            series = rs_df[(ticker, w)].dropna()
            if len(series) < trend_window:
                continue
            recent = series.iloc[-trend_window:].values
            x = np.arange(trend_window, dtype=float)
            # Simple OLS slope
            slope = np.polyfit(x, recent, 1)[0]
            if slope > flat_threshold:
                trend = "Increasing"
            elif slope < -flat_threshold:
                trend = "Decreasing"
            else:
                trend = "Flat"
            rows.append({
                "ticker": ticker,
                "window": w,
                "slope": round(slope, 6),
                "trend": trend,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    classification: str          # "Risk-On", "Risk-Off", "Neutral"
    score: int                   # net score (positive = risk-on)
    risk_on_signals: list[str]   # tickers outperforming
    risk_off_signals: list[str]  # tickers outperforming
    detail: str = ""


def classify_regime(
    rs_df: pd.DataFrame,
    window: str = "60d",
) -> RegimeResult:
    """Classify the current market regime as Risk-On, Risk-Off, or Neutral.

    Scores +1 for each risk-on ticker with positive relative strength and
    -1 for each risk-off ticker with positive relative strength.
    """
    latest = rs_df.xs(window, level="window", axis=1).iloc[-1]

    risk_on_signals = []
    risk_off_signals = []

    for t in latest.index:
        if np.isnan(latest[t]):
            continue
        if t in RISK_ON_TICKERS and latest[t] > 0:
            risk_on_signals.append(t)
        elif t in RISK_OFF_TICKERS and latest[t] > 0:
            risk_off_signals.append(t)

    score = len(risk_on_signals) - len(risk_off_signals)

    if score >= 2:
        classification = "Risk-On"
    elif score <= -2:
        classification = "Risk-Off"
    else:
        classification = "Neutral"

    detail = (
        f"Risk-on outperformers ({len(risk_on_signals)}): "
        f"{', '.join(risk_on_signals) or 'none'}. "
        f"Risk-off outperformers ({len(risk_off_signals)}): "
        f"{', '.join(risk_off_signals) or 'none'}. "
        f"Net score: {score:+d}."
    )

    return RegimeResult(
        classification=classification,
        score=score,
        risk_on_signals=risk_on_signals,
        risk_off_signals=risk_off_signals,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

@dataclass
class PairwiseResult:
    ticker_a: str
    ticker_b: str
    spread: pd.Series              # cumulative return spread over time
    current_spread: float
    spread_trend: str              # "Widening toward A", "Widening toward B", "Flat"
    spread_slope: float


def pairwise_comparison(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    trend_window: int = 20,
    flat_threshold: float = 0.0001,
) -> PairwiseResult:
    """Compare cumulative returns of two tickers and determine spread trend."""
    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        raise ValueError(f"Both {ticker_a} and {ticker_b} must be in price data")

    ret_a = prices[ticker_a].pct_change().fillna(0).add(1).cumprod() - 1
    ret_b = prices[ticker_b].pct_change().fillna(0).add(1).cumprod() - 1
    spread = ret_a - ret_b

    current_spread = float(spread.iloc[-1])

    # Trend of the spread
    recent = spread.iloc[-trend_window:].values
    if len(recent) >= trend_window:
        x = np.arange(len(recent), dtype=float)
        slope = float(np.polyfit(x, recent, 1)[0])
    else:
        slope = 0.0

    if slope > flat_threshold:
        spread_trend = f"Widening toward {ticker_a}"
    elif slope < -flat_threshold:
        spread_trend = f"Widening toward {ticker_b}"
    else:
        spread_trend = "Flat"

    return PairwiseResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        spread=spread,
        current_spread=round(current_spread, 4),
        spread_trend=spread_trend,
        spread_slope=round(slope, 6),
    )


# ---------------------------------------------------------------------------
# Crossover alerts
# ---------------------------------------------------------------------------

def detect_crossovers(
    rs_df: pd.DataFrame,
    window: str = "20d",
    lookback: int = 5,
) -> list[dict]:
    """Detect tickers whose relative strength crossed zero in the last
    *lookback* trading days.

    Returns list of dicts: ticker, direction ("positive" or "negative"),
    cross_date.
    """
    series_block = rs_df.xs(window, level="window", axis=1)
    recent = series_block.iloc[-lookback:]
    alerts = []
    for ticker in recent.columns:
        vals = recent[ticker].dropna()
        if len(vals) < 2:
            continue
        signs = np.sign(vals.values)
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
                direction = "positive" if signs[i] > 0 else "negative"
                alerts.append({
                    "ticker": ticker,
                    "direction": direction,
                    "cross_date": vals.index[i].strftime("%Y-%m-%d"),
                })
    return alerts


# ---------------------------------------------------------------------------
# Summary builder (convenience)
# ---------------------------------------------------------------------------

def build_summary_table(
    prices: pd.DataFrame,
    rs_df: pd.DataFrame,
    etf_names: dict[str, str],
    ranking_window: str = "60d",
) -> pd.DataFrame:
    """Build a single summary DataFrame combining RS, rank, breadth,
    percentile, and trend for display in the dashboard."""
    # Momentum ranking
    rank_df = momentum_ranking(rs_df, window=ranking_window)

    # Breadth
    brd = breadth_proxy(prices)

    # Percentile ranks
    pctile = percentile_ranks(rs_df)

    # Trend direction for ranking window
    trends = trend_direction(rs_df)
    trend_sub = trends[trends["window"] == ranking_window][["ticker", "trend"]].copy()

    # RS values at all windows
    latest_rs = rs_df.iloc[-1]
    rs_cols = {}
    for ticker in rank_df["ticker"]:
        for w in rs_df.columns.get_level_values("window").unique():
            val = latest_rs.get((ticker, w), np.nan)
            rs_cols.setdefault(f"rs_{w}", {})[ticker] = round(val, 4) if not np.isnan(val) else np.nan
    rs_extra = pd.DataFrame(rs_cols)
    rs_extra["ticker"] = rs_extra.index
    rs_extra = rs_extra.reset_index(drop=True)

    # Merge everything
    summary = rank_df.copy()
    summary["name"] = summary["ticker"].map(etf_names)
    summary = summary.merge(brd, on="ticker", how="left")
    summary = summary.merge(pctile, on="ticker", how="left")
    summary = summary.merge(trend_sub, on="ticker", how="left")
    summary = summary.merge(rs_extra, on="ticker", how="left")

    return summary

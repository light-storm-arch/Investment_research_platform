"""
Module 2: Watchlist Alerts & Technical Indicator Scanner

Monitors a watchlist of securities and flags:
- Overbought / oversold conditions (RSI)
- Bollinger Band breakouts
- Unusual volume spikes
- MACD crossovers
- Price gap alerts
- New 52-week highs / lows
- Mean reversion signals (Z-score of returns)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import ta

from data_fetcher import fetch_price_history

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    RSI_OVERBOUGHT = "RSI Overbought"
    RSI_OVERSOLD = "RSI Oversold"
    BOLLINGER_UPPER = "Bollinger Upper Breakout"
    BOLLINGER_LOWER = "Bollinger Lower Breakout"
    VOLUME_SPIKE = "Unusual Volume"
    MACD_BULLISH_CROSS = "MACD Bullish Cross"
    MACD_BEARISH_CROSS = "MACD Bearish Cross"
    NEW_52W_HIGH = "New 52-Week High"
    NEW_52W_LOW = "New 52-Week Low"
    PRICE_GAP_UP = "Price Gap Up"
    PRICE_GAP_DOWN = "Price Gap Down"
    MEAN_REVERSION_OVERSOLD = "Mean Reversion Oversold"
    MEAN_REVERSION_OVERBOUGHT = "Mean Reversion Overbought"


@dataclass
class Alert:
    ticker: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    date: pd.Timestamp
    value: float = 0.0

    def __str__(self):
        return f"[{self.severity.value}] {self.ticker} - {self.alert_type.value}: {self.message}"


@dataclass
class TickerReport:
    """Full technical snapshot for a single ticker."""
    ticker: str
    last_price: float
    daily_change_pct: float
    rsi_14: float
    macd_signal_diff: float
    bb_position: float  # 0 = lower band, 1 = upper band
    volume_vs_avg: float  # ratio of latest volume to 20-day avg
    high_52w: float
    low_52w: float
    alerts: list[Alert] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Technical indicator computation
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    return ta.momentum.RSIIndicator(close, window=window).rsi()


def compute_bollinger(close: pd.Series, window: int = 20, std_dev: float = 2.0):
    bb = ta.volatility.BollingerBands(close, window=window, window_dev=std_dev)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def compute_macd(close: pd.Series):
    macd_ind = ta.trend.MACD(close)
    return macd_ind.macd(), macd_ind.macd_signal(), macd_ind.macd_diff()


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume.rolling(window).mean()
    return volume / avg


def compute_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Z-score of price relative to recent rolling window."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def generate_alerts(
    ticker: str,
    df: pd.DataFrame,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    volume_spike_threshold: float = 2.5,
    zscore_threshold: float = 2.0,
    gap_threshold: float = 0.03,
) -> list[Alert]:
    """Generate all alerts for a ticker based on latest data."""
    alerts: list[Alert] = []
    close = df["Close"]
    volume = df["Volume"]
    latest_date = df.index[-1]

    # RSI
    rsi = compute_rsi(close)
    rsi_val = rsi.iloc[-1]
    if rsi_val >= rsi_overbought:
        sev = AlertSeverity.CRITICAL if rsi_val >= 80 else AlertSeverity.WARNING
        alerts.append(Alert(
            ticker, AlertType.RSI_OVERBOUGHT, sev,
            f"RSI(14) = {rsi_val:.1f}", latest_date, rsi_val,
        ))
    elif rsi_val <= rsi_oversold:
        sev = AlertSeverity.CRITICAL if rsi_val <= 20 else AlertSeverity.WARNING
        alerts.append(Alert(
            ticker, AlertType.RSI_OVERSOLD, sev,
            f"RSI(14) = {rsi_val:.1f}", latest_date, rsi_val,
        ))

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = compute_bollinger(close)
    if close.iloc[-1] > bb_upper.iloc[-1]:
        alerts.append(Alert(
            ticker, AlertType.BOLLINGER_UPPER, AlertSeverity.WARNING,
            f"Price {close.iloc[-1]:.2f} above upper BB {bb_upper.iloc[-1]:.2f}",
            latest_date, close.iloc[-1],
        ))
    elif close.iloc[-1] < bb_lower.iloc[-1]:
        alerts.append(Alert(
            ticker, AlertType.BOLLINGER_LOWER, AlertSeverity.WARNING,
            f"Price {close.iloc[-1]:.2f} below lower BB {bb_lower.iloc[-1]:.2f}",
            latest_date, close.iloc[-1],
        ))

    # MACD crossover (check last 2 days)
    macd_line, macd_signal, macd_diff = compute_macd(close)
    if len(macd_diff) >= 2:
        if macd_diff.iloc[-1] > 0 and macd_diff.iloc[-2] <= 0:
            alerts.append(Alert(
                ticker, AlertType.MACD_BULLISH_CROSS, AlertSeverity.INFO,
                "MACD crossed above signal line", latest_date, macd_diff.iloc[-1],
            ))
        elif macd_diff.iloc[-1] < 0 and macd_diff.iloc[-2] >= 0:
            alerts.append(Alert(
                ticker, AlertType.MACD_BEARISH_CROSS, AlertSeverity.WARNING,
                "MACD crossed below signal line", latest_date, macd_diff.iloc[-1],
            ))

    # Volume spike
    vol_ratio = compute_volume_ratio(volume)
    vol_r = vol_ratio.iloc[-1]
    if vol_r >= volume_spike_threshold:
        alerts.append(Alert(
            ticker, AlertType.VOLUME_SPIKE, AlertSeverity.WARNING,
            f"Volume is {vol_r:.1f}x the 20-day average", latest_date, vol_r,
        ))

    # 52-week high/low
    high_52w = close.rolling(252).max().iloc[-1]
    low_52w = close.rolling(252).min().iloc[-1]
    if close.iloc[-1] >= high_52w:
        alerts.append(Alert(
            ticker, AlertType.NEW_52W_HIGH, AlertSeverity.INFO,
            f"New 52-week high: {close.iloc[-1]:.2f}", latest_date, close.iloc[-1],
        ))
    elif close.iloc[-1] <= low_52w:
        alerts.append(Alert(
            ticker, AlertType.NEW_52W_LOW, AlertSeverity.CRITICAL,
            f"New 52-week low: {close.iloc[-1]:.2f}", latest_date, close.iloc[-1],
        ))

    # Price gap
    if len(df) >= 2:
        prev_close = close.iloc[-2]
        gap_pct = (df["Open"].iloc[-1] - prev_close) / prev_close
        if gap_pct >= gap_threshold:
            alerts.append(Alert(
                ticker, AlertType.PRICE_GAP_UP, AlertSeverity.INFO,
                f"Gapped up {gap_pct:.1%} at open", latest_date, gap_pct,
            ))
        elif gap_pct <= -gap_threshold:
            alerts.append(Alert(
                ticker, AlertType.PRICE_GAP_DOWN, AlertSeverity.WARNING,
                f"Gapped down {gap_pct:.1%} at open", latest_date, gap_pct,
            ))

    # Mean reversion z-score
    zs = compute_zscore(close)
    zs_val = zs.iloc[-1]
    if zs_val <= -zscore_threshold:
        alerts.append(Alert(
            ticker, AlertType.MEAN_REVERSION_OVERSOLD, AlertSeverity.WARNING,
            f"Price Z-score = {zs_val:.2f} (potential bounce)", latest_date, zs_val,
        ))
    elif zs_val >= zscore_threshold:
        alerts.append(Alert(
            ticker, AlertType.MEAN_REVERSION_OVERBOUGHT, AlertSeverity.INFO,
            f"Price Z-score = {zs_val:.2f} (extended)", latest_date, zs_val,
        ))

    return alerts


def build_ticker_report(ticker: str, df: pd.DataFrame) -> TickerReport:
    """Build a full technical report for one ticker."""
    close = df["Close"]
    rsi = compute_rsi(close)
    _, _, macd_diff = compute_macd(close)
    bb_upper, _, bb_lower = compute_bollinger(close)
    vol_ratio = compute_volume_ratio(df["Volume"])

    # BB position: 0 = at lower, 1 = at upper
    bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / bb_range if bb_range > 0 else 0.5

    high_52w = close.rolling(min(252, len(close))).max().iloc[-1]
    low_52w = close.rolling(min(252, len(close))).min().iloc[-1]

    alerts = generate_alerts(ticker, df)

    return TickerReport(
        ticker=ticker,
        last_price=round(float(close.iloc[-1]), 2),
        daily_change_pct=round(float(close.pct_change().iloc[-1]) * 100, 2),
        rsi_14=round(float(rsi.iloc[-1]), 1),
        macd_signal_diff=round(float(macd_diff.iloc[-1]), 4),
        bb_position=round(float(bb_pos), 2),
        volume_vs_avg=round(float(vol_ratio.iloc[-1]), 2),
        high_52w=round(float(high_52w), 2),
        low_52w=round(float(low_52w), 2),
        alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Watchlist scanner
# ---------------------------------------------------------------------------

def scan_watchlist(
    tickers: list[str],
    lookback_days: int = 365,
) -> list[TickerReport]:
    """Scan a full watchlist and return reports sorted by number of alerts."""
    import datetime
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days + 60)).isoformat()
    reports: list[TickerReport] = []
    for t in tickers:
        try:
            df = fetch_price_history(t, start=start)
            if len(df) < 50:
                logger.warning("Not enough data for %s", t)
                continue
            report = build_ticker_report(t, df)
            reports.append(report)
        except Exception as e:
            logger.warning("Error scanning %s: %s", t, e)
    reports.sort(key=lambda r: len(r.alerts), reverse=True)
    return reports

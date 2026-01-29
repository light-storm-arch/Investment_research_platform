#!/usr/bin/env python3
"""
Investment Research Platform - CLI Interface

Three modules:
  1. Historical Pattern Analysis  - study what happened in the past after specific conditions
  2. Watchlist Alerts             - scan securities for technical signals and anomalies
  3. ML Price Prediction          - gradient boosting + LSTM ensemble forecasts

Usage:
    python main.py                         # interactive menu
    python main.py history                 # historical analysis mode
    python main.py alerts                  # watchlist alert scan
    python main.py predict                 # ML prediction mode
"""

import argparse
import logging
import sys

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_pattern_result(result):
    """Pretty-print a PatternResult."""
    console.print(Panel(f"[bold]{result.description}[/bold]", style="cyan"))
    console.print(f"Trigger dates found: [bold]{len(result.trigger_dates)}[/bold]")

    if result.trigger_dates:
        console.print("\nSample trigger dates:")
        for d in result.trigger_dates[:10]:
            console.print(f"  {d.strftime('%Y-%m-%d')}")
        if len(result.trigger_dates) > 10:
            console.print(f"  ... and {len(result.trigger_dates) - 10} more")

    if result.summary is not None and not result.summary.empty:
        console.print()
        table = Table(title="Forward Return Statistics")
        for col in result.summary.columns:
            table.add_column(col, justify="right")
        for _, row in result.summary.iterrows():
            table.add_row(*[str(v) for v in row.values])
        console.print(table)


def _print_ticker_report(report):
    """Pretty-print a TickerReport."""
    color = "green" if report.daily_change_pct >= 0 else "red"
    console.print(Panel(
        f"[bold]{report.ticker}[/bold]  "
        f"${report.last_price}  "
        f"[{color}]{report.daily_change_pct:+.2f}%[/{color}]",
        style="blue",
    ))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("RSI (14)", f"{report.rsi_14}")
    table.add_row("MACD - Signal", f"{report.macd_signal_diff}")
    table.add_row("BB Position", f"{report.bb_position} (0=lower, 1=upper)")
    table.add_row("Volume vs 20d avg", f"{report.volume_vs_avg}x")
    table.add_row("52-Week High", f"${report.high_52w}")
    table.add_row("52-Week Low", f"${report.low_52w}")
    console.print(table)

    if report.alerts:
        console.print(f"\n[bold yellow]Alerts ({len(report.alerts)}):[/bold yellow]")
        for a in report.alerts:
            sev_color = {"INFO": "blue", "WARNING": "yellow", "CRITICAL": "red"}[a.severity.value]
            console.print(f"  [{sev_color}]{a}[/{sev_color}]")
    else:
        console.print("  [dim]No alerts[/dim]")
    console.print()


def _print_ensemble(ens):
    """Pretty-print an EnsemblePrediction."""
    consensus_color = {
        "Bullish": "green", "Bearish": "red", "Mixed": "yellow",
    }[ens.consensus_direction]

    console.print(Panel(
        f"[bold]{ens.ticker}[/bold] - {ens.forward_days}-day forecast",
        style="magenta",
    ))

    table = Table(title="Model Predictions")
    table.add_column("Model")
    table.add_column("Direction", justify="center")
    table.add_column("Predicted Return", justify="right")
    table.add_column("Test Accuracy", justify="right")

    gb_dir = "[green]UP[/green]" if ens.gb_direction == 1 else "[red]DOWN[/red]"
    table.add_row("Gradient Boosting", gb_dir, f"{ens.gb_return:+.2%}", f"{ens.gb_accuracy:.1%}")

    if ens.lstm_direction >= 0:
        lstm_dir = "[green]UP[/green]" if ens.lstm_direction == 1 else "[red]DOWN[/red]"
        table.add_row("LSTM Neural Net", lstm_dir, f"{ens.lstm_return:+.2%}", f"{ens.lstm_accuracy:.1%}")

    console.print(table)
    console.print(
        f"\nConsensus: [{consensus_color}][bold]{ens.consensus_direction}[/bold][/{consensus_color}]"
        f"  |  Avg predicted return: {ens.avg_predicted_return:+.2%}"
    )


# ---------------------------------------------------------------------------
# Mode: Historical Pattern Analysis
# ---------------------------------------------------------------------------

def mode_history(interactive: bool = True):
    from historical_analysis import (
        study_after_annual_gain,
        study_after_annual_loss,
        study_vix_spike,
        study_drawdown_recovery,
        study_sma_cross,
        run_pattern_study,
        condition_annual_return_exceeds,
        condition_annual_return_below,
        condition_vix_above,
        condition_drawdown_exceeds,
        condition_price_crosses_sma,
        condition_consecutive_down_days,
    )
    from data_fetcher import fetch_price_history

    PRESETS = {
        "1": ("S&P 500 up >20% trailing year", lambda: study_after_annual_gain("^GSPC", 0.20)),
        "2": ("S&P 500 down >20% trailing year", lambda: study_after_annual_loss("^GSPC", -0.20)),
        "3": ("VIX crosses above 30", lambda: study_vix_spike(30.0, "^GSPC")),
        "4": ("VIX crosses above 20", lambda: study_vix_spike(20.0, "^GSPC")),
        "5": ("S&P 500 drawdown >10%", lambda: study_drawdown_recovery("^GSPC", -0.10)),
        "6": ("S&P 500 crosses above 200-day SMA", lambda: study_sma_cross("^GSPC", 200, "above")),
        "7": ("S&P 500 crosses below 200-day SMA", lambda: study_sma_cross("^GSPC", 200, "below")),
        "8": ("Custom study", None),
    }

    console.print(Panel("[bold]Historical Pattern Analysis[/bold]", style="cyan"))
    console.print("Choose a preset study or build a custom one:\n")

    for key, (label, _) in PRESETS.items():
        console.print(f"  [bold]{key}[/bold]. {label}")

    choice = Prompt.ask("\nSelect", choices=list(PRESETS.keys()), default="1")

    if choice != "8":
        label, fn = PRESETS[choice]
        console.print(f"\nRunning: [bold]{label}[/bold] ...\n")
        result = fn()
        _print_pattern_result(result)
    else:
        # Custom study builder
        console.print("\n[bold]Custom Study Builder[/bold]\n")
        ticker = Prompt.ask("Ticker to evaluate condition on", default="^GSPC")
        fwd_ticker = Prompt.ask("Ticker to measure forward returns", default=ticker)

        console.print("\nAvailable conditions:")
        console.print("  1. Trailing annual return exceeds threshold")
        console.print("  2. Trailing annual return below threshold")
        console.print("  3. Price crosses SMA")
        console.print("  4. Drawdown exceeds threshold")
        console.print("  5. Close exceeds a level (for VIX, etc.)")
        console.print("  6. Consecutive down days")

        cond_choice = Prompt.ask("Condition", choices=["1", "2", "3", "4", "5", "6"], default="1")

        if cond_choice == "1":
            thresh = FloatPrompt.ask("Return threshold (e.g. 0.20 for 20%)", default=0.20)
            cond_fn = condition_annual_return_exceeds(thresh)
            desc = f"{ticker} trailing 1Y return > {thresh:.0%}"
        elif cond_choice == "2":
            thresh = FloatPrompt.ask("Return threshold (e.g. -0.20 for -20%)", default=-0.20)
            cond_fn = condition_annual_return_below(thresh)
            desc = f"{ticker} trailing 1Y return < {thresh:.0%}"
        elif cond_choice == "3":
            window = IntPrompt.ask("SMA window (days)", default=200)
            direction = Prompt.ask("Direction", choices=["above", "below"], default="above")
            cond_fn = condition_price_crosses_sma(window, direction)
            desc = f"{ticker} crosses {direction} {window}-day SMA"
        elif cond_choice == "4":
            thresh = FloatPrompt.ask("Drawdown threshold (e.g. -0.10 for -10%)", default=-0.10)
            cond_fn = condition_drawdown_exceeds(thresh)
            desc = f"{ticker} drawdown exceeds {thresh:.0%}"
        elif cond_choice == "5":
            level = FloatPrompt.ask("Level (e.g. 30 for VIX > 30)", default=30.0)
            cond_fn = condition_vix_above(level)
            desc = f"{ticker} crosses above {level}"
        else:
            n = IntPrompt.ask("Number of consecutive down days", default=5)
            cond_fn = condition_consecutive_down_days(n)
            desc = f"{ticker} has {n} consecutive down days"

        start = Prompt.ask("Start date", default="1990-01-01")

        console.print(f"\nRunning: [bold]{desc}[/bold] ...\n")
        cond_df = fetch_price_history(ticker, start=start)
        fwd_df = fetch_price_history(fwd_ticker, start=start) if fwd_ticker != ticker else cond_df

        result = run_pattern_study(cond_df, cond_fn, description=desc, forward_ticker_df=fwd_df)
        _print_pattern_result(result)


# ---------------------------------------------------------------------------
# Mode: Watchlist Alerts
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA",     # Major indices
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",  # Mega cap
    "XLF", "XLE", "XLV", "XLK",     # Sectors
    "TLT", "HYG", "GLD", "USO",     # Bonds, gold, oil
]


def mode_alerts():
    from watchlist_alerts import scan_watchlist

    console.print(Panel("[bold]Watchlist Alert Scanner[/bold]", style="blue"))
    console.print(f"Default watchlist: {', '.join(DEFAULT_WATCHLIST)}\n")

    use_default = Confirm.ask("Use default watchlist?", default=True)
    if use_default:
        tickers = DEFAULT_WATCHLIST
    else:
        raw = Prompt.ask("Enter tickers (comma-separated)")
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    console.print(f"\nScanning {len(tickers)} securities ...\n")
    reports = scan_watchlist(tickers)

    # Summary table
    summary_table = Table(title="Watchlist Summary")
    summary_table.add_column("Ticker")
    summary_table.add_column("Price", justify="right")
    summary_table.add_column("Change %", justify="right")
    summary_table.add_column("RSI", justify="right")
    summary_table.add_column("Vol Ratio", justify="right")
    summary_table.add_column("Alerts", justify="center")

    for r in reports:
        chg_color = "green" if r.daily_change_pct >= 0 else "red"
        rsi_color = "red" if r.rsi_14 >= 70 else ("green" if r.rsi_14 <= 30 else "white")
        alert_color = "red" if len(r.alerts) >= 3 else ("yellow" if len(r.alerts) >= 1 else "dim")
        summary_table.add_row(
            r.ticker,
            f"${r.last_price}",
            f"[{chg_color}]{r.daily_change_pct:+.2f}%[/{chg_color}]",
            f"[{rsi_color}]{r.rsi_14}[/{rsi_color}]",
            f"{r.volume_vs_avg}x",
            f"[{alert_color}]{len(r.alerts)}[/{alert_color}]",
        )
    console.print(summary_table)

    # Show detailed reports for tickers with alerts
    tickers_with_alerts = [r for r in reports if r.alerts]
    if tickers_with_alerts:
        console.print(f"\n[bold yellow]Detailed reports for {len(tickers_with_alerts)} securities with alerts:[/bold yellow]\n")
        for r in tickers_with_alerts:
            _print_ticker_report(r)
    else:
        console.print("\n[dim]No alerts triggered across the watchlist.[/dim]")


# ---------------------------------------------------------------------------
# Mode: ML Prediction
# ---------------------------------------------------------------------------

def mode_predict():
    from ml_predictor import train_gradient_boosting, train_lstm, run_ensemble

    console.print(Panel("[bold]ML Price Prediction[/bold]", style="magenta"))

    ticker = Prompt.ask("Ticker to predict", default="SPY")
    forward_days = IntPrompt.ask("Forward horizon (trading days)", default=21)
    use_lstm = Confirm.ask("Include LSTM neural network? (slower)", default=False)

    console.print(f"\nTraining models for [bold]{ticker}[/bold] ({forward_days}-day horizon)...\n")

    if use_lstm:
        ens = run_ensemble(ticker, forward_days=forward_days, train_lstm_model=True)
    else:
        ens = run_ensemble(ticker, forward_days=forward_days, train_lstm_model=False)

    _print_ensemble(ens)

    # Also show feature importance from GB model
    if Confirm.ask("\nShow top feature importances?", default=True):
        gb = train_gradient_boosting(ticker, forward_days=forward_days)
        table = Table(title="Top 15 Feature Importances (Gradient Boosting)")
        table.add_column("Feature")
        table.add_column("Importance", justify="right")
        for feat_name, imp in gb.feature_importance.head(15).items():
            table.add_row(feat_name, f"{imp:.4f}")
        console.print(table)

    console.print(
        "\n[dim]Disclaimer: ML predictions are experimental and should not be used "
        "as the sole basis for investment decisions.[/dim]"
    )


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def interactive_menu():
    console.print(Panel.fit(
        "[bold cyan]Investment Research Platform[/bold cyan]\n\n"
        "  [bold]1[/bold]. Historical Pattern Analysis\n"
        "     Study what happened after specific market conditions\n\n"
        "  [bold]2[/bold]. Watchlist Alerts & Scanner\n"
        "     Scan securities for technical signals and anomalies\n\n"
        "  [bold]3[/bold]. ML Price Prediction\n"
        "     Gradient boosting + LSTM neural network forecasts\n\n"
        "  [bold]4[/bold]. Exit",
        title="Main Menu",
        border_style="cyan",
    ))

    while True:
        choice = Prompt.ask("\nSelect module", choices=["1", "2", "3", "4"], default="1")
        if choice == "1":
            mode_history()
        elif choice == "2":
            mode_alerts()
        elif choice == "3":
            mode_predict()
        elif choice == "4":
            console.print("[dim]Goodbye.[/dim]")
            break

        if not Confirm.ask("\nReturn to main menu?", default=True):
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Investment Research Platform")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["history", "alerts", "predict"],
        help="Run a specific module directly",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if args.mode == "history":
        mode_history()
    elif args.mode == "alerts":
        mode_alerts()
    elif args.mode == "predict":
        mode_predict()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()

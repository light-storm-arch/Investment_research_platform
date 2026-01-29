# Investment Research Platform

A Python-based tool for public investment research with three core modules.

## Modules

### 1. Historical Pattern Analysis (`historical_analysis.py`)
Study what happened historically after specific market conditions:
- **S&P 500 up >20% in trailing year** — what happens next?
- **VIX crosses above 30** — how does the S&P perform over the next 1W, 1M, 3M, 6M, 1Y?
- **Drawdown exceeds 10%** — recovery statistics
- **Price crosses 200-day SMA** — trend-following signals
- Custom conditions with arbitrary tickers and thresholds

Outputs include forward return statistics (mean, median, win rate, best/worst) at multiple horizons.

### 2. Watchlist Alerts (`watchlist_alerts.py`)
Scan a list of securities for technical signals and anomalies:
- RSI overbought/oversold
- Bollinger Band breakouts
- MACD crossovers
- Unusual volume spikes
- New 52-week highs/lows
- Price gaps
- Mean reversion z-score signals

### 3. ML Price Prediction (`ml_predictor.py`)
Two model approaches for predicting forward price direction and magnitude:
- **Gradient Boosting** (scikit-learn) — fast, interpretable, with feature importance
- **LSTM Neural Network** (PyTorch) — sequence-based deep learning
- Ensemble mode combines both models for a consensus forecast

Feature engineering includes 30+ features: multi-horizon returns, volatility, RSI, MACD, Bollinger position, volume ratios, ATR, drawdown, and calendar features.

## Setup

```bash
pip install -r requirements.txt
```

For FRED economic data (optional), set the `FRED_API_KEY` environment variable:
```bash
export FRED_API_KEY=your_key_here
```

## Usage

```bash
# Interactive menu
python main.py

# Run a specific module
python main.py history    # Historical pattern analysis
python main.py alerts     # Watchlist scanner
python main.py predict    # ML prediction

# Verbose logging
python main.py -v alerts
```

## Project Structure

```
├── main.py                 # CLI entry point and interactive menus
├── data_fetcher.py         # Data layer (Yahoo Finance, FRED)
├── historical_analysis.py  # Module 1: pattern studies
├── watchlist_alerts.py     # Module 2: technical alerts
├── ml_predictor.py         # Module 3: ML/neural net prediction
├── requirements.txt
└── README.md
```

## Disclaimer

This tool is for research and educational purposes only. ML predictions are experimental and should not be used as the sole basis for investment decisions.

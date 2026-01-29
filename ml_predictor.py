"""
Module 3: ML / Neural Network Price Prediction

Uses multiple approaches:
1. Feature-engineered gradient boosting (sklearn)
2. LSTM neural network (PyTorch)

Both models predict the *direction* (up/down) and *magnitude* of forward returns
over a configurable horizon.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from data_fetcher import fetch_price_history

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
    """Create a rich feature set from OHLCV data.

    Returns a DataFrame with features and target columns:
    - target_return: forward return over *forward_days*
    - target_direction: 1 if positive, 0 otherwise
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns at multiple horizons
    for d in [1, 5, 10, 21, 63, 126, 252]:
        feat[f"return_{d}d"] = close.pct_change(d)

    # Volatility
    daily_ret = close.pct_change()
    for w in [10, 21, 63]:
        feat[f"volatility_{w}d"] = daily_ret.rolling(w).std()

    # Moving averages ratios
    for w in [10, 20, 50, 100, 200]:
        sma = close.rolling(w).mean()
        feat[f"price_to_sma{w}"] = close / sma - 1

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # Bollinger Band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_position"] = (close - sma20) / (2 * std20)

    # Volume features
    vol_avg = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / vol_avg
    feat["volume_trend"] = volume.rolling(5).mean() / vol_avg

    # Average True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close

    # Day of week / month
    feat["day_of_week"] = df.index.dayofweek
    feat["month"] = df.index.month

    # High-low range
    feat["daily_range"] = (high - low) / close

    # Drawdown from peak
    cummax = close.cummax()
    feat["drawdown"] = (close - cummax) / cummax

    # Targets
    feat["target_return"] = close.pct_change(forward_days).shift(-forward_days)
    feat["target_direction"] = (feat["target_return"] > 0).astype(int)

    return feat.dropna()


# ---------------------------------------------------------------------------
# Gradient Boosting Model
# ---------------------------------------------------------------------------

@dataclass
class GBModelResult:
    """Results from gradient boosting model."""
    ticker: str
    direction_accuracy: float
    mse: float
    feature_importance: pd.Series
    latest_prediction_direction: int
    latest_prediction_return: float
    classification_report: str
    cv_scores: list[float] = field(default_factory=list)


def train_gradient_boosting(
    ticker: str,
    forward_days: int = 21,
    start: str = "2000-01-01",
    test_fraction: float = 0.2,
    n_splits: int = 5,
) -> GBModelResult:
    """Train gradient boosting models for direction and magnitude prediction."""
    df = fetch_price_history(ticker, start=start)
    features_df = engineer_features(df, forward_days=forward_days)

    feature_cols = [c for c in features_df.columns if not c.startswith("target_")]
    X = features_df[feature_cols].values
    y_dir = features_df["target_direction"].values
    y_ret = features_df["target_return"].values

    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]
    y_ret_train, y_ret_test = y_ret[:split_idx], y_ret[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Direction classifier
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train_s, y_dir_train)
    dir_pred = clf.predict(X_test_s)
    dir_acc = accuracy_score(y_dir_test, dir_pred)

    # Time-series cross-validation
    cv_scores = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(X_train_s):
        clf_cv = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        clf_cv.fit(X_train_s[train_idx], y_dir_train[train_idx])
        cv_scores.append(accuracy_score(y_dir_train[val_idx], clf_cv.predict(X_train_s[val_idx])))

    # Magnitude regressor
    reg = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    reg.fit(X_train_s, y_ret_train)
    ret_pred = reg.predict(X_test_s)
    mse = mean_squared_error(y_ret_test, ret_pred)

    # Feature importance
    importance = pd.Series(clf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)

    # Latest prediction
    latest_X = scaler.transform(X[-1:])
    latest_dir = int(clf.predict(latest_X)[0])
    latest_ret = float(reg.predict(latest_X)[0])

    cls_report = classification_report(y_dir_test, dir_pred, target_names=["Down", "Up"])

    return GBModelResult(
        ticker=ticker,
        direction_accuracy=round(dir_acc, 4),
        mse=round(mse, 6),
        feature_importance=importance,
        latest_prediction_direction=latest_dir,
        latest_prediction_return=round(latest_ret, 4),
        classification_report=cls_report,
        cv_scores=cv_scores,
    )


# ---------------------------------------------------------------------------
# LSTM Neural Network Model (PyTorch)
# ---------------------------------------------------------------------------

@dataclass
class LSTMModelResult:
    ticker: str
    direction_accuracy: float
    train_loss_history: list[float]
    latest_prediction_direction: int
    latest_prediction_return: float


def train_lstm(
    ticker: str,
    forward_days: int = 21,
    start: str = "2000-01-01",
    sequence_length: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    test_fraction: float = 0.2,
) -> LSTMModelResult:
    """Train an LSTM model for price direction prediction."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    df = fetch_price_history(ticker, start=start)
    features_df = engineer_features(df, forward_days=forward_days)

    feature_cols = [c for c in features_df.columns if not c.startswith("target_")]
    X_raw = features_df[feature_cols].values
    y_dir = features_df["target_direction"].values
    y_ret = features_df["target_return"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Create sequences
    X_seq, y_dir_seq, y_ret_seq = [], [], []
    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i - sequence_length:i])
        y_dir_seq.append(y_dir[i])
        y_ret_seq.append(y_ret[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_dir_seq = np.array(y_dir_seq, dtype=np.float32)
    y_ret_seq = np.array(y_ret_seq, dtype=np.float32)

    split = int(len(X_seq) * (1 - test_fraction))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_dir_train, y_dir_test = y_dir_seq[:split], y_dir_seq[split:]

    # PyTorch datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_dir_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    # Model definition
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden, layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.fc_dir = nn.Linear(hidden, 1)
            self.fc_ret = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            direction = torch.sigmoid(self.fc_dir(last))
            magnitude = self.fc_ret(last)
            return direction.squeeze(-1), magnitude.squeeze(-1)

    input_size = X_train.shape[2]
    model = LSTMPredictor(input_size, hidden_size, num_layers).to(device)

    criterion_dir = nn.BCELoss()
    criterion_ret = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb_dir in train_loader:
            xb = xb.to(device)
            yb_dir = yb_dir.to(device)

            dir_out, _ = model(xb)
            loss = criterion_dir(dir_out, yb_dir)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).to(device)
        dir_out, ret_out = model(X_test_t)
        dir_preds = (dir_out.cpu().numpy() > 0.5).astype(int)
        dir_acc = accuracy_score(y_dir_test, dir_preds)

        # Latest prediction
        latest = torch.tensor(X_seq[-1:]).to(device)
        lat_dir, lat_ret = model(latest)
        latest_dir = int(lat_dir.cpu().numpy()[0] > 0.5)
        latest_ret = float(lat_ret.cpu().numpy()[0])

    return LSTMModelResult(
        ticker=ticker,
        direction_accuracy=round(dir_acc, 4),
        train_loss_history=loss_history,
        latest_prediction_direction=latest_dir,
        latest_prediction_return=round(latest_ret, 4),
    )


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePrediction:
    ticker: str
    forward_days: int
    gb_direction: int
    gb_return: float
    gb_accuracy: float
    lstm_direction: int
    lstm_return: float
    lstm_accuracy: float
    consensus_direction: str  # "Bullish", "Bearish", "Mixed"
    avg_predicted_return: float


def run_ensemble(
    ticker: str,
    forward_days: int = 21,
    start: str = "2000-01-01",
    train_lstm_model: bool = True,
) -> EnsemblePrediction:
    """Run both GB and LSTM models and combine predictions."""
    gb_result = train_gradient_boosting(ticker, forward_days=forward_days, start=start)

    if train_lstm_model:
        lstm_result = train_lstm(ticker, forward_days=forward_days, start=start, epochs=30)
    else:
        # Placeholder if LSTM training is skipped
        lstm_result = LSTMModelResult(
            ticker=ticker, direction_accuracy=0.0,
            train_loss_history=[], latest_prediction_direction=-1,
            latest_prediction_return=0.0,
        )

    # Consensus
    if train_lstm_model:
        if gb_result.latest_prediction_direction == 1 and lstm_result.latest_prediction_direction == 1:
            consensus = "Bullish"
        elif gb_result.latest_prediction_direction == 0 and lstm_result.latest_prediction_direction == 0:
            consensus = "Bearish"
        else:
            consensus = "Mixed"
        avg_ret = (gb_result.latest_prediction_return + lstm_result.latest_prediction_return) / 2
    else:
        consensus = "Bullish" if gb_result.latest_prediction_direction == 1 else "Bearish"
        avg_ret = gb_result.latest_prediction_return

    return EnsemblePrediction(
        ticker=ticker,
        forward_days=forward_days,
        gb_direction=gb_result.latest_prediction_direction,
        gb_return=gb_result.latest_prediction_return,
        gb_accuracy=gb_result.direction_accuracy,
        lstm_direction=lstm_result.latest_prediction_direction,
        lstm_return=lstm_result.latest_prediction_return,
        lstm_accuracy=lstm_result.direction_accuracy,
        consensus_direction=consensus,
        avg_predicted_return=round(avg_ret, 4),
    )

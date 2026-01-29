"""
Investment Research Platform — Streamlit Web Interface

Run with:
    streamlit run app.py
"""

import datetime
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_fetcher import fetch_price_history, compute_rolling_return, compute_drawdown

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Investment Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Investment Research")
page = st.sidebar.radio(
    "Module",
    ["Home", "Historical Analysis", "Watchlist Alerts", "ML Prediction"],
)

# ===================================================================
# HOME PAGE
# ===================================================================

if page == "Home":
    st.title("Investment Research Platform")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📜 Historical Analysis")
        st.write(
            "Study what happened after specific market conditions. "
            "For example: *What happens after the S&P 500 gains 20% in a year?* "
            "or *How does the market perform after VIX spikes above 30?*"
        )
    with col2:
        st.subheader("🔔 Watchlist Alerts")
        st.write(
            "Scan a list of securities for technical signals: "
            "RSI overbought/oversold, Bollinger Band breakouts, MACD crossovers, "
            "volume spikes, 52-week highs/lows, and more."
        )
    with col3:
        st.subheader("🤖 ML Prediction")
        st.write(
            "Use gradient boosting and LSTM neural networks to predict "
            "future price direction and magnitude. 30+ engineered features "
            "with ensemble consensus forecasts."
        )

    st.markdown("---")
    st.info("Select a module from the sidebar to get started.")

# ===================================================================
# HISTORICAL ANALYSIS PAGE
# ===================================================================

elif page == "Historical Analysis":
    st.title("Historical Pattern Analysis")
    st.markdown("Study forward returns after specific market conditions occur.")
    st.markdown("---")

    from historical_analysis import (
        run_pattern_study,
        condition_annual_return_exceeds,
        condition_annual_return_below,
        condition_vix_above,
        condition_drawdown_exceeds,
        condition_price_crosses_sma,
        condition_consecutive_down_days,
    )

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Study Settings")

    study_type = st.sidebar.selectbox("Study type", [
        "Trailing annual return exceeds threshold",
        "Trailing annual return below threshold",
        "VIX crosses above level",
        "Drawdown exceeds threshold",
        "Price crosses SMA",
        "Consecutive down days",
    ])

    if "VIX" in study_type:
        cond_ticker = st.sidebar.text_input("Condition ticker", value="^VIX")
        fwd_ticker = st.sidebar.text_input("Forward return ticker", value="^GSPC")
    else:
        cond_ticker = st.sidebar.text_input("Ticker", value="^GSPC")
        fwd_ticker = st.sidebar.text_input(
            "Forward return ticker (leave same to use above)", value=""
        )
        if not fwd_ticker.strip():
            fwd_ticker = cond_ticker

    start_date = st.sidebar.text_input("Start date", value="1990-01-01")
    min_gap = st.sidebar.slider("Min gap between triggers (days)", 5, 252, 63)

    # Condition-specific parameters
    if study_type == "Trailing annual return exceeds threshold":
        threshold = st.sidebar.slider("Return threshold", 0.05, 0.50, 0.20, 0.01, format="%.0f%%")
    elif study_type == "Trailing annual return below threshold":
        threshold = st.sidebar.slider("Return threshold", -0.50, -0.05, -0.20, 0.01, format="%.0f%%")
    elif "VIX" in study_type:
        vix_level = st.sidebar.slider("VIX level", 15.0, 80.0, 30.0, 1.0)
    elif "Drawdown" in study_type:
        dd_threshold = st.sidebar.slider("Drawdown threshold", -0.50, -0.05, -0.10, 0.01, format="%.0f%%")
    elif "SMA" in study_type:
        sma_window = st.sidebar.slider("SMA window (days)", 20, 400, 200, 10)
        sma_direction = st.sidebar.selectbox("Cross direction", ["above", "below"])
    elif "Consecutive" in study_type:
        n_days = st.sidebar.slider("Consecutive down days", 3, 10, 5)

    # --- Run study ---
    if st.button("Run Study", type="primary"):
        with st.spinner("Fetching data and running analysis..."):
            try:
                cond_df = fetch_price_history(cond_ticker, start=start_date)
                fwd_df = fetch_price_history(fwd_ticker, start=start_date) if fwd_ticker != cond_ticker else cond_df

                if study_type == "Trailing annual return exceeds threshold":
                    cond_fn = condition_annual_return_exceeds(threshold)
                    desc = f"{cond_ticker} trailing 1Y return > {threshold:.0%}"
                elif study_type == "Trailing annual return below threshold":
                    cond_fn = condition_annual_return_below(threshold)
                    desc = f"{cond_ticker} trailing 1Y return < {threshold:.0%}"
                elif "VIX" in study_type:
                    cond_fn = condition_vix_above(vix_level)
                    desc = f"VIX crosses above {vix_level:.0f}"
                elif "Drawdown" in study_type:
                    cond_fn = condition_drawdown_exceeds(dd_threshold)
                    desc = f"{cond_ticker} drawdown exceeds {dd_threshold:.0%}"
                elif "SMA" in study_type:
                    cond_fn = condition_price_crosses_sma(sma_window, sma_direction)
                    desc = f"{cond_ticker} crosses {sma_direction} {sma_window}-day SMA"
                else:
                    cond_fn = condition_consecutive_down_days(n_days)
                    desc = f"{cond_ticker} has {n_days} consecutive down days"

                result = run_pattern_study(
                    cond_df, cond_fn,
                    description=desc,
                    forward_ticker_df=fwd_df,
                    min_gap_days=min_gap,
                )

                # --- Display results ---
                st.subheader(f"Results: {desc}")
                st.metric("Trigger dates found", len(result.trigger_dates))

                if result.summary is not None and not result.summary.empty:
                    # Summary table
                    st.markdown("#### Forward Return Statistics")
                    st.dataframe(
                        result.summary.set_index("Horizon"),
                        use_container_width=True,
                    )

                    # Bar chart of mean returns
                    fig_mean = go.Figure()
                    colors = [
                        "green" if v >= 0 else "red"
                        for v in result.summary["Mean Return (%)"]
                    ]
                    fig_mean.add_trace(go.Bar(
                        x=result.summary["Horizon"],
                        y=result.summary["Mean Return (%)"],
                        marker_color=colors,
                        text=[f"{v:+.1f}%" for v in result.summary["Mean Return (%)"]],
                        textposition="outside",
                    ))
                    fig_mean.update_layout(
                        title="Mean Forward Return by Horizon",
                        yaxis_title="Return (%)",
                        xaxis_title="Horizon",
                        height=400,
                    )
                    st.plotly_chart(fig_mean, use_container_width=True)

                    # Win rate chart
                    fig_win = go.Figure()
                    fig_win.add_trace(go.Bar(
                        x=result.summary["Horizon"],
                        y=result.summary["Win Rate (%)"],
                        marker_color="steelblue",
                        text=[f"{v:.0f}%" for v in result.summary["Win Rate (%)"]],
                        textposition="outside",
                    ))
                    fig_win.add_hline(y=50, line_dash="dash", line_color="gray")
                    fig_win.update_layout(
                        title="Win Rate (% of positive outcomes) by Horizon",
                        yaxis_title="Win Rate (%)",
                        xaxis_title="Horizon",
                        yaxis_range=[0, 100],
                        height=400,
                    )
                    st.plotly_chart(fig_win, use_container_width=True)

                    # Distribution of forward returns for each horizon
                    st.markdown("#### Return Distributions")
                    horizon_choice = st.selectbox(
                        "Select horizon to view distribution",
                        list(result.forward_returns.keys()),
                    )
                    rets = result.forward_returns[horizon_choice].dropna()
                    if len(rets) > 0:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=rets.values * 100,
                            nbinsx=20,
                            marker_color="steelblue",
                        ))
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                        fig_hist.add_vline(
                            x=rets.mean() * 100,
                            line_dash="dot",
                            line_color="green",
                            annotation_text=f"Mean: {rets.mean() * 100:.1f}%",
                        )
                        fig_hist.update_layout(
                            title=f"{horizon_choice} Forward Return Distribution (n={len(rets)})",
                            xaxis_title="Return (%)",
                            yaxis_title="Count",
                            height=400,
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                # Trigger dates timeline
                if result.trigger_dates:
                    st.markdown("#### Trigger Dates")
                    with st.expander(f"Show all {len(result.trigger_dates)} trigger dates"):
                        dates_df = pd.DataFrame({
                            "Date": [d.strftime("%Y-%m-%d") for d in result.trigger_dates],
                        })
                        st.dataframe(dates_df, use_container_width=True, hide_index=True)

                    # Price chart with trigger markers
                    st.markdown("#### Price Chart with Trigger Points")
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=fwd_df.index,
                        y=fwd_df["Close"],
                        mode="lines",
                        name=fwd_ticker,
                        line=dict(color="steelblue", width=1),
                    ))
                    trigger_prices = []
                    for d in result.trigger_dates:
                        loc = fwd_df.index.get_indexer([d], method="nearest")[0]
                        trigger_prices.append(fwd_df["Close"].iloc[loc])
                    fig_price.add_trace(go.Scatter(
                        x=result.trigger_dates,
                        y=trigger_prices,
                        mode="markers",
                        name="Trigger",
                        marker=dict(color="red", size=8, symbol="triangle-up"),
                    ))
                    fig_price.update_layout(
                        title=f"{fwd_ticker} Price with Trigger Points",
                        yaxis_title="Price",
                        height=500,
                    )
                    st.plotly_chart(fig_price, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ===================================================================
# WATCHLIST ALERTS PAGE
# ===================================================================

elif page == "Watchlist Alerts":
    st.title("Watchlist Alerts & Scanner")
    st.markdown("Scan securities for technical signals and anomalies.")
    st.markdown("---")

    from watchlist_alerts import (
        scan_watchlist,
        generate_alerts,
        build_ticker_report,
        compute_rsi,
        compute_bollinger,
        compute_macd,
        AlertSeverity,
    )

    DEFAULT_WATCHLIST = [
        "SPY", "QQQ", "IWM", "DIA",
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "XLF", "XLE", "XLV", "XLK",
        "TLT", "HYG", "GLD", "USO",
    ]

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Watchlist Settings")

    watchlist_mode = st.sidebar.radio("Watchlist", ["Default", "Custom"])
    if watchlist_mode == "Custom":
        custom_input = st.sidebar.text_area(
            "Enter tickers (one per line or comma-separated)",
            value="SPY, QQQ, AAPL, MSFT, NVDA",
        )
        tickers = [t.strip().upper() for t in custom_input.replace("\n", ",").split(",") if t.strip()]
    else:
        tickers = DEFAULT_WATCHLIST

    st.sidebar.markdown(f"**Scanning {len(tickers)} securities**")

    # --- Run scan ---
    if st.button("Scan Watchlist", type="primary"):
        progress_bar = st.progress(0, text="Scanning...")
        reports = []
        for i, t in enumerate(tickers):
            try:
                start = (datetime.date.today() - datetime.timedelta(days=425)).isoformat()
                df = fetch_price_history(t, start=start)
                if len(df) >= 50:
                    report = build_ticker_report(t, df)
                    reports.append(report)
            except Exception as e:
                st.warning(f"Skipped {t}: {e}")
            progress_bar.progress((i + 1) / len(tickers), text=f"Scanning {t}...")

        progress_bar.empty()
        reports.sort(key=lambda r: len(r.alerts), reverse=True)

        # Store in session state for detail views
        st.session_state["watchlist_reports"] = reports

        # --- Summary dashboard ---
        total_alerts = sum(len(r.alerts) for r in reports)
        critical_alerts = sum(
            1 for r in reports for a in r.alerts if a.severity == AlertSeverity.CRITICAL
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Securities Scanned", len(reports))
        col2.metric("Total Alerts", total_alerts)
        col3.metric("Critical Alerts", critical_alerts)
        col4.metric("Clean (no alerts)", sum(1 for r in reports if not r.alerts))

        st.markdown("---")

        # Summary table
        st.subheader("Watchlist Overview")
        summary_data = []
        for r in reports:
            summary_data.append({
                "Ticker": r.ticker,
                "Price": f"${r.last_price}",
                "Change (%)": r.daily_change_pct,
                "RSI (14)": r.rsi_14,
                "MACD Diff": r.macd_signal_diff,
                "BB Position": r.bb_position,
                "Vol Ratio": r.volume_vs_avg,
                "52W High": f"${r.high_52w}",
                "52W Low": f"${r.low_52w}",
            "Alerts": len(r.alerts),
            })

        summary_df = pd.DataFrame(summary_data)

        def _highlight_alerts(val):
            if isinstance(val, (int, float)):
                if val >= 3:
                    return "background-color: #ff4444; color: white"
                elif val >= 1:
                    return "background-color: #ffaa00; color: black"
            return ""

        def _highlight_rsi(val):
            if isinstance(val, (int, float)):
                if val >= 70:
                    return "color: #ff4444; font-weight: bold"
                elif val <= 30:
                    return "color: #44bb44; font-weight: bold"
            return ""

        def _highlight_change(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #44bb44"
                elif val < 0:
                    return "color: #ff4444"
            return ""

        styled = summary_df.style.map(
            _highlight_alerts, subset=["Alerts"]
        ).map(
            _highlight_rsi, subset=["RSI (14)"]
        ).map(
            _highlight_change, subset=["Change (%)"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(400, 40 + 35 * len(reports)))

        # --- Alert details ---
        tickers_with_alerts = [r for r in reports if r.alerts]
        if tickers_with_alerts:
            st.markdown("---")
            st.subheader(f"Alert Details ({sum(len(r.alerts) for r in tickers_with_alerts)} alerts)")

            for r in tickers_with_alerts:
                severity_icon = {
                    "INFO": "ℹ️",
                    "WARNING": "⚠️",
                    "CRITICAL": "🚨",
                }
                with st.expander(f"{r.ticker} — {len(r.alerts)} alert(s) — ${r.last_price}", expanded=(len(r.alerts) >= 3)):
                    for a in r.alerts:
                        icon = severity_icon.get(a.severity.value, "")
                        st.markdown(f"{icon} **{a.alert_type.value}**: {a.message}")

        # --- RSI heatmap ---
        st.markdown("---")
        st.subheader("RSI Heatmap")
        rsi_data = pd.DataFrame({
            "Ticker": [r.ticker for r in reports],
            "RSI": [r.rsi_14 for r in reports],
        }).set_index("Ticker")

        fig_rsi = go.Figure()
        colors = []
        for rsi_val in rsi_data["RSI"]:
            if rsi_val >= 70:
                colors.append("#ff4444")
            elif rsi_val <= 30:
                colors.append("#44bb44")
            elif rsi_val >= 60:
                colors.append("#ffaa44")
            elif rsi_val <= 40:
                colors.append("#88cc88")
            else:
                colors.append("#888888")

        fig_rsi.add_trace(go.Bar(
            x=rsi_data.index,
            y=rsi_data["RSI"],
            marker_color=colors,
            text=[f"{v:.0f}" for v in rsi_data["RSI"]],
            textposition="outside",
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(
            yaxis_range=[0, 100],
            height=400,
            yaxis_title="RSI (14)",
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # --- Detail view for a single ticker ---
    if "watchlist_reports" in st.session_state and st.session_state["watchlist_reports"]:
        st.markdown("---")
        st.subheader("Detailed Ticker View")
        reports = st.session_state["watchlist_reports"]
        ticker_names = [r.ticker for r in reports]
        selected = st.selectbox("Select ticker for detail view", ticker_names)

        report = next(r for r in reports if r.ticker == selected)

        col1, col2, col3, col4 = st.columns(4)
        change_delta = f"{report.daily_change_pct:+.2f}%"
        col1.metric("Price", f"${report.last_price}", change_delta)
        col2.metric("RSI (14)", f"{report.rsi_14}")
        col3.metric("Volume Ratio", f"{report.volume_vs_avg}x")
        col4.metric("BB Position", f"{report.bb_position}")

        # Fetch fresh data for charting
        try:
            start = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
            df = fetch_price_history(selected, start=start)
            close = df["Close"]

            # Price + Bollinger Bands chart
            bb_upper, bb_mid, bb_lower = compute_bollinger(close)
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df.index, y=bb_upper, mode="lines", name="Upper BB", line=dict(dash="dash", color="rgba(255,0,0,0.3)")))
            fig_bb.add_trace(go.Scatter(x=df.index, y=bb_lower, mode="lines", name="Lower BB", line=dict(dash="dash", color="rgba(0,255,0,0.3)"), fill="tonexty", fillcolor="rgba(100,100,100,0.1)"))
            fig_bb.add_trace(go.Scatter(x=df.index, y=bb_mid, mode="lines", name="SMA(20)", line=dict(color="orange", width=1)))
            fig_bb.add_trace(go.Scatter(x=df.index, y=close, mode="lines", name=selected, line=dict(color="steelblue", width=2)))
            fig_bb.update_layout(title=f"{selected} — Price & Bollinger Bands", height=450, yaxis_title="Price")
            st.plotly_chart(fig_bb, use_container_width=True)

            # RSI chart
            rsi_series = compute_rsi(close)
            fig_rsi_detail = go.Figure()
            fig_rsi_detail.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, mode="lines", name="RSI(14)", line=dict(color="purple")))
            fig_rsi_detail.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi_detail.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi_detail.update_layout(title="RSI (14)", height=300, yaxis_range=[0, 100], yaxis_title="RSI")
            st.plotly_chart(fig_rsi_detail, use_container_width=True)

            # MACD chart
            macd_line, macd_signal, macd_diff = compute_macd(close)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=macd_diff.index, y=macd_diff, name="Histogram", marker_color=["green" if v >= 0 else "red" for v in macd_diff]))
            fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode="lines", name="MACD", line=dict(color="blue")))
            fig_macd.add_trace(go.Scatter(x=macd_signal.index, y=macd_signal, mode="lines", name="Signal", line=dict(color="orange")))
            fig_macd.update_layout(title="MACD", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)

            # Volume chart
            fig_vol = go.Figure()
            vol_colors = ["green" if close.iloc[i] >= close.iloc[i - 1] else "red" for i in range(1, len(close))]
            vol_colors.insert(0, "gray")
            fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"))
            vol_avg = df["Volume"].rolling(20).mean()
            fig_vol.add_trace(go.Scatter(x=df.index, y=vol_avg, mode="lines", name="20-day avg", line=dict(color="orange")))
            fig_vol.update_layout(title="Volume", height=300, yaxis_title="Volume")
            st.plotly_chart(fig_vol, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load detail charts: {e}")


# ===================================================================
# ML PREDICTION PAGE
# ===================================================================

elif page == "ML Prediction":
    st.title("ML Price Prediction")
    st.markdown("Gradient boosting and LSTM neural network ensemble forecasts.")
    st.markdown("---")

    from ml_predictor import (
        train_gradient_boosting,
        train_lstm,
        run_ensemble,
        engineer_features,
    )

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Settings")

    pred_ticker = st.sidebar.text_input("Ticker", value="SPY")
    forward_days = st.sidebar.slider("Forward horizon (trading days)", 5, 126, 21)
    pred_start = st.sidebar.text_input("Training start date", value="2000-01-01")
    use_lstm = st.sidebar.checkbox("Include LSTM neural network (slower)", value=False)

    if use_lstm:
        lstm_epochs = st.sidebar.slider("LSTM epochs", 10, 100, 30)
        lstm_hidden = st.sidebar.slider("LSTM hidden size", 32, 128, 64)

    if st.button("Run Prediction", type="primary"):

        # --- Gradient Boosting ---
        st.subheader("Gradient Boosting Model")
        with st.spinner("Training gradient boosting model..."):
            try:
                gb_result = train_gradient_boosting(
                    pred_ticker,
                    forward_days=forward_days,
                    start=pred_start,
                )

                col1, col2, col3 = st.columns(3)
                dir_label = "UP" if gb_result.latest_prediction_direction == 1 else "DOWN"
                dir_color = "green" if gb_result.latest_prediction_direction == 1 else "red"
                col1.metric("Predicted Direction", dir_label)
                col2.metric("Predicted Return", f"{gb_result.latest_prediction_return:+.2%}")
                col3.metric("Test Accuracy", f"{gb_result.direction_accuracy:.1%}")

                # CV scores
                if gb_result.cv_scores:
                    cv_mean = np.mean(gb_result.cv_scores)
                    cv_std = np.std(gb_result.cv_scores)
                    st.info(f"Cross-validation accuracy: {cv_mean:.1%} ± {cv_std:.1%} (5-fold time-series split)")

                # Classification report
                with st.expander("Classification Report"):
                    st.text(gb_result.classification_report)

                # Feature importance chart
                st.markdown("#### Top 20 Feature Importances")
                top_features = gb_result.feature_importance.head(20)
                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation="h",
                    marker_color="steelblue",
                ))
                fig_imp.update_layout(
                    height=500,
                    xaxis_title="Importance",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            except Exception as e:
                st.error(f"Gradient Boosting error: {e}")
                gb_result = None

        # --- LSTM ---
        lstm_result = None
        if use_lstm:
            st.markdown("---")
            st.subheader("LSTM Neural Network")
            with st.spinner("Training LSTM model (this may take a minute)..."):
                try:
                    lstm_result = train_lstm(
                        pred_ticker,
                        forward_days=forward_days,
                        start=pred_start,
                        epochs=lstm_epochs,
                        hidden_size=lstm_hidden,
                    )

                    col1, col2, col3 = st.columns(3)
                    dir_label = "UP" if lstm_result.latest_prediction_direction == 1 else "DOWN"
                    col1.metric("Predicted Direction", dir_label)
                    col2.metric("Predicted Return", f"{lstm_result.latest_prediction_return:+.2%}")
                    col3.metric("Test Accuracy", f"{lstm_result.direction_accuracy:.1%}")

                    # Training loss chart
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=lstm_result.train_loss_history,
                        mode="lines",
                        name="Training Loss",
                        line=dict(color="orange"),
                    ))
                    fig_loss.update_layout(
                        title="LSTM Training Loss",
                        xaxis_title="Epoch",
                        yaxis_title="Loss (BCE)",
                        height=350,
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                except Exception as e:
                    st.error(f"LSTM error: {e}")

        # --- Ensemble ---
        if gb_result is not None:
            st.markdown("---")
            st.subheader("Ensemble Consensus")

            if lstm_result and lstm_result.latest_prediction_direction >= 0:
                if gb_result.latest_prediction_direction == 1 and lstm_result.latest_prediction_direction == 1:
                    consensus = "Bullish"
                    consensus_icon = "🟢"
                elif gb_result.latest_prediction_direction == 0 and lstm_result.latest_prediction_direction == 0:
                    consensus = "Bearish"
                    consensus_icon = "🔴"
                else:
                    consensus = "Mixed"
                    consensus_icon = "🟡"
                avg_ret = (gb_result.latest_prediction_return + lstm_result.latest_prediction_return) / 2
            else:
                consensus = "Bullish" if gb_result.latest_prediction_direction == 1 else "Bearish"
                consensus_icon = "🟢" if gb_result.latest_prediction_direction == 1 else "🔴"
                avg_ret = gb_result.latest_prediction_return

            col1, col2, col3 = st.columns(3)
            col1.metric("Consensus", f"{consensus_icon} {consensus}")
            col2.metric("Avg Predicted Return", f"{avg_ret:+.2%}")
            col3.metric("Horizon", f"{forward_days} trading days")

            # Model comparison table
            comp_data = {
                "Model": ["Gradient Boosting"],
                "Direction": ["UP" if gb_result.latest_prediction_direction == 1 else "DOWN"],
                "Predicted Return": [f"{gb_result.latest_prediction_return:+.2%}"],
                "Test Accuracy": [f"{gb_result.direction_accuracy:.1%}"],
            }
            if lstm_result and lstm_result.latest_prediction_direction >= 0:
                comp_data["Model"].append("LSTM Neural Net")
                comp_data["Direction"].append("UP" if lstm_result.latest_prediction_direction == 1 else "DOWN")
                comp_data["Predicted Return"].append(f"{lstm_result.latest_prediction_return:+.2%}")
                comp_data["Test Accuracy"].append(f"{lstm_result.direction_accuracy:.1%}")

            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.warning(
            "**Disclaimer:** ML predictions are experimental and should not be used "
            "as the sole basis for investment decisions. Past patterns do not guarantee "
            "future results."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption("Investment Research Platform v1.0")
st.sidebar.caption("Data: Yahoo Finance")
st.sidebar.caption("For research purposes only.")

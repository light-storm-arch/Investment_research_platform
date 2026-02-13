"""
Pair Trend Analysis — Streamlit page module.

Renders the UI for selecting asset pairs and displaying trend analysis,
z-scores, volatility, and return dispersion sections.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pair_analysis import (
    PAIR_CONFIG,
    SMA_PERIODS,
    VOL_SHORT,
    VOL_LONG,
    ZSCORE_THRESHOLD,
    DRAWDOWN_THRESHOLD,
    RSI_DEFAULT_WINDOW,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    fetch_pair,
    compute_sma,
    compute_sma_slopes,
    current_sma_readings,
    compute_ratio_rsi,
    compute_rolling_ratio_returns,
    compute_zscore_table,
    compute_ratio_volatility,
    current_vol_readings,
    compute_dispersion_table,
    compute_non_overlapping_returns,
    compute_combined_signal,
    compute_signal_history,
    get_non_overlapping_zscore_table,
)


def _zscore_color(z: float, threshold: float) -> str:
    if z >= threshold:
        return "color: #22c55e; font-weight: bold"
    elif z <= -threshold:
        return "color: #ef4444; font-weight: bold"
    return ""


def _signal_direction_color(direction: str) -> str:
    """Return CSS color for signal direction."""
    if direction == "FAVOR_VALUE":
        return "color: #22c55e; font-weight: bold"
    elif direction == "FAVOR_GROWTH":
        return "color: #ef4444; font-weight: bold"
    elif direction == "MIXED":
        return "color: #f59e0b; font-weight: bold"
    return ""


def _bool_to_status(active: bool) -> str:
    """Convert boolean to status string."""
    return "✓ Active" if active else "—"


def render_pair_trends_page() -> None:
    """Main entry point called from app.py."""

    st.title("Pair Trend Analysis")
    st.markdown(
        "Analyse relative performance trends and regime shifts between "
        "paired asset classes."
    )
    st.markdown("---")

    # ------------------------------------------------------------------
    # Pair selector
    # ------------------------------------------------------------------
    pair_labels = [p["label"] for p in PAIR_CONFIG]
    selected_label = st.selectbox("Select pair", pair_labels, key="pt_pair")
    pair_cfg = PAIR_CONFIG[pair_labels.index(selected_label)]

    # ------------------------------------------------------------------
    # Fetch data (cached in session state keyed by pair label)
    # ------------------------------------------------------------------
    cache_key = f"pt_data_{selected_label}"
    if st.button("Load / Refresh Data", type="primary"):
        with st.spinner(f"Fetching {pair_cfg['ticker_a']} & {pair_cfg['ticker_b']}..."):
            try:
                data = fetch_pair(
                    pair_cfg["ticker_a"],
                    pair_cfg["ticker_b"],
                    pair_cfg["label"],
                )
                st.session_state[cache_key] = data
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")
                return

    if cache_key not in st.session_state:
        st.info("Click **Load / Refresh Data** to fetch prices for this pair.")
        return

    data = st.session_state[cache_key]
    st.caption(f"Data last fetched: {data.fetched_at}")

    # ------------------------------------------------------------------
    # Section 1: Trend Analysis (SMA)
    # ------------------------------------------------------------------
    st.header("1. Trend Analysis (SMA)")

    chart_years = st.slider(
        "Chart lookback (years)", 1, 15, 3, key="pt_sma_years"
    )
    chart_start = pd.Timestamp.now() - pd.DateOffset(years=chart_years)

    sma_df = compute_sma(data.ratio)
    slope_df = compute_sma_slopes(sma_df)
    sma_plot = sma_df.loc[sma_df.index >= chart_start]
    slope_plot = slope_df.loc[slope_df.index >= chart_start]

    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(
        x=sma_plot.index, y=sma_plot["Log Ratio"],
        name="Log Ratio", line=dict(width=2, color="#3b82f6"),
    ))
    colors = {"SMA 50": "#f59e0b", "SMA 100": "#8b5cf6", "SMA 200": "#ef4444"}
    for col in sma_plot.columns:
        if col.startswith("SMA"):
            slope_col = f"{col} Slope"
            slope_vals = slope_plot[slope_col] if slope_col in slope_plot.columns else None
            if slope_vals is not None:
                hover = [
                    f"{col}: {v:.4f}<br>Slope: {s:+.4f}/yr"
                    if not (pd.isna(v) or pd.isna(s))
                    else f"{col}: {v:.4f}"
                    for v, s in zip(sma_plot[col], slope_vals)
                ]
            else:
                hover = [f"{col}: {v:.4f}" for v in sma_plot[col]]
            fig_sma.add_trace(go.Scatter(
                x=sma_plot.index, y=sma_plot[col],
                name=col, line=dict(width=1.5, dash="dash", color=colors.get(col, "#999")),
                hovertext=hover,
                hoverinfo="text+x",
            ))
    fig_sma.update_layout(
        title=f"{data.ticker_a}/{data.ticker_b} Log Price Ratio with SMAs",
        yaxis_title="Log Ratio",
        height=420,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_sma, use_container_width=True)

    # Current readings
    readings = current_sma_readings(sma_df)
    cols = st.columns(len(SMA_PERIODS) + 1)
    cols[0].metric("Log Ratio", f"{readings['Log Ratio']:.4f}")
    for i, p in enumerate(SMA_PERIODS):
        key = f"SMA {p}"
        signal = readings[f"{key} Signal"]
        slope = readings[f"{key} Slope"]
        ann_slope = slope * 252
        slope_dir = "Rising" if slope > 0 else "Falling" if slope < 0 else "Flat"
        cols[i + 1].metric(
            key,
            f"{readings[key]:.4f}",
            delta=f"{signal} · {slope_dir} ({ann_slope:+.4f}/yr)",
            delta_color="normal" if signal == "Above" else "inverse",
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 1b: RSI Momentum
    # ------------------------------------------------------------------
    st.header("1b. RSI Momentum")

    st.info(
        "**Relative Strength Index (RSI)** applied to the log price ratio. "
        "Readings above the overbought threshold suggest "
        f"**{data.ticker_a}** momentum is extended relative to **{data.ticker_b}**; "
        "readings below oversold suggest the reverse."
    )

    rsi_col1, rsi_col2, rsi_col3, rsi_col4 = st.columns(4)
    with rsi_col1:
        rsi_window = st.number_input(
            "Lookback period",
            min_value=2, max_value=200, value=RSI_DEFAULT_WINDOW, step=1,
            key="pt_rsi_window",
        )
    with rsi_col2:
        rsi_freq = st.selectbox(
            "Frequency",
            options=["Daily", "Weekly"],
            index=0,
            key="pt_rsi_freq",
        )
    with rsi_col3:
        rsi_ob = st.number_input(
            "Overbought",
            min_value=50, max_value=100, value=RSI_OVERBOUGHT, step=1,
            key="pt_rsi_ob",
        )
    with rsi_col4:
        rsi_os = st.number_input(
            "Oversold",
            min_value=0, max_value=50, value=RSI_OVERSOLD, step=1,
            key="pt_rsi_os",
        )

    rsi_series = compute_ratio_rsi(
        data.ratio, window=rsi_window, frequency=rsi_freq.lower(),
    )
    rsi_plot = rsi_series.loc[rsi_series.index >= chart_start].dropna()

    if not rsi_plot.empty:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=rsi_plot.index, y=rsi_plot.values,
            name="RSI", line=dict(width=2, color="#3b82f6"),
        ))
        # Overbought / oversold bands
        fig_rsi.add_hline(
            y=rsi_ob, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Overbought ({rsi_ob})",
            annotation_position="top left",
        )
        fig_rsi.add_hline(
            y=rsi_os, line_dash="dash", line_color="#22c55e",
            annotation_text=f"Oversold ({rsi_os})",
            annotation_position="bottom left",
        )
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="#9ca3af")
        fig_rsi.update_layout(
            title=(
                f"{data.ticker_a}/{data.ticker_b} Ratio RSI"
                f" ({rsi_window}-{rsi_freq.lower()})"
            ),
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            height=340,
            margin=dict(t=40, b=30),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Current reading
        current_rsi = rsi_plot.iloc[-1]
        if current_rsi >= rsi_ob:
            rsi_status = "Overbought"
            rsi_delta_color = "inverse"
        elif current_rsi <= rsi_os:
            rsi_status = "Oversold"
            rsi_delta_color = "normal"
        else:
            rsi_status = "Neutral"
            rsi_delta_color = "off"
        st.metric(
            "Current RSI",
            f"{current_rsi:.1f}",
            delta=rsi_status,
            delta_color=rsi_delta_color,
        )
    else:
        st.warning("Not enough data to compute RSI for the selected parameters.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 2: Z-Scores of Non-Overlapping Returns
    # ------------------------------------------------------------------
    st.header("2. Z-Scores of Non-Overlapping Returns")

    st.info(
        "**Non-overlapping returns** are sampled at period ends to eliminate serial "
        "correlation: quarterly (Mar/Jun/Sep/Dec), semi-annual (Jun/Dec), annual (Dec)."
    )

    z_threshold = st.number_input(
        "Z-score highlight threshold",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        key="pt_z_thresh",
    )

    zscore_rows = get_non_overlapping_zscore_table(data)
    if zscore_rows:
        zdf = pd.DataFrame([
            {
                "Period": r.period,
                "Lookback": f"{r.lookback_obs} obs",
                "Current Log Return": f"{r.current_return:+.2%}",
                "Hist Mean": f"{r.hist_mean:+.2%}",
                "Hist Std": f"{r.hist_std:.2%}",
                "Z-Score": round(r.z_score, 2),
            }
            for r in zscore_rows
        ])

        def _style_z(val):
            if isinstance(val, (int, float)):
                return _zscore_color(val, z_threshold)
            return ""

        styled = zdf.style.map(_style_z, subset=["Z-Score"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient history to compute z-scores.")

    st.caption(
        f"**Interpretation:** A positive z-score means **{data.ticker_a}** is "
        f"outperforming **{data.ticker_b}** relative to historical norms. "
        f"Values beyond ±{z_threshold:.1f} are highlighted."
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 2b: Three-Layer Signal Framework
    # ------------------------------------------------------------------
    st.header("2b. Multi-Layer Signal Framework")

    st.markdown(
        """
        **Signal triggers when ALL three layers are active:**
        - **Layer 1:** Current quarter z-score exceeds ±1.5
        - **Layer 2:** Trailing 4-quarter cumulative z-score exceeds ±1.5
        - **Layer 3:** Drawdown from peak or rally from trough exceeds 10%
        """
    )

    signal_state = compute_combined_signal(data, z_threshold, DRAWDOWN_THRESHOLD)

    if signal_state:
        # Signal Status Banner
        if signal_state.signal_active:
            if signal_state.direction == "FAVOR_VALUE":
                st.success(
                    f"🟢 **SIGNAL ACTIVE: FAVOR VALUE ({data.ticker_a})**\n\n"
                    f"Value has underperformed significantly — expect mean reversion."
                )
            elif signal_state.direction == "FAVOR_GROWTH":
                st.error(
                    f"🔴 **SIGNAL ACTIVE: FAVOR GROWTH ({data.ticker_b})**\n\n"
                    f"Growth has underperformed significantly — expect mean reversion."
                )
            else:
                st.warning(
                    f"🟡 **SIGNAL ACTIVE: MIXED**\n\n"
                    f"All layers triggered but z-score signs are inconsistent."
                )
        else:
            st.info("⚪ **NO SIGNAL** — Not all layers are active.")

        # Layer Details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Layer 1: Single Quarter")
            l1 = signal_state.layer1
            status_color = "green" if l1.is_active else "gray"
            st.markdown(f"**Status:** :{status_color}[{_bool_to_status(l1.is_active)}]")
            st.metric("Current Q Return", f"{l1.current_return:+.2%}")
            st.metric("Z-Score", f"{l1.z_score:+.2f}")
            st.caption(f"Historical: μ={l1.hist_mean:+.2%}, σ={l1.hist_std:.2%}")
            st.caption(f"Based on {l1.lookback_obs} quarters")

        with col2:
            st.subheader("Layer 2: Trailing 4Q")
            l2 = signal_state.layer2
            status_color = "green" if l2.is_active else "gray"
            st.markdown(f"**Status:** :{status_color}[{_bool_to_status(l2.is_active)}]")
            st.metric("4Q Sum", f"{l2.trailing_4q_sum:+.2%}")
            st.metric("Z-Score", f"{l2.z_score:+.2f}")
            st.caption(f"Historical: μ={l2.hist_mean:+.2%}, σ={l2.hist_std:.2%}")
            st.caption(f"Based on {l2.lookback_obs} 4Q periods")

        with col3:
            st.subheader("Layer 3: Drawdown/Rally")
            l3 = signal_state.layer3
            status_color = "green" if l3.is_active else "gray"
            st.markdown(f"**Status:** :{status_color}[{_bool_to_status(l3.is_active)}]")
            st.metric("Cumulative Spread", f"{l3.cumulative_spread:.4f}")
            st.metric("From High", f"{l3.drawdown_from_high:+.2%}")
            st.metric("From Low", f"{l3.rally_from_low:+.2%}")
            st.caption(f"Direction: {l3.direction.title()}")

        # Signal History Chart
        st.subheader("Signal History")
        signal_hist = compute_signal_history(data, z_threshold, DRAWDOWN_THRESHOLD)

        if not signal_hist.empty:
            # Filter to chart period
            signal_hist["date"] = pd.to_datetime(signal_hist["date"])
            hist_plot = signal_hist[signal_hist["date"] >= chart_start]

            if not hist_plot.empty:
                fig_signal = go.Figure()

                # Plot cumulative spread
                fig_signal.add_trace(go.Scatter(
                    x=hist_plot["date"],
                    y=hist_plot["cumulative_spread"],
                    name="Cumulative Spread",
                    line=dict(width=2, color="#3b82f6"),
                ))

                # Add markers for signals
                favor_value = hist_plot[hist_plot["direction"] == "FAVOR_VALUE"]
                favor_growth = hist_plot[hist_plot["direction"] == "FAVOR_GROWTH"]
                mixed = hist_plot[hist_plot["direction"] == "MIXED"]

                if not favor_value.empty:
                    fig_signal.add_trace(go.Scatter(
                        x=favor_value["date"],
                        y=favor_value["cumulative_spread"],
                        mode="markers",
                        name="Favor Value",
                        marker=dict(size=12, color="#22c55e", symbol="triangle-up"),
                    ))

                if not favor_growth.empty:
                    fig_signal.add_trace(go.Scatter(
                        x=favor_growth["date"],
                        y=favor_growth["cumulative_spread"],
                        mode="markers",
                        name="Favor Growth",
                        marker=dict(size=12, color="#ef4444", symbol="triangle-down"),
                    ))

                if not mixed.empty:
                    fig_signal.add_trace(go.Scatter(
                        x=mixed["date"],
                        y=mixed["cumulative_spread"],
                        mode="markers",
                        name="Mixed Signal",
                        marker=dict(size=10, color="#f59e0b", symbol="diamond"),
                    ))

                fig_signal.update_layout(
                    title=f"{data.ticker_a}/{data.ticker_b} Signal History (Quarterly)",
                    yaxis_title="Log Spread",
                    height=400,
                    margin=dict(t=40, b=30),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_signal, use_container_width=True)

                # Show recent signals table
                recent_signals = hist_plot[hist_plot["signal_active"]].tail(10)
                if not recent_signals.empty:
                    st.subheader("Recent Signals")
                    sig_table = pd.DataFrame({
                        "Date": recent_signals["date"].dt.strftime("%Y-%m-%d"),
                        "Direction": recent_signals["direction"],
                        "L1 Z": recent_signals["l1_zscore"].round(2),
                        "L2 Z": recent_signals["l2_zscore"].round(2),
                        "Drawdown": recent_signals["drawdown"].apply(lambda x: f"{x:+.1%}"),
                        "Rally": recent_signals["rally"].apply(lambda x: f"{x:+.1%}"),
                    })
                    st.dataframe(sig_table, use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient quarterly data to compute signal framework (need at least 5 quarters).")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 3: Volatility of the Ratio
    # ------------------------------------------------------------------
    st.header("3. Volatility of the Ratio")

    vol_df = compute_ratio_volatility(data.ratio)
    vol_plot = vol_df.loc[vol_df.index >= chart_start]

    short_col = f"{VOL_SHORT}-Day Vol"
    long_col = f"{VOL_LONG}-Day Vol"

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=vol_plot.index, y=vol_plot[short_col],
        name=short_col, line=dict(width=2, color="#f59e0b"),
    ))
    fig_vol.add_trace(go.Scatter(
        x=vol_plot.index, y=vol_plot[long_col],
        name=long_col, line=dict(width=2, color="#6366f1"),
    ))
    fig_vol.update_layout(
        title="Rolling Annualised Volatility of Log Price Ratio",
        yaxis_title="Volatility",
        yaxis_tickformat=".1%",
        height=360,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    vol_readings = current_vol_readings(vol_df)
    c1, c2 = st.columns(2)
    c1.metric(short_col, f"{vol_readings[short_col]:.2%}")
    c2.metric(long_col, f"{vol_readings[long_col]:.2%}")

    if vol_readings["Short-term elevated"]:
        st.warning(
            f"Short-term volatility ({VOL_SHORT}-day) is elevated "
            f"(> 1.5x the {VOL_LONG}-day), suggesting a recent regime shift "
            "or unusual dispersion in relative performance."
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 4: Return Dispersion
    # ------------------------------------------------------------------
    st.header("4. Return Dispersion")

    disp_rows = compute_dispersion_table(data.price_a, data.price_b)
    if disp_rows:
        ddf = pd.DataFrame([
            {
                "Period": r.period,
                "Lookback": f"{r.lookback_obs} obs (~{r.lookback_obs / 252:.1f} yr)",
                "Current Dispersion": f"{r.current_dispersion:.2%}",
                "Hist Mean": f"{r.hist_mean:.2%}",
                "Hist Std": f"{r.hist_std:.2%}",
                "Z-Score": round(r.z_score, 2),
            }
            for r in disp_rows
        ])

        def _style_disp_z(val):
            if isinstance(val, (int, float)):
                return _zscore_color(val, z_threshold)
            return ""

        styled_disp = ddf.style.map(_style_disp_z, subset=["Z-Score"])
        st.dataframe(styled_disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Insufficient history to compute dispersion metrics.")

    st.caption(
        "**Interpretation:** High dispersion (z > "
        f"{z_threshold:.1f}) means the two assets have diverged "
        "significantly compared to history, which may suggest a "
        "potential mean-reversion setup."
    )

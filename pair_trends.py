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
    fetch_pair,
    compute_sma,
    current_sma_readings,
    compute_rolling_ratio_returns,
    compute_zscore_table,
    compute_ratio_volatility,
    current_vol_readings,
    compute_dispersion_table,
)


def _zscore_color(z: float, threshold: float) -> str:
    if z >= threshold:
        return "color: #22c55e; font-weight: bold"
    elif z <= -threshold:
        return "color: #ef4444; font-weight: bold"
    return ""


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
    sma_plot = sma_df.loc[sma_df.index >= chart_start]

    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(
        x=sma_plot.index, y=sma_plot["Log Ratio"],
        name="Log Ratio", line=dict(width=2, color="#3b82f6"),
    ))
    colors = {"SMA 50": "#f59e0b", "SMA 100": "#8b5cf6", "SMA 200": "#ef4444"}
    for col in sma_plot.columns:
        if col.startswith("SMA"):
            fig_sma.add_trace(go.Scatter(
                x=sma_plot.index, y=sma_plot[col],
                name=col, line=dict(width=1.5, dash="dash", color=colors.get(col, "#999")),
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
        slope_dir = "Rising" if slope > 0 else "Falling" if slope < 0 else "Flat"
        cols[i + 1].metric(
            key,
            f"{readings[key]:.4f}",
            delta=f"{signal} · {slope_dir} ({slope:+.5f}/day)",
            delta_color="normal" if signal == "Above" else "inverse",
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Section 2: Z-Scores of Rolling Returns
    # ------------------------------------------------------------------
    st.header("2. Z-Scores of Rolling Returns")

    z_threshold = st.number_input(
        "Z-score highlight threshold",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        key="pt_z_thresh",
    )

    zscore_rows = compute_zscore_table(data.ratio)
    if zscore_rows:
        zdf = pd.DataFrame([
            {
                "Period": r.period,
                "Lookback": f"{r.lookback_obs} obs (~{r.lookback_obs / 252:.1f} yr)",
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
        f"Values beyond +/-{z_threshold:.1f} are highlighted."
    )

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

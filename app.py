# app.py
# Interactive macro dashboard: color-coded long-term z-score heatmap + compact cards + drillable charts
# Requires FRED_API_KEY environment variable (Streamlit Secrets or env)
# requirements: streamlit, pandas, numpy, plotly, fredapi, requests, lxml

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import requests
from io import StringIO
import plotly.express as px

st.set_page_config(page_title="Macro Snapshot — Heatmap + Charts", layout="wide", initial_sidebar_state="expanded")

# --- FRED setup ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    st.warning("Set FRED_API_KEY environment variable (get one at https://fred.stlouisfed.org)")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

# --- Series mapping (canonical FRED IDs) ---
SERIES = {
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Core PCE": "PCEPILFE",
    "5y5y inflation": "T5YIFR",
    "Real GDP (level)": "GDPC1",
    "Industrial Production": "INDPRO",
    "Real Retail Sales (adv)": "RRSFS",
    "Unemployment": "UNRATE",
    "JOLTS (Job Openings)": "JTSJOL",
    "Consumer Sentiment (UMich)": "UMCSENT",
    "10y Treasury": "DGS10",
    "2y Treasury": "DGS2",
}

# Directionality: +1 means higher = good, -1 means higher = bad
INDICATOR_DIRECTION = {
    "CPI": -1,
    "Core CPI": -1,
    "Core PCE": -1,
    "5y5y inflation": -1,
    "Real GDP (level)": +1,
    "Industrial Production": +1,
    "Real Retail Sales (adv)": +1,
    "Unemployment": -1,
    "JOLTS (Job Openings)": +1,
    "Consumer Sentiment (UMich)": +1,
    "10y Treasury": -1,
    "2y Treasury": -1,
}

# --- Fetching helpers ---
@st.cache_data(ttl=3600)
def fetch_fred(series_id):
    try:
        s = fred.get_series(series_id).dropna()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        return s
    except Exception as e:
        st.error(f"Error fetching {series_id}: {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def fetch_ism_csv_fallback():
    try:
        ism_url = "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/rob_legacy_data/2074_ism_manufacturing_pmi.csv"
        r = requests.get(ism_url, timeout=8)
        if r.status_code == 200 and len(r.text) > 200:
            df = pd.read_csv(StringIO(r.text), parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index)
            return df.iloc[:, 0].dropna().sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)

# Load data
with st.spinner("Fetching FRED series..."):
    data = {name: fetch_fred(sid) for name, sid in SERIES.items()}
ism_series = fetch_ism_csv_fallback()

# --- Timeframe helpers ---
TIMEFRAME_MAP = {
    "1Y": lambda last: last - pd.DateOffset(years=1),
    "3Y": lambda last: last - pd.DateOffset(years=3),
    "5Y": lambda last: last - pd.DateOffset(years=5),
    "10Y": lambda last: last - pd.DateOffset(years=10),
    "Max": lambda last: pd.Timestamp("1900-01-01"),
}

def filter_series_for_timeframe(s: pd.Series, timeframe_key: str):
    if s.empty:
        return s
    last = s.index.max()
    cutoff = TIMEFRAME_MAP.get(timeframe_key, TIMEFRAME_MAP["5Y"])(last)
    return s[s.index >= cutoff]

def compute_cagr(series: pd.Series):
    s = series.dropna()
    if s.empty or len(s) < 2:
        return np.nan
    first_val = float(s.iloc[0])
    last_val = float(s.iloc[-1])
    days = (s.index[-1] - s.index[0]).days
    years = days / 365.25
    if years <= 0 or first_val <= 0:
        return np.nan
    return (last_val / first_val) ** (1 / years) - 1

def compute_long_term_zscore(series: pd.Series, direction: int):
    """
    Latest z-score vs long-term mean. direction flips sign to make green=good.
    """
    s = series.dropna()
    if len(s) < 24:
        return np.nan
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return np.nan
    z = (s.iloc[-1] - mean) / std
    return z * direction

def yoy(series: pd.Series):
    if series.empty:
        return series
    try:
        s = series.resample('M').last() if series.index.inferred_freq is None else series
    except Exception:
        s = series
    return s.pct_change(periods=12) * 100

# --- Sidebar controls ---
st.sidebar.header("Chart Controls")
global_timeframe = st.sidebar.selectbox("Global timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)
st.sidebar.markdown("Heatmap uses *latest value vs long-term average*. Click any card to expand a full chart.")

# Initialize session_state for expanders (stable across reruns)
if "expand_map" not in st.session_state:
    st.session_state.expand_map = {}

# --- Page ---
st.title("Macro Snapshot — Heatmap & Drillable Charts")
st.markdown("Green = better than long-term average • Red = worse than long-term average. Click a card to open the full interactive chart and per-chart timeframe.")

# --- Heatmap: long-term z-scores ---
st.subheader("Snapshot: vs Long-Term Average (z-scores)")
rows = []
for name, s in data.items():
    if name not in INDICATOR_DIRECTION:
        continue
    z = compute_long_term_zscore(s, INDICATOR_DIRECTION[name])
    latest = s.dropna().iloc[-1] if not s.empty else np.nan
    rows.append({"Indicator": name, "Z-Score": z, "Latest": latest})

heat_df = pd.DataFrame(rows).set_index("Indicator").sort_index()

if heat_df.empty:
    st.write("Heatmap data unavailable")
else:
    # Build an Nx1 matrix for px.imshow (one column: Z-Score)
    zvals = heat_df[["Z-Score"]].fillna(0).values
    fig = px.imshow(
        zvals,
        x=["vs long-term avg"],
        y=heat_df.index.tolist(),
        color_continuous_scale="RdYlGn",
        zmin=-2.5,
        zmax=2.5,
        labels={"x": "", "y": "Indicator", "color": "z-score"},
        aspect="auto"
    )
    # add numeric annotations (rounded z and latest value)
    annot_text = [[f"z={heat_df['Z-Score'].loc[idx]:.2f}\nval={heat_df['Latest'].loc[idx]:.2f}" if not np.isnan(heat_df['Z-Score'].loc[idx]) else "n/a" ] for idx in heat_df.index]
    fig.update_traces(text=annot_text, texttemplate="%{text}", textfont_size=11)
    fig.update_layout(height=420, margin=dict(l=140, r=20, t=40, b=20), title="Green = better than historical norm • Red = worse")
    st.plotly_chart(fig, use_container_width=True)

# --- Compact card grid (click to expand) ---
st.subheader("Indicators (compact cards). Click 'View' to expand a full chart.")

snapshot_items = list(heat_df.index)  # use same indicator order as heatmap
cards_per_row = 4
rows_cards = [snapshot_items[i:i+cards_per_row] for i in range(0, len(snapshot_items), cards_per_row)]

for row in rows_cards:
    cols = st.columns(cards_per_row)
    for col, name in zip(cols, row):
        with col:
            s = data.get(name, pd.Series(dtype=float))
            # compute a headline metric: 1-year % change if possible, else latest level
            headline = "n/a"
            try:
                s_drop = s.dropna()
                if len(s_drop) > 12:
                    prev = s_drop.iloc[-13]
                    latest = s_drop.iloc[-1]
                    headline = f"{(latest/prev - 1) * 100:.2f}%"
                elif not s_drop.empty:
                    headline = f"{s_drop.iloc[-1]:.2f}"
            except Exception:
                headline = "n/a"

            st.markdown(f"**{name}**")
            st.metric(label="1yr chg / latest", value=headline)
            # button to toggle expand state
            btn_key = f"btn_{name}"
            if st.button(f"View {name}", key=btn_key):
                st.session_state.expand_map[name] = True

# Render expanders outside the grid for layout stability
for name in snapshot_items:
    if st.session_state.expand_map.get(name, False):
        with st.expander(f"{name} — Full chart (close to collapse)", expanded=True):
            # allow per-chart timeframe override
            chart_tf = st.selectbox("Chart timeframe (overrides global)", ["1Y","3Y","5Y","10Y","Max"],
                                    index=["1Y","3Y","5Y","10Y","Max"].index(global_timeframe),
                                    key=f"tf_{name}")
            # pick series (special handling for ISM & spreads)
            if name == "ISM Manufacturing PMI":
                s = ism_series
            elif name in ["10y Treasury", "2y Treasury"]:
                s = data.get(name, pd.Series(dtype=float))
            elif name == "10y-2y":
                # allow computed spread if user adds it (not in snapshot_items by default)
                s = pd.Series(dtype=float)
            else:
                s = data.get(name, pd.Series(dtype=float))

            s_f = filter_series_for_timeframe(s, chart_tf)

            if s_f.empty:
                st.write("Data unavailable for this indicator")
            else:
                # For GDP, show quarterly series/resampled
                if name == "Real GDP (level)":
                    s_plot = s_f.resample('Q').last() if s_f.index.inferred_freq is None else s_f
                    # QoQ annualized
                    q = s_plot.resample('Q').last()
                    if len(q) > 1:
                        qch = (q.pct_change(periods=1) * 4 * 100).dropna()
                        if not qch.empty:
                            st.metric("Real GDP QoQ annualized (%)", f"{qch.iloc[-1]:.2f}")
                # For unemployment show percent and change
                if name == "Unemployment":
                    latest = s_f.dropna().iloc[-1] if not s_f.dropna().empty else np.nan
                    prev = s_f.dropna().iloc[-13] if len(s_f.dropna())>12 else np.nan
                    if not np.isnan(latest) and not np.isnan(prev):
                        st.metric("Unemployment (latest)", f"{latest:.2f}%", delta=f"{latest-prev:+.2f}")
                # Plot interactive line
                df_plot = s_f.reset_index()
                df_plot.columns = ["date", "value"]
                fig = px.line(df_plot, x="date", y="value", title=f"{name} — {chart_tf}", labels={"value": name, "date":"Date"})
                fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
                st.plotly_chart(fig, use_container_width=True)
                # CAGR where meaningful
                cagr = compute_cagr(s_f)
                if not np.isnan(cagr):
                    st.metric(label=f"{name} CAGR ({chart_tf})", value=f"{cagr*100:.2f}%")
            # close button
            if st.button("Close", key=f"close_{name}"):
                st.session_state.expand_map[name] = False

st.write("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

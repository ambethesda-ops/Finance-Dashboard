# app.py
# Streamlit macro dashboard — interactive Plotly + CAGR + timeframe selector
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

st.set_page_config(page_title="12-Chart US Macro Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Configuration & FRED ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    st.warning("Set FRED_API_KEY environment variable (get one at https://fred.stlouisfed.org)")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

# Series mapping
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

# Cache FRED series for 1 hour
@st.cache_data(ttl=3600)
def fetch_fred(series_id):
    try:
        s = fred.get_series(series_id).dropna()
        # ensure datetime index and sort
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        return s
    except Exception as e:
        st.error(f"Error fetching {series_id}: {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def fetch_ism_csv_fallback():
    # Best-effort ISM PMI CSV fallback (may be outdated); keep safe
    try:
        # Example legacy CSV; if fails, return empty
        ism_url = "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/rob_legacy_data/2074_ism_manufacturing_pmi.csv"
        r = requests.get(ism_url, timeout=8)
        if r.status_code == 200 and len(r.text) > 200:
            df = pd.read_csv(StringIO(r.text), parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index)
            # assume first numeric column is PMI
            return df.iloc[:, 0].dropna().sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)

# load data
with st.spinner("Fetching data from FRED..."):
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

def yoy(series: pd.Series):
    if series.empty:
        return series
    # resample monthly if irregular
    try:
        s = series.resample('M').last() if series.index.inferred_freq is None else series
    except Exception:
        s = series
    return s.pct_change(periods=12) * 100

def qoq_annualized(series: pd.Series):
    if series.empty:
        return series
    q = series.resample('Q').last()
    return (q.pct_change(periods=1) * 100 * 4)

# Sidebar controls
st.sidebar.header("Chart Controls")
timeframe = st.sidebar.selectbox("Timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)
st.sidebar.markdown("Built with FRED series. Timeframe affects all charts.")

# Small helper to draw an interactive line plot
def plot_line(series: pd.Series, title: str, y_label: str = None, show_range_slider: bool = True):
    if series.empty:
        st.write("Data unavailable")
        return
    df = series.reset_index()
    df.columns = ["date", "value"]
    fig = px.line(df, x="date", y="value", title=title, labels={"value": y_label or title, "date":"Date"})
    if show_range_slider:
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

# --- Page header ---
st.title("12-Chart US Macro Dashboard — Interactive")
st.markdown("Use the sidebar to change the timeframe for all charts. CAGR calculates annualized growth over the selected window where meaningful.")

# Layout: 4 columns per row for the main sections
cols = st.columns(4)

# --- Inflation row ---
st.header("Inflation")
infl_items = ["CPI", "Core CPI", "Core PCE", "5y5y inflation"]
for i, item in enumerate(infl_items):
    col = cols[i % 4]
    with col:
        s = data[item]
        s_f = filter_series_for_timeframe(s, timeframe)
        st.subheader(item)
        if s.empty:
            st.write("Data unavailable")
            continue

        if item in ["CPI", "Core CPI", "Core PCE"]:
            # level interactive plot + YoY plot + CAGR
            plot_line(s_f, f"{item} (Index)", y_label="Index")
            # compute monthly series for YoY and CAGR
            s_month = s_f.resample('M').last() if s_f.index.inferred_freq is None else s_f
            yoy_s = yoy(s_month)
            if not yoy_s.dropna().empty:
                plot_line(yoy_s, f"{item} YoY (%)", y_label="YoY %")
            cagr = compute_cagr(s_month)
            if not np.isnan(cagr):
                st.metric(label=f"{item} CAGR ({timeframe})", value=f"{cagr*100:.2f}%")
            else:
                st.write("CAGR: n/a")
        else:
            # 5y5y inflation level
            plot_line(s_f, "5y5y Inflation Expectations (%)", y_label="%")
            cagr = compute_cagr(s_f)
            st.metric(label=f"5y5y CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

# --- Growth row ---
st.header("Growth")
gcols = st.columns(4)

# GDP
with gcols[0]:
    st.subheader("Real GDP")
    s = data["Real GDP (level)"]
    s_f = filter_series_for_timeframe(s, timeframe)
    if s.empty:
        st.write("Data unavailable")
    else:
        # show QoQ annualized using full-quarter series within timeframe
        q = s.resample('Q').last()
        q_f = filter_series_for_timeframe(q, timeframe)
        if not q_f.empty and len(q_f) > 1:
            q_change = (q_f.pct_change(periods=1) * 4 * 100).dropna()
            st.metric("Real GDP QoQ annualized (%)", f"{q_change.iloc[-1]:.2f}")
        else:
            st.metric("Real GDP QoQ annualized (%)", "n/a")
        plot_line(s_f.resample('Q').last() if not s_f.empty else s_f, "Real GDP (level)", y_label="Real GDP")
        cagr = compute_cagr(s_f.resample('Q').last() if not s_f.empty else s_f)
        st.metric(label=f"GDP CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

# ISM PMI (fallback)
with gcols[1]:
    st.subheader("ISM Manufacturing PMI")
    pmi = ism_series
    pmi_f = filter_series_for_timeframe(pmi, timeframe)
    if pmi.empty:
        st.write("ISM PMI not available via fallback CSV. We can add a better source on request.")
    else:
        plot_line(pmi_f, "ISM Manufacturing PMI", y_label="PMI")
        cagr = compute_cagr(pmi_f)
        st.metric(label=f"PMI CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

# Industrial Production
with gcols[2]:
    st.subheader("Industrial Production")
    s = data["Industrial Production"]
    s_f = filter_series_for_timeframe(s, timeframe)
    if s.empty:
        st.write("Data unavailable")
    else:
        plot_line(s_f, "Industrial Production (Index)", y_label="Index")
        yoy_s = yoy(s_f)
        if not yoy_s.dropna().empty:
            plot_line(yoy_s, "Industrial Production YoY (%)", y_label="YoY %")
        cagr = compute_cagr(s_f)
        st.metric(label=f"Industrial Production CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

# Retail sales (real)
with gcols[3]:
    st.subheader("Real Retail Sales")
    s = data["Real Retail Sales (adv)"]
    s_f = filter_series_for_timeframe(s, timeframe)
    if s.empty:
        st.write("Data unavailable")
    else:
        plot_line(s_f, "Real Retail Sales (Index)", y_label="Index")
        yoy_s = yoy(s_f)
        if not yoy_s.dropna().empty:
            plot_line(yoy_s, "Real Retail Sales YoY (%)", y_label="YoY %")
        cagr = compute_cagr(s_f)
        st.metric(label=f"Retail Sales CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

# --- Consumer & Labor row ---
st.header("Consumer & Labor")
lcols = st.columns(4)

with lcols[0]:
    st.subheader("Unemployment Rate")
    s = data["Unemployment"]
    s_f = filter_series_for_timeframe(s, timeframe)
    if s.empty:
        st.write("Data unavailable")
    else:
        plot_line(s_f, "Unemployment Rate (%)", y_label="%")
        # CAGR not meaningful for rate series; show latest & change
        latest = s_f.dropna().iloc[-1] if not s_f.dropna().empty else np.nan
        prev = s_f.dropna().iloc[-13] if len(s_f.dropna()) > 12 else np.nan
        if not np.isnan(latest) and not np.isnan(prev):
            st.metric("Latest Unemployment (%)", f"{latest:.2f}", delta=f"{(latest-prev):+.2f}")

with lcols[1]:
    st.subheader("JOLTS - Job Openings")
    s = data["JOLTS (Job Openings)"]
    s_f = filter_series_for_timeframe(s, timeframe)
    if s.empty:
        st.write("Data unavailable")
    else:
        plot_line(s_f, "Job Openings (total)", y_label="Openings")
        cagr = compute_cagr(s_f)
        st.metric(label=f"JOLTS CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

with lcols[2]:
    st.subheader("Consumer Sentiment (UMich)")
    s = data["Consumer Sentiment (UMich)"]
    s_f = filter_series_for_timeframe(s, timeframe)
    if s.empty:
        st.write("Data unavailable")
    else:
        plot_line(s_f, "UMich Consumer Sentiment", y_label="Index")
        cagr = compute_cagr(s_f)
        st.metric(label=f"Sentiment CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

with lcols[3]:
    st.write("")  # placeholder

# --- Financial Conditions ---
st.header("Financial Conditions")
fcols = st.columns(2)
with fcols[0]:
    y10 = data["10y Treasury"]
    y2 = data["2y Treasury"]
    if y10.empty or y2.empty:
        st.write("Treasury yields data unavailable")
    else:
        df = pd.concat([y10, y2], axis=1).dropna()
        df.columns = ["DGS10", "DGS2"]
        df["10y-2y"] = df["DGS10"] - df["DGS2"]
        df_f = filter_series_for_timeframe(df["10y-2y"], timeframe)
        if df_f.empty:
            st.write("Spread data unavailable for timeframe")
        else:
            # display last spread in bps
            latest = df_f.iloc[-1]
            st.metric("10y − 2y (bps)", f"{latest*100:.1f}")
            # interactive plot
            plot_line(df_f*100, "10y − 2y (basis points)", y_label="bps")
            cagr = compute_cagr(df_f)
            st.metric(label=f"10y-2y CAGR ({timeframe})", value=(f"{cagr*100:.2f}%" if not np.isnan(cagr) else "n/a"))

with fcols[1]:
    st.write("Notes & tips")
    st.markdown("""
    - Timeframe selector affects all charts.  
    - CAGR is annualized and computed on the selected window; for rates and survey indices interpret CAGR cautiously.  
    - ISM PMI is a best-effort CSV fallback; we can add a better ingestion if preferred.  
    - If you want YoY instead of level for any series, we can add toggles per-chart.
    """)

st.write("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

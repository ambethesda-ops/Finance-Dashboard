# app.py
# Macro Indicators Heat Map — with separate tables per bucket (bucket header, quarter row, indicator rows)
# Replace previous app.py with this file. Keep requirements.txt the same.

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import requests
from io import StringIO
import plotly.express as px
import plotly.colors as pc
import html as _html

st.set_page_config(page_title="Macro Indicators Heat Map", layout="wide")

# -----------------------
# Series mapping (edit FRED IDs if you prefer different series)
# -----------------------
SERIES_MAP = {
    # Bucket 1: Inflation & Expectations
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Core PCE": "PCEPILFE",
    "5y5y inflation (market-implied)": "T5YIFR",
    "New Vehicle Sales (proxy)": "TOTALSA",

    # Bucket 2: Growth / Activity
    "Real GDP (level)": "GDPC1",
    "Industrial Production": "INDPRO",
    "Real Retail Sales (adv)": "RRSFS",
    "Housing Starts": "HOUST",

    # Bucket 3: Labor
    "Unemployment": "UNRATE",
    "Initial Claims": "ICSA",
    "JOLTS (Job Openings)": "JTSJOL",
    "Real Wages (YoY)": "LES1252881600Q",

    # Bucket 4: Sentiment / Leading
    "ISM Manufacturing PMI": "PMI_USA_MAN",
    "Consumer Sentiment (UMich)": "UMCSENT",

    # Bucket 5: Rates & Credit
    "10y Treasury": "DGS10",
    "2y Treasury": "DGS2",
    "High Yield OAS": "BAMLH0A0HYM2",
}

YOY_SERIES = {
    "CPI",
    "Core CPI",
    "Core PCE",
    "Real GDP (level)",
    "Industrial Production",
    "Real Retail Sales (adv)",
    "Real Wages (YoY)",
}

INDICATOR_DIRECTION = {
    "CPI": -1, "Core CPI": -1, "Core PCE": -1, "5y5y inflation (market-implied)": -1,
    "New Vehicle Sales (proxy)": +1,
    "Real GDP (level)": +1, "Industrial Production": +1, "Real Retail Sales (adv)": +1, "Housing Starts": +1,
    "Unemployment": -1, "Initial Claims": -1, "JOLTS (Job Openings)": +1, "Real Wages (YoY)": +1,
    "ISM Manufacturing PMI": +1, "Consumer Sentiment (UMich)": +1,
    "10y Treasury": -1, "2y Treasury": -1, "High Yield OAS": -1,
}

DISPLAY_TYPE = {
    "CPI": "pct", "Core CPI": "pct", "Core PCE": "pct", "5y5y inflation (market-implied)": "pct",
    "New Vehicle Sales (proxy)": "float1",
    "Real GDP (level)": "pct", "Industrial Production": "pct", "Real Retail Sales (adv)": "pct",
    "Housing Starts": "int",
    "Unemployment": "pct", "Initial Claims": "int", "JOLTS (Job Openings)": "int", "Real Wages (YoY)": "pct",
    "ISM Manufacturing PMI": "float1", "Consumer Sentiment (UMich)": "float1",
    "10y Treasury": "pct", "2y Treasury": "pct", "High Yield OAS": "float1",
}

# Define buckets (order matters)
BUCKETS = {
    "Inflation & Expectations": [
        "CPI", "Core CPI", "Core PCE", "5y5y inflation (market-implied)", "New Vehicle Sales (proxy)"
    ],
    "Growth / Activity": [
        "Real GDP (level)", "Industrial Production", "Real Retail Sales (adv)", "Housing Starts"
    ],
    "Labor": [
        "Unemployment", "Initial Claims", "JOLTS (Job Openings)", "Real Wages (YoY)"
    ],
    "Sentiment / Leading": [
        "ISM Manufacturing PMI", "Consumer Sentiment (UMich)"
    ],
    "Rates & Credit": [
        "10y Treasury", "2y Treasury", "High Yield OAS"
    ],
}

# -----------------------
# FRED setup
# -----------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    st.warning("Set FRED_API_KEY environment variable (get one at https://fred.stlouisfed.org)")
    st.stop()
fred = Fred(api_key=FRED_API_KEY)

# -----------------------
# Safe fetch helpers
# -----------------------
@st.cache_data(ttl=3600)
def fetch_series_safe(series_id):
    try:
        s = fred.get_series(series_id).dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def fetch_ism_csv_fallback():
    try:
        url = "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/rob_legacy_data/2074_ism_manufacturing_pmi.csv"
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and len(r.text) > 200:
            df = pd.read_csv(StringIO(r.text), parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index)
            return df.iloc[:, 0].dropna().sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)

# Fetch series
with st.spinner("Fetching series from FRED..."):
    raw_data = {name: fetch_series_safe(sid) for name, sid in SERIES_MAP.items()}

# fallback ISM
if raw_data.get("ISM Manufacturing PMI", pd.Series(dtype=float)).empty:
    raw_data["ISM Manufacturing PMI"] = fetch_ism_csv_fallback()

# -----------------------
# Transforms & helper functions
# -----------------------
def to_quarter_label(ts: pd.Timestamp):
    q = ((ts.month - 1) // 3) + 1
    return f"Q{q} {ts.year}"

def to_quarter_series(s: pd.Series):
    if s.empty:
        return s
    try:
        q = s.resample('Q').last()
    except Exception:
        q = s
    return q.dropna()

def compute_yoy(series_q: pd.Series):
    if series_q.empty:
        return series_q
    s = series_q.dropna()
    return (s.pct_change(periods=4) * 100)

def compute_long_term_z(series_q: pd.Series, direction=1, min_obs=20):
    s = series_q.dropna()
    if s.empty or len(s) < min_obs:
        return pd.Series(index=series_q.index, data=[np.nan] * len(series_q))
    mean = s.mean(); std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(index=series_q.index, data=[np.nan] * len(series_q))
    z = (series_q - mean) / std
    return z * direction

def format_value(val, dtype):
    if pd.isna(val):
        return "n/a"
    try:
        if dtype == "pct":
            return f"{val:.1f}%"
        elif dtype == "int":
            return f"{val:,.0f}"
        elif dtype == "float1":
            return f"{val:.1f}"
        else:
            return f"{val}"
    except Exception:
        return str(val)

def z_to_color(z, vmin=-2.5, vmax=2.5, colorscale=pc.diverging.RdYlGn):
    if pd.isna(z):
        return "white"
    t = (z - vmin) / (vmax - vmin); t = max(0.0, min(1.0, t))
    return pc.sample_colorscale(colorscale, [t])[0]

# -----------------------
# Prepare series by quarter
# -----------------------
series_q = {}
for name, raw in raw_data.items():
    series_q[name] = to_quarter_series(raw)

# Real Wages: LES1252881600Q is quarterly real median weekly earnings; compute YoY
if "Real Wages (YoY)" in series_q:
    rw_raw = raw_data.get("Real Wages (YoY)", pd.Series(dtype=float))
    if not rw_raw.empty:
        series_q["Real Wages (YoY)"] = compute_yoy(to_quarter_series(rw_raw))
    else:
        series_q["Real Wages (YoY)"] = pd.Series(dtype=float)

# -----------------------
# Quarter union and labels (newest first -> left)
# -----------------------
all_quarters = set()
for s in series_q.values():
    if not s.empty:
        all_quarters.update(s.index)
if not all_quarters:
    st.error("No quarterly data available. Check FRED series IDs.")
    st.stop()

quarter_index = sorted(list(all_quarters), reverse=True)
quarter_labels = [to_quarter_label(q) for q in quarter_index]

# -----------------------
# Build reported values + z-scores
# -----------------------
rows = list(series_q.keys())
reported_df = pd.DataFrame(index=rows, columns=quarter_labels, dtype=object)
z_df = pd.DataFrame(index=rows, columns=quarter_labels, dtype=float)

# precompute yoy cache
yoy_cache = {name: (compute_yoy(series_q[name]) if name in YOY_SERIES else pd.Series(dtype=float)) for name in rows}

for name in rows:
    qseries = series_q.get(name, pd.Series(dtype=float))
    direction = INDICATOR_DIRECTION.get(name, +1)
    if name in YOY_SERIES:
        zseries = compute_long_term_z(yoy_cache.get(name, pd.Series(dtype=float)), direction)
    else:
        zseries = compute_long_term_z(qseries, direction)
    for i, q_ts in enumerate(quarter_index):
        lbl = quarter_labels[i]
        if q_ts in qseries.index:
            if name in YOY_SERIES:
                val = np.nan
                if q_ts in yoy_cache.get(name, pd.Series(dtype=float)).index:
                    val = yoy_cache[name].loc[q_ts]
                reported_df.loc[name, lbl] = format_value(val, "pct") if not pd.isna(val) else "n/a"
                z_df.loc[name, lbl] = zseries.loc[q_ts] if q_ts in zseries.index else np.nan
            else:
                val = qseries.loc[q_ts]
                dtype = DISPLAY_TYPE.get(name, "float1")
                reported_df.loc[name, lbl] = format_value(val, dtype)
                z_df.loc[name, lbl] = zseries.loc[q_ts] if q_ts in zseries.index else np.nan
        else:
            reported_df.loc[name, lbl] = "n/a"
            z_df.loc[name, lbl] = np.nan

# -----------------------
# Top legend
# -----------------------
st.title("Macro Indicators Heat Map")
legend_html = """
<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#a50026;border:1px solid #000;"></div><div><b>&lt;= -2.0</b></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#f46d43;border:1px solid #000;"></div><div><b>-2.0 to -1.0</b></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#fee08b;border:1px solid #000;"></div><div><b>-1.0 to +1.0</b></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#d9ef8b;border:1px solid #000;"></div><div><b>+

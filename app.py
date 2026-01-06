# app.py
# Macro Indicators Heat Map
# - Separate tables per bucket
# - Newest quarter on LEFT
# - YoY % for key growth/inflation series
# - Real Wages from LES1252881600Q
# - ISM REMOVED
# - No duplicate line under chart
# - Sticky left column fixed

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.express as px
import plotly.colors as pc
import html as _html

st.set_page_config(page_title="Macro Indicators Heat Map", layout="wide")

# -----------------------
# FRED setup
# -----------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    st.warning("Set FRED_API_KEY in Streamlit Secrets")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

@st.cache_data(ttl=3600)
def fetch_series(series_id):
    try:
        s = fred.get_series(series_id).dropna()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)

# -----------------------
# Series & buckets
# -----------------------
BUCKETS = {
    "Inflation & Expectations": {
        "CPI": "CPIAUCSL",
        "Core CPI": "CPILFESL",
        "Core PCE": "PCEPILFE",
        "5y5y inflation (market-implied)": "T5YIFR",
        "New Vehicle Sales (proxy)": "TOTALSA",
    },
    "Growth / Activity": {
        "Real GDP (level)": "GDPC1",
        "Industrial Production": "INDPRO",
        "Real Retail Sales (adv)": "RRSFS",
        "Housing Starts": "HOUST",
    },
    "Labor": {
        "Unemployment": "UNRATE",
        "Initial Claims": "ICSA",
        "JOLTS (Job Openings)": "JTSJOL",
        "Real Wages (YoY)": "LES1252881600Q",
    },
    "Rates & Credit": {
        "10y Treasury": "DGS10",
        "2y Treasury": "DGS2",
        "High Yield OAS": "BAMLH0A0HYM2",
    }
}

YOY_SERIES = {
    "CPI", "Core CPI", "Core PCE",
    "Real GDP (level)", "Industrial Production",
    "Real Retail Sales (adv)", "Real Wages (YoY)"
}

DIRECTION = {
    "CPI": -1, "Core CPI": -1, "Core PCE": -1,
    "5y5y inflation (market-implied)": -1,
    "Real GDP (level)": 1, "Industrial Production": 1,
    "Real Retail Sales (adv)": 1, "Housing Starts": 1,
    "Unemployment": -1, "Initial Claims": -1,
    "JOLTS (Job Openings)": 1, "Real Wages (YoY)": 1,
    "10y Treasury": -1, "2y Treasury": -1,
    "High Yield OAS": -1, "New Vehicle Sales (proxy)": 1,
}

FORMAT = {
    "int": lambda x: f"{int(x):,}",
    "pct": lambda x: f"{x:.1f}%",
    "float": lambda x: f"{x:.1f}",
}

# -----------------------
# Helpers
# -----------------------
def to_quarter(s):
    return s.resample("Q").last().dropna()

def yoy(s):
    return (s.pct_change(4) * 100).dropna()

def zscore(s, direction):
    if len(s.dropna()) < 20:
        return s * np.nan
    z = (s - s.mean()) / s.std()
    return z * direction

def color(z):
    if pd.isna(z):
        return "white"
    t = max(0, min(1, (z + 2.5) / 5))
    return pc.sample_colorscale(pc.diverging.RdYlGn, [t])[0]

# -----------------------
# Load data
# -----------------------
raw = {}
for bucket in BUCKETS.values():
    for name, sid in bucket.items():
        raw[name] = fetch_series(sid)

# Convert to quarterly
qdata = {}
for name, s in raw.items():
    q = to_quarter(s)
    if name in YOY_SERIES:
        q = yoy(q)
    qdata[name] = q

# Build master quarter index (newest left)
all_q = sorted({d for s in qdata.values() for d in s.index}, reverse=True)
qlabels = [f"Q{((d.month-1)//3)+1} {d.year}" for d in all_q]

# -----------------------
# UI
# -----------------------
st.title("Macro Indicators Heat Map")

legend_html = """
<div style="display:flex;gap:14px;align-items:center;margin-bottom:10px;">
  <div style="width:16px;height:16px;background:#006837"></div> Better
  <div style="width:16px;height:16px;background:#fee08b"></div> Neutral
  <div style="width:16px;height:16px;background:#a50026"></div> Worse
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

def render_table(title, indicators):
    st.subheader(title)
    header = ['<th style="position:sticky;left:0;background:#333;color:white;z-index:5">Indicator</th>']
    for i, q in enumerate(qlabels):
        bg = "#cfe2f3" if i == 0 else "#f0f0f0"
        header.append(f'<th style="background:{bg};text-align:center">{q}</th>')
    header_html = "<tr>" + "".join(header) + "</tr>"

    rows_html = []
    for name in indicators:
        s = qdata.get(name, pd.Series(dtype=float))
        z = zscore(s, DIRECTION.get(name, 1))
        row = [f'<td style="position:sticky;left:0;background:white;z-index:4">{_html.escape(name)}</td>']
        for d in all_q:
            if d in s.index:
                val = s.loc[d]
                fmt = FORMAT["pct"] if name in YOY_SERIES else FORMAT["float"]
                cell = fmt(val)
                bg = color(z.loc[d])
            else:
                cell, bg = "n/a", "white"
            row.append(f'<td style="background:{bg};text-align:center">{cell}</td>')
        rows_html.append("<tr>" + "".join(row) + "</tr>")

    table = f"""
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse;width:100%">
      <thead>{header_html}</thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>
    """
    st.markdown(table, unsafe_allow_html=True)

for bucket, indicators in BUCKETS.items():
    render_table(bucket, list(indicators.keys()))

# -----------------------
# Bottom chart
# -----------------------
st.markdown("---")
st.subheader("Single indicator chart")

indicator = st.selectbox("Select indicator", list(qdata.keys()))
years = st.selectbox("Timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)

s = qdata[indicator]
if not s.empty:
    last = s.index.max()
    if years != "Max":
        cutoff = last - pd.DateOffset(years=int(years[:-1]))
        s = s[s.index >= cutoff]

    df = s.reset_index()
    df.columns = ["date", "value"]

    title = f"{indicator} — YoY %" if indicator in YOY_SERIES else f"{indicator} — Level"
    fig = px.line(df, x="date", y="value", markers=True, title=title)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

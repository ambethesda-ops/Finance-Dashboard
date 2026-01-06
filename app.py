# app.py
# Macro Indicators Heat Map — supports monthly + quarterly columns (monthly preferred when available)

import os
from datetime import datetime
import calendar
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
# Buckets + indicators
# -----------------------
BUCKETS = {
    "Inflation": {
        "CPI": "CPIAUCSL",
        "Core CPI": "CPILFESL",
        "Core PCE": "PCEPILFE",
        "5y5y inflation (market-implied)": "T5YIFR",
    },
    "Growth / Activity": {
        "New Vehicle Sales (proxy)": "TOTALSA",
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

DISPLAY_TYPE = {
    "pct": lambda x: f"{x:.1f}%",
    "int": lambda x: f"{int(x):,}",
    "float": lambda x: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.1f}",
}

DISPLAY_KIND = {
    "CPI": "pct", "Core CPI": "pct", "Core PCE": "pct",
    "5y5y inflation (market-implied)": "pct",
    "Real GDP (level)": "pct", "Industrial Production": "pct",
    "Real Retail Sales (adv)": "pct", "Real Wages (YoY)": "pct",
    "Unemployment": "pct", "10y Treasury": "pct", "2y Treasury": "pct",
    "Initial Claims": "int", "JOLTS (Job Openings)": "int", "Housing Starts": "int",
    "High Yield OAS": "float", "New Vehicle Sales (proxy)": "float",
}

# -----------------------
# Helpers
# -----------------------
def month_end(s):
    """Resample to month-end timestamps (M) and return series indexed by month-end."""
    if s.empty:
        return s
    m = s.resample("M").last().dropna()
    # normalize to exact month-end timestamp (already is)
    return m

def quarter_end(s):
    """Resample to quarter-end timestamps (Q) and return series indexed by quarter-end."""
    if s.empty:
        return s
    q = s.resample("Q").last().dropna()
    # canonicalize to quarter-end timestamps (to_timestamp end)
    q.index = q.index.to_period("Q").to_timestamp(how="end")
    return q

def yoy_monthly(s):
    return (s.pct_change(12) * 100).dropna() if not s.empty else s

def yoy_quarterly(s):
    return (s.pct_change(4) * 100).dropna() if not s.empty else s

def zscore(s, direction):
    if s.empty or len(s.dropna()) < 20:
        return pd.Series(index=s.index, data=[np.nan] * len(s))
    return ((s - s.mean()) / s.std()) * direction

def color_for_z(z):
    if pd.isna(z):
        return "white"
    t = max(0, min(1, (z + 2.5) / 5))
    return pc.sample_colorscale(pc.diverging.RdYlGn, [t])[0]

def is_quarter_end_date(dt: pd.Timestamp):
    # consider quarter ends by month number and being last day
    return dt.month in (3, 6, 9, 12) and dt.day == calendar.monthrange(dt.year, dt.month)[1]

def label_for_date(dt: pd.Timestamp):
    if is_quarter_end_date(dt):
        q = ((dt.month - 1) // 3) + 1
        return f"Q{q} {dt.year}"
    else:
        return dt.strftime("%b %Y")  # e.g., "Dec 2025"

# -----------------------
# Load data (create monthly + quarterly views)
# -----------------------
raw = {}
for bucket in BUCKETS.values():
    for name, sid in bucket.items():
        raw[name] = fetch_series(sid)

mdata = {}
qdata = {}
for name, s in raw.items():
    # monthly and quarterly canonical series
    m = month_end(s)
    q = quarter_end(s)
    if name in YOY_SERIES:
        m = yoy_monthly(m)
        q = yoy_quarterly(q)
    mdata[name] = m
    qdata[name] = q

# Build union of all dates (monthly + quarterly), sorted descending
all_dates = sorted({d for s in list(mdata.values()) + list(qdata.values()) for d in s.index}, reverse=True)
labels = [label_for_date(d) for d in all_dates]

# -----------------------
# Render single combined table (sticky header + sticky left column)
# -----------------------
st.title("Macro Indicators Heat Map")

COL_W = 140
IND_W = 320

header = (
    f'<th style="position:sticky;top:0;left:0;z-index:18;background:#222;color:white;min-width:{IND_W}px;padding:10px;text-align:left;">Indicator</th>'
    + "".join(
        f'<th style="position:sticky;top:0;z-index:17;background:#f0f0f0;min-width:{COL_W}px;padding:10px;text-align:center;border-bottom:1px solid #ddd;">{_html.escape(lbl)}</th>'
        for lbl in labels
    )
)

rows_html = []
for bucket, indicators in BUCKETS.items():
    # bucket header row with sticky left cell
    bucket_left_cell = (
        f'<td style="position:sticky; left:0; background:#fafafa; z-index:14;'
        f'min-width:{IND_W}px; padding:10px; border-right:1px solid #e6e6e6;'
        f'font-weight:700; text-decoration:underline;">{_html.escape(bucket)}</td>'
    )
    empty_quarter_cells = ''.join(
        f'<td style="min-width:{COL_W}px; padding:10px; background:#fafafa; border-bottom:1px solid #eee;"></td>'
        for _ in labels
    )
    rows_html.append("<tr>" + bucket_left_cell + empty_quarter_cells + "</tr>")

    for name in indicators:
        mseries = mdata.get(name, pd.Series(dtype=float))
        qseries = qdata.get(name, pd.Series(dtype=float))
        # For zscore we prefer to compute on the quarterly series (keeps previous behavior),
        # but fall back to monthly if quarterly is empty.
        z_series_for_scoring = qseries if not qseries.empty else mseries
        z = zscore(z_series_for_scoring, DIRECTION.get(name, 1))

        left_cell = f'<td style="position:sticky;left:0;background:white;min-width:{IND_W}px;padding:10px;border-right:1px solid #ddd;font-weight:400; z-index:15;">&nbsp;&nbsp;{_html.escape(name)}</td>'
        row = [left_cell]
        for d in all_dates:
            txt, bg = "n/a", "white"
            # prefer monthly value for this date if present, else quarterly
            if d in mseries.index:
                val = mseries.loc[d]
                kind = DISPLAY_KIND.get(name, "float")
                txt = DISPLAY_TYPE[kind](val)
                # for coloring, try to find matching zscore: if d in z, use it; else attempt nearest quarter-end
                if d in z.index:
                    bg = color_for_z(z.loc[d])
                else:
                    # try map to quarter-end timestamp for coloring if possible
                    mapped_q = d.to_period("Q").to_timestamp(how="end")
                    if mapped_q in z.index:
                        bg = color_for_z(z.loc[mapped_q])
            elif d in qseries.index:
                val = qseries.loc[d]
                kind = DISPLAY_KIND.get(name, "float")
                txt = DISPLAY_TYPE[kind](val)
                if d in z.index:
                    bg = color_for_z(z.loc[d])
            row.append(
                f'<td style="background:{bg};min-width:{COL_W}px;padding:10px;text-align:center;border-bottom:1px solid #eee;">{_html.escape(txt)}</td>'
            )
        rows_html.append("<tr>" + "".join(row) + "</tr>")

table_html = f"""
<div style="max-height:70vh; overflow:auto; border:1px solid #eee;">
<table style="border-collapse:separate;border-spacing:0;font-family:Arial,Helvetica,sans-serif; width: max-content; position:relative;">
<thead><tr>{header}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</div>
"""

st.markdown(table_html, unsafe_allow_html=True)

# -----------------------
# Bottom single-indicator chart (unchanged)
# -----------------------
st.markdown("---")
st.subheader("Single indicator chart")
indicator = st.selectbox("Select indicator", list(mdata.keys()))
timeframe = st.selectbox("Timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)

# For the chart prefer monthly series if available (gives higher resolution),
# else fall back to quarterly.
series = mdata.get(indicator, pd.Series(dtype=float))
if series.empty:
    series = qdata.get(indicator, pd.Series(dtype=float))

if not series.empty:
    last = series.index.max()
    if timeframe != "Max":
        years = int(timeframe[:-1])
        cutoff = last - pd.DateOffset(years=years)
        series = series[series.index >= cutoff]
    df = series.reset_index()
    df.columns = ["date", "value"]
    title = f"{indicator} — YoY %" if indicator in YOY_SERIES else f"{indicator} — Level"
    fig = px.line(df, x="date", y="value", markers=True, title=title)
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available for this indicator.")

st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

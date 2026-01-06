# app.py
# Macro Indicators Heat Map — synchronized horizontal scrolling across bucket tables
# Replace your existing app.py with this file.

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import plotly.express as px
import plotly.colors as pc
import html as _html
import streamlit.components.v1 as components

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

DISPLAY_TYPE = {
    "CPI": "pct", "Core CPI": "pct", "Core PCE": "pct", "5y5y inflation (market-implied)": "pct",
    "New Vehicle Sales (proxy)": "float1",
    "Real GDP (level)": "pct", "Industrial Production": "pct", "Real Retail Sales (adv)": "pct",
    "Housing Starts": "int",
    "Unemployment": "pct", "Initial Claims": "int", "JOLTS (Job Openings)": "int", "Real Wages (YoY)": "pct",
    "10y Treasury": "pct", "2y Treasury": "pct", "High Yield OAS": "float1",
}

# -----------------------
# Helpers
# -----------------------
def to_quarter(s):
    return s.resample("Q").last().dropna() if not s.empty else s

def yoy(s):
    return (s.pct_change(4) * 100).dropna() if not s.empty else s

def zscore(s, direction):
    if s.empty or len(s.dropna()) < 20:
        return pd.Series(index=s.index, data=[np.nan] * len(s))
    z = (s - s.mean()) / s.std()
    return z * direction

def color_for_z(z):
    if pd.isna(z):
        return "white"
    t = max(0.0, min(1.0, (z + 2.5) / 5.0))
    return pc.sample_colorscale(pc.diverging.RdYlGn, [t])[0]

def format_cell_value(name, val):
    if pd.isna(val):
        return "n/a"
    dtype = DISPLAY_TYPE.get(name)
    try:
        if dtype == "pct":
            return f"{val:.1f}%"
        if dtype == "int":
            return f"{int(val):,}"
        if dtype == "float1":
            if abs(val) >= 1000:
                return f"{val:,.0f}"
            return f"{val:.1f}"
        if isinstance(val, (int, np.integer)) or abs(val) >= 1000:
            return f"{val:,.0f}"
        return f"{val:.1f}"
    except Exception:
        return str(val)

# -----------------------
# Load data
# -----------------------
raw = {}
for bucket in BUCKETS.values():
    for name, sid in bucket.items():
        raw[name] = fetch_series(sid)

qdata = {}
for name, s in raw.items():
    q = to_quarter(s)
    if name in YOY_SERIES:
        q = yoy(q)
    qdata[name] = q

all_quarters = sorted({d for s in qdata.values() for d in s.index}, reverse=True)
if not all_quarters:
    st.error("No quarterly data available. Check FRED series IDs.")
    st.stop()
quarter_labels = [f"Q{((d.month-1)//3)+1} {d.year}" for d in all_quarters]

# -----------------------
# Page header + legend
# -----------------------
st.title("Macro Indicators Heat Map")
legend_html = """
<div style="display:flex;gap:14px;align-items:center;margin-bottom:12px;">
  <div style="display:flex;gap:8px;align-items:center;"><div style="width:16px;height:16px;background:#006837;border:1px solid #999"></div><div>Better</div></div>
  <div style="display:flex;gap:8px;align-items:center;"><div style="width:16px;height:16px;background:#fee08b;border:1px solid #999"></div><div>Neutral</div></div>
  <div style="display:flex;gap:8px;align-items:center;"><div style="width:16px;height:16px;background:#a50026;border:1px solid #999"></div><div>Worse</div></div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# -----------------------
# Build HTML for all bucket tables and sync JS
# -----------------------
COL_W = 150
IND_W = 320

bucket_html_blocks = []
for bucket_name, indicators in BUCKETS.items():
    # header
    header_cells = [f'<th style="position:sticky; left:0; background:#222; color:white; z-index:5; min-width:{IND_W}px; width:{IND_W}px; text-align:left; padding:10px;">Indicator</th>']
    for i, q in enumerate(quarter_labels):
        bg = "#cfe2f3" if i == 0 else "#f8f8f8"
        header_cells.append(f'<th style="background:{bg}; min-width:{COL_W}px; width:{COL_W}px; text-align:center; padding:10px;">{_html.escape(q)}</th>')
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # rows
    rows_html = []
    for name in indicators.keys():
        s = qdata.get(name, pd.Series(dtype=float))
        z_series = zscore(s, DIRECTION.get(name, 1))
        row_cells = [f'<td style="position:sticky; left:0; background:#fff; z-index:4; min-width:{IND_W}px; width:{IND_W}px; padding:10px; border-right:1px solid #e6e6e6; font-weight:600;">{_html.escape(name)}</td>']
        for d in all_quarters:
            if d in s.index:
                val = s.loc[d]
                display = format_cell_value(name, val)
                bg = color_for_z(z_series.loc[d]) if d in z_series.index else "white"
            else:
                display = "n/a"
                bg = "white"
            row_cells.append(f'<td style="background:{bg}; min-width:{COL_W}px; width:{COL_W}px; text-align:center; padding:10px; border-bottom:1px solid #e9e9e9;">{_html.escape(display)}</td>')
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    table_html = f'''
    <div style="margin-bottom:18px;">
      <h3 style="margin:8px 0 6px 0;">{_html.escape(bucket_name)}</h3>
      <div class="sync-scroll" style="overflow-x:auto; border:1px solid #eee; padding:6px;">
        <table style="border-collapse:separate; border-spacing:0; font-family:Arial, Helvetica, sans-serif;">
          <thead>{header_html}</thead>
          <tbody>{"".join(rows_html)}</tbody>
        </table>
      </div>
    </div>
    '''
    bucket_html_blocks.append(table_html)

# Combine all tables into one HTML blob and add JS to sync scrollLeft across .sync-scroll elements
full_html = """
<html>
  <head>
    <meta charset="utf-8"/>
  </head>
  <body>
    <div style="font-family:Arial, Helvetica, sans-serif;">
      {tables}
    </div>
    <script>
      (function() {{
        const containers = Array.from(document.getElementsByClassName('sync-scroll'));
        if (containers.length < 2) return;
        let isSyncing = false;
        containers.forEach(c => {{
          c.addEventListener('scroll', function() {{
            if (isSyncing) return;
            isSyncing = true;
            const left = c.scrollLeft;
            containers.forEach(other => {{
              if (other !== c) other.scrollLeft = left;
            }});
            // small timeout to avoid recursion
            setTimeout(()=>{{ isSyncing = false; }}, 20);
          }});
        }});
      }})();
    </script>
  </body>
</html>
""".format(tables="".join(bucket_html_blocks))

# Render the HTML (set height so all tables are visible; Streamlit will scroll as needed)
components.html(full_html, height=600, scrolling=True)

# -----------------------
# Single indicator chart (unchanged)
# -----------------------
st.markdown("---")
st.subheader("Single indicator chart")
indicator = st.selectbox("Select indicator", list(qdata.keys()))
timeframe = st.selectbox("Timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)

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

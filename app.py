# app.py
# Quarterly text table (values) with color shading by long-term z-score + single selectable interactive chart
# Requires FRED_API_KEY in Streamlit Secrets or environment
# Requirements: streamlit, pandas, numpy, plotly, fredapi, requests, lxml

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import requests
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc

st.set_page_config(page_title="Quarterly Macro Table + Single Chart", layout="wide")

# --- FRED / series config ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    st.warning("Set FRED_API_KEY environment variable (get one at https://fred.stlouisfed.org)")
    st.stop()

fred = Fred(api_key=FRED_API_KEY)

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

# Which indicators where "higher is good" (+1) vs "higher is bad" (-1)
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

# Display format: 'pct' => "0.0%", 'float1' => one decimal, 'int' => no decimals
DISPLAY_TYPE = {
    "CPI": "float1",
    "Core CPI": "float1",
    "Core PCE": "float1",
    "5y5y inflation": "pct",
    "Real GDP (level)": "float1",
    "Industrial Production": "float1",
    "Real Retail Sales (adv)": "float1",
    "Unemployment": "pct",
    "JOLTS (Job Openings)": "int",
    "Consumer Sentiment (UMich)": "float1",
    "10y Treasury": "pct",
    "2y Treasury": "pct",
}

# --- helpers for fetch & transforms ---
@st.cache_data(ttl=3600)
def fetch_series_from_fred(series_id):
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
        url = "https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/rob_legacy_data/2074_ism_manufacturing_pmi.csv"
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and len(r.text) > 200:
            df = pd.read_csv(StringIO(r.text), parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index)
            return df.iloc[:, 0].dropna().sort_index()
    except Exception:
        pass
    return pd.Series(dtype=float)

# load data
with st.spinner("Fetching series from FRED..."):
    data = {name: fetch_series_from_fred(sid) for name, sid in SERIES.items()}
ism_series = fetch_ism_csv_fallback()

# --- utility functions ---
def to_quarter_label(ts: pd.Timestamp):
    q = ((ts.month - 1) // 3) + 1
    return f"q{q}{ts.year}"

def make_quarterly_series(s: pd.Series):
    if s.empty:
        return s
    try:
        q = s.resample('Q').last()
    except Exception:
        q = s
    q = q.dropna()
    return q

def compute_long_term_z(series_q: pd.Series, direction: int, min_obs=20):
    s = series_q.dropna()
    if s.empty or len(s) < min_obs:
        return pd.Series(index=series_q.index, data=[np.nan]*len(series_q))
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(index=series_q.index, data=[np.nan]*len(series_q))
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
    t = (z - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    hexcol = pc.sample_colorscale(colorscale, [t])[0]
    return hexcol

# --- build quarterly table for each series ---
all_quarters = set()
series_q_map = {}
for name, s in data.items():
    if name == "ISM Manufacturing PMI":
        s2 = ism_series
    else:
        s2 = s
    q = make_quarterly_series(s2)
    series_q_map[name] = q
    all_quarters.update(q.index)

if not ism_series.empty and "ISM Manufacturing PMI" not in series_q_map:
    q = make_quarterly_series(ism_series)
    series_q_map["ISM Manufacturing PMI"] = q
    all_quarters.update(q.index)

if not all_quarters:
    st.error("No quarterly data available.")
    st.stop()

quarter_index = sorted(list(all_quarters))
quarter_labels = [to_quarter_label(q) for q in quarter_index]

reported_df = pd.DataFrame(index=list(series_q_map.keys()), columns=quarter_labels, dtype=object)
z_df = pd.DataFrame(index=list(series_q_map.keys()), columns=quarter_labels, dtype=float)

for name, qseries in series_q_map.items():
    direction = INDICATOR_DIRECTION.get(name, +1)
    zseries = compute_long_term_z(qseries, direction)
    for idx, q_ts in enumerate(quarter_index):
        label = quarter_labels[idx]
        if q_ts in qseries.index:
            val = qseries.loc[q_ts]
            dtype = DISPLAY_TYPE.get(name, "float1")
            reported_df.loc[name, label] = format_value(val, dtype)
            z_df.loc[name, label] = zseries.loc[q_ts] if q_ts in zseries.index else np.nan
        else:
            reported_df.loc[name, label] = "n/a"
            z_df.loc[name, label] = np.nan

# Prepare Plotly table inputs
table_header = ["Indicator"] + quarter_labels

cell_text = []
row_names = list(reported_df.index)
for r in row_names:
    row = [r] + [reported_df.loc[r, col] for col in quarter_labels]
    cell_text.append(row)

cells_color = []
for r in row_names:
    row_colors = ["#f7f7f7"]
    for col in quarter_labels:
        z = z_df.loc[r, col]
        color = z_to_color(z, vmin=-2.5, vmax=2.5)
        row_colors.append(color)
    cells_color.append(row_colors)

cols_text = list(zip(*cell_text))
cols_color = list(zip(*cells_color))
cols_text = [list(c) for c in cols_text]
cols_color = [list(c) for c in cols_color]

# Highlight most recent quarter header
num_quarters = len(quarter_labels)
# header colors: first column dark, middle columns light, last column highlighted
header_colors = ["#333333"] + ["#f0f0f0"] * (num_quarters - 1)
if num_quarters >= 1:
    header_colors[-1] = "#cfe2f3"  # light blue for most recent quarter

# Build Plotly table
fig_table = go.Figure(data=[go.Table(
    header=dict(values=table_header,
                fill_color=header_colors,
                font=dict(color=["white"] + ["#111111"] * num_quarters, size=12),
                align="center"),
    cells=dict(values=cols_text,
               fill_color=cols_color,
               align="center",
               font=dict(color=["#111111"] * (num_quarters+1), size=11),
               height= thirty := 30)
)])

# Adjust width and height for horizontal scroll in Streamlit
table_width = max(1000, 120 * num_quarters)
fig_table.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=table_width, height=420)

# Page
st.title("Quarterly Values Table — colored by long-term z-score")
st.markdown("Numbers are **reported values**. Colors show distance from the series' long-run average (green = better, red = worse). Scroll horizontally to view older quarters. Column names show quarter and year like `q42024` (most recent quarter on the right).")

st.plotly_chart(fig_table, use_container_width=True)

# Legend (z-score ranges)
st.markdown("")
legend_html = """
<div style="display:flex;gap:12px;align-items:center;">
  <div style="display:flex;flex-direction:row;gap:6px;align-items:center;">
    <div style="width:18px;height:18px;background:#a50026;border:1px solid #000;"></div><div><b>&lt;= -2.0</b><br><small>Much worse vs history</small></div>
  </div>
  <div style="display:flex;flex-direction:row;gap:6px;align-items:center;">
    <div style="width:18px;height:18px;background:#f46d43;border:1px solid #000;"></div><div><b>-2.0 to -1.0</b><br><small>Worse</small></div>
  </div>
  <div style="display:flex;flex-direction:row;gap:6px;align-items:center;">
    <div style="width:18px;height:18px;background:#fee08b;border:1px solid #000;"></div><div><b>-1.0 to +1.0</b><br><small>Near normal</small></div>
  </div>
  <div style="display:flex;flex-direction:row;gap:6px;align-items:center;">
    <div style="width:18px;height:18px;background:#d9ef8b;border:1px solid #000;"></div><div><b>+1.0 to +2.0</b><br><small>Better</small></div>
  </div>
  <div style="display:flex;flex-direction:row;gap:6px;align-items:center;">
    <div style="width:18px;height:18px;background:#006837;border:1px solid #000;"></div><div><b>&gt;= +2.0</b><br><small>Much better vs history</small></div>
  </div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# --- Below: single-chart selector (one chart at a time) ---
st.markdown("---")
st.subheader("Single indicator chart (select one)")
indicator_options = list(series_q_map.keys())
selected = st.selectbox("Select indicator", indicator_options, index=indicator_options.index("CPI") if "CPI" in indicator_options else 0)

# timeframe for chart
tf = st.selectbox("Chart timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)

def timeframe_cutoff(series_q, timeframe_key):
    if series_q.empty:
        return series_q
    last = series_q.index.max()
    if timeframe_key == "1Y":
        cutoff = last - pd.DateOffset(years=1)
    elif timeframe_key == "3Y":
        cutoff = last - pd.DateOffset(years=3)
    elif timeframe_key == "5Y":
        cutoff = last - pd.DateOffset(years=5)
    elif timeframe_key == "10Y":
        cutoff = last - pd.DateOffset(years=10)
    else:
        cutoff = pd.Timestamp("1900-01-01")
    return series_q[series_q.index >= cutoff]

if selected == "ISM Manufacturing PMI":
    ser = ism_series
else:
    ser = data.get(selected, pd.Series(dtype=float))

qser = make_quarterly_series(ser)
qser_tf = timeframe_cutoff(qser, tf)

if qser_tf.empty:
    st.write("No data available for this indicator.")
else:
    df_plot = qser_tf.reset_index()
    df_plot.columns = ["date", "value"]
    dtype = DISPLAY_TYPE.get(selected, "float1")
    df_plot["label"] = df_plot["value"].apply(lambda x: format_value(x, dtype))
    fig = px.line(df_plot, x="date", y="value", markers=True, title=f"{selected} — quarterly values", labels={"value": selected, "date":"Date"})
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Value: %{customdata}<extra></extra>", customdata=df_plot["label"])
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

    # compact table for visible window
    small_labels = [to_quarter_label(d) for d in df_plot["date"]]
    small_vals = [format_value(v, dtype) for v in df_plot["value"]]
    table_df = pd.DataFrame([small_vals], index=[selected], columns=small_labels)
    st.markdown("**Quarterly values (visible window):**")
    st.dataframe(table_df.style.set_table_styles([{'selector':'th','props':[('text-align','center')] }]), use_container_width=True)

st.write("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

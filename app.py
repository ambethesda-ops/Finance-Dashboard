# app.py
# Quarterly text table (values) with color shading by long-term z-score + single selectable interactive chart
# - Most recent quarter ON THE LEFT (scroll right for older quarters)
# - Column labels like "Q1 2026"
# - For CPI, Core CPI, Core PCE, Real GDP, Industrial Production, Real Retail Sales: show YoY % (and shade by YoY z-score)
# - Bottom chart: when those series are selected, show YoY % growth (not levels)
# - Removed the second small table under the chart
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
import plotly.express as px
import plotly.colors as pc
import html as _html

st.set_page_config(page_title="Quarterly Macro Table + Single Chart (YoY for key series)", layout="wide")

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

# Series for which we show YoY percent (and shade based on YoY z-score)
YOY_SERIES = {
    "CPI",
    "Core CPI",
    "Core PCE",
    "Real GDP (level)",
    "Industrial Production",
    "Real Retail Sales (adv)",
}

# Which indicators where "higher is good" (+1) vs "higher is bad" (-1)
# direction multiplies z-score so green = good and red = bad
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
    # For YOY_SERIES we'll display pct
    "CPI": "pct",
    "Core CPI": "pct",
    "Core PCE": "pct",
    "5y5y inflation": "pct",
    "Real GDP (level)": "pct",
    "Industrial Production": "pct",
    "Real Retail Sales (adv)": "pct",
    "Unemployment": "pct",
    "JOLTS (Job Openings)": "int",
    "Consumer Sentiment (UMich)": "float1",
    "10y Treasury": "pct",
    "2y Treasury": "pct",
}

# --- fetch helpers ---
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
    # Format like "Q1 2026"
    q = ((ts.month - 1) // 3) + 1
    return f"Q{q} {ts.year}"

def make_quarterly_series(s: pd.Series):
    if s.empty:
        return s
    try:
        q = s.resample('Q').last()
    except Exception:
        q = s
    q = q.dropna()
    return q

def compute_yoy(series_q: pd.Series):
    # returns YoY percent series: (value / value.shift(4) - 1) * 100
    if series_q.empty:
        return series_q
    s = series_q.dropna()
    if s.empty:
        return s
    yoy = s.pct_change(periods=4) * 100
    return yoy

def compute_long_term_z(series_q: pd.Series, direction: int, min_obs=20):
    # Compute z-scores of the series_q (long-run mean/std) and multiply by direction
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
    # allow ISM fallback
    if name == "ISM Manufacturing PMI":
        s2 = ism_series
    else:
        s2 = s
    q = make_quarterly_series(s2)
    series_q_map[name] = q
    all_quarters.update(q.index)

# include ISM explicitly if found
if not ism_series.empty and "ISM Manufacturing PMI" not in series_q_map:
    q = make_quarterly_series(ism_series)
    series_q_map["ISM Manufacturing PMI"] = q
    all_quarters.update(q.index)

if not all_quarters:
    st.error("No quarterly data available.")
    st.stop()

# sort quarters NEWEST -> OLDEST so newest appears on the LEFT
quarter_index = sorted(list(all_quarters), reverse=True)
quarter_labels = [to_quarter_label(q) for q in quarter_index]

# prepare reported values and z-scores; for YOY_SERIES show YoY percent values and shade by YoY z
reported_df = pd.DataFrame(index=list(series_q_map.keys()), columns=quarter_labels, dtype=object)
z_df = pd.DataFrame(index=list(series_q_map.keys()), columns=quarter_labels, dtype=float)

# Precompute yoy series for YOY_SERIES to avoid recomputing in loop
yoy_map = {}
for name, qseries in series_q_map.items():
    if name in YOY_SERIES:
        yoy_map[name] = compute_yoy(qseries)
    else:
        yoy_map[name] = pd.Series(dtype=float)

for name, qseries in series_q_map.items():
    direction = INDICATOR_DIRECTION.get(name, +1)
    # compute series used for shading (z): YoY for selected series, else level
    if name in YOY_SERIES:
        zseries = compute_long_term_z(yoy_map[name], direction)
    else:
        zseries = compute_long_term_z(qseries, direction)

    # fill reported_df: for YOY_SERIES show YoY%, else show reported level formatted
    for idx, q_ts in enumerate(quarter_index):
        label = quarter_labels[idx]
        if q_ts in qseries.index:
            if name in YOY_SERIES:
                yoy_val = np.nan
                if q_ts in yoy_map[name].index:
                    yoy_val = yoy_map[name].loc[q_ts]
                reported_df.loc[name, label] = format_value(yoy_val, "pct") if (not pd.isna(yoy_val)) else "n/a"
                z_df.loc[name, label] = zseries.loc[q_ts] if q_ts in zseries.index else np.nan
            else:
                val = qseries.loc[q_ts]
                dtype = DISPLAY_TYPE.get(name, "float1")
                reported_df.loc[name, label] = format_value(val, dtype)
                z_df.loc[name, label] = zseries.loc[q_ts] if q_ts in zseries.index else np.nan
        else:
            reported_df.loc[name, label] = "n/a"
            z_df.loc[name, label] = np.nan

# -------------------------
# Render HTML table (scrollable). Newest on LEFT.
# -------------------------
def render_html_table(reported_df, z_df, quarter_labels, highlight_first=True, cell_font_size=11, col_min_width=100, name_col_width=240):
    # header: Indicator + quarter columns (newest left)
    header_cells = ['<th style="position: sticky; left:0; background:#333333; color:white; z-index:2; padding:8px;">Indicator</th>']
    for i, q in enumerate(quarter_labels):
        # highlight most recent column (first column in list)
        if highlight_first and i == 0:
            header_cells.append(f'<th style="background:#cfe2f3; min-width:{col_min_width}px; white-space:nowrap; padding:8px;">{_html.escape(q)}</th>')
        else:
            header_cells.append(f'<th style="background:#f0f0f0; min-width:{col_min_width}px; white-space:nowrap; padding:8px;">{_html.escape(q)}</th>')
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    body_rows = []
    for idx in reported_df.index:
        row_html = []
        # sticky indicator name
        row_html.append(f'<td style="position: sticky; left:0; background:#f7f7f7; min-width:{name_col_width}px; text-align:left; padding:8px;">{_html.escape(idx)}</td>')
        for col in quarter_labels:
            txt = reported_df.loc[idx, col] if col in reported_df.columns else "n/a"
            z = z_df.loc[idx, col] if (idx in z_df.index and col in z_df.columns) else np.nan
            color = z_to_color(z, vmin=-2.5, vmax=2.5)
            cell_html = f'<td style="background:{color}; min-width:{col_min_width}px; white-space:nowrap; text-align:center; padding:6px 8px; font-size:{cell_font_size}px;">{_html.escape(str(txt))}</td>'
            row_html.append(cell_html)
        body_rows.append("<tr>" + "".join(row_html) + "</tr>")

    table_html = f"""
    <div style="overflow-x:auto; border:1px solid #eee; padding:6px; margin-bottom:8px;">
      <table style="border-collapse:collapse; font-family:Arial, Helvetica, sans-serif;">
        <thead>{header_html}</thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
    </div>
    """
    return table_html

# Render page header and table
st.title("Quarterly Values Table — colored by long-term z-score")
st.markdown("Numbers show reported values (for key series the table shows YoY %). Colors show distance from the series' long-run average (green = better, red = worse). Scroll horizontally to view older quarters. Column headers read like 'Q1 2026' (most recent quarter is on the left).")

html_table = render_html_table(reported_df, z_df, quarter_labels, highlight_first=True, cell_font_size=11, col_min_width=100, name_col_width=240)
st.markdown(html_table, unsafe_allow_html=True)

# Legend
st.markdown("")
legend_html = """
<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;">
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#a50026;border:1px solid #000;"></div><div><b>&lt;= -2.0</b><br><small>Much worse vs history</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#f46d43;border:1px solid #000;"></div><div><b>-2.0 to -1.0</b><br><small>Worse</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#fee08b;border:1px solid #000;"></div><div><b>-1.0 to +1.0</b><br><small>Near normal</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#d9ef8b;border:1px solid #000;"></div><div><b>+1.0 to +2.0</b><br><small>Better</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#006837;border:1px solid #000;"></div><div><b>&gt;= +2.0</b><br><small>Much better vs history</small></div></div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# --- Single indicator chart (one at a time) ---
st.markdown("---")
st.subheader("Single indicator chart (select one)")
indicator_options = list(series_q_map.keys())
default_idx = 0
if "CPI" in indicator_options:
    default_idx = indicator_options.index("CPI")
selected = st.selectbox("Select indicator", indicator_options, index=default_idx)

# timeframe selector
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

# get series for selected indicator
if selected == "ISM Manufacturing PMI":
    ser = ism_series
else:
    ser = data.get(selected, pd.Series(dtype=float))

qser = make_quarterly_series(ser)
qser_tf = timeframe_cutoff(qser, tf)

if qser_tf.empty:
    st.write("No data available for this indicator.")
else:
    # If the selected series is in YOY_SERIES, show YoY percent line
    if selected in YOY_SERIES:
        yseries = compute_yoy(qser_tf).dropna()
        if yseries.empty:
            st.write("No YoY data available for this timeframe.")
        else:
            df_plot = yseries.reset_index()
            df_plot.columns = ["date", "yoy"]
            fig = px.line(df_plot, x="date", y="yoy", markers=True, title=f"{selected} — YoY %", labels={"yoy":"YoY %", "date":"Date"})
            fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>YoY: %{y:.2f}%<extra></extra>")
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)
    else:
        # plot level series for non-YOY
        df_plot = qser_tf.reset_index()
        df_plot.columns = ["date", "value"]
        dtype = DISPLAY_TYPE.get(selected, "float1")
        df_plot["label"] = df_plot["value"].apply(lambda x: format_value(x, dtype))
        fig = px.line(df_plot, x="date", y="value", markers=True, title=f"{selected} — quarterly values", labels={"value": selected, "date":"Date"})
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Value: %{customdata}<extra></extra>", customdata=df_plot["label"])
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

# (Removed the extra compact table below the chart per request)

st.write("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")


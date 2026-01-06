# app.py
# Macro Indicators Heat Map — updated
# - Newest quarter on LEFT (scroll right for older)
# - YoY values for specified series; Real Wages from LES1252881600Q
# - Added: Initial Claims, High Yield OAS, Housing Starts, New Vehicle Sales (moved to bucket 1)
# - Removed: Existing Home Sales
# - Legend at top (no 5y5y note)
# - Sticky left indicator column anchored, centered headers/values, shading via z-scores
#
# Config: set FRED_API_KEY in env or Streamlit secrets
# Requirements in requirements.txt

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
    "New Vehicle Sales (proxy)": "TOTALSA",  # moved to bucket 1 per request

    # Bucket 2: Growth / Activity
    "Real GDP (level)": "GDPC1",
    "Industrial Production": "INDPRO",
    "Real Retail Sales (adv)": "RRSFS",
    "Housing Starts": "HOUST",

    # Bucket 3: Labor
    "Unemployment": "UNRATE",
    "Initial Claims": "ICSA",  # weekly initial claims (we'll take quarter-end last obs)
    "JOLTS (Job Openings)": "JTSJOL",
    "Real Wages (YoY)": "LES1252881600Q",  # LES1252881600Q used to compute YoY

    # Bucket 4: Sentiment / Leading
    "ISM Manufacturing PMI": "PMI_USA_MAN",  # placeholder; will fallback to a CSV if missing
    "Consumer Sentiment (UMich)": "UMCSENT",

    # Bucket 5: Rates & Credit
    "10y Treasury": "DGS10",
    "2y Treasury": "DGS2",
    "High Yield OAS": "BAMLH0A0HYM2",
}

# The set of series we want to display as YoY percent in the table (and shade by YoY z-score)
YOY_SERIES = {
    "CPI",
    "Core CPI",
    "Core PCE",
    "Real GDP (level)",
    "Industrial Production",
    "Real Retail Sales (adv)",
    "Real Wages (YoY)",
}

# Direction mapping: +1 = higher is good, -1 = higher is bad (used to flip z-score sign)
INDICATOR_DIRECTION = {
    "CPI": -1,
    "Core CPI": -1,
    "Core PCE": -1,
    "5y5y inflation (market-implied)": -1,
    "New Vehicle Sales (proxy)": +1,
    "Real GDP (level)": +1,
    "Industrial Production": +1,
    "Real Retail Sales (adv)": +1,
    "Housing Starts": +1,
    "Unemployment": -1,
    "Initial Claims": -1,
    "JOLTS (Job Openings)": +1,
    "Real Wages (YoY)": +1,
    "ISM Manufacturing PMI": +1,
    "Consumer Sentiment (UMich)": +1,
    "10y Treasury": -1,
    "2y Treasury": -1,
    "High Yield OAS": -1,
}

# Display formatting preferences
DISPLAY_TYPE = {
    "CPI": "pct",
    "Core CPI": "pct",
    "Core PCE": "pct",
    "5y5y inflation (market-implied)": "pct",
    "New Vehicle Sales (proxy)": "float1",
    "Real GDP (level)": "pct",
    "Industrial Production": "pct",
    "Real Retail Sales (adv)": "pct",
    "Housing Starts": "int",
    "Unemployment": "pct",
    "Initial Claims": "int",
    "JOLTS (Job Openings)": "int",
    "Real Wages (YoY)": "pct",
    "ISM Manufacturing PMI": "float1",
    "Consumer Sentiment (UMich)": "float1",
    "10y Treasury": "pct",
    "2y Treasury": "pct",
    "High Yield OAS": "float1",
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
    # fallback for ISM PMI if the FRED ID isn't present
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

# Fetch all configured series
with st.spinner("Fetching series from FRED..."):
    raw_data = {name: fetch_series_safe(sid) for name, sid in SERIES_MAP.items()}

# fallback for ISM if the FRED id wasn't available
if raw_data.get("ISM Manufacturing PMI", pd.Series(dtype=float)).empty:
    raw_data["ISM Manufacturing PMI"] = fetch_ism_csv_fallback()

# -----------------------
# Transforms & helpers
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
    mean = s.mean()
    std = s.std()
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
    t = (z - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return pc.sample_colorscale(colorscale, [t])[0]

# -----------------------
# Build quarterly samples for all series
# -----------------------
series_q = {}
for name, raw in raw_data.items():
    series_q[name] = to_quarter_series(raw)

# For Real Wages: LES1252881600Q is already quarterly and represents real median weekly earnings.
# We'll compute YoY from it.
if "Real Wages (YoY)" in series_q:
    rw = raw_data.get("Real Wages (YoY)", pd.Series(dtype=float))
    if not rw.empty:
        series_q["Real Wages (YoY)"] = compute_yoy(to_quarter_series(rw))
    else:
        # if not present, leave as empty
        series_q["Real Wages (YoY)"] = pd.Series(dtype=float)

# -----------------------
# Determine quarter set (union across series) and sort newest->oldest so newest on LEFT
# -----------------------
all_quarters = set()
for s in series_q.values():
    if not s.empty:
        all_quarters.update(s.index)

if not all_quarters:
    st.error("No quarterly data found. Check FRED series IDs.")
    st.stop()

quarter_index = sorted(list(all_quarters), reverse=True)  # newest first (left)
quarter_labels = [to_quarter_label(q) for q in quarter_index]

# -----------------------
# Build reported text table + z-score map
# -----------------------
rows = list(series_q.keys())
reported_df = pd.DataFrame(index=rows, columns=quarter_labels, dtype=object)
z_df = pd.DataFrame(index=rows, columns=quarter_labels, dtype=float)

# Precompute yoy for YOY_SERIES to avoid repeated compute
yoy_cache = {}
for name in rows:
    if name in YOY_SERIES:
        yoy_cache[name] = compute_yoy(series_q[name])
    else:
        yoy_cache[name] = pd.Series(dtype=float)

for name in rows:
    qseries = series_q.get(name, pd.Series(dtype=float))
    direction = INDICATOR_DIRECTION.get(name, +1)
    # choose series to compute z (YoY if requested else level)
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
# Render header: legend at top (no 5y5y note)
# -----------------------
st.title("Macro Indicators Heat Map")

legend_html = """
<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#a50026;border:1px solid #000;"></div><div><b>&lt;= -2.0</b><br><small>Much worse vs history</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#f46d43;border:1px solid #000;"></div><div><b>-2.0 to -1.0</b><br><small>Worse</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#fee08b;border:1px solid #000;"></div><div><b>-1.0 to +1.0</b><br><small>Near normal</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#d9ef8b;border:1px solid #000;"></div><div><b>+1.0 to +2.0</b><br><small>Better</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#006837;border:1px solid #000;"></div><div><b>&gt;= +2.0</b><br><small>Much better vs history</small></div></div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# -----------------------
# Render scrollable HTML table (newest on left). Sticky left indicator column anchored.
# -----------------------
def render_html_table(reported_df, z_df, quarter_labels, highlight_first=True, col_min_width=110, name_col_width=260, font_size=12):
    header_cells = ['<th style="position: sticky; left:0; background:#333333; color:white; z-index:4; padding:8px; text-align:center;">Indicator</th>']
    for i, q in enumerate(quarter_labels):
        if highlight_first and i == 0:
            header_cells.append(f'<th style="background:#cfe2f3; min-width:{col_min_width}px; white-space:nowrap; padding:8px; text-align:center;">{_html.escape(q)}</th>')
        else:
            header_cells.append(f'<th style="background:#f0f0f0; min-width:{col_min_width}px; white-space:nowrap; padding:8px; text-align:center;">{_html.escape(q)}</th>')
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    body_rows = []
    for idx in reported_df.index:
        row_cells = []
        # Sticky label on left (solid white background, higher z-index)
        row_cells.append(f'<td style="position: sticky; left:0; background:#ffffff; z-index:5; min-width:{name_col_width}px; text-align:left; padding:10px; border-right:1px solid #e6e6e6;">{_html.escape(idx)}</td>')
        for col in quarter_labels:
            txt = reported_df.loc[idx, col] if col in reported_df.columns else "n/a"
            z = z_df.loc[idx, col] if (idx in z_df.index and col in z_df.columns) else np.nan
            color = z_to_color(z, vmin=-2.5, vmax=2.5)
            row_cells.append(f'<td style="background:{color}; min-width:{col_min_width}px; white-space:nowrap; text-align:center; padding:8px; font-size:{font_size}px; border-right:1px solid #ffffff;">{_html.escape(str(txt))}</td>')
        body_rows.append("<tr>" + "".join(row_cells) + "</tr>")

    html = f"""
    <div style="overflow-x:auto; border:1px solid #eee; padding:6px; margin-bottom:8px;">
      <table style="border-collapse:collapse; font-family:Arial, Helvetica, sans-serif; width:100%;">
        <thead>{header_html}</thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
    </div>
    """
    return html

st.markdown(render_html_table(reported_df, z_df, quarter_labels, highlight_first=True, col_min_width=110, name_col_width=260, font_size=12), unsafe_allow_html=True)

# -----------------------
# Single indicator chart (select one) — YoY plotted for YOY_SERIES; single line only
# -----------------------
st.markdown("---")
st.subheader("Single indicator chart (select one)")
indicator_options = list(series_q.keys())
default_idx = 0
if "CPI" in indicator_options:
    default_idx = indicator_options.index("CPI")
selected = st.selectbox("Select indicator", indicator_options, index=default_idx)

# timeframe selector
tf = st.selectbox("Chart timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)

def timeframe_cutoff(series_q: pd.Series, timeframe_key: str):
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

sel_q = series_q.get(selected, pd.Series(dtype=float))
sel_cut = timeframe_cutoff(sel_q, tf)

if selected in YOY_SERIES:
    plot_s = compute_yoy(sel_cut)
    plot_s = plot_s.dropna()
    if plot_s.empty:
        st.write("No YoY data available for this timeframe.")
    else:
        dfp = plot_s.reset_index()
        dfp.columns = ["date", "value_pct"]
        fig = px.line(dfp, x="date", y="value_pct", markers=True, title=f"{selected} — YoY %", labels={"value_pct":"YoY %", "date":"Date"})
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>YoY: %{y:.2f}%<extra></extra>")
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        st.plotly_chart(fig, use_container_width=True)
else:
    if sel_cut.empty:
        st.write("No data available for this indicator.")
    else:
        dfp = sel_cut.reset_index()
        dfp.columns = ["date", "value"]
        dtype = DISPLAY_TYPE.get(selected, "float1")
        dfp["label"] = dfp["value"].apply(lambda x: format_value(x, dtype))
        fig = px.line(dfp, x="date", y="value", markers=True, title=f"{selected} — Level", labels={"value": selected, "date":"Date"})
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Value: %{customdata}<extra></extra>", customdata=dfp["label"])
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

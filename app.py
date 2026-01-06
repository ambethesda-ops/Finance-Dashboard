# app.py
# Macro Indicators Heat Map
# - Quarterly text table (values) with color shading by long-term z-score (green=good, red=bad)
# - Newest quarter on LEFT (scroll right for older)
# - Column labels like "Q1 2026"
# - Key & legend at top; clarifying note for 5y5y inflation (market-implied expectations)
# - Bottom chart: YoY % for selected key series; no extra secondary/duplicates
#
# IMPORTANT:
# - This uses FRED. Put your FRED API key into Streamlit Secrets or set env var FRED_API_KEY.
# - Some series IDs may differ by country/format; see the SERIES_MAP below and change IDs if necessary.
#
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

st.set_page_config(page_title="Macro Indicators Heat Map", layout="wide")

# -----------------------
# Config: edit FRED series IDs here if you prefer alternatives
# Notes: default IDs are common FRED IDs; replace any that don't exist in your account
# -----------------------
# Core set (previously used)
SERIES_MAP = {
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "Core PCE": "PCEPILFE",
    "5y5y inflation (market-implied)": "T5YIFR",
    "Real GDP (level)": "GDPC1",
    "Industrial Production": "INDPRO",
    "Real Retail Sales (adv)": "RRSFS",
    "Unemployment": "UNRATE",
    "JOLTS (Job Openings)": "JTSJOL",
    "Consumer Sentiment (UMich)": "UMCSENT",
    "10y Treasury": "DGS10",
    "2y Treasury": "DGS2",
}

# Additional requested series — tweak IDs here if you'd like other measures
EXTRA_SERIES = {
    # Weekly initial unemployment claims (seasonally adjusted) — shown at quarter-end last observation
    "Initial Claims": "ICSA",  # common FRED ID for initial claims (weekly)

    # High-yield OAS (BofA US HY option-adjusted spread) — common FRED ID
    "High Yield OAS": "BAMLH0A0HYM2",

    # Real wages: we'll attempt to compute Real Wages YoY = AHE YoY - CPI YoY.
    # Default AHE series candidates (replace with exact series if you prefer):
    # "Average Hourly Earnings: Total Private" : "CES0500000003"  (may be available under a different code)
    # We'll try a few candidates below; update if needed.
    "AHE_candidate_1": "AHEW",  # placeholder candidate (replace if absent)
    "AHE_candidate_2": "AHETPI",  # placeholder candidate (replace if absent)

    # Housing starts (total)
    "Housing Starts": "HOUST",

    # Existing home sales (attempt a common FRED id; adjust if needed)
    "Existing Home Sales": "EXHOSLUSM495S",  # may need replacing depending on your FRED access

    # New vehicle sales / auto purchases — using a proxy retail series if exact ID unavailable
    "New Vehicle Sales (proxy)": "TOTALSA",  # placeholder; replace with preferred FRED id
}

# Merge into main mapping for easy iteration
ALL_SERIES_MAP = {**SERIES_MAP, **EXTRA_SERIES}

# The series for which we show YoY percent in the table (and shade by YoY z-score)
YOY_SERIES = {
    "CPI",
    "Core CPI",
    "Core PCE",
    "Real GDP (level)",
    "Industrial Production",
    "Real Retail Sales (adv)",
}

# Direction mapping: +1 = higher is good, -1 = higher is bad (used to flip z-score sign)
INDICATOR_DIRECTION = {
    "CPI": -1,
    "Core CPI": -1,
    "Core PCE": -1,
    "5y5y inflation (market-implied)": -1,
    "Real GDP (level)": +1,
    "Industrial Production": +1,
    "Real Retail Sales (adv)": +1,
    "Unemployment": -1,
    "JOLTS (Job Openings)": +1,
    "Consumer Sentiment (UMich)": +1,
    "10y Treasury": -1,
    "2y Treasury": -1,
    "Initial Claims": -1,
    "High Yield OAS": -1,
    "Housing Starts": +1,
    "Existing Home Sales": +1,
    "New Vehicle Sales (proxy)": +1,
}

# Display formatting preferences (overridable)
DISPLAY_TYPE = {
    "CPI": "pct",
    "Core CPI": "pct",
    "Core PCE": "pct",
    "5y5y inflation (market-implied)": "pct",
    "Real GDP (level)": "pct",
    "Industrial Production": "pct",
    "Real Retail Sales (adv)": "pct",
    "Unemployment": "pct",
    "JOLTS (Job Openings)": "int",
    "Consumer Sentiment (UMich)": "float1",
    "10y Treasury": "pct",
    "2y Treasury": "pct",
    "Initial Claims": "int",
    "High Yield OAS": "float1",
    "Housing Starts": "int",
    "Existing Home Sales": "float1",
    "New Vehicle Sales (proxy)": "float1",
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
# Utility: fetch from FRED safely
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

# Fetch all series (some may be empty if ID doesn't exist; that's ok)
with st.spinner("Fetching series from FRED..."):
    raw_data = {name: fetch_series_safe(sid) for name, sid in ALL_SERIES_MAP.items()}
ism_series = fetch_ism_csv_fallback()

# -----------------------
# Helper transforms
# -----------------------
def to_quarter_end_label(ts: pd.Timestamp):
    q = ((ts.month - 1) // 3) + 1
    return f"Q{q} {ts.year}"

def to_quarter_indexed_series(s: pd.Series):
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
    return (s.pct_change(periods=4) * 100).dropna()

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
# Build quarterly series map (quarter-end sampling)
# -----------------------
series_q = {}
for name, raw in raw_data.items():
    # if this was the AHE candidate placeholders, skip adding them directly as rows — they are for computing real wages only
    if name.startswith("AHE_candidate"):
        continue
    # fallback: if empty, but it's ISM series name, use ism_series
    if raw.empty and name == "Consumer Sentiment (UMich)":
        raw = ism_series
    series_q[name] = to_quarter_indexed_series(raw)

# Compute Real Wages YoY if possible:
# Strategy: attempt candidate AHE series IDs (user can edit above). If found, compute YoY of nominal AHE minus CPI YoY.
ahe_series = None
for cand in ["AHE_candidate_1", "AHE_candidate_2"]:
    cand_id = ALL_SERIES_MAP.get(cand)
    if cand_id:
        s = fetch_series_safe(cand_id)
        if not s.empty:
            ahe_series = to_quarter_indexed_series(s)
            break

if ahe_series is not None and not ahe_series.empty and "CPI" in raw_data and not raw_data["CPI"].empty:
    # compute nominal AHE YoY and CPI YoY, then compute "real wages growth" = AHE YoY - CPI YoY
    ahe_yoy = compute_yoy(ahe_series)
    cpi_q = to_quarter_indexed_series(raw_data["CPI"])
    cpi_yoy = compute_yoy(cpi_q)
    # align indices and subtract where possible
    common_idx = ahe_yoy.index.intersection(cpi_yoy.index)
    real_wage_yoy = pd.Series(index=common_idx, data=(ahe_yoy.loc[common_idx].values - cpi_yoy.loc[common_idx].values))
    # Add to series_q map with label
    series_q["Real Wages (YoY)"] = real_wage_yoy
    # display type
    DISPLAY_TYPE["Real Wages (YoY)"] = "pct"
    INDICATOR_DIRECTION["Real Wages (YoY)"] = +1
else:
    # fallback: if can't compute real wages, try to find any AHE series and show its YoY as placeholder
    if ahe_series is not None and not ahe_series.empty:
        series_q["Average Hourly Earnings (YoY)"] = compute_yoy(ahe_series)
        DISPLAY_TYPE["Average Hourly Earnings (YoY)"] = "pct"
        INDICATOR_DIRECTION["Average Hourly Earnings (YoY)"] = +1

# -----------------------
# Determine quarter index (union across series) and sort NEWEST->OLDEST so newest on LEFT
# -----------------------
all_quarters = set()
for s in series_q.values():
    if not s.empty:
        all_quarters.update(s.index)

if not all_quarters:
    st.error("No quarterly data available for any series. Check FRED series IDs.")
    st.stop()

quarter_index = sorted(list(all_quarters), reverse=True)  # newest first
quarter_labels = [to_quarter_end_label(q) for q in quarter_index]

# -----------------------
# Build reported table (text) and z-score table used for coloring
# For YOY_SERIES we display YoY values (and color by YoY z). For others we show levels and color by level z.
# -----------------------
row_names = list(series_q.keys())
reported_df = pd.DataFrame(index=row_names, columns=quarter_labels, dtype=object)
z_df = pd.DataFrame(index=row_names, columns=quarter_labels, dtype=float)

# Precompute YoY maps for series where relevant
yoy_map = {name: compute_yoy(series_q[name]) if name in YOY_SERIES and not series_q[name].empty else pd.Series(dtype=float)
           for name in row_names}

for name in row_names:
    qseries = series_q[name]
    direction = INDICATOR_DIRECTION.get(name, +1)
    # choose series for z calculation
    if name in YOY_SERIES:
        zseries = compute_long_term_z(yoy_map.get(name, pd.Series(dtype=float)), direction)
    else:
        zseries = compute_long_term_z(qseries, direction)

    for i, q_ts in enumerate(quarter_index):
        lbl = quarter_labels[i]
        if q_ts in qseries.index:
            if name in YOY_SERIES:
                val = np.nan
                if q_ts in yoy_map.get(name, pd.Series(dtype=float)).index:
                    val = yoy_map[name].loc[q_ts]
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
# Render UI top: legend (moved to top), small note about 5y5y
# -----------------------
st.title("Macro Indicators Heat Map")

legend_html = """
<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#a50026;border:1px solid #000;"></div><div><b>&lt;= -2.0</b><br><small>Much worse vs history</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#f46d43;border:1px solid #000;"></div><div><b>-2.0 to -1.0</b><br><small>Worse</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#fee08b;border:1px solid #000;"></div><div><b>-1.0 to +1.0</b><br><small>Near normal</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#d9ef8b;border:1px solid #000;"></div><div><b>+1.0 to +2.0</b><br><small>Better</small></div></div>
  <div style="display:flex;gap:6px;align-items:center;"><div style="width:18px;height:18px;background:#006837;border:1px solid #000;"></div><div><b>&gt;= +2.0</b><br><small>Much better vs history</small></div></div>
  <div style="margin-left:24px; font-size:13px; color:#333;">Note: <b>5y5y inflation</b> is a market-implied expectation (5-year forward, 5-year average) and not a realized CPI series.</div>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)

# -----------------------
# Render the scrollable HTML table (newest on left).
# Sticky left column anchored; higher z-index to avoid color bleed.
# -----------------------
def render_html_table(reported_df, z_df, quarter_labels, highlight_first=True, col_min_width=110, name_col_width=260, font_size=12):
    # Header row
    header_cells = ['<th style="position: sticky; left:0; background:#333333; color:white; z-index:4; padding:8px; text-align:center;">Indicator</th>']
    for i, q in enumerate(quarter_labels):
        if highlight_first and i == 0:
            header_cells.append(f'<th style="background:#cfe2f3; min-width:{col_min_width}px; white-space:nowrap; padding:8px; text-align:center;">{_html.escape(q)}</th>')
        else:
            header_cells.append(f'<th style="background:#f0f0f0; min-width:{col_min_width}px; white-space:nowrap; padding:8px; text-align:center;">{_html.escape(q)}</th>')
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # Body rows
    body_rows = []
    for idx in reported_df.index:
        row_cells = []
        # Sticky left label with solid background and right border
        row_cells.append(f'<td style="position: sticky; left:0; background:#ffffff; z-index:5; min-width:{name_col_width}px; text-align:left; padding:10px; border-right:1px solid #e6e6e6;">{_html.escape(idx)}</td>')
        for col in quarter_labels:
            txt = reported_df.loc[idx, col] if col in reported_df.columns else "n/a"
            z = z_df.loc[idx, col] if (idx in z_df.index and col in z_df.columns) else np.nan
            color = z_to_color(z, vmin=-2.5, vmax=2.5)
            # center column text
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
# Single indicator chart (select one). New: for YOY_SERIES plot YoY%, else plot levels.
# Removed the range slider / extra line (single line only).
# -----------------------
st.markdown("---")
st.subheader("Single indicator chart (select one)")
indicator_options = list(series_q.keys())
# default to CPI if present
default_idx = 0
if "CPI" in indicator_options:
    default_idx = indicator_options.index("CPI")
selected = st.selectbox("Select indicator", indicator_options, index=default_idx)

# timeframe selector
tf = st.selectbox("Chart timeframe", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2)

def timeframe_filter(s: pd.Series, timeframe_key: str):
    if s.empty:
        return s
    last = s.index.max()
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
    return s[s.index >= cutoff]

sel_series_q = series_q.get(selected, pd.Series(dtype=float))
sel_series_cut = timeframe_filter(sel_series_q, tf)

# If selected is in the YOY set, compute YoY for plotting
if selected in YOY_SERIES:
    plot_series = compute_yoy(sel_series_cut)
    if plot_series.empty:
        st.write("No YoY data available for this timeframe.")
    else:
        dfp = plot_series.reset_index()
        dfp.columns = ["date", "value_pct"]
        fig = px.line(dfp, x="date", y="value_pct", markers=True, title=f"{selected} — YoY %", labels={"value_pct":"YoY %", "date":"Date"})
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>YoY: %{y:.2f}%<extra></extra>")
        # ensure no rangeslider/duplicate line
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        st.plotly_chart(fig, use_container_width=True)
else:
    # plot levels
    if sel_series_cut.empty:
        st.write("No data available for this indicator.")
    else:
        dfp = sel_series_cut.reset_index()
        dfp.columns = ["date", "value"]
        dtype = DISPLAY_TYPE.get(selected, "float1")
        dfp["label"] = dfp["value"].apply(lambda x: format_value(x, dtype))
        fig = px.line(dfp, x="date", y="value", markers=True, title=f"{selected} — Level", labels={"value": selected, "date":"Date"})
        fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>Value: %{customdata}<extra></extra>", customdata=dfp["label"])
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
        st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

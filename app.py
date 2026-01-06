# (Only the modified/added sections are shown inline here for brevity; paste the full file if you prefer.)
# Full file is below — drop it in place of your existing app.py

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
def to_quarter(s):
    return s.resample("Q").last().dropna() if not s.empty else s

def yoy(s):
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

# -----------------------
# Load data
# -----------------------
raw = {}
for bucket in BUCKETS.values():
    for name, sid in bucket.items():
        raw[name] = fetch_series(sid)

# quarterly data as before
qdata = {}
for name, s in raw.items():
    q = to_quarter(s)
    if name in YOY_SERIES:
        q = yoy(q)
    qdata[name] = q

# compute quarters and labels (existing behavior)
quarters = sorted({d for s in qdata.values() for d in s.index}, reverse=True)
q_labels = [f"Q{((d.month-1)//3)+1} {d.year}" for d in quarters]

# -----------------------
# NEW: monthly columns (latest 2 months) logic
# -----------------------
# gather month-end dates from raw series
all_month_ends = sorted(
    {pd.to_datetime(idx).to_period('M').to_timestamp('M') for s in raw.values() for idx in s.index},
    reverse=True
)

monthly_cols = []
show_monthly = False
if len(all_month_ends) >= 1:
    # take up to 2 most recent month-ends
    monthly_cols = all_month_ends[:2]
    # hide monthly columns when the most recent month is the final month of a quarter (e.g., March)
    # AND that quarter date is present in the quarterly labels (meaning we have the quarterly read).
    most_recent_month = monthly_cols[0]
    is_final_month_of_quarter = (most_recent_month.month % 3 == 0)
    quarter_from_month = most_recent_month.to_period('Q').to_timestamp('Q')  # quarter-end timestamp
    has_quarter = quarter_from_month in quarters
    # show monthly columns unless the final-month-of-quarter AND we already have that quarter
    show_monthly = not (is_final_month_of_quarter and has_quarter)

# monthly labels formatted like "Jan 2026"
monthly_labels = [d.strftime("%b %Y") for d in monthly_cols] if show_monthly else []

# combine labels: monthly (if any) first, then quarters
labels = monthly_labels + q_labels

# -----------------------
# Render table (updated to include monthly cells)
# -----------------------
st.title("Macro Indicators Heat Map")

COL_W = 140
IND_W = 320

header = (
    f'<th style="position:sticky;top:0;left:0;z-index:18;background:#222;color:white;min-width:{IND_W}px;padding:10px;text-align:left;">Indicator</th>'
    + "".join(
        f'<th style="position:sticky;top:0;z-index:17;background:#f0f0f0;min-width:{COL_W}px;padding:10px;text-align:center;border-bottom:1px solid #ddd;">{_html.escape(q)}</th>'
        for q in labels
    )
)

rows_html = []
for bucket, indicators in BUCKETS.items():
    # bucket header row (left sticky)
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

    # indicator rows
    for name in indicators:
        # quarterly row data (same as before)
        s_q = qdata.get(name, pd.Series(dtype=float))

        # monthly data — use raw series resampled to month-ends
        s_raw = raw.get(name, pd.Series(dtype=float))
        if not s_raw.empty:
            s_m = s_raw.resample("M").last().dropna()
            # if indicator is in YOY_SERIES, compute monthly YoY% (12-month change)
            if name in YOY_SERIES:
                try:
                    m_vals = (s_m.pct_change(12) * 100).dropna()
                except Exception:
                    m_vals = s_m
            else:
                m_vals = s_m
        else:
            m_vals = pd.Series(dtype=float)

        # z-scores for quarterly and monthly (use same direction mapping)
        z_q = zscore(s_q, DIRECTION.get(name, 1))
        z_m = zscore(m_vals, DIRECTION.get(name, 1))

        row = [
            f'<td style="position:sticky;left:0;background:white;min-width:{IND_W}px;padding:10px;border-right:1px solid #ddd;font-weight:400; z-index:15;">&nbsp;&nbsp;{_html.escape(name)}</td>'
        ]

        # monthly cells first (if shown)
        if show_monthly:
            for md in monthly_cols:
                if md in m_vals.index:
                    val = m_vals.loc[md]
                    kind = DISPLAY_KIND.get(name, "float")
                    txt = DISPLAY_TYPE[kind](val)
                    bg = color_for_z(z_m.loc[md]) if md in z_m.index else "white"
                else:
                    txt, bg = "n/a", "white"
                row.append(
                    f'<td style="background:{bg};min-width:{COL_W}px;padding:10px;text-align:center;border-bottom:1px solid #eee;">{_html.escape(txt)}</td>'
                )

        # then quarterly cells
        for d in quarters:
            if d in s_q.index:
                val = s_q.loc[d]
                kind = DISPLAY_KIND.get(name, "float")
                txt = DISPLAY_TYPE[kind](val)
                bg = color_for_z(z_q.loc[d]) if d in z_q.index else "white"
            else:
                txt, bg = "n/a", "white"
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

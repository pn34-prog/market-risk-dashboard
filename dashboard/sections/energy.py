"""Energy: US oil and gas production, rigs, prices and drilling efficiency.

Primary data are the sponsor's saved monthly Bloomberg series (all charts in the Energy tab, per
the Process document). Analytics add headline tiles with Y/Y changes, a rigs-follow-prices lag
study, Y/Y comparisons and production per rig for gas. FRED WTI and Henry Hub spot prices give an
independent check on the sponsor's front-month futures (CL1, NG1).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.sections.common import (
    clip,
    cross_check,
    line_figure,
    load,
    lookback_start,
    page_tabs,
    refresh_controls,
    render_workbook,
    snapshot_table,
    sources_table,
    workbook_metric,
    workbook_series,
)
from dashboard.sections.extras import workbook_unavailable
from src.analytics.commodities import best_lag, change_table, lead_lag, monthly_yoy, per_unit

IDS = ["DCOILWTICO", "DHHNGSP", "USREC"]
TABLE = [("DCOILWTICO", "pct"), ("DHHNGSP", "pct")]
SERIES = {
    "oil_production": ("energy_0_0", "US oil production", "thousand bbl/d"),
    "oil_rigs": ("energy_1_0", "Oil rigs", "rigs"),
    "wti": ("energy_1_1", "WTI (CL1)", "USD/bbl"),
    "oil_per_rig": ("energy_2_0", "Oil production per rig", "thousand bbl/d per rig"),
    "marcellus": ("energy_3_0", "Marcellus", "Bcf/d"),
    "haynesville": ("energy_3_1", "Haynesville", "Bcf/d"),
    "gas_production": ("energy_3_2", "US gas marketed production", "Bcf per month"),
    "gas_rigs": ("energy_4_0", "Gas rigs", "rigs"),
    "henry_hub": ("energy_4_1", "Henry Hub (NG1)", "USD/MMBtu"),
}
NOTES = [
    (
        "All sponsor series are saved monthly Bloomberg values: US Lower-48 crude production "
        "(DOETCL48), Baker Hughes oil and gas rig counts (BAKEOIL, BAKEGAS), Marcellus and "
        "Haynesville output (SHNGMARC, SHNGHAYN), US marketed gas production (NGMPTOTL), and "
        "front-month WTI (CL1) and Henry Hub (NG1) futures."
    ),
    (
        "Oil production per rig is the sponsor's saved calculation. Gas output per rig (DERIVED) "
        "divides Marcellus + Haynesville Bcf/d by the gas rig count; those two basins are not all "
        "US gas, so it is an indicator of trend, not a national efficiency measure."
    ),
    (
        "Rigs-follow-prices study: correlation of rig-count Y/Y change with price Y/Y change "
        "shifted 0–12 months earlier. Requires 24 overlapping months. Descriptive, not causal."
    ),
    (
        "Independent cross-check: FRED WTI Cushing spot and Henry Hub spot (daily, EIA). Spot and "
        "front-month futures differ, and month-end vs. mid-month sampling adds small gaps; series "
        "are shown on separate axes and never merged."
    ),
]


def render(ctx, render_section):
    data = load(IDS)
    workbook_tab, analytics_tab, indicators_tab, method_tab = page_tabs(ctx.title)
    with workbook_tab:
        render_workbook(ctx.title, render_section)
    with analytics_tab:
        refresh_controls(ctx, IDS)
        if not workbook_unavailable(ctx.title):
            saved = _saved(ctx.title)
            _tiles(saved)
            _yoy(ctx, saved)
            _lag_study(ctx, saved)
            _gas_efficiency(ctx, saved)
        cross_check(
            ctx,
            data,
            ctx.title,
            [
                ("energy_1_1", "WTI (CL1)", "DCOILWTICO"),
                ("energy_4_1", "Henry Hub (NG1)", "DHHNGSP"),
            ],
            "energy_xcheck",
        )
    with indicators_tab:
        _workbook_table(ctx)
        snapshot_table(ctx, data, TABLE, "energy_indicators.csv")
    with method_tab:
        sources_table(ctx, IDS, NOTES)


def _saved(title):
    return {key: workbook_series(title, spec[0]) for key, spec in SERIES.items()}


def _yoy_text(series):
    change = monthly_yoy(series)
    if change.dropna().empty:
        return None
    return f"{change.dropna().iloc[-1]:+.1f}% Y/Y"


def _tiles(saved):
    st.subheader("Where US energy stands")
    rows = (
        [
            ("oil_production", "{:,.0f}", False),
            ("oil_rigs", "{:,.0f}", False),
            ("wti", "${:,.2f}", False),
            ("oil_per_rig", "{:,.1f}", False),
        ],
        [
            ("gas_production", "{:,.0f}", False),
            ("gas_rigs", "{:,.0f}", False),
            ("henry_hub", "${:,.2f}", False),
            ("haynesville", "{:,.1f}", False),
        ],
    )
    for row in rows:
        columns = st.columns(len(row))
        for column, (key, fmt, inverse) in zip(columns, row, strict=True):
            _, label, unit = SERIES[key]
            workbook_metric(column, label, saved[key], fmt, unit, _yoy_text(saved[key]), inverse)
    st.caption("Workbook values; tiles show the latest saved month and its Y/Y change.")


def _yoy(ctx, saved):
    st.subheader("Year-over-year change: production, rigs and price")
    start = lookback_start("energy_yoy", "20Y")
    left = {
        "US oil production": clip(monthly_yoy(saved["oil_production"]), start),
        "Oil rigs": clip(monthly_yoy(saved["oil_rigs"]), start),
        "WTI (CL1)": clip(monthly_yoy(saved["wti"]), start),
    }
    if not any(len(s.dropna()) for s in left.values()):
        st.info("UNAVAILABLE · Saved production, rig or price history missing.")
        return
    fig = line_figure("DERIVED · Y/Y change (%)", left, left_title="Percent", zero=True)
    ctx.chart(fig, key="energy_yoy")
    st.caption(
        "Production usually reacts to rigs and price with a lag; the study below measures it."
    )


def _lag_study(ctx, saved):
    st.subheader("Do rigs follow prices?")
    pairs = {
        "Oil rigs vs WTI": (saved["oil_rigs"], saved["wti"]),
        "Gas rigs vs Henry Hub": (saved["gas_rigs"], saved["henry_hub"]),
    }
    fig = go.Figure()
    summary = []
    for name, (rigs, price) in pairs.items():
        table = lead_lag(rigs, price)
        best = best_lag(table)
        if table["Correlation"].notna().any():
            fig.add_trace(go.Bar(x=table["Lag (months)"], y=table["Correlation"], name=name))
        summary.append(
            {
                "Pair": name,
                "Best lag (months)": int(best["Lag (months)"]) if best else np.nan,
                "Correlation at best lag": best["Correlation"] if best else np.nan,
                "Correlation at lag 0": table["Correlation"].iloc[0],
                "Overlapping months": int(best["Observations"]) if best else 0,
            }
        )
    if not fig.data:
        st.info("UNAVAILABLE · Needs at least 24 overlapping months of saved rigs and prices.")
        return
    fig.update_layout(
        title="DERIVED · Correlation of rig Y/Y with price Y/Y, price shifted earlier",
        barmode="group",
        xaxis_title="Price leads rigs by (months)",
        yaxis_title="Correlation",
    )
    ctx.chart(fig, key="energy_lag")
    ctx.table_view(pd.DataFrame(summary), hide_index=True)
    st.caption(
        "A peak at lag N means rig counts have tended to move N months after prices. "
        "Descriptive only; correlations are not forecasts."
    )


def _gas_efficiency(ctx, saved):
    st.subheader("Gas output per rig (Marcellus + Haynesville)")
    basins = (saved["marcellus"] + saved["haynesville"]).dropna()
    per_rig = per_unit(basins, saved["gas_rigs"])
    start = lookback_start("energy_gas_eff", "Max")
    fig = line_figure(
        "DERIVED · Marcellus + Haynesville Bcf/d per gas rig, with oil output per rig",
        {"Gas: Bcf/d per gas rig": clip(per_rig, start)},
        {"Oil: thousand bbl/d per oil rig (sponsor)": clip(saved["oil_per_rig"], start)},
        "Bcf/d per rig",
        "thousand bbl/d per rig",
    )
    if per_rig.empty and saved["oil_per_rig"].empty:
        st.info("UNAVAILABLE · Basin output, gas rigs or oil efficiency missing.")
        return
    ctx.chart(fig, key="energy_gas_efficiency")
    st.caption("Two basins only; see Methodology. Separate axes because units differ.")


def _workbook_table(ctx):
    st.subheader("Sponsor series: latest and changes")
    title = ctx.title
    table = change_table(
        {spec[1]: workbook_series(title, spec[0]) for spec in SERIES.values()},
        {"Change vs prior month": 1, "Change vs 12 months ago": 12},
    )
    if "Latest" not in table.columns:
        st.info("UNAVAILABLE · No validated workbook snapshot.")
        return
    table.insert(1, "Units", [spec[2] for spec in SERIES.values()])
    ctx.table_view(table, hide_index=True)
    st.download_button(
        "Export sponsor energy table (CSV)",
        table.to_csv(index=False),
        "energy_workbook.csv",
        "text/csv",
        key="export_energy_workbook",
    )
    st.caption("Percentile and z-score are measured against each series' full saved history.")

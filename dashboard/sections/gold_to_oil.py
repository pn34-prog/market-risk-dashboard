"""Gold to Oil: gold priced in barrels of Brent, versus its long-term average.

Primary data are the sponsor's saved monthly ratio (H:L) and Brent (C01). Analytics add distance
from the long-term average, historical percentiles and z-score, and gold/oil prices implied by a
return to the average, recomputed from the latest saved row. The sponsor's own implied-price block
(N23:Q29) references fixed row 463 in Excel, so the page shows both and flags any difference.
FRED Brent spot gives an independent check on the sponsor's futures series.
"""

import numpy as np
import pandas as pd
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
    workbook_cell,
    workbook_series,
)
from dashboard.sections.extras import add_reference, workbook_unavailable
from src.analytics.commodities import episodes_above, implied_prices, zscore
from src.analytics.empirical import deviation
from src.analytics.sectors import ordinal
from src.analytics.signals import percentile_rank

IDS = ["DCOILBRENTEU", "USREC"]
TABLE = [("DCOILBRENTEU", "pct")]
RATIO, AVERAGE, BRENT = "gold_to_oil_0_0", "gold_to_oil_0_1", "gold_to_oil_0_2"
SAVED_CELLS = {"implied_gold": "Q23", "gold": "Q24", "implied_oil": "Q27", "oil": "Q28"}
TOLERANCE = 0.01  # 1% relative difference before the sponsor block is flagged
NOTES = [
    (
        "Gold/Oil is the sponsor's saved monthly gold price (USD/oz) ÷ Brent front-month futures "
        "(CO1, USD/bbl). The long-term average is the sponsor's saved AVERAGE over the full "
        "history. Gold (DERIVED) = ratio × Brent, which reproduces the sponsor's gold column."
    ),
    (
        "Implied prices: gold if the ratio returned to its long-term average with Brent unchanged "
        "(average × Brent), and Brent if gold were unchanged (gold ÷ average). These are "
        "arithmetic reference points, not forecasts."
    ),
    (
        "The sponsor's implied-price block (N23:Q29) points at fixed row 463 and its oil rows "
        "point at X16:X18. The page recomputes from the latest saved row and lists both; a "
        "difference above 1% is flagged so the Excel formulas can be updated."
    ),
    (
        "Independent cross-check: FRED Brent spot (DCOILBRENTEU, EIA). Spot and front-month "
        "futures differ; series are shown on separate axes and never merged. FRED no longer "
        "redistributes LBMA gold prices, so gold has no independent series here."
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
            _summary(ctx, saved)
            _implied(ctx, saved)
            _history(ctx, saved)
        cross_check(ctx, data, ctx.title, [(BRENT, "Brent (CO1)", "DCOILBRENTEU")], "gold_xcheck")
    with indicators_tab:
        _workbook_table(ctx)
        snapshot_table(ctx, data, TABLE, "gold_to_oil_indicators.csv")
    with method_tab:
        sources_table(ctx, IDS, NOTES)


def _saved(title):
    ratio = workbook_series(title, RATIO)
    brent = workbook_series(title, BRENT)
    pair = pd.concat([ratio, brent], axis=1, keys=["r", "b"]).dropna()
    return {
        "ratio": ratio,
        "average": workbook_series(title, AVERAGE),
        "brent": brent,
        "gold": pair["r"] * pair["b"],
    }


def _latest(series):
    s = series.dropna()
    return (float(s.iloc[-1]), s.index[-1].date()) if len(s) else (np.nan, None)


def _summary(ctx, saved):
    st.subheader("Where the ratio stands")
    ratio, month = _latest(saved["ratio"])
    average, _ = _latest(saved["average"])
    if np.isnan(ratio):
        st.info("UNAVAILABLE · Saved ratio history missing.")
        return
    gap = (ratio / average - 1) * 100 if average > 0 else np.nan
    pct = percentile_rank(saved["ratio"])
    z = zscore(saved["ratio"])
    above = episodes_above(saved["ratio"], ratio)
    a, b, c, d = st.columns(4)
    a.metric("Gold / Brent", f"{ratio:.1f}x", help="Barrels of Brent one ounce of gold buys")
    a.caption(f"Workbook · {month}")
    b.metric("Long-term average", f"{average:.1f}x")
    b.caption("Sponsor's saved full-history average")
    c.metric(
        "vs long-term average",
        "—" if np.isnan(gap) else f"{gap:+.0f}%",
        help="Latest ratio ÷ long-term average − 1",
    )
    c.caption(f"{ordinal(pct)} percentile since {saved['ratio'].index[0].year}")
    d.metric("Z-score", "—" if np.isnan(z) else f"{z:+.1f}σ")
    d.caption(f"{above['months']} of {len(saved['ratio'])} months at or above today's level")
    st.caption(
        "A high ratio means gold is expensive relative to oil (or oil cheap relative to gold)."
    )


def _implied(ctx, saved):
    st.subheader("Prices implied by a return to the long-term average")
    gold, gold_date = _latest(saved["gold"])
    brent, _ = _latest(saved["brent"])
    average, _ = _latest(saved["average"])
    levels = implied_prices(gold, brent, average)
    if levels is None:
        st.info("UNAVAILABLE · Latest gold, Brent or average missing.")
        return
    left, right = st.columns(2)
    left.metric(
        "Implied gold (Brent unchanged)",
        f"${levels['implied_gold']:,.0f}",
        f"{levels['gold_change']:+.0f}% vs ${gold:,.0f}",
    )
    right.metric(
        "Implied Brent (gold unchanged)",
        f"${levels['implied_oil']:,.0f}",
        f"{levels['oil_change']:+.0f}% vs ${brent:,.0f}",
    )
    st.caption(f"Recomputed from the latest saved row ({gold_date}). Not forecasts.")
    rows, flagged = [], []
    computed = {
        "implied_gold": levels["implied_gold"],
        "gold": gold,
        "implied_oil": levels["implied_oil"],
        "oil": brent,
    }
    labels = {
        "implied_gold": "Implied gold at LT average",
        "gold": "Gold spot",
        "implied_oil": "Implied Brent at LT average",
        "oil": "Brent spot",
    }
    for key, coordinate in SAVED_CELLS.items():
        sponsor = workbook_cell(ctx.title, coordinate)
        diff = (
            (sponsor / computed[key] - 1) * 100
            if sponsor is not None and computed[key] > 0
            else np.nan
        )
        if not np.isnan(diff) and abs(diff) > TOLERANCE * 100:
            flagged.append(labels[key])
        rows.append(
            {
                "Metric": labels[key],
                "Recomputed (latest row)": computed[key],
                "Sponsor block": sponsor,
                "Sponsor cell": coordinate,
                "Difference (%)": diff,
            }
        )
    ctx.table_view(pd.DataFrame(rows), hide_index=True)
    if flagged:
        st.warning(
            "Sponsor implied-price block differs from the latest row for: "
            + ", ".join(flagged)
            + ". In Excel, N23:Q29 still references row 463 and X16:X18; update those formulas "
            "after adding each month."
        )


def _history(ctx, saved):
    st.subheader("History and distance from the average")
    start = lookback_start("gold_to_oil", "Max")
    ratio = saved["ratio"]
    if ratio.empty:
        return
    fig = line_figure(
        "Gold ÷ Brent with historical bands",
        {"Gold / Brent": clip(ratio, start), "Long-term average": clip(saved["average"], start)},
        left_title="Ratio (x)",
    )
    for q, label in ((0.1, "10th pct"), (0.9, "90th pct")):
        add_reference(fig, float(ratio.quantile(q)), label, dash="dot")
    ctx.chart(fig, key="gold_to_oil_bands")
    gap = deviation(ratio, saved["average"])
    fig = line_figure(
        "DERIVED · Ratio versus its long-term average (%)",
        {"Gold / Brent vs average": clip(gap, start)},
        left_title="Percent above (+) / below (−) average",
        zero=True,
    )
    ctx.chart(fig, key="gold_to_oil_deviation")
    fig = line_figure(
        "Underlying prices: gold (DERIVED) and Brent (CO1)",
        {"Gold (USD/oz)": clip(saved["gold"], start)},
        {"Brent (USD/bbl)": clip(saved["brent"], start)},
        "USD per troy ounce",
        "USD per barrel",
    )
    ctx.chart(fig, key="gold_to_oil_prices")


def _workbook_table(ctx):
    st.subheader("Sponsor series: latest and changes")
    saved = _saved(ctx.title)
    rows = []
    for name, key, unit in (
        ("Gold / Brent", "ratio", "x"),
        ("Gold (DERIVED)", "gold", "USD/oz"),
        ("Brent (CO1)", "brent", "USD/bbl"),
    ):
        s = saved[key].dropna()
        if s.empty:
            rows.append({"Series": name, "Status": "UNAVAILABLE"})
            continue
        rows.append(
            {
                "Series": name,
                "Units": unit,
                "Latest": float(s.iloc[-1]),
                "As of": s.index[-1].date(),
                "1M change (%)": (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) > 1 else np.nan,
                "12M change (%)": (s.iloc[-1] / s.iloc[-13] - 1) * 100 if len(s) > 12 else np.nan,
                "Percentile": percentile_rank(s),
                "Status": "AVAILABLE",
            }
        )
    table = pd.DataFrame(rows)
    ctx.table_view(table, hide_index=True)
    st.download_button(
        "Export sponsor gold/oil table (CSV)",
        table.to_csv(index=False),
        "gold_to_oil_workbook.csv",
        "text/csv",
        key="export_gold_to_oil_workbook",
    )

"""S&P 500 Fundamentals: valuation (P/E) against profitability (operating margin, ROE).

Primary data are the sponsor's saved quarterly Bloomberg BDH series for the SPX index. Analytics
add Q/Q and Y/Y changes with historical percentiles, the P/E a simple fit on margin or ROE would
suggest, and the earnings yield against the 10-year Treasury (FRED) as an equity-premium gauge.
The 2026-11 workbook no longer carries the dividends & buybacks block, so it is not shown.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.sections.common import (
    clip,
    derived_metric,
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
from src.analytics.commodities import change_table, earnings_yield, fair_multiple
from src.analytics.sectors import align_asof

IDS = ["DGS10", "USREC"]
TABLE = [("DGS10", "pp")]
SERIES = {
    "pe": ("sandp_500_fundamentals_0_0", "S&P 500 P/E", "x"),
    "margin": ("sandp_500_fundamentals_0_1", "Operating margin", "%"),
    "roe": ("sandp_500_fundamentals_1_1", "Return on equity", "%"),
}
DRIVERS = {"margin": "Operating margin", "roe": "Return on equity"}
NOTES = [
    (
        "Sponsor series are saved quarterly Bloomberg BDH values for SPX Index: BEST_PE_RATIO "
        "(P/E on consensus earnings), OPER_MARGIN and RETURN_COM_EQY. The sponsor refreshes them "
        "quarterly."
    ),
    (
        "Margin- and ROE-implied P/E: an ordinary least-squares line of P/E on the driver over the "
        "saved history, evaluated at the latest driver value. The premium is the actual P/E vs "
        "that line. A descriptive cross-check of whether profitability explains the multiple, "
        "not a valuation model; R² shows how much it explains."
    ),
    (
        "Earnings yield (DERIVED) = 100 ÷ P/E. Yield gap = earnings yield − 10-year Treasury "
        "yield (FRED DGS10, quarter-end). Mixes a forward-earnings yield with a nominal bond "
        "yield; read as a relative gauge."
    ),
    (
        "The dividends & buybacks charts were removed from the mapping because the 2026-11 "
        "workbook no longer contains W:AB. Ask the sponsor whether they should return."
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
            saved = {key: workbook_series(ctx.title, spec[0]) for key, spec in SERIES.items()}
            _tiles(saved)
            _fair_pe(ctx, saved)
            _yield_gap(ctx, saved, data)
    with indicators_tab:
        _workbook_table(ctx)
        snapshot_table(ctx, data, TABLE, "sp500_fundamentals_indicators.csv")
    with method_tab:
        sources_table(ctx, IDS, NOTES)


def _change_text(series, back, suffix):
    s = series.dropna()
    if len(s) <= back:
        return None
    sep = "" if suffix.startswith("x") else " "
    return f"{s.iloc[-1] - s.iloc[-1 - back]:+.1f}{sep}{suffix}"


def _tiles(saved):
    st.subheader("Valuation and profitability")
    ey = earnings_yield(saved["pe"])
    columns = st.columns(4)
    workbook_metric(
        columns[0],
        "S&P 500 P/E",
        saved["pe"],
        "{:.1f}x",
        "x",
        _change_text(saved["pe"], 4, "x Y/Y"),
    )
    derived_metric(
        columns[1], "Earnings yield", ey, "{:.2f}%", "% (100 ÷ P/E)", _change_text(ey, 4, "pp Y/Y")
    )
    workbook_metric(
        columns[2],
        "Operating margin",
        saved["margin"],
        "{:.1f}%",
        "%",
        _change_text(saved["margin"], 4, "pp Y/Y"),
    )
    workbook_metric(
        columns[3],
        "Return on equity",
        saved["roe"],
        "{:.1f}%",
        "%",
        _change_text(saved["roe"], 4, "pp Y/Y"),
    )
    st.caption("Changes are versus four quarters earlier. Earnings yield is calculated from P/E.")


def _fair_pe(ctx, saved):
    st.subheader("Is the P/E explained by profitability?")
    rows, fits = [], {}
    for key, label in DRIVERS.items():
        fit = fair_multiple(saved["pe"], saved[key])
        if fit is None:
            rows.append({"Driver": label, "Status": "UNAVAILABLE · fewer than 8 quarters"})
            continue
        fits[key] = fit
        rows.append(
            {
                "Driver": label,
                "Latest driver (%)": fit["driver"],
                "Actual P/E": fit["actual"],
                "P/E implied by fit": fit["implied"],
                "Premium vs fit (%)": fit["premium"],
                "Slope (P/E per pp)": fit["slope"],
                "R²": fit["r2"],
                "Quarter": fit["date"],
                "Status": "AVAILABLE",
            }
        )
    if not fits:
        st.info("UNAVAILABLE · Saved P/E, margin or ROE history missing.")
        return
    columns = st.columns(len(fits))
    for column, (key, fit) in zip(columns, fits.items(), strict=True):
        column.metric(
            f"P/E implied by {DRIVERS[key].lower()}",
            f"{fit['implied']:.1f}x",
            f"actual {fit['premium']:+.0f}% vs fit",
            delta_color="inverse",
        )
        column.caption(f"R² {fit['r2']:.2f} · {fit['date']}")
    ctx.table_view(pd.DataFrame(rows), hide_index=True)
    left, right = st.columns(2)
    for column, (key, fit) in zip((left, right), fits.items(), strict=False):
        pairs = align_asof(saved["pe"], saved[key], 50)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pairs["right"],
                y=pairs["left"],
                mode="markers",
                name="Quarters",
                text=[d.date().isoformat() for d in pairs.index],
                hovertemplate="%{text}<br>%{x:.1f}% → %{y:.1f}x<extra></extra>",
            )
        )
        xs = np.linspace(pairs["right"].min(), pairs["right"].max(), 20)
        fig.add_trace(
            go.Scatter(x=xs, y=fit["slope"] * xs + fit["intercept"], mode="lines", name="Fit")
        )
        fig.add_trace(
            go.Scatter(
                x=[fit["driver"]],
                y=[fit["actual"]],
                mode="markers",
                marker={"size": 14, "symbol": "star"},
                name="Latest",
            )
        )
        fig.update_layout(
            title=f"P/E vs {DRIVERS[key].lower()}",
            xaxis_title=f"{DRIVERS[key]} (%)",
            yaxis_title="P/E (x)",
        )
        with column:
            ctx.chart(fig, key=f"fundamentals_scatter_{key}")
    start = lookback_start("fundamentals_premium", "Max")
    lines = {
        f"vs {DRIVERS[key].lower()} fit": clip(fit["residuals"], start) for key, fit in fits.items()
    }
    fig = line_figure(
        "DERIVED · P/E above (+) / below (−) the profitability fit (multiple points)",
        lines,
        left_title="P/E points",
        zero=True,
    )
    ctx.chart(fig, key="fundamentals_premium")
    st.caption("Positive values: the market pays more than profitability alone has implied.")


def _yield_gap(ctx, saved, data):
    st.subheader("Earnings yield vs 10-year Treasury")
    ey = earnings_yield(saved["pe"])
    ten = data.get("DGS10", pd.Series(dtype=float))
    pairs = align_asof(ey, ten, 7)
    if ey.empty:
        st.info("UNAVAILABLE · Saved P/E missing.")
        return
    gap = (pairs["left"] - pairs["right"]) if len(pairs) else pd.Series(dtype=float)
    start = lookback_start("fundamentals_gap", "Max")
    fig = line_figure(
        "DERIVED · Earnings yield, 10Y Treasury and the gap (%)",
        {
            "Earnings yield (100 ÷ P/E)": clip(ey, start),
            "10Y Treasury (FRED, quarter-end)": clip(pairs["right"], start)
            if len(pairs)
            else pd.Series(dtype=float),
            "Gap": clip(gap, start),
        },
        left_title="Percent",
        zero=True,
    )
    ctx.chart(fig, key="fundamentals_yield_gap")
    if gap.empty:
        st.caption(
            "10Y Treasury not ingested; run `uv run python -m scripts.refresh_market` for the gap."
        )
    else:
        st.caption(
            f"Latest gap {gap.iloc[-1]:+.2f} pp ({gap.index[-1].date()}); "
            f"{(gap <= gap.iloc[-1]).mean() * 100:.0f}th percentile of the overlapping history."
        )


def _workbook_table(ctx):
    st.subheader("Sponsor series: latest and changes")
    table = change_table(
        {spec[1]: workbook_series(ctx.title, spec[0]) for spec in SERIES.values()},
        {"Change vs prior quarter": 1, "Change vs 4 quarters ago": 4},
    )
    if "Latest" not in table.columns:
        st.info("UNAVAILABLE · No validated workbook snapshot.")
        return
    table.insert(1, "Units", [spec[2] for spec in SERIES.values()])
    ctx.table_view(table, hide_index=True)
    st.download_button(
        "Export sponsor fundamentals table (CSV)",
        table.to_csv(index=False),
        "sp500_fundamentals_workbook.csv",
        "text/csv",
        key="export_fundamentals_workbook",
    )

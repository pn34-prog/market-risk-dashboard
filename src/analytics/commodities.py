"""Pure calculations for the Energy, Gold to Oil and S&P 500 Fundamentals pages.

Every function takes saved series and returns derived values; nothing is interpolated or filled.
Monthly workbook rows can sit anywhere in the month, so alignment uses the 35-day tolerance from
``empirical``.
"""

import numpy as np
import pandas as pd

from src.analytics.empirical import MONTH_GAP, clean
from src.analytics.sectors import align_asof, yoy_percent

QUARTER_GAP = 50


def per_unit(numerator, denominator, max_gap_days=MONTH_GAP):
    """numerator ÷ denominator on aligned dates (e.g., production per rig)."""
    pairs = align_asof(clean(numerator), clean(denominator), max_gap_days)
    if pairs.empty:
        return pd.Series(dtype=float)
    out = pairs["left"] / pairs["right"].where(pairs["right"] > 0)
    return out.dropna()


def monthly_yoy(series):
    """Y/Y percent change for a monthly series (bounded 35-day lookup)."""
    s = clean(series)
    return yoy_percent(s, MONTH_GAP) if len(s) else pd.Series(dtype=float)


def lead_lag(follower, leader, max_lag=12, months=12):
    """Correlation of Y/Y changes in ``follower`` with ``leader`` shifted 0..max_lag months.

    Used for "rigs follow prices": returns a frame of lag (months) and correlation over the
    overlapping observations. Descriptive only; no causality is implied.
    """
    f = monthly_yoy(follower) if months else clean(follower)
    lead = monthly_yoy(leader) if months else clean(leader)
    f = f.resample("ME").last().dropna()
    lead = lead.resample("ME").last().dropna()
    rows = []
    for lag in range(max_lag + 1):
        shifted = lead.copy()
        shifted.index = shifted.index + pd.offsets.MonthEnd(lag)
        pair = pd.concat([f, shifted], axis=1, keys=["f", "l"]).dropna()
        corr = float(pair["f"].corr(pair["l"])) if len(pair) >= 24 else np.nan
        rows.append({"Lag (months)": lag, "Correlation": corr, "Observations": len(pair)})
    return pd.DataFrame(rows)


def best_lag(table):
    """Lag with the highest correlation, or None when nothing could be evaluated."""
    valid = table.dropna(subset=["Correlation"])
    if valid.empty:
        return None
    return valid.loc[valid["Correlation"].idxmax()].to_dict()


def implied_prices(gold, oil, average):
    """Gold and oil implied by the gold/oil ratio returning to ``average``, holding the other fixed.

    Implied gold = average × oil; implied oil = gold ÷ average. Changes are in percent.
    """
    if not all(np.isfinite(v) and v > 0 for v in (gold, oil, average)):
        return None
    implied_gold = average * oil
    implied_oil = gold / average
    return {
        "implied_gold": implied_gold,
        "gold_change": (implied_gold / gold - 1) * 100,
        "implied_oil": implied_oil,
        "oil_change": (implied_oil / oil - 1) * 100,
    }


def zscore(series):
    """Latest value in standard deviations from the full-history mean (sample SD)."""
    s = clean(series)
    if len(s) < 3:
        return np.nan
    sd = s.std(ddof=1)
    return float((s.iloc[-1] - s.mean()) / sd) if sd > 0 else np.nan


def episodes_above(series, threshold):
    """Months the series spent at/above ``threshold`` and the most recent such date."""
    s = clean(series)
    hit = s[s >= threshold]
    return {"months": len(hit), "last": hit.index[-1].date() if len(hit) else None}


def earnings_yield(pe):
    """Earnings yield (%) = 100 ÷ P/E for positive multiples."""
    s = clean(pe)
    return (100 / s.where(s > 0)).dropna()


def fair_multiple(pe, driver, max_gap_days=QUARTER_GAP):
    """Linear fit of P/E on a profitability driver (operating margin or ROE).

    Returns slope, intercept, R², the P/E implied by the latest driver value, and the latest
    actual P/E's premium (%) over it. A descriptive cross-check, not a valuation model.
    """
    pairs = align_asof(clean(pe), clean(driver), max_gap_days)
    if len(pairs) < 8:
        return None
    x, y = pairs["right"].to_numpy(), pairs["left"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    implied = slope * x[-1] + intercept
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "driver": float(x[-1]),
        "actual": float(y[-1]),
        "implied": float(implied),
        "premium": (y[-1] / implied - 1) * 100 if implied > 0 else np.nan,
        "date": pairs.index[-1].date() if isinstance(pairs.index, pd.DatetimeIndex) else None,
        "residuals": pd.Series(y - fitted, index=pairs.index),
    }


def change_table(series_map, periods):
    """Latest value and change over ``periods`` (label → observations back) for each series."""
    rows = []
    for name, series in series_map.items():
        s = clean(series)
        if s.empty:
            rows.append({"Series": name, "Status": "UNAVAILABLE"})
            continue
        row = {"Series": name, "Latest": float(s.iloc[-1]), "As of": s.index[-1].date()}
        for label, back in periods.items():
            row[label] = float(s.iloc[-1] - s.iloc[-1 - back]) if len(s) > back else np.nan
        row["Percentile"] = float((s <= s.iloc[-1]).mean() * 100)
        row["Z-score"] = zscore(s)
        row["Status"] = "AVAILABLE"
        rows.append(row)
    return pd.DataFrame(rows)

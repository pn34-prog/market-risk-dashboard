import numpy as np
import pandas as pd
import pytest

from src.analytics.commodities import (
    best_lag,
    change_table,
    earnings_yield,
    episodes_above,
    fair_multiple,
    implied_prices,
    lead_lag,
    monthly_yoy,
    per_unit,
    zscore,
)


def monthly(values, start="2005-01-31"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="ME"), dtype=float
    )


def quarterly(values, start="2010-03-31"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="QE"), dtype=float
    )


def test_per_unit_aligns_and_skips_zero():
    production = monthly([100, 120, 130])
    rigs = monthly([10, 0, 13])
    out = per_unit(production, rigs)
    assert list(out.round(6)) == [10.0, 10.0]  # zero-rig month dropped, not divided


def test_monthly_yoy_bounded():
    s = monthly(np.arange(1, 26))
    yoy = monthly_yoy(s)
    assert yoy.iloc[-1] == pytest.approx((25 / 13 - 1) * 100)
    assert yoy.iloc[:12].isna().all()


def test_lead_lag_recovers_known_lag():
    t = np.arange(200)
    price = monthly(60 + 20 * np.sin(t / 10))
    rigs = monthly(500 + 150 * np.sin((t - 4) / 10))  # rigs trail price by 4 months
    best = best_lag(lead_lag(rigs, price))
    assert best["Lag (months)"] == 4
    assert best["Correlation"] > 0.95


def test_lead_lag_needs_overlap():
    table = lead_lag(monthly([1, 2, 3]), monthly([1, 2, 3]))
    assert table["Correlation"].isna().all()
    assert best_lag(table) is None


def test_implied_prices_mirror_ratio():
    out = implied_prices(gold=4000, oil=100, average=20)
    assert out["implied_gold"] == 2000 and out["gold_change"] == pytest.approx(-50)
    assert out["implied_oil"] == 200 and out["oil_change"] == pytest.approx(100)
    assert implied_prices(4000, 0, 20) is None
    assert implied_prices(np.nan, 100, 20) is None


def test_zscore_and_episodes():
    s = monthly([1, 2, 3, 4, 10])
    assert zscore(s) == pytest.approx((10 - s.mean()) / s.std(ddof=1))
    assert np.isnan(zscore(monthly([1, 2])))
    assert episodes_above(s, 4)["months"] == 2


def test_earnings_yield_positive_only():
    out = earnings_yield(quarterly([20, -5, 25]))
    assert list(out) == [5.0, 4.0]


def test_fair_multiple_exact_line():
    margin = quarterly(np.linspace(8, 14, 12))
    pe = 2 * margin + 1
    fit = fair_multiple(pe, margin)
    assert fit["slope"] == pytest.approx(2)
    assert fit["intercept"] == pytest.approx(1)
    assert fit["r2"] == pytest.approx(1)
    assert fit["premium"] == pytest.approx(0, abs=1e-9)
    assert fair_multiple(pe.iloc[:5], margin.iloc[:5]) is None


def test_change_table():
    table = change_table({"A": quarterly([1, 2, 3, 4, 6]), "B": pd.Series(dtype=float)}, {"Y/Y": 4})
    a = table.set_index("Series").loc["A"]
    assert a["Latest"] == 6 and a["Y/Y"] == 5 and a["Percentile"] == 100
    assert table.set_index("Series").loc["B", "Status"] == "UNAVAILABLE"

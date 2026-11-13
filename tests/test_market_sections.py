"""Page renders for S&P 500 Fundamentals, Gold to Oil and Energy.

Every value below is synthetic and lives only in an isolated temporary database.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """from unittest.mock import patch
import runpy
with patch('streamlit.navigation'), patch('src.refresh.enqueue'):
    namespace = runpy.run_path({path!r})
with patch.dict(namespace, enqueue=lambda ids: None):
    namespace['render']({title!r})
"""
TITLES = ["S&P 500 Fundamentals", "Gold to Oil", "Energy"]
MONTHS = pd.date_range("2003-09-30", "2026-09-30", freq="ME")
QUARTERS = pd.date_range("2000-03-31", "2026-09-30", freq="QE")


@pytest.fixture
def database(monkeypatch, tmp_path):
    from src.database import store

    engine = create_engine("sqlite:///" + str(tmp_path / "fixtures.db"))
    monkeypatch.setattr(store, "engine", engine)
    store.initialize()
    return store


def synthetic_values(series_id, n):
    t = np.arange(n)
    price = 60 + 20 * np.sin(t / 10)
    values = {
        # Fundamentals (quarterly): P/E loosely tied to margin.
        "sandp_500_fundamentals_0_1": 10 + 2 * np.sin(t / 8),
        "sandp_500_fundamentals_1_1": 14 + 3 * np.sin(t / 8),
        # Gold to Oil (monthly).
        "gold_to_oil_0_0": 20 + 8 * np.sin(t / 25),
        "gold_to_oil_0_1": np.full(n, 20.0),
        "gold_to_oil_0_2": price,
        # Energy (monthly): rigs trail price by four months.
        "energy_0_0": 5000 + 30 * t,
        "energy_1_0": 500 + 150 * np.sin((t - 4) / 10),
        "energy_1_1": price,
        "energy_2_0": (5000 + 30 * t) / (500 + 150 * np.sin((t - 4) / 10)),
        "energy_3_0": 10 + 0.06 * t,
        "energy_3_1": 6 + 0.03 * t,
        "energy_3_2": 2500 + 5 * t,
        "energy_4_0": 150 + 40 * np.sin((t - 3) / 10),
        "energy_4_1": 3 + np.sin(t / 10),
    }
    if series_id in ("sandp_500_fundamentals_0_0", "sandp_500_fundamentals_1_0"):
        return 2 * values["sandp_500_fundamentals_0_1"] - 2 + 0.5 * np.cos(t)
    return values[series_id]


def synthetic_snapshot():
    from src.providers.excel import load_mapping

    mapping = load_mapping()["sections"]
    sections = {}
    for title in TITLES:
        config = mapping[title]
        dates = QUARTERS if title == "S&P 500 Fundamentals" else MONTHS
        charts = []
        for chart in config["charts"]:
            out = {"id": chart["id"], "title": chart["title"], "series": []}
            for spec in chart["series"]:
                values = synthetic_values(spec["id"], len(dates))
                points = [
                    {
                        "date": d.date().isoformat(),
                        "value": float(v),
                        "cell": f"Z{i}",
                        "forward": False,
                        "formula": None,
                    }
                    for i, (d, v) in enumerate(zip(dates, values, strict=True))
                ]
                out["series"].append({**spec, "points": points, "latest_reported": None})
            charts.append(out)
        tables = []
        for table in config["tables"]:
            # Sponsor implied-price block deliberately stale (as in the 2026-11 workbook).
            saved = {"Q23": 1.0, "Q24": 1.0, "Q27": 1.0, "Q28": 1.0}
            rows = [
                [
                    {
                        "cell": c,
                        "value": v,
                        "format": "General",
                        "formula": False,
                        "font_color": None,
                    }
                ]
                for c, v in saved.items()
            ]
            tables.append({**table, "rows": rows, "as_of": "2026-09-30"})
        sections[title] = {"notes": config["notes"], "charts": charts, "tables": tables}
    return {
        "sections": sections,
        "metric_count": 1,
        "latest_observation": "2026-09-30",
        "diagnostics": {},
        "issues": [],
    }


def seed_snapshot(store):
    with Session(store.engine) as session, session.begin():
        session.add(
            store.WorkbookImport(
                imported_at=datetime.now(UTC),
                path="synthetic.xlsx",
                sha256="0" * 64,
                status="success",
                payload=json.dumps(synthetic_snapshot()),
            )
        )


def run(title):
    return AppTest.from_string(SCRIPT.format(path=str(ROOT / "dashboard/app.py"), title=title)).run(
        timeout=120
    )


def frames(app):
    return [d.value.data if hasattr(d.value, "data") else d.value for d in app.dataframe]


@pytest.mark.parametrize(
    "title,charts", [("S&P 500 Fundamentals", 5), ("Gold to Oil", 3), ("Energy", 3)]
)
def test_page_with_snapshot(title, charts, database):
    seed_snapshot(database)
    app = run(title)
    assert not app.exception
    assert app.title[0].value == title
    assert len(app.tabs) == 0  # one-page layout: sections are stacked, not tabbed
    headers = [h.value for h in app.header]
    assert headers == [
        "Sponsor workbook",
        "Independent analytics",
        "Indicators",
        "Methodology & sources",
    ]
    assert len(app.get("plotly_chart")) >= charts


@pytest.mark.parametrize("title", TITLES)
def test_pages_without_snapshot(title, database):
    app = run(title)
    assert not app.exception
    assert any("UNAVAILABLE" in w.value for w in app.warning)
    # Without a snapshot the independent analytics lead.
    assert app.header[0].value == "Independent analytics"


def test_energy_lag_study_finds_rig_lag(database):
    seed_snapshot(database)
    app = run("Energy")
    summary = next(f for f in frames(app) if "Best lag (months)" in f.columns)
    oil = summary.set_index("Pair").loc["Oil rigs vs WTI"]
    assert oil["Best lag (months)"] == 4


def test_gold_to_oil_flags_stale_sponsor_block(database):
    seed_snapshot(database)
    app = run("Gold to Oil")
    assert any("Sponsor implied-price block differs" in w.value for w in app.warning)
    labels = {m.label for m in app.metric}
    assert {"Gold / Brent", "Implied gold (Brent unchanged)"} <= labels


def test_fundamentals_fair_pe(database):
    seed_snapshot(database)
    app = run("S&P 500 Fundamentals")
    table = next(f for f in frames(app) if "P/E implied by fit" in f.columns)
    assert table["R²"].min() > 0.5
    assert not any("Dividends" in str(c.value) for c in app.get("plotly_chart"))

"""Tests for the Streamlit app (app/) and the shared chart builders (insureport.charts): every builder
returns a titled plotly Figure with the expected traces on tiny inputs (axis titles with units, hover
text on every trace, the frontier's stacked max-DD panel instead of a second scale and its T-bill
overlay clipped to the sleeve's weights, the shaded rejection region and regime bins, the back-test
hatching); edited candidates round-trip through dataclasses.astuple (the cache keys); uploads over
1 MB or not UTF-8 are refused; the Portfolio page builds a preset (weights or shares, valued at the
last prices) into session state, lists unknown tickers instead of building, keeps extra cash in its
unit across presets and proxy-fills a late ticker when asked; pages without a portfolio stop with a
pointer to the Portfolio page; Diagnostics shows its metric cards and draws the annual bars and both
means from full calendar years; Candidates evaluates a selection, with edited parameters reaching the
engine, into a ranked table with a verdict column and the return on premium of a premium sleeve, drops
it once the portfolio changes, hides per-candidate rebalance selectors the sidebar overrides and lists
only csv / zip chains; every Stress lab tab renders (the book bootstrap's three modes, roll offsets that
differ, rolling windows that fill on a long sample), the estimated dice reads the regime table off the
backtest and keeps its editor across reruns, and the on-demand sensitivity sweep draws its tornados
(16 variants, the base among them), kept only for the sample it ran on; Methodology renders
METHODOLOGY.md or a placeholder, the validation table, the coverage table's back-fill flags, and a
warning when validation cannot run; the entry point's sidebar holds the settings and the validation
banner on its assumptions, and an unloadable market panel stops the app with an error, not a traceback.

Why: the app is the product's face; these smoke tests catch a page that raises, a missing title or
a builder that silently drops a trace. Pages run under streamlit.testing.v1.AppTest with
common.market, prices_cached, pput3m, haven_px and validation_cached monkeypatched to the 2018-2019
fixture panel and synthetic frames; no network.
"""

import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from insureport import catalog, charts, metrics, portfolio, pricing, scenarios
from insureport.catalog import CANDIDATES
from insureport.portfolio import Portfolio

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))  # `streamlit run app/streamlit_app.py` puts app/ on sys.path; the pages import common
import common  # noqa: E402  (the module object the pages import, patched below)

FIXTURE = Path(__file__).parent / "fixtures" / "market_2018_2019.csv"
RAW = pd.read_csv(FIXTURE, index_col="date", parse_dates=True)
MKT = pricing.fill_term_structure(RAW)
MKT.attrs["first_dates"] = {c: RAW[c].first_valid_index() for c in RAW}
TIMEOUT = 120  # seconds per AppTest run (the default is 3)


def fake_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Synthetic adjusted closes on the fixture's dates: the S&P 500 TR path with a per-ticker drift; tickers
    starting with ZZ are unknown (no column), as data.prices drops them."""
    base = MKT["spxtr"] / MKT["spxtr"].iloc[0] * 100
    return pd.DataFrame({t: base * (1 + 0.0001 * i) ** np.arange(len(base))
                         for i, t in enumerate(tickers) if not t.startswith("ZZ")}, index=MKT.index)


HAVEN_PX = pd.concat([fake_prices(common.HAVEN_ETFS), pd.Series(2.5, MKT.index, name="DGS10"),
                      MKT["pput"].rename("pput3m")], axis=1)  # haven_px on the fixture: ETFs, a flat 2.5% DGS10, PPUT3M
VALIDATION = pd.DataFrame(
    {"window_start": pd.Timestamp("2016-01-04"), "window_end": pd.Timestamp("2019-12-31"), "te": [0.012, 0.009, 0.036],
     "dcagr": [-0.017, -0.013, -0.007], "corr": [0.98, 0.98, 0.95], "dmaxdd": 0.0, "n_months": 47,
     "te_roll": [0.05, 0.03, 0.04], "te_nonroll": [0.01, 0.008, 0.035], "passed": [False, True, False]},
    index=["PPUT", "PPUT3M", "CLL"])


@pytest.fixture
def patched(monkeypatch):
    """The app's loaders on the fixture: market panel, synthetic prices, PPUT3M and haven sources (no FRED fetch), a
    canned validation table; the unpatched caches emptied before and after, so no test reads another's results."""
    monkeypatch.setattr(common, "market", lambda: MKT)
    monkeypatch.setattr(common, "prices_cached", fake_prices)
    monkeypatch.setattr(common, "haven_px", lambda: HAVEN_PX)
    monkeypatch.setattr(common, "pput3m", lambda: MKT["pput"].rename("pput3m"))
    monkeypatch.setattr(common, "validation_cached", lambda m, nu, h: VALIDATION)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / "pages" / f"{name}.py"), default_timeout=TIMEOUT)


def _core(spy: float = 0.9) -> tuple[Portfolio, pd.Series]:
    """A SPY / cash portfolio on the fixture and its core returns."""
    p = Portfolio({"SPY": spy, "CASH": 1 - spy}, 1_000_000.0)
    return p, portfolio.core_returns(p, fake_prices(("SPY",)), MKT["rf"])[0]


def _seeded(name: str) -> AppTest:
    """A page with a 90% SPY / 10% cash portfolio already built into session state."""
    p, core = _core()
    at = _page(name)
    at.session_state["portfolio"], at.session_state["core_r"], at.session_state["proxied"] = p, core, []
    return at


def _spec(chart: Any) -> dict:
    """The plotly figure dict behind an AppTest plotly_chart element."""
    return json.loads(chart.proto.spec)


def _y(trace: dict) -> np.ndarray:
    """A trace's y values, decoding plotly's typed-array JSON."""
    y = trace["y"]
    return np.frombuffer(base64.b64decode(y["bdata"]), y["dtype"]) if isinstance(y, dict) else np.asarray(y, float)


# ---------------------------------------------------------------- charts.py

IDX = pd.bdate_range("2019-01-01", periods=10)
NAV = pd.Series([1, 1.1, 1.05, 0.9, 0.95, 1.0, 1.2, 1.15, 1.25, 1.3], IDX)
TABLE = pd.DataFrame({"cagr": [0.08, 0.085, 0.07], "max_dd": [-0.35, -0.29, -0.25]},
                     index=pd.Index([0.0, 0.05, 0.1], name="w"))
CASH = pd.DataFrame({"cagr": [0.08, 0.07, 0.05, 0.02], "max_dd": [-0.35, -0.3, -0.18, 0.0]},
                    index=pd.Index([0.0, 0.08, 0.5, 1.0], name="w"))  # a T-bill frontier on a wider grid


def _boot() -> pd.DataFrame:
    """A small block bootstrap: 10 years of monthly returns, w in {0, 0.1, 0.2}, horizons 5 and 10."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2000-01-31", periods=120, freq="ME")
    return scenarios.bootstrap(pd.Series(rng.normal(0.007, 0.04, 120), months),
                               pd.Series(rng.normal(0.002, 0.01, 120), months), np.array([0.0, 0.1, 0.2]), 12,
                               horizons=(5, 10), n_paths=50)


def _figures() -> dict[str, tuple[go.Figure, int]]:
    """Every builder on tiny inputs, with the trace count it must draw."""
    years = pd.Series([0.2, -0.1, 0.05, 0.3], index=[2016, 2017, 2018, 2019])
    crashes = pd.DataFrame({"start": IDX[:2], "end": IDX[2:4], "ret": [-0.1, -0.2], "bench_ret": [-0.15, -0.3]},
                           index=["A", "B"])
    rows = pd.DataFrame({"candidate": ["tbills", "universa_model"], "arith_cost_bp": [-40.0, -10.0],
                         "d_cagr_bp": [-20.0, 15.0], "verdict": ["not cost-effective", "cost-effective (robust)"]})
    boot = _boot()
    w = np.linspace(0, 0.2, 5)
    p, c, h = np.array([0.1, 0.9]), np.array([-0.3, 0.1]), np.array([9.0, -1.0])
    P = scenarios.book_portfolio_returns(c, h, w)
    sweep = scenarios.crash_frequency_sweep(p, c, h, np.array([0.0, 0.05]))
    rolling = pd.DataFrame({0.0: 0.0, 0.05: np.linspace(-0.01, 0.01, 10)}, index=IDX)
    sens = pd.DataFrame({"low": [-30.0, -5.0], "high": [20.0, 4.0]}, index=["m", "h"])
    synth = pd.Series(np.linspace(100, 120, 10), IDX)
    return {
        "growth": (charts.growth_chart({"Portfolio": NAV, "S&P 500 TR": NAV * 1.01}), 2),
        "underwater": (charts.underwater_chart(NAV), 1),
        "annual": (charts.annual_bars(years, 0.1125, 0.1), 3),
        "crash": (charts.crash_table_fig(crashes), 2),
        "frontier": (charts.frontier_chart(TABLE, 0.30), 3),
        "frontier+cash": (charts.frontier_chart(TABLE, 0.30, cash_table=CASH, w_star=0.05), 6),
        "ce": (charts.ce_plane(rows), 1),
        "payoff": (charts.payoff_scatter(years * 2, years, "Gold"), 1),
        "fan": (charts.fan_chart(boot, 10), 5),
        "dice": (charts.dice_chart(w, scenarios.dice_growth(p, P), scenarios.dice_arith(p, P), 0.05), 3),
        "crash_freq": (charts.crash_freq_chart(sweep), 2),
        "rolling": (charts.rolling_chart(rolling, 0.05, years=10), 1),
        "tornado": (charts.tornado(sens), 2),
        "replication": (charts.replication_chart(synth, synth * 1.01, "2019-01-08", "PPUT3M"), 3),
        "regime": (charts.regime_table_fig(scenarios.regime_table(scenarios.load_sp500_annual())), 1),
    }


def test_chart_builders_return_titled_figures_with_their_traces():
    for name, (fig, n) in _figures().items():
        assert isinstance(fig, go.Figure), name
        assert len(fig.data) == n, name
        assert fig.layout.title.text, name
        assert any(ax.title.text for ax in fig.select_xaxes()), name  # stacked panels title the bottom axis
        assert all(ax.title.text for ax in fig.select_yaxes()), name
        for tr in fig.data:
            assert tr.hovertemplate or tr.hoverinfo == "skip", (name, tr.name)
        colours = set()
        for tr in fig.data:
            for c in (getattr(tr.marker, "color", None), getattr(getattr(tr, "line", None), "color", None)):
                points = c if isinstance(c, (list, tuple)) else ()  # per-point colours, e.g. the annual bars'
                colours |= {c} if isinstance(c, str) else {x for x in points if isinstance(x, str)}
        assert len(colours) <= 10, name


def test_chart_units_and_shapes():
    f = _figures()
    assert f["growth"][0].layout.yaxis.type == "log"
    assert charts.growth_chart({"a": NAV}, log=False).layout.yaxis.type != "log"
    assert "%" in f["underwater"][0].layout.yaxis.title.text
    assert (np.asarray(f["underwater"][0].data[0].y) <= 0).all()
    assert [t.name for t in f["annual"][0].data[1:]] == ["arithmetic mean", "geometric (CAGR)"]
    assert "bp" in f["ce"][0].layout.xaxis.title.text and "bp" in f["ce"][0].layout.yaxis.title.text
    assert list(f["ce"][0].data[0].text) == ["tbills", "universa_model"]
    assert any(s.y1 == 0 for s in f["ce"][0].layout.shapes)  # the rejection region y <= 0
    assert sum(s.type == "rect" for s in f["payoff"][0].layout.shapes) == 4  # the four regime bins
    # max drawdown sits in its own panel under the CAGR panel, not on a second scale over it
    fr = f["frontier+cash"][0]
    assert fr.layout.yaxis2.title.text and not fr.layout.yaxis.overlaying and not fr.layout.yaxis2.overlaying
    assert {t.yaxis for t in fr.data} == {"y", "y2"}
    # the 0-100% T-bill overlay and the cap line stop at the sleeve's last weight, the overlay interpolated there
    assert max(max(t.x) for t in fr.data) == 0.1
    cash = [t for t in fr.data if t.name.startswith("T-bills")]
    np.testing.assert_allclose(cash[0].x, [0.0, 0.08, 0.1])
    np.testing.assert_allclose([c.y[-1] for c in cash], [0.07 + 0.02 / 0.42 * -0.02, -0.3 + 0.02 / 0.42 * 0.12])
    # the PPUT3M fixture window is all before the 2021 launch: hatched as back-tested
    hatch = f["replication"][0].data[2]
    assert hatch.fillpattern.shape and "back-tested" in hatch.name
    assert len(charts.replication_chart(NAV, NAV, "2019-01-08", "PPUT").data) == 2  # launched 2015: no hatch
    fan = f["fan"][0].data[-1]
    assert "P(beat w = 0)" in fan.hovertemplate
    assert list(fan.customdata) == _boot().xs(10, level="horizon")["p_beat"].tolist()
    # the tornado's bars sit side by side (same-sign changes both show); xfmt sets the unit
    assert f["tornado"][0].layout.barmode == "group"
    w_tornado = charts.tornado(pd.DataFrame({"low": [0.0], "high": [0.01]}, index=["m"]), "w*", xfmt=".2%")
    assert w_tornado.layout.xaxis.tickformat == ".2%" and "%{x:+.2%}" in w_tornado.data[0].hovertemplate


# ---------------------------------------------------------------- app, part (a)


def test_common_candidate_round_trips_through_astuple():
    from dataclasses import astuple

    for c in CANDIDATES.values():
        assert common.candidate_from_tuple(astuple(c)) == c


def test_portfolio_page_builds_the_preset(patched):
    at = _page("portfolio").run()
    assert not at.exception
    assert at.title[0].value == "Portfolio"
    assert at.selectbox(key="preset").options[:1] == ["sp500.csv"]
    at.button(key="build").click().run()
    assert not at.exception
    p = at.session_state["portfolio"]
    assert p.weights == {"SPY": 1.0} and p.value == 1_000_000
    core = at.session_state["core_r"]
    assert core.index[0] == MKT.index[0] and core.index[-1] == MKT.index[-1]
    assert at.session_state["proxied"] == []


def test_portfolio_page_shares_preset_and_unknown_tickers(patched, monkeypatch, tmp_path):
    at = _page("portfolio").run()
    at.selectbox(key="preset").set_value("tech_concentrated.csv").run()
    at.button(key="build").click().run()
    assert not at.exception
    p = at.session_state["portfolio"]
    shares = {"AAPL": 100, "MSFT": 60, "NVDA": 150, "AMZN": 100, "GOOGL": 100}  # samples/tech_concentrated.csv
    last = fake_prices(tuple(sorted(shares))).iloc[-1]  # each ticker drifts differently: weights are not equal
    value = sum(n * last[t] for t, n in shares.items())
    assert p.value == pytest.approx(value) and set(p.weights) == set(shares)
    assert p.weights["NVDA"] == pytest.approx(150 * last["NVDA"] / value)
    # an unknown ticker is listed and nothing is built
    (tmp_path / "bad.csv").write_text("ticker,weight\nSPY,0.5\nZZZZ,0.5\n")
    monkeypatch.setattr(common, "SAMPLES", tmp_path)
    at = _page("portfolio").run()
    at.button(key="build").click().run()
    assert not at.exception
    assert "portfolio" not in at.session_state
    assert any("ZZZZ" in e.value for e in at.error)


def test_portfolio_page_extra_cash_keeps_its_unit(patched):
    at = _page("portfolio").run()
    at.selectbox(key="preset").set_value("tech_concentrated.csv").run()
    at.number_input(key="cash_usd").set_value(50_000.0).run()
    at.selectbox(key="preset").set_value("sp500.csv").run()
    assert at.number_input(key="cash_w").value == 0.0  # $50,000 does not carry over as a weight
    at.number_input(key="cash_w").set_value(0.2).run()
    at.button(key="build").click().run()
    assert not at.exception
    assert at.session_state["portfolio"].weights == pytest.approx({"SPY": 1 / 1.2, "CASH": 0.2 / 1.2})


def test_portfolio_page_proxy_fill(patched, monkeypatch):
    def late_bnd(tickers: tuple[str, ...]) -> pd.DataFrame:  # BND's first price in 2019
        px = fake_prices(tickers)
        return px.assign(BND=px["BND"].where(px.index >= "2019-01-01"))

    monkeypatch.setattr(common, "prices_cached", late_bnd)
    at = _page("portfolio").run()
    at.selectbox(key="preset").set_value("voo_bnd_60_40.csv").run()
    at.button(key="build").click().run()
    assert at.session_state["proxied"] == [] and at.session_state["core_r"].index[0] >= pd.Timestamp("2019-01-01")
    at.toggle(key="proxy").set_value(True).run()
    at.button(key="build").click().run()
    assert not at.exception
    assert at.session_state["proxied"] == ["BND"] and at.session_state["core_r"].index[0] == MKT.index[0]
    assert any("BND" in c.value and "proxy" in c.value for c in at.caption)  # the back-fill is labelled


class _Upload:
    """The two attributes of st.file_uploader's UploadedFile that read_upload uses."""

    def __init__(self, data: bytes):
        self.size, self._data = len(data), data

    def getvalue(self) -> bytes:
        return self._data


def test_read_upload_is_a_trust_boundary():
    assert common.read_upload(_Upload("\ufeffticker,weight\nSPY,1\n".encode())) == "ticker,weight\nSPY,1\n"
    with pytest.raises(ValueError, match="1 MB"):
        common.read_upload(_Upload(b"x" * (common.MAX_UPLOAD + 1)))
    with pytest.raises(ValueError, match="UTF-8"):
        common.read_upload(_Upload(b"\xff\xfe\x00bad"))


def test_samples_are_valid_holdings():
    from insureport.portfolio import parse_holdings

    files = sorted((ROOT / "samples").glob("*.csv"))
    assert [f.name for f in files] == ["sp500.csv", "tech_concentrated.csv", "voo_bnd_60_40.csv"]
    assert parse_holdings((ROOT / "samples" / "voo_bnd_60_40.csv").read_text())["ticker"].tolist() == ["VOO", "BND"]


def test_diagnostics_page_shows_its_metric_cards(patched):
    at = _seeded("diagnostics").run()
    assert not at.exception
    assert at.title[0].value == "Diagnostics"
    labels = {m.label for m in at.metric}
    assert {"CAGR", "Arithmetic mean", "Volatility tax", "Max drawdown", "Time under water", "Longest recovery",
            "Beta", "Downside beta", "S&P 500 CAGR", "S&P 500 max drawdown", "HHI", "Effective N", "Top 5",
            "Hedge notional", "SPX contracts", "XSP contracts"} <= labels
    # the annual bars and both lines come from the full calendar years (2019 alone: 2018 starts on the first row)
    full = metrics.full_year_returns(metrics.nav_from_returns(_core()[1]))
    bars = next(_spec(c) for c in at.get("plotly_chart") if "arithmetic mean" in json.dumps(_spec(c)["data"]))
    bar, arith, geo = bars["data"]
    np.testing.assert_allclose(_y(bar), full.to_numpy(), rtol=1e-12)
    assert _y(arith)[0] == pytest.approx(full.mean())
    assert _y(geo)[0] == pytest.approx((1 + full).prod() ** (1 / len(full)) - 1)
    assert next(m.value for m in at.metric if m.label == "Arithmetic mean") == f"{full.mean():.2%}"


def test_pages_stop_without_a_portfolio(patched):
    for name in ("diagnostics", "candidates", "stress_lab"):
        at = _page(name).run()
        assert not at.exception, name
        assert any("Portfolio" in w.value for w in at.warning), name
        assert not at.metric, name


# ---------------------------------------------------------------- app, part (b)


def _charts(node: Any) -> int:
    """Plotly charts drawn under an AppTest node (the page or one tab)."""
    return len(node.get("plotly_chart"))


def test_candidates_page_evaluates_into_a_ranked_table(patched):
    at = _seeded("candidates").run()
    assert not at.exception
    assert at.title[0].value == "Candidates"
    picks = at.multiselect(key="picks")
    assert {"tbills", "universa_model", "gold"} <= set(picks.value)  # the shortlist, pre-selected (downside beta 0.9)
    picks.set_value(["tbills", "universa_model", "prot_put"]).run()
    # the expanders' edits reach the engine: OTM in percent, monetize 0 = off, an overlay's coverage
    at.number_input(key="universa_model:otm0").set_value(25.0)
    at.number_input(key="universa_model:monetize").set_value(5.0)
    at.number_input(key="prot_put:coverage").set_value(0.5)
    at.button(key="evaluate").click().run()
    assert not at.exception
    ev = at.session_state["evaluated"]
    assert set(ev) == {"tbills", "universa_model", "prot_put"}
    spec = common.candidate_from_tuple(ev["universa_model"]["args"].params).spec
    assert spec.legs[0].otm == 0.25 and spec.monetize_mult == 5.0
    assert ev["prot_put"]["args"].cover == 0.5
    assert common.candidate_from_tuple(ev["prot_put"]["args"].params).spec.monetize_mult is None
    tables = [df.value for df in at.dataframe]
    ranked = next(t for t in tables if "verdict" in t.columns and "d_cagr_bp" in t.columns)
    assert set(ranked["candidate"]) == {"tbills", "universa_model", "prot_put"}
    assert any("critic_sleeve_sharpe" in t.columns for t in tables)  # the critic's scorecard
    # the three denominators (option sleeves): the premium one for the premium-mode sleeve, n/a for an overlay
    den = next(t for t in tables if "net return on premium" in t.columns).set_index("candidate")
    a = ev["universa_model"]["args"]
    sleeve, cal = common.sleeve_cached(at.session_state["core_r"], a.params, a.pricing, a.chain_path, a.cover)
    rop = den.at["universa_model", "net return on premium"]
    assert -1 <= rop and rop == pytest.approx(catalog.return_on_premium(sleeve, MKT["rf"], cal))
    assert np.isnan(den.at["prot_put", "net return on premium"])
    assert any("universa_reported" in t.columns for t in tables)  # the self-reported known-answer table
    assert _charts(at) >= 4  # the cost-effectiveness plane and one payoff scatter per candidate
    # a rebuilt portfolio (here a 30% SPY core) no longer shows the old ranking: Evaluate again
    at.session_state["core_r"] = _core(0.3)[1]
    at.run()
    assert not at.exception
    assert any("Evaluate" in i.value for i in at.info)
    assert not any("verdict" in df.value.columns for df in at.dataframe) and _charts(at) == 0


def test_candidates_sidebar_rebalance_replaces_the_preset_selectors(patched):
    at = _seeded("candidates")
    at.session_state["settings"] = {**common.DEFAULTS, "rebalance": "monthly"}
    at.run()
    assert not at.exception
    assert not any(sb.key.endswith(":rebalance") for sb in at.selectbox)  # nothing to edit that would be ignored
    assert any("Rebalance: monthly" in c.value for c in at.caption)


def test_candidates_chain_picker_lists_csv_and_zip_files(patched, monkeypatch, tmp_path):
    (tmp_path / "chains").mkdir()
    for name in ("a.csv", "b.ZIP", "c.txt"):
        (tmp_path / "chains" / name).write_text("x")
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    at = _seeded("candidates").run()
    assert not at.exception
    assert at.selectbox(key="chain").options == [
        "model-priced (VIX / SKEW surface)", "real end-of-day chain: a.csv", "real end-of-day chain: b.ZIP"]


def test_stress_lab_tabs_render(patched):
    at = _seeded("stress_lab").run()
    assert not at.exception
    assert at.title[0].value == "Stress lab"
    assert at.selectbox(key="stress_candidate").value == "universa_model"
    assert [t.label for t in at.tabs] == ["Frontier", "Bootstrap", "Dice", "Robustness", "Sensitivity"]
    frontier, boot, dice, robust, sens = at.tabs
    assert _charts(frontier) == 1 and len(frontier.metric) >= 4  # the AQR twins as metric cards
    assert _charts(boot) == 3 and len(boot.dataframe) == 1  # fan charts 5 / 10 / 20 years, the book bootstrap
    bt = boot.dataframe[0].value
    assert bt.index.tolist() == ["joint", "reshuffle", "no_crash"] and np.isfinite(bt["cagr_p50"]).all()
    assert _charts(dice) == 3  # the regime frequencies, the dice and the crash-frequency sweep
    assert _charts(robust) == 2 and len(robust.dataframe) == 2  # rolling 5 / 10 years; decades, roll offsets
    # the page's run, from the cache: the preset at the default settings, coverage = the core's downside beta
    core = at.session_state["core_r"]
    beta = metrics.downside_beta(core.iloc[1:], MKT["spxtr"].pct_change())
    args = common.eval_args(CANDIDATES["universa_model"], common.DEFAULTS, common.coverage(beta))
    row = common.evaluate_cached(core, *args)
    w_s, table = row["w_scores"], row["_table"]
    base = (table.at[w_s, "cagr"] - table.at[0.0, "cagr"]) * 1e4  # dCAGR at w_s, bp/yr
    assert [_spec(c)["data"][0]["name"] for c in robust.get("plotly_chart")] == [f"w = {w_s:.1%}"] * 2
    offs = robust.dataframe[1].value.iloc[:, 0]  # each offset rebuilds the sleeve: path dependence shows
    assert offs.index.tolist() == [0, 5, 10, 15] and offs.nunique() == 4 and offs[0] == pytest.approx(base)
    assert _charts(sens) == 0  # on demand
    at.button(key="run_sensitivity").click().run()
    assert not at.exception
    assert _charts(at.tabs[4]) == 2  # tornados of dCAGR at w* and of w*
    res = at.session_state["sensitivity"][("universa_model", args, common.fingerprint(core))]
    assert len(res) == 16  # m, h, nu and offset, four settings each
    assert res.set_index("setting").at["offset 0", f"dCAGR at w = {w_s:.2%} (bp/yr)"] == pytest.approx(base)
    # estimated mode reads the dice table off this backtest: only 2019 is a full year (> 15%), the other regimes
    # are left for the user to fill, so the page warns and draws only the regime frequencies
    at.radio(key="dice_mode").set_value("estimated").run()
    assert not at.exception
    dice = at.tabs[2]
    assert any("fill it in" in w.value for w in dice.warning) and _charts(dice) == 1
    tbl = dice.dataframe[0].value
    returns = tbl[["core", "haven"]].to_numpy(float)
    assert np.isfinite(returns[-1]).all() and np.isnan(returns[:-1]).all()
    # the editor keeps its identity on the next run (the cartoon widgets are gone by then): a first edit sticks
    editor_id = dice.dataframe[0].proto.id
    at.run()
    assert at.tabs[2].dataframe[0].proto.id == editor_id
    # a sweep belongs to its sample: kept across reruns, but another start date leaves the Sensitivity tab empty
    assert _charts(at.tabs[4]) == 2
    at.session_state["core_r"] = at.session_state["core_r"].loc["2019-01-01":]
    at.run()
    assert not at.exception and _charts(at.tabs[4]) == 0


def test_stress_lab_rolling_windows_fill_on_a_long_sample(patched, monkeypatch):
    idx = pd.bdate_range("2004-01-01", "2019-12-31")  # 16 years: both rolling windows fill
    spxtr = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0003, 0.01, len(idx))), idx)
    long = pd.DataFrame({"spx": spxtr, "spxtr": spxtr, "rf": 0.02})
    monkeypatch.setattr(common, "market", lambda: long)
    monkeypatch.setattr(common, "haven_px", lambda: pd.DataFrame(index=idx))  # T-bills need none
    at = _page("stress_lab")
    at.session_state["portfolio"], at.session_state["proxied"] = Portfolio({"SPY": 1.0}, 1_000_000.0), []
    at.session_state["core_r"] = spxtr.pct_change().fillna(0.0)
    at.session_state["stress_candidate"] = "tbills"
    at.run()
    assert not at.exception
    rolling = [_y(_spec(c)["data"][0]) for c in at.tabs[3].get("plotly_chart")]
    assert [len(y) for y in rolling] == [len(idx) - 252 * 5, len(idx) - 252 * 10]  # one point per full window
    assert all(np.isfinite(y).all() for y in rolling)


def _coverage(at: AppTest) -> pd.DataFrame:
    """The Methodology page's data-coverage table."""
    return next(df.value for df in at.dataframe if "first date in the panel" in df.value.columns)


def test_methodology_renders_the_validation_table(patched, monkeypatch, tmp_path):
    monkeypatch.setattr(common, "ROOT", tmp_path)  # no METHODOLOGY.md: the placeholder
    at = _page("methodology").run()
    assert not at.exception
    assert at.title[0].value == "Methodology"
    assert any("METHODOLOGY.md" in i.value for i in at.info)
    tables = [df.value for df in at.dataframe]
    v = next(t for t in tables if "result" in t.columns)
    assert v["result"].tolist() == ["FAIL", "PASS", "FAIL"]
    assert (_coverage(at)["before it"] == "").all()  # the fixture's term structure is native throughout
    assert _charts(at) == 3  # PPUT, PPUT3M and CLL replication
    # a METHODOLOGY.md at the root is rendered; a back-filled series is flagged in the coverage table
    (tmp_path / "METHODOLOGY.md").write_text("# How InsurePort works", encoding="utf-8")
    raw = RAW.assign(vix1y=RAW["vix1y"].where(np.arange(len(RAW)) >= 200))  # VIX1Y from row 200 (300 overlap rows)
    mkt = pricing.fill_term_structure(raw)
    mkt.attrs["first_dates"] = {c: raw[c].first_valid_index() for c in raw}
    monkeypatch.setattr(common, "market", lambda: mkt)
    at = _page("methodology").run()
    assert not at.exception and not at.info
    assert any("How InsurePort works" in m.value for m in at.markdown)
    cov = _coverage(at)
    assert cov.at["vix1y", "first date in the panel"] == RAW.index[200]
    assert cov.at["vix1y", "before it"].startswith("regression-filled") and cov.at["vix9d", "before it"] == ""


def test_methodology_warns_when_validation_cannot_run(patched, monkeypatch):
    def fail(m: float, nu: float, h: float) -> pd.DataFrame:
        raise ValueError("the CLL replica and mkt['cll'] share no date")

    monkeypatch.setattr(common, "validation_cached", fail)
    at = _page("methodology").run()
    assert not at.exception
    assert any("Validation not available" in w.value for w in at.warning)
    assert not any("result" in df.value.columns for df in at.dataframe)
    assert _charts(at) == 3 and len(_coverage(at))  # the rest still renders


def test_entry_point_sidebar_holds_the_settings(patched):
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "Portfolio"  # the default page
    s = at.session_state["settings"]
    assert (s["dd_cap"], s["funding"], s["h"], s["m"], s["rebalance"]) == (
        0.30, "carve_out", catalog.H_DEFAULT, pricing.SKEW_MULT, "preset")
    assert any("validation **FAIL**" in m.value for m in at.sidebar.markdown)
    # relever needs the leverage toggle: off, the radio is disabled and a relever choice left on it is ignored
    assert at.radio(key="set_funding").disabled
    at.toggle(key="set_leverage").set_value(True).run()
    at.radio(key="set_funding").set_value("relever").run()
    assert at.session_state["settings"]["funding"] == "relever"
    at.toggle(key="set_leverage").set_value(False).run()
    assert at.session_state["settings"]["funding"] == "carve_out"
    at.switch_page("pages/diagnostics.py").run()
    assert not at.exception and at.title[0].value == "Diagnostics"


def test_entry_point_banner_passes_on_the_sidebar_assumptions(patched, monkeypatch):
    seen = []
    monkeypatch.setattr(common, "validation_cached",
                        lambda m, nu, h: seen.append((m, nu, h)) or VALIDATION.assign(passed=True))
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("validation **PASS**" in m.value for m in at.sidebar.markdown)
    assert seen == [(pricing.SKEW_MULT, 1.0, 0.0)]  # the sidebar's m and nu; h 0, Cboe's own fill convention


def test_entry_point_stops_with_an_error_when_the_market_panel_cannot_load(patched, monkeypatch):
    def down() -> pd.DataFrame:  # data.load_market on a first run with no network and no cache
        raise RuntimeError("load_market: required series 'spx' has no data from 1990-01-01")

    monkeypatch.setattr(common, "market", down)
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Market data unavailable" in e.value and "'spx'" in e.value for e in at.error)
    assert not at.title  # no page ran

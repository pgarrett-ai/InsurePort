"""Tests for scripts/case_study.py, the case study's table builders: the three cores (the S&P 500 TR, the
monthly 60/40 on the 10-year proxy, the proxy-filled sample portfolio), the cores and candidate tables, the
frontier, sleeve-score, cost-effectiveness-plane, verdict-matrix, AQR, relever, robustness (rolling, block
bootstrap, decades, roll offsets, the book bootstrap reshuffling the haven's years rather than the
portfolio's, with the book's cartoon as its reference row), dice (the book's Part One cartoons reproduced on
the packaged annual table, the estimated mode with its breakeven crash frequency and the cartoon's reference
row, the crash-frequency sweep), pricing-sensitivity,
validation and crisis-window tables, each run on the 2018-2019 fixture panel with a tiny grid and checked
for its columns and one row per (core, candidate) it covers; the weight the post-processing reads a
candidate at; the markdown writer's cell formats and its round trip through the reader.

Why: CASE_STUDY.md copies every number from these tables, so each builder must run end to end on a
known panel and keep its columns. The script is loaded from its file (scripts/ is not a package); its
main() (network, figures) is not run here. Fixture panel and synthetic prices, no I/O.
"""

import importlib.util
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insureport import backtest, catalog, metrics, pricing, scenarios, sleeves
from insureport.catalog import CANDIDATES

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("case_study", ROOT / "scripts" / "case_study.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

FIXTURE = Path(__file__).parent / "fixtures" / "market_2018_2019.csv"
MKT = pricing.fill_term_structure(pd.read_csv(FIXTURE, index_col="date", parse_dates=True))
_t = np.arange(len(MKT))
PX = pd.DataFrame({"DGS10": 2.5 + 0.5 * np.sin(_t / 40), "GLD": 120 * 1.0002**_t,
                   "AAA": 50 * 1.0004**_t, "BBB": np.where(_t >= 100, 20 * 1.0001**_t, np.nan)}, index=MKT.index)
HOLDINGS = "ticker,shares\nAAA,10\nBBB,30\n"
TINY = {  # a tiny grid per candidate: one option sleeve at its preset weight, an overlay, a series and an ETF
    "universa_model": replace(CANDIDATES["universa_model"], w_grid=(catalog.W_UNIVERSA,)),
    "prot_put": replace(CANDIDATES["prot_put"], w_grid=(0.1,)),
    "tbills": replace(CANDIDATES["tbills"], w_grid=(0.1, 0.5)),
    "gold": replace(CANDIDATES["gold"], w_grid=(0.1,)),
}
PAIRS = {(c, k) for c in ("spx", "6040", "tech") for k in TINY}


@cache
def cores() -> dict:
    """The fixture's three cores (built once)."""
    return cs.build_cores(MKT, PX, HOLDINGS)[0]


@cache
def rows() -> dict:
    """evaluate_all on the fixture: every (core, tiny candidate), 100 bootstrap paths (built once)."""
    return cs.evaluate_all(cores(), MKT, PX, TINY, paths=100)


def test_build_cores_spx_6040_and_the_proxy_filled_sample():
    c, proxied = cs.build_cores(MKT, PX, HOLDINGS)
    assert set(c) == {"spx", "6040", "tech"} and proxied == ["BBB"]  # BBB starts on row 100: S&P 500 TR before
    for r in c.values():
        assert r.index[0] == MKT.index[0] and r.iloc[0] == 0.0 and r.notna().all()
    spx = MKT["spxtr"].pct_change()
    np.testing.assert_allclose(c["spx"].iloc[1:], spx.iloc[1:])
    # the 60/40 holds 0.6 / 0.4 on the first day of each month (rebalanced at the previous month's last close)
    day = MKT.index[MKT.index.to_period("M") == pd.Period("2018-03")][0]
    bond = sleeves.bond_proxy(PX["DGS10"])
    assert c["6040"][day] == pytest.approx(0.6 * spx[day] + 0.4 * bond[day], rel=1e-6)
    # before BBB's first price the sample portfolio holds the S&P 500 TR in its place
    w = PX[["AAA", "BBB"]].iloc[-1] * [10, 30]
    w /= w.sum()
    day = MKT.index[50]
    aaa = PX["AAA"].pct_change()[day]
    assert c["tech"][day] == pytest.approx(w["AAA"] * aaa + w["BBB"] * spx[day])


def test_core_tables():
    t = cs.core_table(cores(), MKT)
    assert t["core"].tolist() == ["spx", "6040", "tech"]
    assert {"start", "end", "years", "cagr", "arith_mean", "vol_tax", "vol_tax_approx", "ann_vol", "max_dd", "worst_year",
            "tuw", "downside_beta"} <= set(t.columns)
    assert t.set_index("core").at["spx", "downside_beta"] == pytest.approx(1.0)
    crash = cs.core_crash_table(cores())
    assert crash["core"].tolist() == ["spx", "6040", "tech"] and "crash:2018 Q4" in crash.columns
    nav = metrics.nav_from_returns(cores()["spx"])
    assert crash.at[0, "crash:2018 Q4"] == pytest.approx(metrics.crash_returns(nav).at["2018 Q4", "ret"])


def test_evaluate_all_rows_carry_their_core_and_calendar():
    r = rows()
    assert set(r) == PAIRS
    for (core, key), row in r.items():
        assert row["core"] == core and row["candidate"] == key and row["funding"] == "carve_out"
        assert len(row["_cal"]) > 0 and np.isfinite(row["_beta"])
    assert r["spx", "tbills"]["verdict"] == "insufficient history"  # two years of fixture


def test_candidate_frontier_and_score_tables():
    for build, cols in (
        (cs.candidate_table, {"archetype", "kind", "model_priced", "sample_start", "sample_end", "years",
                              "n_crashes_in_sample"}),
        (cs.frontier_table, {"w_star", "w_cap", "cagr_unhedged", "cagr_star", "d_cagr_bp", "arith_cost_bp",
                             "vol_tax_saved_bp", "maxdd_unhedged", "maxdd_star", "d_cagr_p50_bp", "verdict", "robustness"}),
        (cs.sleeve_score_table, {"w_scores", "calm_year_bleed", "crash_payoff_mean", "avg_premium_yr", "avg_coverage",
                                 "attr_crash_bp", "attr_noncrash_bp", "shock20_pnl"}),
    ):
        t = build(rows())
        assert {"core", "candidate"} | cols <= set(t.columns)
        assert set(zip(t["core"], t["candidate"])) == PAIRS and len(t) == len(PAIRS)
    f = cs.frontier_table(rows()).set_index(["core", "candidate"])
    row = rows()["spx", "gold"]
    tab, w = row["_table"], row["w_star"]
    assert f.at[("spx", "gold"), "vol_tax_saved_bp"] == pytest.approx((tab.at[0.0, "vol_tax"] - tab.at[w, "vol_tax"]) * 1e4)
    ce = cs.ce_table(rows()).set_index(["core", "candidate"])
    assert list(ce.columns) == ["verdict", "w_eval", "arith_cost_bp", "vol_tax_saved_bp", "d_cagr_bp"]
    assert ce.at[("spx", "gold"), "d_cagr_bp"] == pytest.approx((tab.at[0.1, "cagr"] - tab.at[0.0, "cagr"]) * 1e4)
    m = cs.verdict_matrix(cs.frontier_table(rows()))
    assert sorted(m["candidate"]) == sorted(TINY) and {f"{c}: {x}" for c in ("spx", "6040", "tech")
                                                       for x in ("w_star", "d_cagr_bp", "verdict")} <= set(m.columns)


def test_aqr_table_twins_at_w_cap_or_w_star():
    t = cs.aqr_table(rows())
    assert {"core", "candidate", "w_twin", "d_cagr_vs_dd_matched_cash_bp", "w_cash_twin", "maxdd_cash_twin",
            "dd_gap_vs_cagr_matched_cash", "beats_deleverage", "critic_sleeve_cagr", "critic_sleeve_sharpe",
            "critic_alpha", "critic_beta"} <= set(t.columns) and len(t) == len(PAIRS)
    for _, r in t.iterrows():
        row = rows()[r["core"], r["candidate"]]
        assert r["w_twin"] == (row["w_star"] if np.isnan(row["w_cap"]) else row["w_cap"])
        twin = backtest.cagr_matched_cash_dd(row["_table"], row["_cash"], r["w_twin"])
        assert r["w_cash_twin"] == pytest.approx(twin["w_cash"], nan_ok=True)


def test_relever_table_puts_both_fundings_side_by_side():
    opts = {k: c for k, c in TINY.items() if c.kind == "option"}
    lev = cs.evaluate_all(cores(), MKT, PX, opts, funding="relever", paths=100)
    t = cs.relever_table(rows(), lev)
    assert {"core", "candidate", "funding", "w_star", "gross_exposure", "cagr_star", "d_cagr_bp", "maxdd_star",
            "d_cagr_vs_dd_matched_cash_bp", "beats_deleverage", "verdict"} <= set(t.columns)
    assert len(t) == 2 * 3 * len(opts) and set(t["funding"]) == {"carve_out", "relever"}
    assert (t.loc[t["funding"] == "relever", "gross_exposure"] >= 1).all()


def test_w_eval_rule():
    base = {"candidate": "gold", "w_star": 0.0, "w_scores": 0.1, "_nav": pd.DataFrame(columns=[0.0, 0.0333, 0.1])}
    assert cs.w_eval(base | {"w_star": 0.1}) == 0.1  # w* when positive
    assert cs.w_eval(base) == 0.1  # else the smallest grid weight
    assert cs.w_eval(base | {"candidate": "universa_model"}) == catalog.W_UNIVERSA  # a preset's own weight
    assert cs.w_eval(base | {"candidate": "dao_ch9"}) == 0.1  # its 0.5% is off this grid


def test_robustness_tables():
    t = cs.robustness_table(rows(), paths=100)
    assert {"core", "candidate", "w_eval", "rolling10_win_rate", "p_beat_5y", "p_beat_10y", "p_beat_20y"} <= set(t.columns)
    assert len(t) == len(PAIRS) and t["rolling10_win_rate"].isna().all()  # no 10-year window in two years
    assert t[["p_beat_5y", "p_beat_10y", "p_beat_20y"]].apply(lambda s: s.between(0, 1)).all().all()
    d = cs.decade_table(rows())
    assert list(d.columns) == ["core", "candidate", "w_eval", "d_1990s_bp", "d_2000s_bp", "d_2010s_bp", "d_2020s_bp"]
    row = rows()["spx", "gold"]
    w = cs.w_eval(row)
    sub = backtest.subperiods(row["_nav"], w)
    got = d.set_index(["core", "candidate"]).loc[("spx", "gold")]
    assert got["d_2010s_bp"] == pytest.approx(sub.at[2010, "d_cagr"] * 1e4) and np.isnan(got["d_1990s_bp"])
    o = cs.offset_table(rows(), cores(), MKT, PX, TINY)
    assert {"core", "candidate", "w_eval", "d_cagr_off0_bp", "d_cagr_off5_bp", "d_cagr_off10_bp",
            "d_cagr_off15_bp", "spread_bp"} <= set(o.columns)
    assert set(zip(o["core"], o["candidate"])) == {(c, k) for c, k in PAIRS if TINY[k].spec is not None}
    b = cs.book_bootstrap_table(rows(), paths=100)
    assert {"core", "candidate", "w_eval", "years"} | {f"{m}_{x}" for m in cs.MODES for x in ("d_cagr_p50_bp", "p_beat")} \
        <= set(b.columns)
    ins = {(c, k) for c, k in PAIRS if TINY[k].archetype == "insurance"}
    assert ins | {(cs.CARTOON_CORE, "insurance cartoon")} == set(zip(b["core"], b["candidate"]))
    spx = scenarios.load_sp500_annual()  # the reference row: the zero-mean cartoon at 3% on 1928-latest
    cart = scenarios.book_haven("insurance", scenarios.regime_table(spx).at[0, "freq"])[scenarios.regime_of(spx.to_numpy())]
    want = scenarios.book_bootstrap(spx, pd.Series(cart, spx.index), np.array([0.0, 0.03]), n_paths=100, seed=0).loc[0.03]
    got = b.set_index("core").loc[cs.CARTOON_CORE]
    assert got["joint_d_cagr_p50_bp"] == pytest.approx(want["d_cagr_p50"] * 1e4) and got["years"] == len(spx)
    # reshuffle draws the haven's year apart from the core's: the haven as the portfolio held it, mixed back at w
    # (a synthetic 12-year pair: the fixture has one full year, where every mode draws the same year)
    idx = pd.bdate_range("2005-01-03", "2016-12-30")
    rng = np.random.default_rng(1)
    nav = pd.DataFrame({0.0: np.cumprod(1 + rng.normal(4e-4, 0.012, len(idx))),
                        0.1: np.cumprod(1 + rng.normal(3e-4, 0.010, len(idx)))}, idx)
    row = {"core": "spx", "candidate": "gold", "archetype": "insurance", "w_star": 0.1, "w_scores": 0.1, "_nav": nav}
    b = cs.book_bootstrap_table({("spx", "gold"): row}, paths=100)
    w = 0.1
    a = metrics.full_year_returns(nav)
    want = scenarios.book_bootstrap(a[0.0], scenarios.implied_haven(a[0.0], a[w], w), np.array([0.0, w]), n_paths=100,
                                    seed=0, mode="reshuffle").loc[w]
    got = b.set_index(["core", "candidate"]).loc[("spx", "gold")]
    assert got["reshuffle_p_beat"] == pytest.approx(want["p_beat"])
    assert got["reshuffle_d_cagr_p50_bp"] == pytest.approx(want["d_cagr_p50"] * 1e4)


def test_cartoons_reproduce_part_one_on_1997_2016():
    t = cs.cartoon_table()
    assert {"haven", "w", "window", "d_cagr_bp", "book_reported_bp"} <= set(t.columns)
    part_one = t[t["window"] == "1997-2016"].set_index(["haven", "w"])
    # Universa Safe Haven Investing Part One, 1997-2016: +67 / +18 / -17 bp as reported (test_scenarios pins these)
    assert part_one.at[("insurance", 0.03), "d_cagr_bp"] == pytest.approx(67, abs=2)
    assert part_one.at[("alpha", 0.10), "d_cagr_bp"] == pytest.approx(18, abs=2)
    assert part_one.at[("store of value (CPI + 2%)", 0.10), "d_cagr_bp"] == pytest.approx(-18.4, abs=0.5)
    assert part_one["book_reported_bp"].notna().sum() == 3 and t["book_reported_bp"].notna().sum() == 3
    assert set(t["window"]) == {"1997-2016", f"1928-{scenarios.load_sp500_annual().index[-1]}"}


def test_dice_and_crash_sweep_tables():
    opts = {k for k, c in TINY.items() if c.spec is not None}
    d = cs.dice_table(rows())
    assert {"core", "candidate", "w_eval", "years_per_regime", "cost_effective", "kelly_w", "breakeven_pi", "hist_pi",
            "dice_d_cagr_bp"} | {f"haven {lbl}" for lbl in scenarios.REGIME_LABELS} <= set(d.columns)
    assert set(zip(d["core"], d["candidate"])) == {(c, k) for c, k in PAIRS if k in opts} | {(cs.CARTOON_CORE,
                                                                                               "insurance cartoon")}
    sleeves_ = d[d["core"] != cs.CARTOON_CORE]
    assert sleeves_["kelly_w"].isna().all()  # one full year (2019) leaves regimes without a year: n/a, not a guess
    cart = d.set_index("core").loc[cs.CARTOON_CORE]
    reg = scenarios.regime_table(scenarios.load_sp500_annual())
    p, cj = reg["freq"].to_numpy(), reg["mean_ret"].to_numpy()
    hj = scenarios.book_haven("insurance", p[0])
    assert cart["kelly_w"] == pytest.approx(scenarios.kelly_w(p, cj, hj)) and cart["cost_effective"]
    assert cart["breakeven_pi"] == pytest.approx(scenarios.breakeven_crash_freq(p, cj, hj))
    assert (d["hist_pi"] == scenarios.regime_table(scenarios.load_sp500_annual()).at[0, "freq"]).all()
    s = cs.crash_sweep_table(rows())
    assert s.index.name == "pi" and list(s.index) == list(cs.PI_GRID)
    assert set(s.columns) == {"insurance cartoon at 3%", *(f"{k} at {cs.w_eval(rows()['spx', k]):.2%}" for k in opts)}
    # the cartoon: growth at 3% minus at 0 on the 1928-latest regimes, rising with the crash frequency through 0
    reg = scenarios.regime_table(scenarios.load_sp500_annual())
    p, cj = reg["freq"].to_numpy(), reg["mean_ret"].to_numpy()
    sweep = scenarios.crash_frequency_sweep(p, cj, scenarios.book_haven("insurance", p[0]), np.array([0.0, 0.03]),
                                            grid=np.array(cs.PI_GRID))
    col = s["insurance cartoon at 3%"]
    np.testing.assert_allclose(col, sweep[0.03] - sweep[0.0])
    assert col.is_monotonic_increasing and col.iloc[0] < 0 < col.iloc[-1]


def test_sensitivity_table_sweeps_every_pricing_knob_one_at_a_time():
    c, core = TINY["universa_model"], cores()["spx"]
    t = cs.sensitivity_table(c, core, MKT, PX, 1.0, catalog.W_UNIVERSA)
    assert list(t.columns) == ["parameter", "setting", "w_star", "d_cagr_bp"]
    assert t["parameter"].tolist() == ["base", *(p for p, _, _ in cs.VARIANTS)]
    n = dict(t["parameter"].value_counts())
    assert n == {"base": 1, "m": 5, "nu": 5, "h": 4, "z_cap": 3, "beta": 3}
    base = t.at[0, "d_cagr_bp"]
    for p in ("m", "nu", "beta"):  # x1 / 1 / 0.5 is the base setting itself
        assert t.loc[t["parameter"] == p, "d_cagr_bp"].sub(base).abs().min() == pytest.approx(0, abs=1e-9)
    assert t.loc[t["parameter"] == "nu", "d_cagr_bp"].nunique() > 1
    bars = backtest.tornado_rows(t.iloc[1:], "d_cagr_bp", base)
    assert len(bars) == 5


def test_validation_and_scenario_tables():
    v = cs.validation_table(MKT, None)
    assert {"target", "window_start", "window_end", "n_months", "te", "te_tol", "dcagr", "dcagr_tol", "corr", "corr_tol",
            "te_roll", "te_nonroll", "passed"} <= set(v.columns)
    assert v["target"].tolist() == ["PPUT", "CLL"]  # no PPUT3M levels passed
    navs = {"unhedged": metrics.nav_from_returns(cores()["spx"]), "60/40": metrics.nav_from_returns(cores()["6040"])}
    s = cs.scenario_table(navs)
    assert list(s.columns) == ["window", "start", "end", "unhedged", "60/40"]
    assert s["window"].tolist() == [n for n, _, _ in cs.SCENARIOS]
    q4 = s.set_index("window").loc["2018 Q4"]
    assert q4["unhedged"] == pytest.approx(metrics.window_return(navs["unhedged"], q4["start"], q4["end"]))
    assert s.set_index("window").loc["2008", ["unhedged", "60/40"]].isna().all()  # outside the fixture


def test_markdown_writer_formats_cells_and_round_trips():
    df = pd.DataFrame({"candidate": ["a|b", "c"], "w_star": [0.0333, np.nan], "d_cagr_bp": [12.4, -3.6],
                       "years": [36.77, 2.0], "beats_deleverage": [True, False],
                       "start": pd.to_datetime(["1990-01-02", "2018-01-02"]), "n": [3, 4]})
    text = cs.to_markdown(df)
    lines = text.splitlines()
    assert lines[0] == "| candidate | w_star | d_cagr_bp | years | beats_deleverage | start | n |"
    assert lines[1] == "|---|---|---|---|---|---|---|"
    assert lines[2] == r"| a\|b | 3.33% | +12 | 36.8 | yes | 1990-01-02 | 3 |"
    assert lines[3] == "| c | n/a | -4 | 2 | no | 2018-01-02 | 4 |"
    back = cs.from_markdown(text)
    assert back.columns.tolist() == df.columns.tolist() and back.iloc[0, 0] == "a|b"
    pd.testing.assert_frame_equal(cs.from_markdown(cs.to_markdown(back)), back)  # strings round-trip exactly
    assert back.at[1, "w_star"] == "n/a" and back.at[0, "beats_deleverage"] == "yes"

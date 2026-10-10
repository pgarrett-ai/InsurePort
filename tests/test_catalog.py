"""Tests for insureport.catalog: the candidate presets (grids, notes citing a source, archetypes and
kinds, validation, the Universa preset's calendar, budget and year-end harvest, dao_ch9's delta strike,
the TAIL proxy's legs and constants), rebalance calendars (a trailing partial period has no date),
build_sleeve on the 2018-2019 fixture (T-bills, the 10-year proxy, trend, an ETF and a Cboe index
aligned to the panel; the dao_ch9 and universa_model option sleeves with their monthly events and the
anchored 0.20%-of-NAV budget funded through 2019 by the year-end harvest; the deep-OTM spread passed to
the pricer; the overlay sized on downside beta; the TAIL proxy rebuilt by hand, spending 1% of NAV at
every roll, its bond share drifting, and its calendar-year check; the 60/40 bond leg using the DGS10
proxy only before AGG; the CLL splice; missing sources), the shortlist rules (basis-risk and
concentration warnings on every index-put key, reasons joined), evaluate on synthetic histories (a
dominant insurance payoff is cost-effective, its attribution, crash payoffs, deleverage twins
(including a finite divested twin) and benchmarks pinned; the verdict at the uncapped Kelly w* and the
twins at the capped w_cap, or at w* when no weight meets the cap; robust with n/a checks on a
post-2009 or short-calendar sample; fragile when a robustness check fails; not cost-effective when the
bootstrap median is not positive; a cash sleeve is not, its bootstrap columns n/a; short or crash-free
histories are insufficient; relever funding is labelled, raises gross exposure and refuses a grid
above 1/3; option re-runs, the roll-offset skip for hold-to-expiry overlays and px reaching them; the
put preset's shock and bleed at its own weight; the option sleeve's coverage marked daily; the 60/40
benchmark from px, skipped on an empty AGG column; the index launch and bad-mark notes), the
robustness helpers on hand-built years (the best crash year of two, the crash edge), the calm-year
bleed, the critic's stand-alone view (excess returns over cash), ranking, the Universa known-answer
table and the -20% one-month shock P&L (its formula through the option_sleeve seam; the calm state
from mkt's last year; the addendum's figure at the 2026-10-08 inputs; the fixture's values pinned).

Why: the Candidates page and the case study read every verdict from here, so each rule is pinned on
inputs whose answer is known by construction. Synthetic series plus the committed fixture, no I/O.
"""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insureport import backtest, catalog, metrics, pricing, scenarios, sleeves
from insureport.catalog import CANDIDATES, Candidate

FIXTURE = Path(__file__).parent / "fixtures" / "market_2018_2019.csv"
MKT = pd.read_csv(FIXTURE, index_col="date", parse_dates=True)
CORE = MKT["spxtr"].pct_change().fillna(0.0)  # the 0 placeholder first row core_returns emits
PX = pd.DataFrame({"DGS10": 2.5 + 0.5 * np.sin(np.arange(len(MKT)) / 40), "GLD": 120 * 1.0002 ** np.arange(len(MKT))},
                  index=MKT.index)
KEYS = {"candidate", "archetype", "kind", "model_priced", "funding", "sample_start", "sample_end", "years",
        "n_crashes_in_sample", "w_star", "w_cap", "cagr_unhedged", "cagr_star", "d_cagr_bp", "arith_cost_bp", "maxdd_unhedged",
        "maxdd_star", "calmar_star", "worst_year_star", "d_cagr_vs_dd_matched_cash_bp", "dd_gap_vs_cagr_matched_cash",
        "beats_deleverage", "p_beat_10y", "rolling10_win_rate", "crash_payoff_mean", "calm_year_bleed",
        "avg_premium_yr", "avg_coverage", "critic_sleeve_cagr", "critic_sleeve_sharpe", "critic_alpha", "critic_beta",
        "verdict", "notes", "_table", "_nav", "bench_spxtr_cagr", "bench_spxtr_maxdd", "bench_6040_cagr",
        "bench_6040_maxdd"}
ANNUAL = pd.to_datetime(["2018-12-31", "2019-12-31"])  # the fixture's year-ends
TOY = Candidate("toy", "Toy haven", "insurance", "series", w_grid=(0.02, 0.05, 0.1), rebalance="annual", note="test")


def _history(start: str = "1990-01-01", end: str = "2019-12-31", sigma: float = 0.012, seed: int = 0):
    """(mkt, core_r): a random-walk S&P 500 total return on business days (drift 4 bp a day), rf 2%."""
    idx = pd.bdate_range(start, end)
    r = pd.Series(np.random.default_rng(seed).normal(0.0004, sigma, len(idx)), idx)
    r.iloc[0] = 0.0
    tr = 1000 * (1 + r).cumprod()
    return pd.DataFrame({"spx": tr, "spxtr": tr, "rf": 0.02}, index=idx), r


def _dominant(core_r: pd.Series) -> pd.Series:
    """An insurance payoff that never loses: +1.5x the core's loss on every down day, flat otherwise."""
    return 1.5 * (-core_r).clip(lower=0)


def test_candidates_are_well_formed():
    assert {"universa_model", "universa_2023", "dao_ch9", "put_spread", "prot_put", "collar", "pput_idx", "pput3m_idx",
            "cll_idx", "vxth_idx", "tail", "tail_proxy", "tbills", "ust10", "tlt", "ief", "gold", "trend", "dbmf",
            "sixty_forty"} == set(CANDIDATES)
    source = re.compile(r"addendum|universa-practice|dao-of-capital|critics-and-literature|data-and-implementation|gap-\d",
                        re.IGNORECASE)  # every note cites its research note
    for key, c in CANDIDATES.items():
        assert c.key == key and c.w_grid and source.search(c.note), key
        assert c.archetype in catalog.ARCHETYPES and c.kind in catalog.KINDS
        assert all(0 <= w <= 1 for w in c.w_grid) and c.rebalance in catalog.REBALANCE
        assert (c.spec is not None) == (c.kind == "option" or key == "tail_proxy") == c.model_priced
    u = CANDIDATES["universa_model"]
    leg = u.spec.legs[0]
    assert (leg.otm, leg.tenor_months, leg.hold_months, u.spec.roll_when, u.spec.budget_base, u.rebalance) == (
        0.30, 2, 2, "month_start", "anchor", "annual")
    assert u.spec.premium_frac * catalog.W_UNIVERSA == pytest.approx(0.0020, abs=1e-5)  # 0.20% of NAV a month
    assert {0.01, 0.02, 0.0333, 0.05} <= set(u.w_grid) and max(u.w_grid) == 0.06 and u.spec.monetize_mult is None
    # the put spread runs on universa_model's calendar, budget and year-end harvest, only its legs differ
    assert u.spec.harvest_at_anchor and CANDIDATES["put_spread"].spec == replace(
        u.spec, legs=CANDIDATES["put_spread"].spec.legs)
    d = CANDIDATES["dao_ch9"]
    assert (d.spec.legs[0].otm, d.spec.legs[0].delta, d.spec.legs[0].tenor_months, d.spec.legs[0].hold_months) == (
        None, -0.005, 2, 1) and (d.spec.premium_frac, d.rebalance) == (1.0, "events")
    assert CANDIDATES["gold"].archetype == "insurance" and CANDIDATES["tbills"].archetype == "store_of_value"
    assert CANDIDATES["trend"].archetype == "alpha" and CANDIDATES["sixty_forty"].w_grid == (0.4,)
    with pytest.raises(ValueError, match="kind"):
        replace(TOY, kind="etf ")  # an app user's typo
    with pytest.raises(ValueError, match="w_grid"):
        replace(TOY, w_grid=(1.5,))


def test_rebalance_calendar():
    idx = pd.bdate_range("2018-01-02", "2019-12-31").drop(pd.Timestamp("2018-12-31"))  # a year-end holiday
    assert catalog.rebalance_calendar(idx, "annual").tolist() == pd.to_datetime(["2018-12-28", "2019-12-31"]).tolist()
    q = catalog.rebalance_calendar(idx, "quarterly")
    assert len(q) == 8 and q[0] == pd.Timestamp("2018-03-30") and q[3] == pd.Timestamp("2018-12-28")
    m = catalog.rebalance_calendar(idx, "monthly")
    assert len(m) == 24 and m[1] == pd.Timestamp("2018-02-28") and m[-1] == idx[-1]
    # an index ending before its period's last weekday (2019-12-31) has no date for that partial period, so an
    # option sleeve does not harvest its legs at the bid on the panel's last row
    short = idx[idx <= "2019-12-30"]
    assert catalog.rebalance_calendar(short, "annual").tolist() == [pd.Timestamp("2018-12-28")]
    assert len(catalog.rebalance_calendar(short, "quarterly")) == 7
    assert catalog.rebalance_calendar(short, "monthly")[-1] == pd.Timestamp("2019-11-29")
    assert catalog.rebalance_calendar(idx[idx <= "2019-06-28"], "quarterly")[-1] == pd.Timestamp("2019-06-28")
    ev = pd.DatetimeIndex(["2018-03-16", "2018-04-20"])
    assert catalog.rebalance_calendar(idx, "events", ev) is ev and catalog.rebalance_calendar(idx, "events").empty
    with pytest.raises(ValueError, match="rebalance"):
        catalog.rebalance_calendar(idx, "weekly")


def test_build_sleeve_series_etf_and_index_on_the_fixture():
    annual = pd.to_datetime(["2018-12-31", "2019-12-31"])
    for key, want in (("tbills", sleeves.tbill_sleeve(MKT)), ("ust10", sleeves.bond_proxy(PX["DGS10"])),
                      ("trend", sleeves.tsmom_sleeve(MKT["spxtr"], MKT["rf"]))):
        r, cal = catalog.build_sleeve(CANDIDATES[key], MKT, PX, CORE, 1.0)
        assert r.index.equals(MKT.index) and cal.tolist() == annual.tolist()
        np.testing.assert_allclose(r, want, rtol=0, atol=1e-15, err_msg=key)
    gold, _ = catalog.build_sleeve(CANDIDATES["gold"], MKT, PX, CORE, 1.0)
    assert gold.iloc[0] == 0 and gold.iloc[1:].to_numpy() == pytest.approx(0.0002, abs=1e-12)
    cll, _ = catalog.build_sleeve(CANDIDATES["cll_idx"], MKT, PX, CORE, 1.0)  # after the 2009-10-16 splice: all of it
    np.testing.assert_allclose(cll, MKT["cll"].pct_change().fillna(0.0), rtol=0, atol=1e-15)
    late = CORE.loc["2019-01-01":]  # sleeves are cut to the core's span
    assert catalog.build_sleeve(CANDIDATES["tbills"], MKT, PX, late, 1.0)[0].index[0] == late.index[0]
    with pytest.raises(ValueError, match="px"):
        catalog.build_sleeve(CANDIDATES["tail"], MKT, PX, CORE, 1.0)  # no TAIL prices passed


def test_sixty_forty_uses_the_bond_proxy_only_before_agg():
    # AGG flat from 2018-06-01; FRED also prints DGS10 on 2018-12-05 (NYSE shut), a 20 bp jump AGG's next print holds
    idx = MKT.index.union([pd.Timestamp("2018-12-05")])
    dgs10 = pd.Series(np.where(idx >= "2018-12-05", 2.7, 2.5), idx)
    px = pd.DataFrame({"DGS10": dgs10, "AGG": pd.Series(100.0, MKT.index[MKT.index >= "2018-06-01"])}, index=idx)
    r, _ = catalog.build_sleeve(CANDIDATES["sixty_forty"], MKT, px, CORE, 1.0)
    assert (r["2018-06-04":] == 0).all() and pd.Timestamp("2018-12-05") not in r.index  # no double-counted move
    # the proxy up to AGG's first print (its NaN first return)
    np.testing.assert_allclose(r[:"2018-06-01"], sleeves.bond_proxy(dgs10)[:"2018-06-01"], rtol=0, atol=1e-15)


def test_build_option_sleeves_with_monthly_events():
    dao, cal = catalog.build_sleeve(CANDIDATES["dao_ch9"], MKT, None, CORE, 1.0)
    assert dao.index.equals(MKT.index) and cal.equals(sleeves.roll_dates(MKT.index))  # rebalanced on its rolls
    assert (dao.attrs["premium_paid"][cal] == 1.0).all()  # f = 1: all the sleeve every month
    u, cal = catalog.build_sleeve(CANDIDATES["universa_model"], MKT, None, CORE, 1.0)
    assert u.index.equals(MKT.index) and cal.tolist() == ANNUAL.tolist()
    paid, nav = u.attrs["premium_paid"], metrics.nav_from_returns(u)
    month_starts = sleeves.roll_dates(MKT.index, when="month_start")
    assert paid.index[paid > 0].isin(month_starts).all() and (paid[month_starts] > 0).all()
    # 6% of the sleeve's NAV at the last year-end (the start, in 2018) each month: 0.20% of NAV at w = 3.33%
    spent = (paid * nav)[month_starts]
    np.testing.assert_allclose(spent[:"2018-12-31"], 0.06, rtol=1e-12)
    # the 2018-12-31 rebalance sells the Q4-marked tranches into T-bills, so every 2019 tranche gets 6% of that NAV
    # and the sleeve is never wiped out (with the marks as its base, cash ran out in June and July hit -100%)
    np.testing.assert_allclose(spent["2019"], 0.06 * nav["2018-12-31"], rtol=1e-12)
    assert (u > -1).all() and u.attrs["coverage"]["2018-12-31"] == 0
    # the overlay holds the core plus puts on downside beta x its value
    # (on its rolls: between them coverage drifts with S / the core, e.g. 0.804 on 2019-07-03, when ^SP500TR's
    # half-day close lagged ^GSPC's by 0.5%)
    pp, cal = catalog.build_sleeve(CANDIDATES["prot_put"], MKT, None, CORE, 0.8)
    assert 0.75 < pp.attrs["coverage"][cal].max() < 0.8
    with pytest.raises(ValueError, match="coverage"):
        catalog.build_sleeve(CANDIDATES["prot_put"], MKT, None, CORE, -0.1)


def test_build_sleeve_prices_with_the_deep_otm_spread(monkeypatch):
    # option_sleeve is the seam: h = H_DEFAULT (an 8%-of-mid spread) unless pricing_kw overrides it
    seen = {}

    def fake(panel, spec, **kw):
        seen.update(kw)
        return pd.DataFrame({"ret": 0.0, "event": False, "premium_paid": 0.0, "coverage": 0.0}, index=panel.index)

    monkeypatch.setattr(sleeves, "option_sleeve", fake)
    catalog.build_sleeve(CANDIDATES["universa_model"], MKT, None, CORE, 1.0)
    assert seen["h"] == catalog.H_DEFAULT == 0.04 and seen["anchors"].equals(ANNUAL) and seen["carry"] is None
    catalog.build_sleeve(CANDIDATES["universa_model"], MKT, None, CORE, 1.0, pricing_kw={"h": 0.0})
    assert seen["h"] == 0.0


def test_tail_proxy_spends_one_percent_of_nav_a_month_on_a_bond_backed_ladder():
    spec = catalog.TAIL_SPEC
    assert [(L.cp, L.otm, L.tenor_months, L.hold_months, L.qty) for L in spec.legs] == [
        ("P", 0.08, 14, 10, 0.75), ("P", 0.08, 14, 13, 0.25)]  # 75% sold with ~4 months left, the rest ~1 month
    assert (spec.premium_frac, spec.budget_base, catalog.TAIL_BUDGET, catalog.TAIL_BOND, catalog.TAIL_FEE) == (
        0.01, "anchor", 0.01, 0.90, 0.006)
    r, cal = catalog.build_sleeve(CANDIDATES["tail_proxy"], MKT, PX, CORE, 1.0)
    assert r.index.equals(MKT.index) and cal.tolist() == ANNUAL.tolist()
    # by hand: one put sleeve (anchors None) whose cash earns 90% bond proxy + 10% T-bills, less the fee by day
    carry = 0.9 * sleeves.bond_proxy(PX["DGS10"]) + 0.1 * sleeves.tbill_sleeve(MKT)
    df = sleeves.option_sleeve(pricing.fill_term_structure(MKT), spec, h=0.04, carry=carry)
    days = MKT.index.to_series().diff().dt.days.fillna(0)
    np.testing.assert_allclose(r, df["ret"] - 0.006 * days / 365, rtol=1e-12, atol=1e-15)
    # the puts' marks move the bond share: not the pure 90 / 10 bond and bill mix (its fee taken alike)
    bonds = metrics.nav_from_returns(carry - 0.006 * days / 365).iloc[-1]
    assert abs(metrics.nav_from_returns(r).iloc[-1] - bonds) > 0.05
    # 1% of NAV at every monthly roll: the bond funds it, so the cash cap never binds and no rung is rescaled
    assert df["event"].sum() == 24
    np.testing.assert_allclose(r.attrs["premium_paid"][df["event"]], 0.01, rtol=1e-12)
    t = catalog.tail_checks(r)
    assert list(t.columns) == ["year", "tail_reported", "ours", "gap"] and t["year"].tolist()[:2] == [2018, 2019]
    assert np.isnan(t.at[0, "ours"]) and t.at[1, "gap"] == pytest.approx(t.at[1, "ours"] + 0.1399, abs=1e-12)


def test_suggest_high_beta_concentrated():
    out = catalog.suggest({"downside_beta": 1.1, "eff_n": 3.0, "top1": 0.4})
    keys = [k for k, _ in out]
    assert keys[0] == "tbills" and len(keys) == len(set(keys))
    assert {"dao_ch9", "universa_model", "put_spread", "pput_idx", "collar", "tail", "tail_proxy", "dbmf"} <= set(keys)
    puts = (*catalog.OPTION_PICKS, "tail", "tail_proxy")
    assert all("single-name" in why for k, why in out if k in puts)
    assert not any("single-name" in why for k, why in out if k not in puts)
    assert "short history" in dict(out)["tail"] and "short history" in dict(out)["dbmf"]
    assert not any("basis risk" in why for _, why in out)


def test_suggest_concentration_warning_whatever_the_beta():
    out = dict(catalog.suggest({"downside_beta": 0.5, "eff_n": 1.0, "top1": 1.0}))  # a single stock
    assert not set(catalog.OPTION_PICKS) & set(out)
    assert "single-name" in out["tail"] and "single-name" in out["tail_proxy"] and "single-name" not in out["gold"]


def test_suggest_low_beta_flags_basis_risk():
    out = dict(catalog.suggest({"downside_beta": 0.2, "eff_n": 10.0, "top1": 0.1}))
    assert "basis risk" in out["trend"] and "basis risk" in out["ust10"] and "tbills" in out
    assert "basis risk" in out["tail"] and "basis risk" in out["tail_proxy"]  # the index puts it warns about
    assert "basis risk" not in out["gold"] and "basis risk" not in out["dbmf"]
    assert "2022" in out["ust10"]  # the reasons are joined: the basis-risk flag does not hide the 2022 caveat
    assert not {"universa_model", "dao_ch9", "put_spread", "pput_idx", "collar"} & set(out)
    assert not any("single-name" in why for why in out.values())


def test_evaluate_dominant_insurance_is_cost_effective():
    mkt, core = _history()
    cal = catalog.rebalance_calendar(mkt.index, "annual")
    cash_table, _ = backtest.frontier(core, sleeves.tbill_sleeve(mkt), catalog.CASH_GRID, cal)
    row = catalog.evaluate(TOY, core, _dominant(core), cal, mkt, cash_table=cash_table, bootstrap_paths=300)
    assert KEYS <= set(row) and row["candidate"] == "toy" and row["funding"] == "carve_out"
    assert row["w_star"] > 0 and row["d_cagr_bp"] > 0 and row["verdict"].startswith("cost-effective")
    assert row["verdict"] == "cost-effective (robust)" and "proven" not in row["verdict"]
    assert row["years"] == pytest.approx(30, abs=0.01) and row["n_crashes_in_sample"] == 7  # 1990 .. 2018 Q4
    assert row["beats_deleverage"] and row["p_beat_10y"] > 0.9 and row["rolling10_win_rate"] == 1.0
    assert row["crash_payoff_mean"] > 0 and row["calm_year_bleed"] > 0  # it never bleeds
    assert row["_table"].index.tolist() == [0.0, 0.02, 0.05, 0.1] and row["w_star"] in row["_nav"].columns
    assert row["cagr_star"] - row["cagr_unhedged"] == pytest.approx(row["d_cagr_bp"] / 1e4, rel=1e-12)
    # not an option sleeve: no premium, and its capital share is its notional
    assert np.isnan(row["avg_premium_yr"]) and row["avg_coverage"] == pytest.approx(
        catalog._sleeve_share(row["_nav"][row["w_star"]], _dominant(core), cal, row["w_star"]).mean(), rel=1e-12)
    assert row["bleed_notional"] == pytest.approx(row["calm_year_bleed"] / row["avg_coverage"])
    w, t, nav = row["w_star"], row["_table"], row["_nav"]
    # w* is the uncapped CAGR maximum; here it also meets the 30% cap, so w_cap = w*
    assert w == backtest.best_allocation(t, None) == 0.1 and row["w_cap"] == backtest.best_allocation(t, 0.30) == w
    # the AQR twins against the cash frontier passed in; this sleeve's CAGR is above every cash mix's, so it has
    # no divested twin
    assert row["d_cagr_vs_dd_matched_cash_bp"] == pytest.approx(1e4 * backtest.dd_matched_cash_gap(t, cash_table, w))
    assert row["cagr_star"] > cash_table["cagr"].max() and np.isnan(row["dd_gap_vs_cagr_matched_cash"])
    # attribution: crash-bin and other years add up to the mean full-year difference; the crash years pay
    a = scenarios.full_year_returns(nav[[0.0, w]])
    assert row["attr_crash_bp"] + row["attr_noncrash_bp"] == pytest.approx(1e4 * (a[w] - a[0.0]).mean(), rel=1e-12)
    assert row["attr_crash_bp"] > 0
    crash = [k for k in t.columns if k.startswith("crash:")]
    pay, speed = t.loc[w, crash] - t.loc[0.0, crash], pd.Series({k: metrics.CRASH_SPEED[k[6:]] for k in crash})
    assert row["crash_payoff_fast"] == pytest.approx(pay[speed == "fast"].mean(), rel=1e-12)
    assert row["crash_payoff_slow"] == pytest.approx(pay[speed == "slow"].mean(), rel=1e-12)
    assert row["d_cagr_nocrash_bp"] < row["d_cagr_p50_bp"]  # without crash years the insurance earns less
    # D7 benchmarks: the S&P 500 TR is this core; 60/40 needs AGG and DGS10 in px
    assert row["bench_spxtr_cagr"] == pytest.approx(row["cagr_unhedged"], rel=1e-9)
    assert row["bench_spxtr_maxdd"] == pytest.approx(row["maxdd_unhedged"], rel=1e-9)
    assert np.isnan(row["bench_6040_cagr"]) and "60/40 benchmark not run" in row["notes"]
    # a cash frontier richer at every drawdown (CAGR 10 x |max DD|, up to 200% at -20%): w* > 0, yet holding less
    # equity beats it, and its divested twin at the same CAGR falls cagr / 10, less than the hedged mix
    rich = pd.DataFrame({"cagr": [2.0, 0.0], "max_dd": [-0.2, 0.0]}, index=[0.0, 1.0])
    row = catalog.evaluate(TOY, core, _dominant(core), cal, mkt, cash_table=rich, bootstrap_paths=300)
    assert row["w_star"] > 0 and not row["beats_deleverage"]
    assert row["d_cagr_vs_dd_matched_cash_bp"] == pytest.approx(1e4 * (row["cagr_star"] + 10 * row["maxdd_star"]))
    assert row["dd_gap_vs_cagr_matched_cash"] == pytest.approx(row["maxdd_star"] + row["cagr_star"] / 10, rel=1e-12)
    assert row["dd_gap_vs_cagr_matched_cash"] < 0
    assert row["dd_gap_vs_cagr_matched_cash"] == pytest.approx(
        backtest.cagr_matched_cash_dd(row["_table"], rich, row["w_star"])["maxdd_gap"], rel=1e-12)


def test_evaluate_scores_at_the_kelly_weight_and_compares_twins_at_the_cap():
    # every candidate is judged at its uncapped CAGR maximum, whether or not it can meet the drawdown cap
    mkt, core = _history()
    cal = catalog.rebalance_calendar(mkt.index, "annual")
    cash = pd.DataFrame({"cagr": [0.1, 0.0], "max_dd": [-0.6, 0.0]}, index=[0.0, 1.0])
    # a levered equity sleeve: more of it raises CAGR and deepens the drawdown, so a 56% cap stops it at 2%
    row = catalog.evaluate(TOY, core, 2 * core, cal, mkt, dd_cap=0.56, cash_table=cash, bootstrap_paths=200)
    t = row["_table"]
    assert (row["w_star"], row["w_cap"]) == (0.1, 0.02) and t.at[0.05, "max_dd"] < -0.56 < t.at[0.02, "max_dd"]
    assert row["d_cagr_bp"] == pytest.approx(1e4 * (t.at[0.1, "cagr"] - t.at[0.0, "cagr"]), rel=1e-12)
    assert row["d_cagr_vs_dd_matched_cash_bp"] == pytest.approx(1e4 * backtest.dd_matched_cash_gap(t, cash, 0.02))
    assert row["dd_gap_vs_cagr_matched_cash"] == pytest.approx(backtest.cagr_matched_cash_dd(t, cash, 0.02)["maxdd_gap"])
    # no weight meets a 5% cap: w_cap is n/a and the twins fall back to w*
    row = catalog.evaluate(TOY, core, _dominant(core), cal, mkt, dd_cap=0.05, cash_table=cash, bootstrap_paths=200)
    t = row["_table"]
    assert row["w_star"] == 0.1 and np.isnan(row["w_cap"]) and "w_cap n/a" in row["notes"]
    assert row["d_cagr_vs_dd_matched_cash_bp"] == pytest.approx(1e4 * backtest.dd_matched_cash_gap(t, cash, 0.1))
    assert row["verdict"] == "cost-effective (robust)"


def test_evaluate_sixty_forty_benchmark_from_px():
    mkt, core = _history("2000-01-03", "2012-12-31")
    agg = pd.Series(1.0001 ** np.arange(len(mkt)), mkt.index).loc["2003-09-29":]  # AGG from its launch
    px = pd.DataFrame({"AGG": agg, "DGS10": 4.0}, index=mkt.index)
    row = catalog.evaluate(TOY, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt, px=px,
                           bootstrap_paths=200)
    bond = sleeves.bond_proxy(px["DGS10"])[:"2003-09-29"].combine_first(agg.pct_change())
    nav, _ = backtest.simulate(core, bond, 0.4, catalog.rebalance_calendar(mkt.index, "annual"))
    assert row["bench_6040_cagr"] == pytest.approx(metrics.cagr(nav[0.4]), rel=1e-12)
    assert row["bench_6040_maxdd"] == pytest.approx(metrics.max_drawdown(nav[0.4]), rel=1e-12)
    assert "60/40" not in row["notes"]
    # a sample starting after AGG's launch needs no DGS10
    late, cal = core.loc["2004-01-02":], catalog.rebalance_calendar(core.loc["2004-01-02":].index, "annual")
    row = catalog.evaluate(TOY, late, _dominant(late), cal, mkt, px=px[["AGG"]], bootstrap_paths=200)
    nav, _ = backtest.simulate(late, agg.pct_change().loc["2004-01-02":].fillna(0.0), 0.4, cal)
    assert row["bench_6040_cagr"] == pytest.approx(metrics.cagr(nav[0.4]), rel=1e-12) and "60/40" not in row["notes"]
    # an AGG column with no price (a failed fetch framed with the others) skips the benchmark instead of crashing
    empty = px.assign(AGG=np.nan)
    row = catalog.evaluate(TOY, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt, px=empty,
                           bootstrap_paths=200)
    assert np.isnan(row["bench_6040_cagr"]) and "60/40 benchmark not run" in row["notes"]
    with pytest.raises(ValueError, match="AGG"):
        catalog.build_sleeve(CANDIDATES["sixty_forty"], mkt, empty, core, 1.0)


def test_evaluate_na_checks_do_not_fail_a_dominant_sleeve():
    # no S&P 500 year at or below -20% after 2008: that dice regime has no sample year, so the check is n/a
    mkt, core = _history("2009-10-16", "2025-12-31")
    row = catalog.evaluate(TOY, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt,
                           bootstrap_paths=200)
    assert "crash_edge_-20% n/a" in row["robustness"] and "crash_edge_-10% pass" in row["robustness"]
    assert row["verdict"] == "cost-effective (robust)"
    # just over 10 calendar years on a ~252-day calendar: fewer rows than a 2,520-day window, so rolling10 is n/a
    mkt, core = _history("2009-10-01", "2019-10-08")
    keep = np.arange(len(core)) % 25 != 24
    mkt, core = mkt[keep], core[keep]
    row = catalog.evaluate(TOY, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt,
                           bootstrap_paths=200)
    assert row["years"] > 10 and len(row["_nav"]) < 2520 and np.isnan(row["rolling10_win_rate"])
    assert "rolling10 n/a" in row["robustness"] and row["verdict"] == "cost-effective (robust)"


def test_evaluate_base_case_needs_a_positive_bootstrap_median(monkeypatch):
    # the live trend row: dCAGR > 0 in history but a negative book-bootstrap median -> not cost-effective, no checks
    mkt, core = _history()

    def book(c, h, w, n_paths, seed, mode):
        return pd.DataFrame({"d_cagr_p50": [0.0, -1e-4 if mode == "joint" else -2e-4]}, index=[0.0, 1.0])

    monkeypatch.setattr(scenarios, "book_bootstrap", book)
    row = catalog.evaluate(TOY, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt,
                           bootstrap_paths=200)
    assert row["d_cagr_bp"] > 0 and row["d_cagr_p50_bp"] == pytest.approx(-1.0)
    assert row["d_cagr_nocrash_bp"] == pytest.approx(-2.0)
    assert row["verdict"] == "not cost-effective" and row["robustness"] == "not run"


def test_evaluate_fragile_when_a_robustness_check_fails():
    mkt, core = _history()
    sleeve = _dominant(core).where(core.index < "2015-01-01", -0.002)  # bleeds 40% a year from 2015 on
    row = catalog.evaluate(TOY, core, sleeve, catalog.rebalance_calendar(mkt.index, "annual"), mkt, bootstrap_paths=300)
    assert row["verdict"] == "cost-effective (fragile)" and "since2015 fail" in row["robustness"]


def test_evaluate_cash_sleeve_is_not_cost_effective():
    mkt, core = _history(sigma=0.005)  # a calm core: w = 0 already meets the 30% drawdown cap
    sleeve, cal = catalog.build_sleeve(CANDIDATES["tbills"], mkt, None, core, 1.0)
    row = catalog.evaluate(CANDIDATES["tbills"], core, sleeve, cal, mkt, bootstrap_paths=300)
    assert row["maxdd_unhedged"] > -0.30 and row["w_star"] == 0 and row["d_cagr_bp"] == 0
    assert row["verdict"] == "not cost-effective" and not row["beats_deleverage"]
    assert np.isnan(row["p_beat_10y"]) and np.isnan(row["rolling10_win_rate"])  # nothing tested at w* = 0: n/a
    assert "smallest grid weight" in row["notes"] and row["calm_year_bleed"] < 0  # scored at w = 0.05


def test_evaluate_short_or_crash_free_history_is_insufficient(monkeypatch):
    sleeve, cal = catalog.build_sleeve(CANDIDATES["tbills"], MKT, None, CORE, 1.0)
    row = catalog.evaluate(CANDIDATES["tbills"], CORE, sleeve, cal, MKT, bootstrap_paths=200)
    assert KEYS <= set(row) and row["years"] < 10 and row["verdict"] == "insufficient history"
    mkt, core = _history()
    monkeypatch.setattr(metrics, "CRASHES", {})  # 30 years, but no crash window inside them
    row = catalog.evaluate(TOY, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt,
                           bootstrap_paths=200)
    assert row["n_crashes_in_sample"] == 0 and row["verdict"] == "insufficient history"


def test_evaluate_relever_is_labelled_and_raises_exposure():
    mkt, core = _history()
    toy = replace(TOY, w_grid=(0.02, 0.1, 0.5, 1.0))
    row = catalog.evaluate(toy, core, _dominant(core), catalog.rebalance_calendar(mkt.index, "annual"), mkt,
                           funding="relever", bootstrap_paths=200)
    assert row["funding"] == "relever" and "levered" in row["notes"]
    assert "grid weights above 1/3 dropped (0.5, 1)" in row["notes"]
    assert row["w_star"] > 0 and row["gross_exposure"] == pytest.approx(1 + row["w_star"])
    assert row["_table"].index.max() <= 1 / 3  # L = 1 / (1 - w) <= 1.5
    # the w = 0.1 column is the core levered to 1 / 0.9 plus 10% of the sleeve: equity stays at 100%
    lev = catalog.portfolio.lever(core, mkt["rf"], 1 / 0.9)
    nav, _ = backtest.simulate(lev, _dominant(core), 0.1, catalog.rebalance_calendar(mkt.index, "annual"))
    np.testing.assert_allclose(row["_nav"][0.1], nav[0.1], rtol=1e-12)
    with pytest.raises(ValueError, match="funding"):
        catalog.evaluate(TOY, core, _dominant(core), mkt.index[:0], mkt, funding="overlay")
    with pytest.raises(ValueError, match="relever"):  # sixty_forty's (0.4,): no weight left to test
        catalog.evaluate(replace(TOY, w_grid=(0.4,)), core, _dominant(core), mkt.index[:0], mkt, funding="relever")


def test_evaluate_reruns_option_sleeves_for_the_robustness_checks(monkeypatch):
    # build_sleeve is the seam: record the re-run overrides; a variant that bleeds fails its check
    mkt, core = _history()
    opt = replace(TOY, kind="option", spec=CANDIDATES["universa_model"].spec, model_priced=True)
    cal = catalog.rebalance_calendar(mkt.index, "annual")
    calls, pxs, bleeds = [], [], {}
    px = pd.DataFrame(index=mkt.index)  # no AGG: the 60/40 benchmark is skipped, the frame still reaches the rebuilds

    def fake_build(c, m, px, core_r, beta, *, pricing_kw=None, chain=None):
        calls.append(pricing_kw)
        pxs.append(px)
        bad = any(pricing_kw.get(k) == v for k, v in bleeds.items())
        return (pd.Series(-0.002, core_r.index) if bad else _dominant(core_r)), cal

    monkeypatch.setattr(catalog, "build_sleeve", fake_build)
    monkeypatch.setattr(catalog, "shock_pnl", lambda c, m, w, **kw: dict(zip(catalog.SHOCK_VIX, (-0.12, -0.08, -0.03))))

    def run(c: Candidate = opt) -> dict:
        return catalog.evaluate(c, core, _dominant(core), cal, mkt, bootstrap_paths=200, pricing_kw={"m": 0.5}, px=px)

    row = run()
    assert calls == [{"m": 0.5, "offset_bdays": 5}, {"m": 0.5, "offset_bdays": 10}, {"m": 0.5, "offset_bdays": 15},
                     {"m": 0.5, "h": 0.0}, {"m": 0.5, "z_cap": 3.0}]
    assert all(p is px for p in pxs)
    assert row["verdict"] == "cost-effective (robust)" and "roll_offsets pass, mid_fills pass, wing_z_cap pass" in row[
        "robustness"]
    assert row["shock20_pnl"] == -0.08 and "-12.0% / -8.0% / -3.0% with VIX 40 / 55 / 70" in row["notes"]
    bleeds["h"] = 0.0  # mid fills turn the gain into a loss
    row = run()
    assert row["verdict"] == "cost-effective (fragile)" and "mid_fills fail" in row["robustness"]
    # a protective put holds its puts to expiry: an offset would change coverage and tenor, so no sweep, said so
    calls.clear()
    bleeds.clear()
    row = run(replace(opt, spec=CANDIDATES["prot_put"].spec))
    assert calls == [{"m": 0.5, "h": 0.0}, {"m": 0.5, "z_cap": 3.0}] and "roll_offsets n/a" in row["robustness"]
    assert "roll-offset sweep skipped" in row["notes"] and row["verdict"] == "cost-effective (robust)"
    with pytest.raises(ValueError, match="DGS10"):  # the TAIL proxy's rebuilds need px: refused before the frontier
        catalog.evaluate(replace(TOY, kind="composite", spec=catalog.TAIL_SPEC, source="DGS10", model_priced=True),
                         core, _dominant(core), cal, mkt)


def test_evaluate_option_sleeve_scores_on_the_fixture():
    c = CANDIDATES["universa_model"]
    s, cal = catalog.build_sleeve(c, MKT, None, CORE, 1.0)
    row = catalog.evaluate(c, CORE, s, cal, MKT, bootstrap_paths=200)
    assert row["verdict"] == "insufficient history" and row["robustness"] == "not run"
    w = row["w_scores"]
    # 12 tranches a year, each 6% of the sleeve at its last year-end (the harvest keeps 2019 funded): about the
    # full 12 x 6% budget a year, in portfolio terms
    assert 0.9 * 0.72 * w < row["avg_premium_yr"] < 1.1 * 0.72 * w and row["avg_coverage"] > 0
    # notional / portfolio NAV day by day: the sleeve's daily coverage x its share, not the last roll's ratio
    # carried forward while the sleeve's share drifts
    r = s.copy()
    r.attrs = {}
    share = catalog._sleeve_share(row["_nav"][w], r, cal, w)
    assert row["avg_coverage"] == pytest.approx((s.attrs["coverage"] * share).mean(), rel=1e-12)
    assert np.isfinite(row["shock20_pnl"]) and "-20% month at w = 0.0333" in row["notes"]
    assert "calm-year bleed" in row["notes"] and "at w = 0.0333" in row["notes"]
    assert np.isfinite([row[k] for k in ("critic_sleeve_cagr", "critic_beta", "d_beta", "calm_year_bleed")]).all()


def test_evaluate_scores_the_put_preset_at_its_own_weight(monkeypatch):
    # w* = 0 scores the sleeve at the smallest grid weight, but the shock and the 2.6% bleed warning are Universa's
    # 3.33% figures; relever keeps the equity at 100% through the shock
    mkt, core = _history()
    opt = replace(TOY, kind="option", spec=CANDIDATES["universa_model"].spec, w_grid=(0.01, catalog.W_UNIVERSA),
                  model_priced=True)
    cal = catalog.rebalance_calendar(mkt.index, "annual")
    seen = []

    def shock(c, m, w, equity=None, **kw):
        seen.append((w, equity))
        return dict(zip(catalog.SHOCK_VIX, (-0.12, -0.08, -0.03)))

    monkeypatch.setattr(catalog, "shock_pnl", shock)
    for kept, warned in ((0.1, True), (0.8, False)):  # the sleeve keeps 10% / 80% of itself a year: ~3.3% / ~1%
        sleeve = pd.Series(kept ** (1 / 261) - 1, core.index)
        row = catalog.evaluate(opt, core, sleeve, cal, mkt, bootstrap_paths=200)
        assert row["w_star"] == 0 and row["w_scores"] == 0.01 and "-20% month at w = 0.0333" in row["notes"]
        assert ("above the 2.6% warning" in row["notes"]) is warned and "/yr of NAV at w = 0.0333" in row["notes"]
    catalog.evaluate(opt, core, sleeve, cal, mkt, bootstrap_paths=200, funding="relever")
    assert seen == [(0.0333, None), (0.0333, None), (0.0333, 1.0)]


def test_index_candidates_cut_at_the_cll_splice_and_flag_launch_and_bad_marks():
    mkt, core = _history("2008-01-01", "2020-06-30")
    step = np.where(mkt.index == pd.Timestamp("2009-10-16"), 0.802, 1.0).cumprod()  # the file's -19.8% splice
    mkt["cll"] = 800 * (1 + core / 2).cumprod() * step
    mkt["pput"] = 900 * (1 + core / 2).cumprod()
    cll, cal = catalog.build_sleeve(CANDIDATES["cll_idx"], mkt, None, core, 1.0)
    assert cll.index[0] == pd.Timestamp("2009-10-16") and cll.iloc[0] == 0 and cll.min() > -0.1
    row = catalog.evaluate(CANDIDATES["cll_idx"], core, cll, cal, mkt, bootstrap_paths=200)
    assert row["sample_start"] == pd.Timestamp("2009-10-16") and "2020-02-28, 2020-03-02" in row["notes"]
    assert "launch" not in row["notes"]  # CLL launched 2008-09-04, before the splice cut
    pput, cal = catalog.build_sleeve(CANDIDATES["pput_idx"], mkt, None, core, 1.0)
    row = catalog.evaluate(CANDIDATES["pput_idx"], core, pput, cal, mkt, bootstrap_paths=200)
    assert "back-test levels before the 2015-08-03 launch" in row["notes"] and "bad Cboe marks" not in row["notes"]


def test_robustness_helpers_on_hand_built_years():
    ann = pd.DataFrame({"u": [0.05, -0.37, 0.26], "h": [0.03, 0.10, 0.20]}, index=[2007, 2008, 2009])
    assert catalog._ex_best_crash_year(ann) is False  # without 2008: 1.03 x 1.20 < 1.05 x 1.26
    ann["h"] = [0.06, 0.10, 0.27]
    assert catalog._ex_best_crash_year(ann) is True
    assert catalog._ex_best_crash_year(ann.set_axis([2011, 2012, 2013])) is None  # neither 2008 nor 2020
    # both crash years: 2020 pays the hedge most (+50 points over u, 2008 +7), so 2020 goes and the rest loses
    # (0.70 x 1.05 x 1.20 < 0.63 x 1.15 x 1.30); dropping 2008 instead would keep 2020's gain and pass
    both = pd.DataFrame({"u": [-0.37, 0.15, -0.10, 0.30], "h": [-0.30, 0.05, 0.40, 0.20]}, index=[2008, 2012, 2020, 2021])
    g = (1 + both.drop(2008)).prod()
    assert catalog._ex_best_crash_year(both) is False and g["h"] > g["u"]
    # the dice on four hand-made years, one per bin at a -10% crash edge (each 1/4): growth = the 4th root of
    # the product
    spx = pd.Series({2001: -0.30, 2002: -0.05, 2003: 0.10, 2004: 0.20})
    dates = pd.to_datetime(["2000-12-29", "2001-12-31", "2002-12-31", "2003-12-31", "2004-12-31"])

    def pair(h: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"u": np.cumprod([1, 0.70, 0.95, 1.10, 1.20]), "h": np.cumprod([1, *np.add(1, h)])}, dates)

    assert catalog._dice_beats(pair([-0.10, -0.06, 0.08, 0.18]), spx, -0.10) is True  # 1.078 > 0.878
    assert catalog._dice_beats(pair([-0.35, -0.06, 0.08, 0.18]), spx, -0.10) is False  # 0.779 < 0.878
    assert catalog._dice_beats(pair([-0.10, -0.06, 0.08, 0.18]).iloc[:-1], spx, -0.10) is None  # 2004's bin: no year
    # the edge moves a -12% year: at -10% it shares the crash bin with 2001 and the hedge's +70% / -70% pair is
    # pooled to its arithmetic mean (0 > the core's -21%); at -15% each year is its own bin and the pair compounds
    # (1.7 x 0.3 < 0.70 x 0.88)
    spx[2002] = -0.12
    u = np.cumprod([1, 0.70, 0.88, 1.10, 1.20])
    edge = pd.DataFrame({"u": u, "h": np.cumprod([1, 1.7, 0.3, 1.10, 1.20])}, dates)
    assert catalog._dice_beats(edge, spx, -0.10) is True and catalog._dice_beats(edge, spx, -0.15) is False
    sold = {k for k, c in CANDIDATES.items() if c.spec is not None and catalog._sold_before_expiry(c.spec)}
    assert sold == {"universa_model", "universa_2023", "dao_ch9", "put_spread", "tail_proxy"}  # not prot_put, collar


def test_sleeve_share_drifts_between_rebalances():
    idx = pd.bdate_range("2020-01-01", periods=10)
    sleeve, core = pd.Series(0.01, idx), pd.Series(0.0, idx)  # the sleeve grows 1% a day, the core is flat
    reb = idx[5:6]
    nav, _ = backtest.simulate(core, sleeve, 0.1, reb, backtest.Policy(tc_core=0.0, tc_sleeve=0.0))
    g = 1.01 ** np.r_[np.arange(5), np.arange(5)]  # growth since the start, then since the day-5 reset
    np.testing.assert_allclose(catalog._sleeve_share(nav[0.1], sleeve, reb, 0.1), 0.1 * g / (0.9 + 0.1 * g), rtol=1e-12)


def test_calm_year_bleed_hand_values():
    dates = pd.to_datetime(["2009-12-31", "2010-12-31", "2011-12-30", "2012-12-31", "2013-12-31", "2014-12-31"])
    nav_0 = pd.Series(np.cumprod([1, 1.10, 0.80, 1.05, 1.30, 1.20]), dates)
    nav_w = pd.Series(np.cumprod([1, 1.08, 0.95, 1.10, 1.28, 1.18]), dates)
    spx = pd.Series({2010: 0.15, 2011: -0.20, 2012: -0.15, 2013: 0.32, 2014: 0.14})  # -15% is a crash year
    # calm years 2010, 2013, 2014 each cost 2 points; the 2011 (+15) and 2012 (+5) gains are crash years
    assert catalog.calm_year_bleed(nav_w, nav_0, spx) == pytest.approx(-0.02, abs=1e-12)


def test_critic_view_beta_of_twice_the_market():
    m = pd.Series(np.random.default_rng(4).normal(0.0003, 0.01, 600), pd.bdate_range("2015-01-01", periods=600))
    m.iloc[0] = 0.0
    v = catalog.critic_view(2 * m, m, pd.Series(0.0, m.index))
    assert set(v) == {"cagr", "ann_vol", "max_dd", "sharpe", "alpha", "beta"}
    assert v["beta"] == pytest.approx(2, abs=1e-12) and v["alpha"] == pytest.approx(0, abs=1e-12)
    nav = metrics.nav_from_returns(2 * m)
    assert v["cagr"] == pytest.approx(metrics.cagr(nav)) and v["max_dd"] == pytest.approx(metrics.max_drawdown(nav))
    assert v["sharpe"] == pytest.approx((2 * m).iloc[1:].mean() / (2 * m).iloc[1:].std() * np.sqrt(252), rel=1e-12)


def test_critic_view_works_on_excess_returns_over_cash():
    idx = pd.date_range("2015-01-01", periods=600)  # calendar days: the same cash return every day
    m = pd.Series(np.random.default_rng(5).normal(0.0003, 0.01, 600), idx)
    m.iloc[0] = 0.0
    rf, cash = pd.Series(0.05, idx), 0.05 / 360
    v = catalog.critic_view(2 * m, m, rf)  # unfinanced leverage: its Jensen alpha is the cash rate it skips
    assert v["beta"] == pytest.approx(2, rel=1e-9) and v["alpha"] == pytest.approx(252 * cash, rel=1e-9)
    v = catalog.critic_view(2 * m - cash, m, rf)  # financed at the cash rate: 2 x the excess return, no alpha
    assert v["beta"] == pytest.approx(2, rel=1e-9) and v["alpha"] == pytest.approx(0, abs=1e-12)
    ex = m.iloc[1:] - cash
    assert v["sharpe"] == pytest.approx(ex.mean() / ex.std() * np.sqrt(252), rel=1e-9)


def test_rank_orders_tiers_then_dcagr_and_drops_private_keys():
    rows = [{"candidate": k, "verdict": v, "d_cagr_bp": d, "_table": None, "_nav": None} for k, v, d in (
        ("a", "not cost-effective", 5.0), ("b", "cost-effective (fragile)", 10.0),
        ("c", "cost-effective (robust)", 1.0), ("d", "insufficient history", 50.0),
        ("e", "cost-effective (robust)", 3.0), ("f", "not cost-effective", 9.0))]
    out = catalog.rank(rows)
    assert out["candidate"].tolist() == ["e", "c", "b", "f", "a", "d"] and list(out.columns) == ["candidate", "verdict",
                                                                                                   "d_cagr_bp"]
    assert catalog.rank([]).empty  # the Candidates page with nothing selected


def test_universa_checks_known_answer_table():
    idx = pd.bdate_range("2007-06-01", "2020-06-30")
    t = np.arange(len(idx))
    nav, spx = pd.Series(1.0003**t, idx), pd.Series(1.0002**t, idx)
    out = catalog.universa_checks(nav, spx)
    assert len(out) == 4 and {"check", "universa_reported", "ours", "gap"} <= set(out.columns)
    assert (out["label"] == "self-reported, hypothetical, window-dependent").all()  # D15 provenance
    assert out["gap"].notna().all() and np.isfinite(out["gap"].astype(float)).all()
    assert out["universa_reported"].tolist() == [0.004, -0.063, -0.0169, 0.099]
    windows = {0: ("2020-02-28", "2020-03-31"), 1: ("2019-12-31", "2020-03-31"), 3: ("2008-02-29", "2008-12-31")}
    for i, (a, b) in windows.items():  # each window from the close before it, as reported
        ret = nav[b] / nav[a] - 1
        assert out.at[i, "ours"] == pytest.approx(ret) and out.at[i, "gap"] == pytest.approx(
            ret - out.at[i, "universa_reported"])
        assert out.at[i, "spx_ours"] == pytest.approx(spx[b] / spx[a] - 1)
    ret = {k: x["2019-12-31"] / x["2018-12-31"] - 1 for k, x in (("nav", nav), ("spx", spx))}
    drag = ret["nav"] - (1 - catalog.W_UNIVERSA) * ret["spx"]
    assert out.at[2, "ours"] == pytest.approx(drag)  # the sleeve's 2019 drag in portfolio points
    short = catalog.universa_checks(nav[:"2020-03-30"], spx)  # ends before Q1 2020 does
    assert np.isnan(short.at[1, "ours"]) and np.isfinite(short.at[2, "ours"])


def test_shock_pnl_formula_over_the_last_roll_month(monkeypatch):
    # option_sleeve is the seam: a sleeve earning 1% a day pins the window (rows i0 + 1 .. i1) and the weights
    idx = pd.bdate_range("2001-03-01", periods=300)
    i0, i1 = idx.get_indexer(sleeves.roll_dates(idx, when="month_start")[-2:])
    assert (idx[i0], idx[i1]) == (pd.Timestamp("2002-03-01"), pd.Timestamp("2002-04-01"))  # clear of the harvest
    panels = []

    def fake(panel, spec, **kw):
        panels.append(panel)
        return pd.DataFrame({"ret": 0.01}, panel.index)

    monkeypatch.setattr(sleeves, "option_sleeve", fake)
    w, month = catalog.W_UNIVERSA, 1.01 ** (i1 - i0) - 1
    pnl = catalog.shock_pnl(CANDIDATES["universa_model"], MKT, w)
    assert pnl == pytest.approx(dict.fromkeys(catalog.SHOCK_VIX, (1 - w) * -0.20 + w * month), rel=1e-12)
    pnl = catalog.shock_pnl(CANDIDATES["universa_model"], MKT, w, equity=1.0)  # relever: all the equity falls
    assert pnl[55.0] == pytest.approx(-0.20 + w * month, rel=1e-12)
    # the calm state is the last year's medians (2019's SKEW 123, not the two years' 126), spot the last close;
    # by the shock's end S is 20% lower and every pillar at the VIX reached
    calm, p = MKT.iloc[-catalog.CALM_ROWS:].median(), panels[0]
    assert (p["skew"] == calm["skew"]).all() and (p.loc[: idx[i0], ["vix", "vix3m"]] == calm[["vix", "vix3m"]]).all().all()
    assert p["spx"].iloc[0] == MKT["spx"].iloc[-1] and p["spx"].iloc[i1] == pytest.approx(0.8 * MKT["spx"].iloc[-1])
    assert (p.loc[idx[i1] :, ["vix9d", "vix", "vix6m"]] == 40.0).all().all() and calm["skew"] < MKT["skew"].median()


def test_shock_pnl_at_the_addendum_calm_day():
    # the addendum's figure (section 3): -11.5% / -8.3% / -3.2% at 3.33% from the calm-day premium of the
    # 2026-10-08 XSP chain (gap-3-q9 A: XSP 776.54, VIX 15.41, VIX9D 12.21, VIX3M 18.08, VIX6M 20.04, SKEW 149.19,
    # 3-month bill 4.043%). A year of those inputs after a 1990s-like stretch (SKEW 120): only the last year counts
    calm = pd.DataFrame({"spx": 7765.4, "vix9d": 12.21, "vix": 15.41, "vix3m": 18.08, "vix6m": 20.04, "skew": 149.19,
                         "rf": 0.04043, "q": 0.012}, index=pd.bdate_range(end="2026-10-08", periods=catalog.CALM_ROWS))
    old = pd.DataFrame(calm.iloc[0].to_dict() | {"skew": 120.0}, index=pd.bdate_range(end="2025-10-08", periods=2000))
    pnl = catalog.shock_pnl(CANDIDATES["universa_model"], pd.concat([old, calm]), catalog.W_UNIVERSA)
    assert pnl == catalog.shock_pnl(CANDIDATES["universa_model"], calm, catalog.W_UNIVERSA)
    for v, want in zip(catalog.SHOCK_VIX, (-0.115, -0.083, -0.032)):
        assert pnl[v] == pytest.approx(want, abs=0.03), v  # within 3 points: about -13.8% / -8.8% / -3.2%


def test_shock_pnl_of_the_universa_preset():
    # the real pricer on the fixture's calm state (2019: VIX 14.9, SKEW 123): +65% / +155% / +260% for the
    # portfolio. At SKEW 123 the model's far wing is cheap, so the 0.20% budget buys many times the notional it
    # buys at today's SKEW 149 (test above); the same SKEW sensitivity sizes the sleeve inside every backtest.
    # Pinned so that any pricer or sleeve change shows here.
    c = CANDIDATES["universa_model"]
    pnl = catalog.shock_pnl(c, MKT, catalog.W_UNIVERSA)
    assert pnl == pytest.approx({40.0: 0.6496, 55.0: 1.5518, 70.0: 2.6029}, rel=1e-3)
    for key in ("prot_put", "collar", "tail_proxy"):  # every put preset (addendum section 3)
        assert np.isfinite(list(catalog.shock_pnl(CANDIDATES[key], MKT, 0.5).values())).all(), key
    # dao_ch9 spends all its cash each roll, so the month-old put sold at a zero bid wipes it out and it restarts
    # from unit capital on the roll: the month compounds from there (a NAV ratio would be 0 / 0)
    assert np.isfinite(list(catalog.shock_pnl(CANDIDATES["dao_ch9"], MKT, 0.005).values())).all()
    with pytest.raises(ValueError, match="spec"):
        catalog.shock_pnl(CANDIDATES["tbills"], MKT, 0.1)

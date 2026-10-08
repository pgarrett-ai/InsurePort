"""Tests for insureport.backtest: the two-asset engine vectorized over the allocation grid
(w = 0 / w = 1 reproduce the core / sleeve after date alignment, returns and rebalance dates
on dates the other series lacks carried to the next common date, buy-and-hold without
rebalancing, a scalar w as a one-point grid, the relative band firing on the exact crossing
day and only there (not at the boundary itself; calendar dates still fire with a band set),
transaction costs on the rebalance flows with turnover and cost drag), the Kelly check (a
+100% / -50% sleeve is best at w = 0.5 with growth sqrt(1.125) per day), the frontier table
wired to metrics.py (crash windows incl. ones touching the sample edges, calmar NaN at zero
drawdown), best_allocation under a drawdown cap, the drawdown-matched and CAGR-matched
comparisons with the T-bill frontier (raw grid values find their rounded labels; the
deleverage twin on a humped frontier), rolling excess CAGR, period compounding and the
9,000-day x 41-point speed budget.

Why: every frontier, verdict and robustness panel runs through this engine, so each rule is
pinned to a hand-checkable number. Small synthetic series with explicit dates, no I/O.
"""

import time

import numpy as np
import pandas as pd
import pytest

from insureport import backtest, metrics
from insureport.backtest import Policy

NO_COST = Policy(tc_core=0.0, tc_sleeve=0.0)
NO_DATES = pd.DatetimeIndex([])


def _daily(values: list[float] | np.ndarray, start: str = "2020-01-01") -> pd.Series:
    """Daily returns on consecutive calendar days from `start` (the first is the ignored placeholder)."""
    return pd.Series(values, pd.date_range(start, periods=len(values)), dtype=float)


def test_w0_is_the_core_and_w1_the_sleeve():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=300)
    core = pd.Series(rng.normal(0.0004, 0.01, 300), idx)
    sleeve = pd.Series(rng.normal(-0.001, 0.05, 300), idx).iloc[3:]  # starts later ...
    sleeve.iloc[10] = np.nan  # ... and has a hole: the core's return there compounds into the next date
    common = sleeve.dropna().index
    w = np.array([0.0, 0.3, 1.0])
    nav, stats = backtest.simulate(core, sleeve, w, idx[::21], Policy(band=0.2))  # default costs too
    assert nav.index.equals(common) and list(nav.columns) == [0.0, 0.3, 1.0]
    for x, r in ((0.0, core), (1.0, sleeve)):  # each standalone NAV, seen on the common dates
        level = metrics.nav_from_returns(r.dropna()).reindex(common)
        np.testing.assert_allclose(nav[x], level / level.iloc[0], rtol=0, atol=1e-12)
    assert stats.loc[[0.0, 1.0]].eq(0).all().all()  # nothing to trade at the ends of the grid
    assert stats.loc[0.3, "turnover"] > 0 and stats.loc[0.3, "cost_drag"] > 0
    assert w.tolist() == [0.0, 0.3, 1.0]  # the caller's grid is not mutated
    with pytest.raises(ValueError, match="share 1 dates"):
        backtest.simulate(core.iloc[:4], sleeve, w, idx)  # only the sleeve's first date is shared


def test_returns_and_rebalances_on_dates_the_other_lacks_carry_to_the_next_common_date():
    # The core misses day 3 (a missing print: its day-4 return spans both days). The sleeve, on
    # the full calendar but missing day 1, spends its whole premium on day 3, the roll date, and
    # doubles on day 5. Nothing is lost: the core's +10% on day 1 lands on day 2, the sleeve's
    # -100% on day 4, and the day-3 refill moves to day 4, so w = 0.05 earns 5% on day 5.
    core = _daily([0, 0.1, 0, 0, 0, 0]).iloc[[0, 1, 2, 4, 5]]
    sleeve = _daily([0, 0, 0, -1.0, 0, 1.0]).iloc[[0, 2, 3, 4, 5]]
    nav, _ = backtest.simulate(core, sleeve, np.array([0.0, 0.05, 1.0]), pd.DatetimeIndex(["2020-01-04"]), NO_COST)
    assert nav.index.equals(pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-05", "2020-01-06"]))
    want = [[1, 1, 1], [1.1, 0.95 * 1.1 + 0.05, 1], [1.1, 0.95 * 1.1, 0], [1.1, 0.95 * 1.1 * 1.05, 0]]
    np.testing.assert_allclose(nav, want, rtol=0, atol=1e-14)


def test_no_rebalance_no_band_is_buy_and_hold():
    rng = np.random.default_rng(3)
    core, sleeve = _daily(rng.normal(0.0005, 0.01, 60)), _daily(rng.normal(0, 0.03, 60))
    w = np.linspace(0, 1, 21)
    nav, stats = backtest.simulate(core, sleeve, w, NO_DATES)  # default costs, never charged
    C, S = metrics.nav_from_returns(core), metrics.nav_from_returns(sleeve)
    np.testing.assert_allclose(nav, np.outer(C, 1 - w) + np.outer(S, w), rtol=0, atol=1e-12)
    assert stats.eq(0).all().all()
    assert nav[0.3].equals(nav.iloc[:, 6])  # w[6] is 0.30000000000000004; the label is rounded to 0.3
    one, _ = backtest.simulate(core, sleeve, w[6], NO_DATES)  # a scalar w is a one-point grid
    assert one.columns.tolist() == [0.3] and one[0.3].equals(nav[0.3])


def test_band_fires_on_the_crossing_day_only():
    # Core flat; sleeve +10% a day for 5 days, then -10%, then flat. At w = 0.1 the sleeve share
    # s/v is 0.13992 after day 4 and 0.15179 after day 5: past w + 0.5 w = 0.15 on day 5. After
    # the reset to 0.1, day 6 takes it to 0.0909, inside the band. At w = 0.2 it peaks at
    # 0.28705 < 0.3 (the band is relative: an absolute 0.05 would fire), and w = 0 never fires.
    core, sleeve = _daily([0.0] * 9), _daily([0, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0, 0])
    w = np.array([0.0, 0.1, 0.2])
    costs = Policy(tc_core=0.001, tc_sleeve=0.001)  # a cost makes the trigger day visible in the NAV
    banded = Policy(band=0.5, tc_core=0.001, tc_sleeve=0.001)
    nav, stats = backtest.simulate(core, sleeve, w, NO_DATES, banded)
    hold, _ = backtest.simulate(core, sleeve, w, NO_DATES, costs)
    day5, day5_stats = backtest.simulate(core, sleeve, w, core.index[5:6], costs)  # a calendar rebalance on day 5
    assert (nav[0.1].iloc[:5] == hold[0.1].iloc[:5]).all() and nav[0.1].iloc[5] < hold[0.1].iloc[5]
    np.testing.assert_allclose(nav[0.1], day5[0.1], rtol=0, atol=1e-15)  # same path as one day-5 rebalance ...
    assert stats.loc[0.1].tolist() == pytest.approx(day5_stats.loc[0.1].tolist(), rel=1e-15)  # ... and same trades
    np.testing.assert_allclose(nav[[0.0, 0.2]], hold[[0.0, 0.2]], rtol=0, atol=1e-15)
    assert stats.loc[[0.0, 0.2]].eq(0).all().all() and stats.loc[0.1, "turnover"] > 0
    # with a band set a calendar date still fires: at w = 0.2 the band never does, day 7 must
    both, _ = backtest.simulate(core, sleeve, w, core.index[7:8], banded)
    cal7, _ = backtest.simulate(core, sleeve, w, core.index[7:8], costs)
    np.testing.assert_allclose(both[0.2], cal7[0.2], rtol=0, atol=1e-15)
    assert both[0.2].iloc[7] < hold[0.2].iloc[7]
    # the boundary itself does not fire: at w = 0.5 a +200% sleeve day puts s/v at 0.75, and
    # |0.75 - 0.5| equals band * w = 0.25 exactly in binary
    _, edge = backtest.simulate(_daily([0, 0, 0]), _daily([0, 2.0, 0]), np.array([0.5]), NO_DATES, Policy(band=0.5))
    assert edge.loc[0.5, "turnover"] == 0


def test_rebalance_cost_is_tc_times_the_flows():
    core, sleeve = _daily([0, 0.1, 0.1]), _daily([0, -0.5, 0])
    w = np.array([0.0, 0.2, 1.0])
    nav, stats = backtest.simulate(core, sleeve, w, core.index[1:2], Policy(tc_core=0.001, tc_sleeve=0.01))
    # w = 0.2 on day 1: c = 0.8 * 1.1 = 0.88, s = 0.2 * 0.5 = 0.1, v = 0.98; targets 0.784 / 0.196
    cost = 0.001 * abs(0.784 - 0.88) + 0.01 * abs(0.196 - 0.1)
    np.testing.assert_allclose(nav.iloc[1], [1.1, 0.98 - cost, 0.5], rtol=0, atol=1e-15)
    # day 2 (core +10%, sleeve flat) grows the post-cost value split 0.8 / 0.2
    assert nav.iloc[2, 1] == pytest.approx((0.98 - cost) * (0.8 * 1.1 + 0.2), rel=1e-14)
    years = 2 / 365.25
    assert stats.loc[0.2, "turnover"] == pytest.approx(0.096 / 0.98 / years, rel=1e-12)
    assert stats.loc[0.2, "cost_drag"] == pytest.approx(cost / 0.98 / years, rel=1e-12)
    assert stats.loc[[0.0, 1.0]].eq(0).all().all()


def test_wiped_out_sleeve_is_refilled_at_the_next_rebalance():
    core, sleeve = _daily([0, 0, 0, 0]), _daily([0, -1.0, 0.5, 0.5])  # a premium sleeve loses it all on day 1
    nav, stats = backtest.simulate(core, sleeve, np.array([0.1, 1.0]), core.index[2:3], NO_COST)
    # w = 0.1: 0.9 left after day 1, the sleeve refilled with 0.09 on day 2 earns 50% on day 3
    np.testing.assert_allclose(nav[0.1], [1.0, 0.9, 0.9, 0.9 + 0.045], rtol=0, atol=1e-15)
    assert nav[1.0].iloc[1:].eq(0).all() and stats.notna().all().all()  # a dead w = 1 stays at 0, no NaN stats


def test_kelly_alternating_sleeve():
    n = 200  # an even number of daily returns after the ignored first row
    core, sleeve = _daily([0.0] * (n + 1)), _daily([0.0] + [1.0, -0.5] * (n // 2))
    table, nav = backtest.frontier(core, sleeve, np.linspace(0, 1, 101), core.index, NO_COST)
    assert backtest.best_allocation(table, None) == 0.5
    assert nav[0.5].iloc[-1] ** (1 / n) == pytest.approx(np.sqrt(1.125), abs=1e-9)  # (1 + w)(1 - w / 2) per pair


def test_frontier_columns_and_crash_windows():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2018-10-01", "2020-06-30")  # three calendar years: their mean is not their median
    core = pd.Series(rng.normal(0.0005, 0.01, len(idx)), idx)
    covid = (idx > "2020-02-19") & (idx <= "2020-03-23")
    core[covid] = 0.0
    core["2020-03-16"] = -0.5  # the core NAV halves exactly over the COVID window
    sleeve = pd.Series(0.0001, idx)  # T-bill-like: never falls
    w = np.array([0.0, 0.5, 1.0])
    table, nav = backtest.frontier(core, sleeve, w, idx[::21])
    assert list(table.columns) == [
        "cagr", "ann_vol", "arith_mean", "vol_tax", "max_dd", "calmar", "tuw", "longest_dd_days",
        "worst_year", "turnover", "cost_drag", "crash:2020 COVID",
    ]  # fmt: skip
    assert table.index.tolist() == [0.0, 0.5, 1.0]
    assert table.loc[0.0, "crash:2020 COVID"] == pytest.approx(-0.5, abs=1e-12)
    assert table.loc[1.0, "crash:2020 COVID"] > 0
    nav2, stats = backtest.simulate(core, sleeve, w, idx[::21])
    pd.testing.assert_frame_equal(nav, nav2)
    annual = metrics.annual_returns(nav)
    expected = {
        "cagr": metrics.cagr(nav),
        "ann_vol": metrics.ann_vol(nav.pct_change().iloc[1:]),
        "arith_mean": annual.mean(),
        "vol_tax": metrics.vol_tax(nav)["tax"],
        "max_dd": metrics.max_drawdown(nav),
        "tuw": metrics.time_under_water(nav),
        "longest_dd_days": [metrics.longest_recovery_days(nav[x]) for x in w],
        "worst_year": annual.min(),
        "turnover": stats["turnover"],
        "cost_drag": stats["cost_drag"],
    }
    for col, want in expected.items():
        np.testing.assert_allclose(table[col], want, rtol=1e-12, err_msg=col)
    for x in w:  # the vectorized crash columns follow metrics.crash_returns' window rule
        assert table.loc[x, "crash:2020 COVID"] == pytest.approx(metrics.crash_returns(nav[x]).at["2020 COVID", "ret"])
    # off-calendar window ends: first date >= start (a Saturday), last date <= end (a Sunday)
    odd = {"weekend": ("2020-02-15", "2020-03-22")}
    t2, _ = backtest.frontier(core, sleeve, w, idx[::21], crashes=odd)
    want = nav.loc["2020-03-20", 0.5] / nav.loc["2020-02-17", 0.5] - 1
    assert t2.loc[0.5, "crash:weekend"] == pytest.approx(want, rel=1e-12)
    cut, _ = backtest.frontier(core, sleeve, w, idx[::21], crashes={"cut": ("2018-09-01", "2018-12-31")})
    assert "crash:cut" not in cut.columns  # a window cut by the sample start is omitted
    late, _ = backtest.frontier(core, sleeve, w, idx[::21], crashes={"late": ("2020-06-01", "2020-07-31")})
    assert "crash:late" not in late.columns  # ... and one cut by the sample end
    # a window starting on the first date and ending on the last is inside the sample
    edge, _ = backtest.frontier(core, sleeve, w, idx[::21], crashes={"edge": ("2018-10-01", "2020-06-30")})
    assert edge.loc[0.5, "crash:edge"] == pytest.approx(nav[0.5].iloc[-1] / nav[0.5].iloc[0] - 1, rel=1e-12)
    assert len(annual) == 3 and table.loc[0.0, "worst_year"] == annual[0.0].min()
    assert table.loc[0.0, "calmar"] == pytest.approx(table.loc[0.0, "cagr"] / -table.loc[0.0, "max_dd"])
    assert table.loc[1.0, "max_dd"] == 0 and np.isnan(table.loc[1.0, "calmar"])


def test_best_allocation_cap_and_ties():
    t = pd.DataFrame(
        {"cagr": [0.05, 0.07, 0.07, 0.06], "max_dd": [-0.5, -0.45, -0.40, -0.2]}, pd.Index([0.0, 0.01, 0.02, 0.03])
    )
    assert backtest.best_allocation(t, None) == 0.01  # a tie goes to the smaller w
    assert backtest.best_allocation(t.iloc[::-1], None) == 0.01  # whatever the row order
    assert backtest.best_allocation(t, 0.42) == 0.02  # 0 and 0.01 breach a 42% cap
    assert backtest.best_allocation(t, 0.40) == 0.02  # max_dd == -cap is allowed
    assert backtest.best_allocation(t, 0.1) is None


# Hand-built frontiers. Cash (T-bill sleeve): CAGR falls and max-DD shrinks as w rises.
CASH = pd.DataFrame({"cagr": [0.10, 0.07, 0.03], "max_dd": [-0.50, -0.26, 0.0]}, pd.Index([0.0, 0.5, 1.0]))
INSURED = pd.DataFrame({"cagr": [0.10, 0.085, 0.12], "max_dd": [-0.50, -0.30, -0.55]}, pd.Index([0.0, 0.02, 0.04]))


def test_dd_matched_cash_gap():
    # w = 0.02: max-DD -0.30 is 0.20 / 0.24 of the way from -0.50 to -0.26, where cash CAGR is
    # 0.10 - (5/6) 0.03 = 0.075, so the gap is 0.085 - 0.075
    assert backtest.dd_matched_cash_gap(INSURED, CASH, 0.02) == pytest.approx(0.01, abs=1e-12)
    assert backtest.dd_matched_cash_gap(INSURED, CASH, 0.0) == pytest.approx(0.0, abs=1e-12)  # range end included
    assert np.isnan(backtest.dd_matched_cash_gap(INSURED, CASH, 0.04))  # -0.55 is deeper than any cash max-DD
    assert backtest.dd_matched_cash_gap(INSURED, CASH.iloc[::-1], 0.02) == pytest.approx(0.01, abs=1e-12)  # any order
    with pytest.raises(KeyError):
        backtest.dd_matched_cash_gap(INSURED, CASH, 0.03)  # not on the grid
    on_grid = INSURED.set_axis([0.0, 0.3, 0.6])
    w6 = np.linspace(0, 1, 21)[6]  # 0.30000000000000004; simulate labels it 0.3
    assert backtest.dd_matched_cash_gap(on_grid, CASH, w6) == pytest.approx(0.01, abs=1e-12)


def test_cagr_matched_cash_dd():
    # w = 0.02: CAGR 0.085 is halfway from cash 0.07 (w 0.5, max-DD -0.26) to 0.10 (w 0, -0.50)
    out = backtest.cagr_matched_cash_dd(INSURED, CASH, 0.02)
    assert out == pytest.approx({"w_cash": 0.25, "maxdd_cash": -0.38, "maxdd_gap": 0.08}, abs=1e-12)
    assert backtest.cagr_matched_cash_dd(INSURED, CASH, 0.0) == pytest.approx(
        {"w_cash": 0.0, "maxdd_cash": -0.5, "maxdd_gap": 0.0}, abs=1e-12
    )
    out = backtest.cagr_matched_cash_dd(INSURED, CASH, 0.04)  # CAGR 0.12 beats every cash allocation
    assert set(out) == {"w_cash", "maxdd_cash", "maxdd_gap"} and all(np.isnan(v) for v in out.values())
    out = backtest.cagr_matched_cash_dd(INSURED.set_axis([0.0, 0.3, 0.6]), CASH, np.linspace(0, 1, 21)[6])
    assert out["w_cash"] == pytest.approx(0.25, abs=1e-12)  # a raw grid value finds its rounded label


def test_cagr_matched_cash_dd_takes_the_deleverage_twin_on_a_humped_frontier():
    # Cash CAGR rises from w = 0 to a peak at w = 0.5, then falls (a weak decade). Interpolating
    # the whole frontier sorted by CAGR blends non-neighbours: 0.035 lands between w = 0.25 and
    # w = 0.75 and gives w = 0.5, whose CAGR is 0.05.
    hump = pd.DataFrame(
        {"cagr": [0.01, 0.03, 0.05, 0.04, 0.02], "max_dd": [-0.5, -0.4, -0.3, -0.2, 0.0]},
        pd.Index([0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    sleeve = pd.DataFrame({"cagr": [0.035, 0.015], "max_dd": [-0.1, -0.4]}, pd.Index([0.01, 0.02]))
    # CAGR 0.035 has twins at w 0.3125 and 0.8125; the deleverage one is 1/4 of the way from w 0.75 to 1
    out = backtest.cagr_matched_cash_dd(sleeve, hump, 0.01)
    assert out == pytest.approx({"w_cash": 0.8125, "maxdd_cash": -0.15, "maxdd_gap": 0.05}, abs=1e-12)
    # CAGR 0.015 is below the w = 1 end's 0.02, so only the rising side reaches it: 1/4 of the way from w 0 to 0.25
    out = backtest.cagr_matched_cash_dd(sleeve, hump.iloc[::-1], 0.02)  # any row order
    assert out == pytest.approx({"w_cash": 0.0625, "maxdd_cash": -0.475, "maxdd_gap": 0.075}, abs=1e-12)


def test_period_returns_compound_into_periods():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])  # Thu, Fri, Mon
    r = pd.Series([0.01, 0.01, 0.02], idx)
    out = backtest.period_returns(r, idx[1:2])  # a period end after day 2; day 3 is the trailing period
    assert isinstance(out, pd.DataFrame) and out.index.equals(idx[1:])  # labelled by each period's last row
    np.testing.assert_allclose(out.iloc[:, 0], [1.01**2 - 1, 0.02], rtol=1e-14)
    # a Saturday end and an end after the sample give the same periods, still labelled by their last rows
    pd.testing.assert_frame_equal(backtest.period_returns(r, pd.DatetimeIndex(["2020-01-31", "2020-01-04"])), out)
    two = backtest.period_returns(pd.DataFrame({"a": r, "b": -r}), idx[1:2])
    np.testing.assert_allclose(two["b"], [0.99**2 - 1, -0.02], rtol=1e-14)
    r5 = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], pd.bdate_range("2020-01-06", periods=5))
    ends = pd.DatetimeIndex(["2020-01-09", "2020-01-06", "2020-01-07"])  # unsorted ends are sorted first
    pd.testing.assert_frame_equal(backtest.period_returns(r5, ends), backtest.period_returns(r5, ends.sort_values()))
    # an all-NaN period (a pct_change of a NAV that starts on a period end) is NaN, not 0%
    nan_first = backtest.period_returns(pd.Series([np.nan, 0.1, 0.2], idx), idx[:1])
    assert np.isnan(nan_first.iloc[0, 0]) and nan_first.iloc[1, 0] == pytest.approx(1.1 * 1.2 - 1, rel=1e-14)


def test_rolling_excess_constant_gap():
    idx = pd.bdate_range("2015-01-01", periods=800)
    t = np.asarray((idx - idx[0]).days / 365.25)
    nav = pd.DataFrame({0.0: 1.05**t, 0.02: 1.06**t}, idx)  # CAGR 5% and 6% over any calendar window
    ex = backtest.rolling_excess(nav, years=1)
    assert ex.iloc[:252].isna().all().all()  # the 252-return window is not full yet
    np.testing.assert_allclose(ex[0.02].iloc[252:], 0.01, rtol=0, atol=1e-6)
    assert ex[0.0].iloc[252:].eq(0).all()
    ex2 = backtest.rolling_excess(nav, years=2)  # the window scales with years: 504 returns
    assert ex2.iloc[:504].isna().all().all() and ex2.iloc[504:].notna().all().all()


def test_speed_9000_days_41_points():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("1990-01-01", periods=9000)
    core = pd.Series(rng.normal(0.0004, 0.011, 9000), idx)
    sleeve = pd.Series(rng.normal(-0.002, 0.08, 9000), idx).clip(lower=-1)
    t0 = time.perf_counter()
    nav, _ = backtest.simulate(core, sleeve, np.linspace(0, 0.05, 41), idx[::21], Policy(band=0.5))
    dt = time.perf_counter() - t0
    print(f"simulate 9,000 days x 41 w: {dt:.3f} s")
    assert nav.shape == (9000, 41) and dt < 5

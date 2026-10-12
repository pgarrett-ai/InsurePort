"""Tests for insureport.metrics: NAV from returns, CAGR on calendar days, annualized volatility,
calendar-year returns, the volatility tax (the +50% / -50% example and the two-point lognormal
closed form against its variance / 2 approximation; on full calendar years only, a 10%-a-year line
keeps its 10% mean), drawdown, time under water and the longest recovery span, OLS beta and downside
beta, the crash-window library and its fast / slow tags, crash-window returns (end over start,
off-calendar boundaries, sparse NAVs), concentration, hedge notional, and diagnose() on the 2018-2019
fixture panel (every value wired to its function, the volatility tax on full years, p.value, the
window shared with the S&P 500).

Why: these are the numbers the Diagnostics page shows and the backtest frontier reuses, so each
is pinned to a hand-checkable literal. Synthetic inputs plus the committed fixture CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insureport import metrics
from insureport.portfolio import Portfolio

FIXTURE = Path(__file__).parent / "fixtures" / "market_2018_2019.csv"
MKT = pd.read_csv(FIXTURE, index_col="date", parse_dates=True)
SPX_R = MKT["spxtr"].pct_change()  # first row NaN, as the app builds it


def _nav_from_annual(rets: list[float] | np.ndarray, first_year: int = 2010) -> pd.Series:
    """NAV 1.0 on Jan 1 of first_year, then one year-end NAV per annual return."""
    idx = pd.to_datetime([f"{first_year}-01-01"] + [f"{first_year + i}-12-31" for i in range(len(rets))])
    return pd.Series(np.cumprod([1.0, *(1 + np.asarray(rets))]), idx)


def test_nav_from_returns_ignores_first_return():
    r = pd.Series([0.5, 0.1, -0.5], pd.bdate_range("2020-01-01", periods=3))
    np.testing.assert_allclose(metrics.nav_from_returns(r), [1.0, 1.1, 0.55], rtol=1e-15)
    df = metrics.nav_from_returns(pd.DataFrame({"a": r, "b": 2 * r}))
    np.testing.assert_allclose(df["b"], [1.0, 1.2, 0.0], rtol=1e-15)


def test_cagr_calendar_days():
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01") + pd.Timedelta(days=730)])
    assert metrics.cagr(pd.Series([1.0, 2.0], idx)) == pytest.approx(2 ** (365.25 / 730) - 1, rel=1e-14)
    both = metrics.cagr(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 8.0]}, idx))
    assert both["b"] == pytest.approx(8 ** (365.25 / 730) - 1, rel=1e-14)


def test_ann_vol_ddof1_sqrt252():
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    assert metrics.ann_vol(r) == pytest.approx(0.183303, abs=1e-6)  # sqrt(4e-4 / 3 * 252): ddof 1, 252 days


def test_annual_returns_from_year_end_navs():
    idx = pd.to_datetime(["2019-06-03", "2019-09-30", "2019-12-31", "2020-06-30", "2020-12-30", "2021-03-31"])
    nav = pd.Series([1.0, 1.5, 1.2, 0.5, 0.9, 1.08], idx)  # 2020's last row is Dec 30
    a = metrics.annual_returns(nav)
    assert a.index.tolist() == [2019, 2020, 2021]  # partial first and last years included
    np.testing.assert_allclose(a, [0.2, -0.25, 0.2], rtol=1e-12)
    df = metrics.annual_returns(pd.DataFrame({"x": nav, "y": nav**2}))
    np.testing.assert_allclose(df["y"], [0.44, -0.4375, 0.44], rtol=1e-12)
    stub = pd.Series([1.0, 1.1, 1.21], pd.to_datetime(["2017-12-29", "2018-12-31", "2019-12-31"]))
    assert metrics.annual_returns(stub).to_dict() == pytest.approx({2018: 0.1, 2019: 0.1})  # no 0% 2017


def test_vol_tax_plus_minus_fifty():
    vt = metrics.vol_tax(_nav_from_annual([0.5, -0.5]))
    assert vt["arith"] == pytest.approx(0, abs=1e-15)
    assert vt["geo"] == pytest.approx(-0.133975, abs=1e-6)  # sqrt(1.5 * 0.5) - 1
    assert vt["tax"] == pytest.approx(0.133975, abs=1e-6)
    assert vt["approx"] == pytest.approx(0.125, abs=1e-15)  # var([0.5, -0.5]) / 2
    nav = _nav_from_annual([0.5, -0.5])
    np.testing.assert_allclose(metrics.vol_tax(pd.DataFrame({"a": nav, "b": nav}))["tax"], [0.133975] * 2, atol=1e-6)


def test_vol_tax_full_years_drops_partial_years():
    idx = pd.date_range("2010-09-09", "2026-10-09")  # calendar days: a zero-volatility 10% a year
    nav = pd.Series(1.1 ** ((idx - idx[0]).days / 365.25), idx)
    assert metrics.cagr(nav) == pytest.approx(0.10, rel=1e-12)
    assert metrics.full_year_returns(nav).index.tolist() == list(range(2011, 2026))
    part, full = metrics.vol_tax(nav), metrics.vol_tax(nav, full_years=True)
    assert part["arith"] < 0.095  # the partial 2010 and 2026 count as years and pull the mean below the CAGR
    assert full["arith"] == pytest.approx(0.10, abs=2e-4) and 0 <= full["tax"] < 1e-7
    short = metrics.vol_tax(nav.loc[:"2011-06-30"], full_years=True)  # no full calendar year
    assert np.isnan(short["arith"]) and np.isnan(short["geo"]) and np.isnan(short["tax"])


def test_vol_tax_two_point_lognormal():
    mu, sig = 0.07, 0.2
    vt = metrics.vol_tax(_nav_from_annual(np.exp(mu + sig * np.array([1, -1] * 5)) - 1))
    assert vt["tax"] == pytest.approx(np.exp(mu) * (np.cosh(sig) - 1), abs=1e-12)
    assert vt["tax"] == pytest.approx(0.021522, abs=1e-5)
    assert vt["approx"] == pytest.approx(0.023314, abs=1e-5)  # (e^mu sinh sigma)^2 / 2
    assert abs(vt["approx"] - vt["tax"]) < 0.002


def test_drawdown_stats_hand_counts():
    nav = pd.Series([1, 2, 1, 1.5, 2.5], pd.date_range("2020-01-01", periods=5))
    np.testing.assert_allclose(metrics.drawdown(nav), [0, 0, -0.5, -0.25, 0])
    assert metrics.max_drawdown(nav) == -0.5
    assert metrics.time_under_water(nav) == 0.4
    assert metrics.longest_recovery_days(nav) == 3  # peak on 01-02, back at or above it on 01-05
    both = pd.DataFrame({"a": nav, "b": nav})
    assert metrics.max_drawdown(both).tolist() == [-0.5, -0.5] and metrics.time_under_water(both).tolist() == [0.4, 0.4]
    open_span = pd.Series([1, 2, 1.5, 1.8], pd.bdate_range("2020-01-02", periods=4))  # Thu, Fri, Mon, Tue
    assert metrics.longest_recovery_days(open_span) == 4  # never recovers: Fri peak to the last date, calendar days
    assert metrics.longest_recovery_days(pd.Series([1.0, 1.0, 2.0], pd.date_range("2020-01-01", periods=3))) == 0


def test_beta_and_downside_beta():
    m = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500), pd.bdate_range("2020-01-01", periods=500))
    assert metrics.beta(2 * m, m) == pytest.approx(2, abs=1e-12)
    assert metrics.beta(2 * m.iloc[5:], m) == pytest.approx(2, abs=1e-12)  # aligned on common dates
    r = pd.Series(np.where(m < 0, 2 * m + 0.003, 0.5 * m - 0.003), m.index)  # r's sign differs from m's
    assert metrics.downside_beta(r, m) == pytest.approx(2, abs=1e-12)  # filtering on r < 0 gives 1.42
    assert 0.5 < metrics.beta(r, m) < 1.9


def test_crash_returns_windows_inside_sample():
    idx = pd.bdate_range("2018-01-02", "2020-12-31")
    nav, bench = pd.Series(100.0, idx), pd.Series(10.0, idx)
    nav.loc[["2018-09-20", "2018-12-24", "2020-02-19", "2020-03-23"]] = [100.0, 80.0, 120.0, 90.0]
    nav.loc["2018-11-15"] = 70.0  # an interior trough: the window return is end over start, not the minimum
    nav.loc[["2018-09-19", "2018-12-21", "2018-12-26"]] = [95.0, 85.0, 75.0]  # just outside the pinned dates
    bench.loc[["2018-09-20", "2018-12-24", "2020-02-19", "2020-03-23"]] = [10.0, 8.5, 10.0, 6.6]
    out = metrics.crash_returns(nav, bench=bench)
    assert out.index.tolist() == ["2018 Q4", "2020 COVID"]  # 1987-2011 and 2022 lie outside the sample
    assert list(out.columns) == ["start", "end", "ret", "bench_ret"]
    np.testing.assert_allclose(out["ret"], [-0.2, -0.25], rtol=1e-12)
    np.testing.assert_allclose(out["bench_ret"], [-0.15, -0.34], rtol=1e-12)
    assert out.loc["2020 COVID", "start"] == pd.Timestamp("2020-02-19")
    assert out.loc["2020 COVID", "end"] == pd.Timestamp("2020-03-23")
    assert metrics.crash_returns(nav)["bench_ret"].isna().all()
    # a window cut by the sample edge is omitted, not truncated; same for a short bench
    assert metrics.crash_returns(nav.loc["2018-10-01":]).index.tolist() == ["2020 COVID"]
    short = metrics.crash_returns(nav, bench=bench.loc["2019-01-01":])["bench_ret"]
    assert np.isnan(short.iloc[0]) and short.iloc[1] == pytest.approx(-0.34)
    assert metrics.crash_returns(nav.loc["2019-01-01":"2019-12-31"]).empty
    # boundaries off the calendar: first date >= start (not 09-19's 95), last date <= end (not 12-26's 75)
    assert metrics.crash_returns(nav.drop(pd.Timestamp("2018-09-20"))).loc["2018 Q4", "ret"] == pytest.approx(-0.2)
    assert metrics.crash_returns(nav.drop(pd.Timestamp("2018-12-24"))).loc["2018 Q4", "ret"] == pytest.approx(-0.15)
    sparse = pd.Series(1.0, pd.to_datetime(["2020-01-02", "2020-04-01"]))  # COVID lies inside but holds no row
    assert metrics.crash_returns(sparse).empty


def test_crash_library_and_speed_tags():
    # D14: Harvey et al.'s eight >15% S&P 500 drawdowns 1985-2018, COVID, the 2022 grind and the 2025 tariff shock
    assert len(metrics.CRASHES) == 11 and metrics.CRASHES["2025 Tariff shock"] == ("2025-02-19", "2025-04-08")
    assert metrics.CRASHES["2010 Flash crash"] == ("2010-04-23", "2010-07-02")
    assert set(metrics.CRASH_SPEED) == set(metrics.CRASHES)
    assert {k for k, v in metrics.CRASH_SPEED.items() if v == "slow"} == {"2000-02 Dot-com", "2007-09 GFC", "2022 Rates"}


def test_concentration_excludes_cash():
    c = metrics.concentration({"A": 0.5, "B": 0.25, "C": 0.25, "CASH": 0.1})
    assert c == pytest.approx({"hhi": 0.375, "eff_n": 8 / 3, "top1": 0.5, "top5": 1.0})
    six = metrics.concentration({t: 1 / 6 for t in "ABCDEF"})
    assert six == pytest.approx({"hhi": 1 / 6, "eff_n": 6, "top1": 1 / 6, "top5": 5 / 6})
    renorm = metrics.concentration({"A": 0.4, "B": 0.2, "C": 0.2, "CASH": 0.2})  # non-CASH sums to 0.8
    assert renorm == pytest.approx({"hhi": 0.375, "eff_n": 8 / 3, "top1": 0.5, "top5": 1.0})
    assert all(np.isnan(v) for v in metrics.concentration({"CASH": 1.0}).values())


def test_hedge_notional():
    h = metrics.hedge_notional(1_000_000, 1.2, 5000)
    assert h == pytest.approx({"notional": 1.2e6, "spx_contracts": 2.4, "xsp_contracts": 24})
    assert metrics.hedge_notional(1_000_000, -0.2, 5000)["spx_contracts"] == 0  # negative beta: no puts


def test_diagnose_on_fixture():
    mkt, spx_r = MKT, SPX_R
    core = (0.8 * spx_r).fillna(0.0)  # first row is the 0 placeholder core_returns emits
    d = metrics.diagnose(core, spx_r, Portfolio({"SPY": 0.8, "CASH": 0.2}), mkt)
    assert set(d) == {
        "start", "end", "years", "cagr", "arith", "vol_tax", "vol_tax_approx", "ann_vol", "max_dd", "tuw",
        "longest_recovery_days", "beta", "downside_beta", "hhi", "eff_n", "top1", "top5", "spx_cagr",
        "spx_max_dd", "crashes", "hedge",
    }  # fmt: skip
    assert d["years"] == pytest.approx(2, abs=0.01)
    nav = metrics.nav_from_returns(core)
    assert d["start"] == mkt.index[0] and d["end"] == mkt.index[-1]
    assert d["cagr"] == pytest.approx(metrics.cagr(nav), rel=1e-12) and d["cagr"] != pytest.approx(d["spx_cagr"])
    assert d["max_dd"] == pytest.approx(metrics.max_drawdown(nav), rel=1e-12)
    assert d["tuw"] == metrics.time_under_water(nav) > 0
    assert d["longest_recovery_days"] == metrics.longest_recovery_days(nav) > 0
    assert d["vol_tax_approx"] == pytest.approx(metrics.vol_tax(nav, full_years=True)["approx"], rel=1e-12)
    assert d["crashes"].index.tolist() == ["2018 Q4"]
    spx_q4 = mkt.loc["2018-12-24", "spxtr"] / mkt.loc["2018-09-20", "spxtr"] - 1  # bench = SPX total return
    assert d["crashes"].loc["2018 Q4", "bench_ret"] == pytest.approx(spx_q4, rel=1e-12)
    assert d["beta"] == pytest.approx(0.8, abs=1e-12) and d["downside_beta"] == pytest.approx(0.8, abs=1e-12)
    assert d["spx_cagr"] == pytest.approx(metrics.cagr(mkt["spxtr"]), rel=1e-12)
    assert d["spx_max_dd"] == pytest.approx(metrics.max_drawdown(mkt["spxtr"]), rel=1e-12)
    assert (d["hhi"], d["eff_n"], d["top1"], d["top5"]) == (1.0, 1.0, 1.0, 1.0)  # CASH excluded
    full = metrics.vol_tax(nav, full_years=True)  # 2019 alone: 2018 starts on the panel's first row
    assert d["arith"] == pytest.approx(full["arith"]) and d["vol_tax"] == pytest.approx(full["tax"], abs=1e-15)
    assert d["arith"] == pytest.approx(nav["2019-12-31"] / nav["2018-12-31"] - 1, rel=1e-12)
    assert d["ann_vol"] == pytest.approx(metrics.ann_vol(core.iloc[1:]))
    n, spx = 0.8e6, mkt["spx"].iloc[-1]  # value None -> $1M, times downside beta 0.8, at the last SPX
    hedge = {"notional": n, "spx_contracts": n / (100 * spx), "xsp_contracts": n / (10 * spx)}
    assert d["hedge"] == pytest.approx(hedge)
    h = metrics.diagnose(core, spx_r, Portfolio({"SPY": 1.0}, value=250_000.0), mkt)["hedge"]
    assert h["notional"] == pytest.approx(250_000 * 0.8)  # p.value, times downside beta 0.8


def test_diagnose_uses_the_window_spx_covers():
    early = pd.Series(0.0004, pd.bdate_range("2017-06-01", "2017-12-29"))  # the core starts 7 months earlier
    core = pd.concat([early, (0.8 * SPX_R).fillna(0.0)])
    p = Portfolio({"SPY": 1.0})
    d, same = metrics.diagnose(core, SPX_R, p, MKT), metrics.diagnose((0.8 * SPX_R).fillna(0.0), SPX_R, p, MKT)
    assert d["start"] == MKT.index[0] and d["years"] == pytest.approx(2, abs=0.01)
    assert d["cagr"] == pytest.approx(same["cagr"], rel=1e-12) and d["ann_vol"] == pytest.approx(same["ann_vol"])
    assert d["spx_cagr"] == pytest.approx(metrics.cagr(MKT["spxtr"]), rel=1e-12)
    # S&P TR prints only from row 4 (a panel starting before 1988): its level, not its first return, sets start
    late = SPX_R.copy()
    late.iloc[:5] = np.nan
    d = metrics.diagnose(core, late, p, MKT)
    assert d["start"] == MKT.index[4] and d["spx_cagr"] == pytest.approx(metrics.cagr(MKT["spxtr"].iloc[4:]), rel=1e-12)
    assert d["cagr"] == pytest.approx(metrics.cagr(metrics.nav_from_returns(core.loc[MKT.index[4] :])), rel=1e-12)
    with pytest.raises(ValueError, match="fewer than two dates"):
        metrics.diagnose(early, SPX_R, p, MKT)

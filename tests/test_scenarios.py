"""Tests for insureport.scenarios: the book's regime bins (right-closed edges) and the regime table,
the cartoon haven tables (insurance X = (1 - p)/p, store of value CPI + 2% real or the book's 7%),
book-mode and estimated-mode portfolio returns per regime (full calendar years only, empty regimes
NaN), geometric vs arithmetic dice growth (and ruin), the Kelly allocation (closed form, the w_max
cap, the feasibility bound), cost-effectiveness, the breakeven crash frequency and the
crash-frequency sweep (one and two crash regimes, a window with no crash year, invalid crash sets),
the book-style annual bootstrap (joint / reshuffle / no-crash, annual vs no rebalancing, paired
paths, seeds), the stationary block bootstrap (index paths, constant returns, the ergodicity panel,
a dominant sleeve, drawdown depth per horizon and breaches, seeds, the 2,000 x 240 x 41 speed
budget) and the packaged S&P 500 annual series (source line, its CPI column, the Part One 1997-2016
cartoon on it).

Why: the Dice tab, the Kelly sizing and the bootstrap panels run through these functions, so
each is pinned to the book's own numbers (source next to each) or a hand-checkable literal.
Pure functions on small synthetic inputs plus the packaged CSV, no network.
"""

import time

import numpy as np
import pandas as pd
import pytest

from insureport import scenarios as sc

# Safe Haven (2021) ch. 2-3 die, faces 1..6, each 1/6 ("Kelly optimal bet size", idx pp. 81-87)
DIE = np.array([-0.5, 0.05, 0.05, 0.05, 0.05, 0.5])
SIXTH = np.full(6, 1 / 6)
# Universa "Safe Haven Investing" Part One (2017) / Safe Haven Part Three cartoon, two regimes,
# crash first: a crash year in 10, SPX -30% / +10%, insurance +900% / -100% (zero mean at p = 0.1)
P2, CORE2, INS2 = np.array([0.1, 0.9]), np.array([-0.30, 0.10]), np.array([9.0, -1.0])
CAGR_COLS = ["cagr_p5", "cagr_p25", "cagr_p50", "cagr_p75", "cagr_p95"]


def _years(values: list[float], first: int = 2000) -> pd.Series:
    """Annual returns indexed by consecutive integer years from `first`."""
    return pd.Series(values, index=range(first, first + len(values)), dtype=float)


def _monthly(values: float | np.ndarray, n: int) -> pd.Series:
    """n month-end period returns from Jan 2000 (a scalar is repeated)."""
    return pd.Series(np.broadcast_to(values, n), pd.date_range("2000-01-31", periods=n, freq="ME"), dtype=float)


def test_two_outcome_volatility_tax():
    # Universa, "The Volatility Tax" (Safe Haven Investing Part Four, 2018): -50% then +100%
    # averages +25% and compounds to 0%
    p, P = np.array([0.5, 0.5]), np.array([-0.5, 1.0])
    assert sc.dice_arith(p, P) == pytest.approx(0.25, abs=1e-12)
    assert sc.dice_growth(p, P) == pytest.approx(0.0, abs=1e-12)


def test_dice_growth_ruin():
    # 1 + P_j <= 0 in a regime that can happen is ruin (-1); a regime with p_j = 0 cannot ruin
    assert sc.dice_growth(np.array([0.5, 0.5]), np.array([-1.0, 1.0])) == -1
    g = sc.dice_growth(np.array([0.5, 0.5]), np.array([[-2.0, 0.1], [1.0, 0.1]]))
    np.testing.assert_allclose(g, [-1, 0.1], rtol=0, atol=1e-12)
    assert sc.dice_growth(np.array([0.0, 1.0]), np.array([-2.0, 0.1])) == pytest.approx(0.1, abs=1e-12)


def test_book_die_and_its_kelly_bet():
    # Safe Haven ch. 2-3: the die averages +3.33% a roll but compounds at -1.53%; staked on a 0%
    # core (cash) it is a bet of a fraction w of wealth: Kelly near 0.379, +0.642% a roll at 40%
    assert sc.dice_arith(SIXTH, DIE) == pytest.approx(0.033333, abs=1e-5)
    assert sc.dice_growth(SIXTH, DIE) == pytest.approx(-0.015298, abs=1e-5)  # (0.5 * 1.05^4 * 1.5)^(1/6) - 1
    cash = np.zeros(6)
    assert sc.kelly_w(SIXTH, cash, DIE) == pytest.approx(0.379, abs=0.002)
    P40 = sc.book_portfolio_returns(cash, DIE, np.array([0.4]))
    assert P40.shape == (6, 1)
    assert sc.dice_growth(SIXTH, P40)[0] == pytest.approx(0.00642, abs=1e-4)


def test_book_91_9_insurance_blend():
    # Safe Haven ch. 3 ("side bets: insurance safe haven", idx pp. 87-95): 91% die + 9% insurance
    # paying +500% net on face 1 and -100% otherwise (a zero-mean bet) averages 3.033% and
    # compounds at 2.089%; 9% is the Kelly weight (1/11)
    ins = np.array([5.0, -1, -1, -1, -1, -1])
    P = sc.book_portfolio_returns(DIE, ins, np.array([0.09]))
    np.testing.assert_allclose(P[:, 0], [-0.005, -0.0445, -0.0445, -0.0445, -0.0445, 0.365], rtol=0, atol=1e-15)
    assert sc.dice_arith(SIXTH, P)[0] == pytest.approx(0.030333, abs=1e-5)
    assert sc.dice_growth(SIXTH, P)[0] == pytest.approx(0.020894, abs=1e-5)
    assert sc.kelly_w(SIXTH, DIE, ins) == pytest.approx(0.091, abs=0.003)


def test_part_three_cartoon_insurance():
    # Universa Part One / Safe Haven Part Three: 2% insurance turns the -30% crash year into
    # -11.4% and costs 2% of the +10% years, and still raises the growth rate
    P = sc.book_portfolio_returns(CORE2, INS2, np.array([0.0, 0.02]))
    g = sc.dice_growth(P2, P)
    assert g[0] == pytest.approx(0.70**0.1 * 1.10**0.9 - 1, abs=1e-12)  # 0.051388...
    assert g[1] == pytest.approx(0.886**0.1 * 1.078**0.9 - 1, abs=1e-12)  # 0.057062...
    assert g[1] > g[0]
    assert sc.cost_effective(P2, CORE2, INS2) is True
    # G'(0; pi) = pi 9.3/0.7 - (1 - pi) 1.1/1.1 = 0 at pi = 0.7/10
    assert sc.breakeven_crash_freq(P2, CORE2, INS2) == pytest.approx(0.07, abs=1e-12)
    np.testing.assert_allclose(sc.book_haven("insurance", 0.1), [9.0, -1, -1, -1], rtol=0, atol=1e-12)
    assert sc.book_haven("insurance", 1 / 11, insurance_payoff=10.0)[0] == 10.0  # Part Three's "tenbagger"
    np.testing.assert_array_equal(sc.book_haven("alpha", 0.1), [0.20, 0.10, 0.05, 0.05])
    np.testing.assert_array_equal(sc.book_haven("store_of_value_book", 0.1), [0.07] * 4)  # the book's 7% nominal
    with pytest.raises(ValueError, match="p_crash"):
        sc.book_haven("insurance", 0.0)


def test_store_of_value_is_cpi_plus_two_percent_real(monkeypatch):
    # addendum D5 (Universa Part One): CPI_y + 2% real, averaged per regime over the packaged table's years
    s, cpi = sc.load_sp500_annual(), sc._annual_table()["cpi"]
    assert cpi.index.equals(s.index) and cpi[2022] == pytest.approx(0.0645, abs=5e-4)  # Dec/Dec CPI-U
    crash = cpi[s <= -0.15]  # 1930, 1931, 1937, 1974, 2002, 2008, 2022
    assert sc.book_haven("store_of_value", 0.1)[0] == pytest.approx(crash.mean() + 0.02, abs=1e-12)
    # _annual_table is the seam: hand-made years, one or two per bin
    annual = pd.DataFrame({"total_return": [-0.30, -0.05, 0.10, 0.20, 0.25], "cpi": [0.01, 0.02, 0.03, 0.04, 0.06]},
                          index=range(2000, 2005))
    monkeypatch.setattr(sc, "_annual_table", lambda: annual)
    np.testing.assert_allclose(sc.book_haven("store_of_value", 0.1), [0.03, 0.04, 0.05, 0.07], atol=1e-15)


def test_store_of_value_is_not_cost_effective():
    # a flat +2% store of value on the Part Three cartoon; Part One's -17.9 bp used CPI + 2% real at a
    # 10% weight on 1997-2016
    sov = np.full(2, 0.02)
    assert sc.cost_effective(P2, CORE2, sov) is False
    assert sc.kelly_w(P2, CORE2, sov) == 0.0


def test_kelly_two_outcome_closed_form():
    # Kelly (1956): a bet winning b with probability p, losing a otherwise -> w* = p/a - (1 - p)/b
    p, cash, coin = np.array([0.6, 0.4]), np.zeros(2), np.array([1.0, -1.0])
    assert sc.kelly_w(p, cash, coin) == pytest.approx(0.2, abs=1e-9)
    assert sc.kelly_w(p, cash, coin, w_max=0.1) == 0.1  # G' still > 0 at the cap
    assert sc.kelly_w(np.array([0.5, 0.5]), cash, np.array([0.1, 0.0])) == 1.0  # never trails the core: w_max


def test_kelly_stays_inside_the_feasible_range():
    cash = np.zeros(2)
    lever = np.array([3.0, -2.0])  # a sleeve that can lose 200%: 1 + P_j = 1 - 2w, ruin at w = 1/2 < 1
    w = sc.kelly_w(np.array([0.99, 0.01]), cash, lever)
    assert w == pytest.approx(0.99 / 2 - 0.01 / 3, abs=1e-9)  # the closed form, inside the bound
    # a near-sure payoff keeps G' > 0 up to the bound: w* sits just inside it, never on it
    sure = np.array([1 - 1e-12, 1e-12])
    w = sc.kelly_w(sure, cash, lever)
    assert 0.5 - 1e-6 < w < 0.5
    assert (1 + sc.book_portfolio_returns(cash, lever, np.array([w])) > 0).all()
    # h_j = -1 (a premium sleeve expiring worthless): 1 + P_j = (1 - w)(1 + c_j) caps w below 1
    core, prem = np.array([0.0, 0.1]), np.array([5.0, -1.0])
    w = sc.kelly_w(sure, core, prem)
    assert 1 - 1e-6 < w < 1
    assert (1 + sc.book_portfolio_returns(core, prem, np.array([w])) > 0).all()
    # a calm sample (p_crash = 0) and a short-vol haven losing 150% in the crash bin: G' > 0 on
    # the live regimes, but the crash bin still caps w at 0.7 / 1.2 (ruin beyond it)
    p, core, short_vol = np.array([0.0, 0.1, 0.5, 0.4]), np.array([-0.30, -0.05, 0.08, 0.25]), np.array([-1.5, 0.05, 0.12, 0.30])
    w = sc.kelly_w(p, core, short_vol)
    assert 0.7 / 1.2 - 1e-6 < w < 0.7 / 1.2
    assert (1 + sc.book_portfolio_returns(core, short_vol, np.array([w])) > 0).all()


def test_regime_of_right_closed_bins():
    # the book's four bins of calendar-year SPX total return: <= -15% (crash), -15..0, 0..15, > 15
    assert [int(sc.regime_of(x)) for x in (-0.15, 0.0, 0.15, 0.151)] == [0, 1, 2, 3]
    np.testing.assert_array_equal(sc.regime_of(np.array([-0.6, -0.1499, -1e-9, 1e-9, 2.0])), [0, 1, 1, 2, 3])
    np.testing.assert_array_equal(sc.regime_of(np.array([-0.1, 0.1]), edges=(-np.inf, 0.0, np.inf)), [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        sc.regime_of(np.array([0.1, np.nan]))


def test_regime_table_counts_and_freqs():
    rets = _years([-0.30, -0.15, -0.10, 0.0, 0.05, 0.15, 0.151, 0.20, 0.25, -0.05, np.nan], 1990)
    t = sc.regime_table(rets)  # the NaN year is dropped
    assert list(t.columns) == ["label", "lo", "hi", "count", "freq", "mean_ret"]
    assert t.index.tolist() == [0, 1, 2, 3]
    assert tuple(t["label"]) == sc.REGIME_LABELS == ("< -15%", "-15%..0%", "0%..15%", "> 15%")
    assert t["count"].tolist() == [2, 3, 2, 3]
    np.testing.assert_allclose(t["freq"], [0.2, 0.3, 0.2, 0.3], rtol=1e-15)
    np.testing.assert_allclose(t["mean_ret"], [-0.225, -0.05, 0.10, 0.601 / 3], rtol=1e-12)
    assert t.loc[0, "lo"] == -np.inf and t.loc[3, "hi"] == np.inf and t.loc[1, "hi"] == 0.0
    sparse = sc.regime_table(_years([0.05, 0.25]))  # no crash or down year: count 0, mean NaN
    assert sparse["count"].tolist() == [0, 0, 1, 1] and sparse["mean_ret"].isna().tolist() == [True, True, False, False]


def test_regime_portfolio_returns_estimated_mode():
    # w = 0 earns +10%, -20%, +5%, +30%, +20% in 2010-2014; the first (2009) and last (2015)
    # years are partial and dropped. SPX puts 2011 in the crash bin, no year in -15..0,
    # 2010 (exactly +15%), 2012 and 2014 in 0..15, 2013 above 15%.
    a0 = [0.10, -0.20, 0.05, 0.30, 0.20]
    a1 = [0.0, 0.10, -0.02, 0.01, -0.03]
    dates = pd.to_datetime(
        ["2009-06-30", "2009-12-31", "2010-12-31", "2011-12-30", "2012-12-31", "2013-12-31", "2014-12-31", "2015-06-30"]
    )  # 2011-12-30 is that year's last business day
    nav = pd.DataFrame({w: np.cumprod([1.0, 1.07, *(1 + np.array(a)), 1.04]) for w, a in ((0.0, a0), (1.0, a1))}, dates)
    spx = _years([0.26, 0.15, -0.20, 0.05, 0.32, 0.14, 0.01], 2009)
    p, P, n = sc.regime_portfolio_returns(nav, spx)
    assert n.tolist() == [1, 0, 3, 1]
    np.testing.assert_allclose(p, [0.2, 0.0, 0.6, 0.2], rtol=1e-15)
    assert P.shape == (4, 2) and np.isnan(P[1]).all()
    np.testing.assert_allclose(P[[0, 2, 3]], [[-0.20, 0.10], [0.35 / 3, -0.05 / 3], [0.30, 0.01]], rtol=1e-12)
    # a NAV ending on the year's last business day keeps that year
    p2, P2_, n2 = sc.regime_portfolio_returns(nav.loc[:"2011-12-30"], spx)
    assert n2.tolist() == [1, 0, 1, 0]
    np.testing.assert_allclose(P2_[[0, 2]], [[-0.20, 0.10], [0.10, 0.0]], rtol=1e-12, atol=1e-15)
    # so does a calendar month-end NAV ending on Saturday 2011-12-31 (2010 crash-free, 2011 a crash)
    me = pd.DataFrame({0.0: np.linspace(1, 2, 25)}, pd.date_range("2009-12-31", "2011-12-31", freq="ME"))
    assert sc.regime_portfolio_returns(me, _years([0.1, 0.05, -0.2], 2009))[2].tolist() == [1, 0, 1, 0]
    # the dice engine skips the empty regime (p = 0, P NaN), and so do cost_effective and Kelly
    assert np.isfinite(sc.dice_growth(p, P)).all() and np.isfinite(sc.dice_arith(p, P)).all()
    p3, c3, h3 = np.array([0.5, 0.0, 0.5]), np.array([-0.3, np.nan, 0.1]), np.array([1.0, np.nan, 0.0])
    assert sc.cost_effective(p3, c3, h3) is True
    assert sc.kelly_w(p3, c3, h3) == sc.kelly_w(p3[[0, 2]], c3[[0, 2]], h3[[0, 2]])
    # a full NAV year the S&P 500 series does not have yet (2012) is dropped, not a KeyError
    late = pd.DataFrame({0.0: [1.0, 1.1, 1.2, 1.3]}, pd.to_datetime(["2009-12-31", "2010-12-31", "2011-12-30", "2012-12-31"]))
    assert sc.regime_portfolio_returns(late, _years([0.1, 0.2], 2010))[2].sum() == 2
    with pytest.raises(ValueError, match="full calendar year"):
        sc.regime_portfolio_returns(nav.loc[:"2010-06-30"], spx)


def test_breakeven_and_sweep_with_two_crash_regimes():
    # crash mass pi is split over regimes 0 and 1 in their 1:3 proportion, regime 2 gets 1 - pi
    p, core, hav = np.array([0.1, 0.3, 0.6]), np.array([-0.5, -0.2, 0.1]), np.array([4.0, 0.3, -0.1])
    x = (hav - core) / (1 + core)  # 9, 0.625, -0.181818...
    A, B = (0.1 * x[0] + 0.3 * x[1]) / 0.4, x[2]
    assert sc.breakeven_crash_freq(p, core, hav, crash=(0, 1)) == pytest.approx(B / (B - A), abs=1e-12)
    w = np.array([0.0, 0.1])
    sweep = sc.crash_frequency_sweep(p, core, hav, w, crash=(0, 1), grid=np.array([0.2]))
    want = sc.dice_growth(np.array([0.05, 0.15, 0.8]), sc.book_portfolio_returns(core, hav, w))
    np.testing.assert_allclose(sweep.loc[0.2], want, rtol=0, atol=1e-12)
    # undefined: a haven equal to the core (B - A = 0), or one worse in every regime (pi* < 0)
    assert np.isnan(sc.breakeven_crash_freq(P2, CORE2, CORE2))
    assert np.isnan(sc.breakeven_crash_freq(P2, CORE2, CORE2 - 0.05))
    # no crash regime, one out of range, or every regime: no split, so no breakeven (not 100% / 0%)
    for bad in ((), (2,), (0, 1)):
        with pytest.raises(ValueError, match="crash"):
            sc.breakeven_crash_freq(P2, CORE2, INS2, crash=bad)
    with pytest.raises(ValueError, match="crash"):
        sc.crash_frequency_sweep(P2, CORE2, INS2, w, crash=())


def test_breakeven_and_sweep_without_crash_years():
    # a calm window (e.g. 2010-2019: no crash year) with the Part Three insurance (+900% / -100%):
    # the crash regime still gets weight 1 within its group, so pi* = 0.7/10 as in the cartoon
    # (x = 9.3/0.7 on the crash, -1 elsewhere) and the pi = 0 row is the sample's own growth
    p, core, ins = np.array([0.0, 0.1, 0.5, 0.4]), np.array([-0.30, -0.05, 0.08, 0.25]), sc.book_haven("insurance", 0.1)
    assert sc.breakeven_crash_freq(p, core, ins) == pytest.approx(0.07, abs=1e-12)
    w = np.array([0.0, 0.02])
    P = sc.book_portfolio_returns(core, ins, w)
    sweep = sc.crash_frequency_sweep(p, core, ins, w, grid=np.array([0.0, 0.1]))
    np.testing.assert_allclose(sweep.loc[0.0], sc.dice_growth(p, P), rtol=0, atol=1e-12)
    np.testing.assert_allclose(sweep.loc[0.1], sc.dice_growth(np.array([0.1, 0.09, 0.45, 0.36]), P), rtol=0, atol=1e-12)


def test_crash_frequency_sweep():
    w = np.array([0.0, 0.02, 0.05])
    sweep = sc.crash_frequency_sweep(P2, CORE2, INS2, w, grid=np.array([0.0, 0.05, 0.1, 0.2]))
    assert sweep.shape == (4, 3) and sweep.columns.tolist() == [0.0, 0.02, 0.05]
    assert sweep.index.name == "pi" and sweep.columns.name == "w"
    at_p = sc.dice_growth(P2, sc.book_portfolio_returns(CORE2, INS2, w))
    np.testing.assert_allclose(sweep.loc[0.1], at_p, rtol=0, atol=1e-12)
    assert (np.diff(sweep[0.0].to_numpy()) < 0).all()  # more crash years, less core growth
    # below the breakeven pi* = 0.07 insurance costs growth, above it adds growth
    assert sweep.loc[0.05, 0.02] < sweep.loc[0.05, 0.0] and sweep.loc[0.1, 0.02] > sweep.loc[0.1, 0.0]
    default = sc.crash_frequency_sweep(P2, CORE2, INS2, w)
    assert default.index.tolist() == pytest.approx(np.linspace(0, 0.3, 31).tolist())


def test_book_bootstrap_two_year_history():
    # -50% / +100% years compound to 0% at the median (an even horizon: k = 10 of 20 good years)
    out = sc.book_bootstrap(_years([-0.5, 1.0]), _years([0.0, 0.0]), np.array([0.0]), horizon=20, n_paths=20_000)
    assert list(out.columns) == [*CAGR_COLS, "tw_p50", "p_beat", "d_cagr_p50"]
    assert out.index.tolist() == [0.0] and out.index.name == "w"
    assert out.loc[0.0, "cagr_p50"] == pytest.approx(0.0, abs=0.02)
    assert out.loc[0.0, "tw_p50"] == pytest.approx(1.0, abs=1e-12)


def test_book_bootstrap_modes():
    # a perfect hedge: the haven doubles when the core halves and halves when it doubles, so
    # 50/50 grows 25% every year when drawn jointly; reshuffled, the link (the gain) is gone
    core, hav, w = _years([-0.5, 1.0]), _years([1.0, -0.5]), np.array([0.0, 0.5])
    joint = sc.book_bootstrap(core, hav, w, n_paths=4000)
    shuf = sc.book_bootstrap(core, hav, w, n_paths=4000, mode="reshuffle")
    np.testing.assert_allclose(joint.loc[0.5, CAGR_COLS], 0.25, rtol=0, atol=1e-12)
    assert joint.loc[0.5, "d_cagr_p50"] > shuf.loc[0.5, "d_cagr_p50"] + 0.1
    # one crash year in four: no_crash never draws it, so every core path compounds above 0
    hist, ins = _years([-0.40, 0.05, 0.10, 0.20]), _years([3.0, -0.1, -0.1, -0.1])
    nc = sc.book_bootstrap(hist, ins, w, n_paths=2000, mode="no_crash")
    j = sc.book_bootstrap(hist, ins, w, n_paths=2000)
    assert nc.loc[0.0, "cagr_p5"] > 0 > j.loc[0.0, "cagr_p5"]
    assert nc.loc[0.5, "d_cagr_p50"] < 0 < j.loc[0.5, "d_cagr_p50"]  # the hedge only costs when no crash comes
    # -15% is a crash year (right-closed bin): no_crash leaves only the +5% year
    edge = sc.book_bootstrap(_years([-0.15, 0.05]), _years([0.0, 0.0]), np.array([0.0]), n_paths=200, mode="no_crash")
    np.testing.assert_allclose(edge.loc[0.0, CAGR_COLS], 0.05, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="no common year"):
        sc.book_bootstrap(_years([-0.3]), _years([1.0]), w, mode="no_crash")
    for t in (joint, shuf, nc, j):  # paired paths: w = 0 never beats itself
        assert t["p_beat"].between(0, 1).all() and t.loc[0.0, "p_beat"] == 0 and t.loc[0.0, "d_cagr_p50"] == 0


def test_book_bootstrap_rebalance_and_seeds():
    rng = np.random.default_rng(2)
    core, hav = _years(rng.normal(0.08, 0.18, 40).tolist()), _years(rng.normal(0.0, 0.5, 40).clip(-1).tolist())
    w = np.array([0.0, 0.3, 1.0])
    a = sc.book_bootstrap(core, hav, w, n_paths=500, seed=4)
    pd.testing.assert_frame_equal(a, sc.book_bootstrap(core, hav, w, n_paths=500, seed=4))
    assert not a.equals(sc.book_bootstrap(core, hav, w, n_paths=500, seed=5))
    none = sc.book_bootstrap(core, hav, w, n_paths=500, seed=4, rebalance="none")
    pd.testing.assert_frame_equal(none.loc[[0.0, 1.0]], a.loc[[0.0, 1.0]], check_exact=False, rtol=1e-12)
    assert none.loc[0.3, "cagr_p50"] != pytest.approx(a.loc[0.3, "cagr_p50"], abs=1e-6)
    # one year of history, so every path draws it twice: 50/50 of +10% / -10% is flat each year
    # when rebalanced, 0.5 * 1.1^2 + 0.5 * 0.9^2 = 1.01 when left alone
    one = (_years([0.10]), _years([-0.10]), np.array([0.5]))
    assert sc.book_bootstrap(*one, horizon=2, n_paths=10).loc[0.5, "tw_p50"] == pytest.approx(1.0, abs=1e-12)
    assert sc.book_bootstrap(*one, horizon=2, n_paths=10, rebalance="none").loc[0.5, "tw_p50"] == pytest.approx(
        1.01, abs=1e-12
    )
    with pytest.raises(ValueError, match="mode"):
        sc.book_bootstrap(core, hav, w, mode="blocks")
    with pytest.raises(ValueError, match="rebalance"):
        sc.book_bootstrap(core, hav, w, rebalance="monthly")


def test_bootstraps_pair_every_w_on_the_same_paths():
    # a haven equal to the core makes every w the core itself, so on shared paths every row is
    # the w = 0 row; a w drawing its own paths would differ by sampling noise
    core = _years(np.random.default_rng(2).normal(0.08, 0.18, 40).tolist())
    b = sc.book_bootstrap(core, core, np.array([0.0, 0.5, 1.0]), n_paths=500, seed=4)[[*CAGR_COLS, "tw_p50"]]
    for x in (0.5, 1.0):
        np.testing.assert_allclose(b.loc[x], b.loc[0.0], rtol=1e-12)
    rc = _monthly(np.random.default_rng(3).normal(0.007, 0.045, 240), 240)
    o = sc.bootstrap(rc, rc, np.array([0.0, 0.5, 1.0]), 12, n_paths=300)[[*CAGR_COLS, "maxdd_p50", "maxdd_p95"]]
    for h in (5, 10, 20):
        for x in (0.5, 1.0):
            np.testing.assert_allclose(o.loc[(h, x)], o.loc[(h, 0.0)], rtol=1e-12)


def test_stationary_bootstrap_idx():
    I = sc.stationary_bootstrap_idx(7, 50, 20, np.inf, seed=3)  # infinite mean block: one wrap-around block
    assert I.shape == (50, 20) and I.dtype.kind == "i"
    np.testing.assert_array_equal(I, (I[:, :1] + np.arange(20)) % 7)
    K = sc.stationary_bootstrap_idx(100, 4000, 50, 6, seed=1)
    assert K.min() == 0 and K.max() == 99
    # continue w.p. 5/6, plus a fresh draw landing on the next index by chance (1/6 * 1/100)
    assert (K[:, 1:] == (K[:, :-1] + 1) % 100).mean() == pytest.approx(5 / 6 + 1 / 600, abs=0.01)
    np.testing.assert_array_equal(K, sc.stationary_bootstrap_idx(100, 4000, 50, 6, seed=1))


def test_bootstrap_constant_returns():
    rc, rs = _monthly(0.01, 120), _monthly(0.0, 120)
    out = sc.bootstrap(rc, rs, np.array([0.0, 0.5]), 12, horizons=(5, 10), n_paths=200)
    assert out.index.names == ["horizon", "w"]
    assert out.index.tolist() == [(5, 0.0), (5, 0.5), (10, 0.0), (10, 0.5)]
    assert list(out.columns) == [
        *CAGR_COLS, "p_beat", "ensemble_growth", "median_growth", "maxdd_p50", "maxdd_p95", "p_dd_breach",
    ]  # fmt: skip
    for h in (5, 10):  # every path compounds the same: 1.005 a month at w = 0.5
        g = out.loc[(h, 0.5), [*CAGR_COLS, "ensemble_growth", "median_growth"]]
        np.testing.assert_allclose(g, 1.005**12 - 1, rtol=0, atol=1e-12)
        assert out.loc[(h, 0.0), "cagr_p50"] == pytest.approx(1.01**12 - 1, abs=1e-12)
    assert (out[["maxdd_p50", "maxdd_p95", "p_dd_breach", "p_beat"]] == 0).all().all()  # never falls; 0.5 never beats


def test_bootstrap_ergodicity_panel():
    # the -50% / +100% volatility-tax fixture (Universa, "The Volatility Tax"), drawn i.i.d. for two
    # years: W in {0.25, 1, 4} w.p. 1/4, 1/2, 1/4, so the ensemble E[W] = 1.25^2 grows 25% a year
    # while the median path (and the median CAGR, n_paths odd) grows 0%
    r = _years([-0.5, 1.0])
    out = sc.bootstrap(r, r, np.array([0.0]), 1, horizons=(2,), n_paths=20_001, mean_block=1).loc[(2, 0.0)]
    assert out["ensemble_growth"] == pytest.approx(0.25, abs=0.01)
    assert out["median_growth"] == pytest.approx(0.0, abs=1e-12) and out["cagr_p50"] == out["median_growth"]
    with pytest.raises(ValueError, match="no common period"):
        sc.bootstrap(r, r.set_axis(r.index + 10), np.array([0.0]), 1)


def test_bootstrap_dominant_sleeve_always_beats():
    rng = np.random.default_rng(5)
    rc = _monthly(rng.normal(0.007, 0.045, 240), 240)
    out = sc.bootstrap(rc, rc + 0.002, np.array([0.0, 0.1, 1.0]), 12, n_paths=500)
    assert (out.xs(0.0, level="w")["p_beat"] == 0).all()
    assert (out.drop(0.0, level="w")["p_beat"] == 1).all()


def test_bootstrap_drawdown_breach():
    r = _monthly(-0.01, 60)  # a 1% loss every month: max-DD = 0.99^60 - 1 = -45.3% on every 5-year path
    out = sc.bootstrap(r, r, np.array([0.0, 0.5]), 12, horizons=(5,), n_paths=100)
    np.testing.assert_allclose(out[["maxdd_p50", "maxdd_p95"]], 0.99**60 - 1, rtol=1e-12)
    assert (out["p_dd_breach"] == 1).all()
    assert (sc.bootstrap(r, r, np.array([0.5]), 12, horizons=(5,), n_paths=100, dd_cap=0.5)["p_dd_breach"] == 0).all()
    # -50% / 0% drawn i.i.d.: over 2 years max-DD is -75% w.p. 1/4, -50% w.p. 1/2, 0 w.p. 1/4, so
    # p95 (the 5th percentile) is -75% and p50 -50%; over 1 year it cannot pass -50%
    r = _years([-0.5, 0.0])
    d = sc.bootstrap(r, r, np.array([0.0]), 1, horizons=(1, 2), n_paths=20_001, mean_block=1)
    assert d.loc[(2, 0.0), "maxdd_p95"] == -0.75 and d.loc[(2, 0.0), "maxdd_p50"] == -0.5
    assert d.loc[(1, 0.0), "maxdd_p95"] == -0.5


def test_bootstrap_seeds():
    rng = np.random.default_rng(9)
    rc, rs = _monthly(rng.normal(0.007, 0.045, 300), 300), _monthly(rng.normal(-0.01, 0.2, 300).clip(-1), 300)
    w = np.array([0.0, 0.03])
    a = sc.bootstrap(rc, rs, w, 12, n_paths=300, seed=1)
    pd.testing.assert_frame_equal(a, sc.bootstrap(rc, rs, w, 12, n_paths=300, seed=1))
    assert not a.equals(sc.bootstrap(rc, rs, w, 12, n_paths=300, seed=2))


def test_bootstrap_speed_2000_paths_240_periods_41_w():
    rng = np.random.default_rng(0)
    rc, rs = _monthly(rng.normal(0.007, 0.045, 420), 420), _monthly(rng.normal(-0.02, 0.3, 420).clip(-1), 420)
    t0 = time.perf_counter()
    out = sc.bootstrap(rc, rs, np.linspace(0, 0.2, 41), 12, horizons=(5, 10, 20), n_paths=2000)
    dt = time.perf_counter() - t0
    print(f"bootstrap 2,000 paths x 240 months x 41 w: {dt:.3f} s")
    # ponytail: loose bound for loaded machines (the backtest speed test's convention); budget ~1 s
    assert out.shape == (3 * 41, 11) and dt < 5


def test_load_sp500_annual():
    s = sc.load_sp500_annual()
    assert s.index[0] == 1928 and s.index[-1] >= 2024 and s.index.dtype.kind == "i"
    assert (np.diff(s.index) == 1).all() and s.notna().all()
    assert -0.40 < s[2008] < -0.35
    assert sc.SP500_CSV.read_text(encoding="utf-8").startswith("# source: ")  # the attribution line


def test_part_one_cartoon_on_1997_2016():
    # Universa "Safe Haven Investing" Part One (2017): over 1997-2016 (bins 2/2/8/8), annually
    # reset, 90% SPX + 10% alpha (+20/+10/+5/+5%) beats SPX by about +18 bp of CAGR and 97% SPX
    # + 3% insurance (+900% / -100%) by about +67 bp
    s = sc.load_sp500_annual().loc[1997:2016]
    assert sc.regime_table(s)["count"].tolist() == [2, 2, 8, 8]
    r = s.to_numpy()
    j = sc.regime_of(r)

    def gain(name: str, w: float) -> float:
        h = sc.book_haven(name, 0.1)[j]
        return np.prod((1 - w) * (1 + r) + w * (1 + h)) ** (1 / 20) - np.prod(1 + r) ** (1 / 20)

    assert gain("alpha", 0.10) == pytest.approx(0.0018, abs=0.0002)
    assert gain("insurance", 0.03) == pytest.approx(0.0067, abs=0.0002)
    # and 90% SPX + 10% store of value earning each year's CPI + 2% real trails SPX by about -17.9 bp (-18.4 here);
    # book_haven's per-bin 1928-2025 mean gives about half that (-9.4 bp, its docstring)
    cpi = sc._annual_table()["cpi"].loc[1997:2016].to_numpy()
    sov = np.prod(0.9 * (1 + r) + 0.1 * (1 + cpi + 0.02)) ** (1 / 20) - np.prod(1 + r) ** (1 / 20)
    assert sov == pytest.approx(-0.0018, abs=0.0002) and gain("store_of_value", 0.10) == pytest.approx(-0.00094, abs=5e-5)

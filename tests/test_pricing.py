"""Tests for insureport.pricing: Black-Scholes-Merton prices and deltas (Hull's textbook values,
dividend-yield hand values, put-call parity, expiry and zero vol, NaN propagation), the SKEW
slope and the smile factor A(c), the smile shape (flat at SKEW 100, the z = +/-2 values, the
call-wing floor, the put-wing cap at z_cap, the nu scaling, the slope cap c_max at nu 0.8, 1 and
1.2 and at z_cap 0), the closed-form floored and capped factor A_f and variance-swap consistency
of the smile, both by numerical integration, put prices non-decreasing in strike (0.01 S and
standardized-moneyness grids, and the two slice-3 review counterexamples), a far-OTM put
non-decreasing in SKEW where c_max binds, the variance-swap term structure
(total-variance interpolation, flat extrapolation, NaN pillars, the 9 d and 184 d pillars), the regression
back-fill of VIX3M / VIX6M / VIX9D (chain order, zero prints, the 250-row overlap threshold,
warnings), the bid/ask spread rule (h = 0 mid fills), keyword pass-through in quote /
strike_for_delta, strike_for_delta, and elementwise vectorization.

Why: every option sleeve, replication and backtest is priced through this module, so its
formulas are pinned here to hand-checkable literals. Pure functions on synthetic inputs, no I/O.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from insureport import pricing

# A 1-month SPX option: r = ln(1 + rf) as callers pass it, q from the panel.
S, T, VS, R, Q = 4000.0, 30 / 365, 0.20, np.log1p(0.05), 0.015
F = S * np.exp((R - Q) * T)


def test_bs_hull_values():
    # Hull: S = K = 100, T = 1, r = 5%, q = 0, sigma = 20%
    assert pricing.bs_price(100, 100, 1, 0.2, 0.05, 0, "C") == pytest.approx(10.4506, abs=1e-4)
    assert pricing.bs_price(100, 100, 1, 0.2, 0.05, 0, "P") == pytest.approx(5.5735, abs=1e-4)
    assert pricing.bs_delta(100, 100, 1, 0.2, 0.05, 0, "C") == pytest.approx(0.6368, abs=1e-4)
    assert pricing.bs_delta(100, 100, 1, 0.2, 0.05, 0, "P") == pytest.approx(0.6368 - 1, abs=1e-4)
    # q = 3%: d1 = (0.05 - 0.03 + 0.02) / 0.2 = 0.2, d2 = 0, so N(d2) = 1/2. Call = 100 e^-.03 N(.2) - 50 e^-.05,
    # put = 50 e^-.05 - 100 e^-.03 N(-.2), delta call = e^-.03 N(.2), delta put = -e^-.03 N(-.2)
    assert pricing.bs_price(100, 100, 1, 0.2, 0.05, 0.03, "C") == pytest.approx(8.652529, abs=1e-6)
    assert pricing.bs_price(100, 100, 1, 0.2, 0.05, 0.03, "P") == pytest.approx(6.730918, abs=1e-6)
    assert pricing.bs_delta(100, 100, 1, 0.2, 0.05, 0.03, "C") == pytest.approx(0.562140, abs=1e-6)
    assert pricing.bs_delta(100, 100, 1, 0.2, 0.05, 0.03, "P") == pytest.approx(-0.408306, abs=1e-6)
    with pytest.raises(ValueError):
        pricing.bs_price(100, 100, 1, 0.2, 0.05, 0, "put")


def test_bs_parity_dividends_expiry_and_nan():
    K = np.array([80.0, 95, 100, 105, 130])
    c = pricing.bs_price(100, K, 0.5, 0.25, 0.04, 0.02, "C")
    p = pricing.bs_price(100, K, 0.5, 0.25, 0.04, 0.02, "P")
    np.testing.assert_allclose(c - p, 100 * np.exp(-0.02 * 0.5) - K * np.exp(-0.04 * 0.5), rtol=0, atol=1e-10)
    dc, dp = (pricing.bs_delta(100, K, 0.5, 0.25, 0.04, 0.02, cp) for cp in "CP")
    np.testing.assert_allclose(dc - dp, np.exp(-0.02 * 0.5), rtol=1e-12)
    assert pricing.bs_price(100, 100, 1, 0.2, 0.05, 0.03, "C") < pricing.bs_price(100, 100, 1, 0.2, 0.05, 0, "C")
    K = np.array([90.0, 100, 110])
    for t, sig in [(0, 0.2), (1, 0.0), (-0.1, 0.2)]:  # expired or zero vol: plain intrinsic
        assert pricing.bs_price(100, K, t, sig, 0.05, 0, "P").tolist() == [0, 0, 10]
        assert pricing.bs_price(100, K, t, sig, 0.05, 0, "C").tolist() == [10, 0, 0]
        assert pricing.bs_delta(100, K, t, sig, 0.05, 0, "P").tolist() == [0, 0, -1]
        assert pricing.bs_delta(100, K, t, sig, 0.05, 0, "C").tolist() == [1, 0, 0]
    # a missing vol is not "zero vol": NaN must propagate, not silently become intrinsic
    assert np.isnan(pricing.bs_price(100, 90, 1, np.nan, 0.05, 0, "P"))
    assert np.isnan(pricing.bs_delta(100, 90, 1, np.nan, 0.05, 0, "P"))


def test_skew_slope_and_smile_factor():
    assert pricing.skew_slope(130, m=1) == pytest.approx(0.5)  # 1 * 30 / 10 / 6
    assert pricing.skew_slope(90) == 0
    # A(0.5) = 1 + 0.5/sqrt(2 pi) + 5/32 = 1 + 0.19947114 + 0.15625 = 1.35572114
    assert pricing.smile_factor(0.5) == pytest.approx(1.355721, abs=1e-6)
    assert pricing.smile_factor(0) == 1


def test_flat_smile_at_skew_100():
    K = np.array([2000.0, 3600, 4000, 4400, 6000])
    np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, 100, R, Q, nu=0.9), 0.9 * VS, rtol=1e-14)


def test_smile_shape_and_call_wing_floor():
    c = 0.25  # SKEW 130, m = 0.5: 0.5 * 30 / 10 / 6
    s0 = float(pricing.smile_vol(S, F, T, VS, 130, R, Q, m=0.5))  # K = F: z = 0, the ATM vol sigma0
    # A_f(0.25, 3) = A(c) 1.138798 + cap -3.471e-4 [2c(3Q - phi) + c^2(8Q - 3 phi), Q = N(-3), phi = phi(3)]
    # + floor 8e-7 (binds below z = -4) = 1.138452
    assert s0 == pytest.approx(VS / np.sqrt(1.138452), rel=1e-6)
    K = F * np.exp(-np.array([2.0, 5.0, -2.0, -10.0]) * s0 * np.sqrt(T))  # z = +2, +5, -2, -10
    hand = np.array([s0 * (1 + 2 * c), s0 * (1 + pricing.Z_CAP * c), s0 * (1 - c), 0.5 * s0])  # z = 5: capped at 3
    np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, 130, R, Q, m=0.5), hand, rtol=1e-12)
    # nu scales the whole smile, floor and cap included (c = 0.25 < c_max(0.8, 3) = 0.763, so the slope
    # cap does not bind); z is still standardised by sigma0, not nu sigma0
    np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, 130, R, Q, m=0.5, nu=0.8), 0.8 * hand, rtol=1e-12)
    assert K[1] < K[0] < F < K[2] < K[3]  # z > 0 is the OTM-put side
    # expired: the ATM vol; a missing T is not expiry and stays NaN
    np.testing.assert_allclose(pricing.smile_vol(S, 3000.0, np.array([0, -0.1]), VS, 130, R, Q, m=0.5), s0, rtol=1e-14)
    assert np.isnan(pricing.smile_vol(S, F, np.nan, VS, 130, R, Q))


@pytest.mark.parametrize("z_cap", [2.5, 3.0, 4.0])
@pytest.mark.parametrize("c", [0.1, 0.3, 0.6])
def test_floored_smile_factor_closed_form(c, z_cap):
    # A_f(c, b) = E[max(1 + c min(z+, b) - (c/2) z-, 0.5)^2], directly (smile_vol clips c at c_max)
    def f(z: float) -> float:
        return max(1 + c * min(max(z, 0), z_cap) - c / 2 * max(-z, 0), 0.5) ** 2 * norm.pdf(z)

    ev, _ = quad(f, -12, 12, points=[-1 / c, 0, z_cap], epsabs=1e-14, epsrel=1e-12)
    assert pricing._floored_smile_factor(c, z_cap) == pytest.approx(ev, rel=1e-9)


@pytest.mark.parametrize("z_cap", [2.5, 3.0, 4.0])
@pytest.mark.parametrize("c", [0.1, 0.3, 0.6])
def test_smile_is_variance_swap_consistent(c, z_cap):
    skew = 100 + 60 * c  # with m = 1, c = (SKEW - 100) / 60
    kw = {"m": 1, "z_cap": z_cap}
    s0 = float(pricing.smile_vol(S, F, T, VS, skew, R, Q, **kw))  # K = F: z = 0, the ATM vol sigma0
    ce = min(c, pricing._max_slope(1.0, z_cap))  # c_max(1, 4) = 0.5991 binds at c = 0.6, z_cap 4

    def vol(z: float) -> float:  # the module's smile at standardized moneyness z, floor and cap included
        return float(pricing.smile_vol(S, F * np.exp(-z * s0 * np.sqrt(T)), T, VS, skew, R, Q, **kw))

    ev, _ = quad(lambda z: vol(z) ** 2 * norm.pdf(z), -12, 12, points=[-1 / ce, 0, z_cap], epsabs=1e-14, epsrel=1e-12)
    assert ev == pytest.approx(VS**2, rel=1e-9)  # stricter than the brief's 1e-8


@pytest.mark.parametrize(
    "skew, nu, c_max",
    [(160, 1.0, 0.649827533), (183, 0.8, 0.762818097), (175, 1.2, 0.569180559)],  # m 0.7: c = 0.7, 0.968, 0.875
)
def test_steep_slope_capped_where_puts_would_fall(skew, nu, c_max):
    # dP/dK >= 0 at every T on the wing up to Z_CAP = 3 needs N(-x)/phi(x) >= nu c at x = 3/(nu(1 + 3c));
    # c_max is where that holds with equality (hand-solved by brentq)
    x = 3 / (nu * (1 + 3 * c_max))
    assert norm.sf(x) / norm.pdf(x) == pytest.approx(nu * c_max, rel=1e-8)
    s0 = float(pricing.smile_vol(S, F, T, VS, skew, R, Q, m=0.7, nu=nu)) / nu
    K = F * np.exp(-np.array([1.0, 5.0, -1.0]) * s0 * np.sqrt(T))  # z = 1, 5 (flat past 3), -1
    hand = nu * s0 * np.array([1 + c_max, 1 + 3 * c_max, 1 - c_max / 2])
    np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, skew, R, Q, m=0.7, nu=nu), hand, rtol=1e-8)
    # z_cap = 0: x = 0 for every c, so c_max = R(0)/nu = sqrt(pi/2)/nu (the root sits on R(0)/nu exactly)
    assert pricing._max_slope(1.17, 0.0) == pytest.approx(np.sqrt(np.pi / 2) / 1.17, rel=1e-12)


def test_nu_not_a_pure_scale_where_slope_is_capped():
    # SKEW 175, m 0.6: c = 0.75 > c_max(1) > c_max(1.2) (pinned above), so nu 1 -> 1.2 also flattens the wing.
    # ATM (K = the 61 d forward, z = 0) the ratio is nu sigma0(c_max(1.2)) / sigma0(c_max(1))
    # = 1.2 sqrt(A_f(c_max(1), 3) / A_f(c_max(1.2), 3)) = 1.239674 > 1.2; the 25% OTM vol by less than 20%
    F61 = S * np.exp((R - Q) * 61 / 365)
    vol = [pricing.smile_vol(S, np.array([F61, 3000.0]), 61 / 365, VS, 175, R, Q, m=0.6, nu=nu) for nu in (1.2, 1.0)]
    ratio = vol[0] / vol[1]
    a1, a12 = pricing._floored_smile_factor(0.649827533, 3), pricing._floored_smile_factor(0.569180559, 3)
    assert ratio[0] == pytest.approx(1.2 * np.sqrt(a1 / a12), rel=1e-8)
    assert 1 < ratio[1] < 1.2


@pytest.mark.parametrize(
    "skew, m, kw",
    [
        (120, 0.5, {}),
        (120, 0.7, {}),
        (160, 0.5, {}),
        (160, 0.7, {}),
        (183, 0.7, {"z_cap": 4.0, "nu": 1.2}),
        (183, 0.7, {"nu": 0.8}),
        (183, 0.7, {"nu": 1.3}),
    ],
)
def test_put_mids_non_decreasing_in_strike(skew, m, kw):
    # the 183 rows are stress cases: SKEW's 2025 high, the widest swept cap, vol premia either side of 1
    K = S * np.arange(30, 101) / 100  # 0.30 S ... 1.00 S, step 0.01 S
    tenors = np.array([1 / 365, 7 / 365, 30 / 365, 60 / 365, 0.5])[:, None]  # one row per tenor
    mid = pricing.quote(S, K, tenors, 0.15, skew, R, Q, "P", m=m, **kw)[0]
    assert mid.shape == (5, 71)
    assert (np.diff(mid, axis=1) >= -1e-9).all()
    # that grid holds only a few points of the 1 d and 7 d wing, so also step in standardized moneyness,
    # z from 8 down to -2 (K rising), across the linear wing and the cap at every tenor
    z = np.linspace(8, -2, 1001)
    Kz = S * np.exp((R - Q) * tenors) * np.exp(-z * 0.15 * np.sqrt(tenors))
    mid = pricing.quote(S, Kz, tenors, 0.15, skew, R, Q, "P", m=m, **kw)[0]
    assert (np.diff(mid, axis=1) >= -1e-9).all()


def test_far_put_non_decreasing_in_skew():
    # a higher SKEW never cheapens the 25% OTM 2-month put (Universa preset), also past SKEW 156 where
    # c_max binds at m 0.7 and the slope stops steepening
    mid = pricing.quote(S, 0.75 * S, 61 / 365, VS, np.arange(120, 201), R, Q, m=0.7)[0]
    assert (np.diff(mid) >= -1e-9).all()


def test_review_counterexamples_fixed():
    # slice-3 review, r = q = 0: the uncapped wing gave P(0.30 S) = 64.5 > P(0.96 S) = 17.5 at 7 d
    p30, p96 = pricing.quote(S, np.array([0.30, 0.96]) * S, 7 / 365, 0.15, 160, 0, 0, "P", m=0.7)[0]
    assert p30 < p96
    # ... and a 25% OTM put that kept 1.128 of value as T -> 0+; now it decays to 0
    mid = pricing.quote(4000.0, 3000.0, np.array([1 / 365, 1e-6]), 0.15, 150, 0, 0, "P", m=0.5)[0]
    assert mid[0] < 1e-3 and mid[1] < 1e-9


def test_var_swap_term_structure():
    days = np.array([61.5, 5, 365, 30, 61.5])
    vix = np.array([20, 20, 20, 20, np.nan])  # pillar 30 d
    vix3m = np.array([25, 25, 25, 25, np.nan])  # pillar 93 d; vix9d and vix6m missing
    v = pricing.var_swap_vol(days / 365, np.nan, vix, vix3m, np.nan)
    w30, w93, w184 = 0.20**2 * 30, 0.25**2 * 93, 0.25**2 * 184  # total variance x 365
    hand = np.sqrt((w30 + (w93 - w30) * (61.5 - 30) / (93 - 30)) / 61.5)  # 0.238772
    assert v[0] == pytest.approx(hand, rel=1e-12) and 0.20 < v[0] < 0.25
    assert v[1] == pytest.approx(0.20, rel=1e-12)  # before the first pillar: flat vol
    assert v[2] == pytest.approx(0.25, rel=1e-12)  # after the last pillar: flat vol
    assert v[3] == pytest.approx(0.20, rel=1e-12)  # on a pillar
    assert np.isnan(v[4])  # no pillar at all
    one = pricing.var_swap_vol(np.array([5, 30, 61.5, 365]) / 365, np.nan, 20, np.nan, np.nan)
    np.testing.assert_allclose(one, 0.20, rtol=1e-12)  # a single pillar is used flat
    # a NaN 93 d pillar is skipped: 30 d interpolates straight to 184 d
    gap = pricing.var_swap_vol(61.5 / 365, np.nan, 20, np.nan, 25)
    assert gap == pytest.approx(np.sqrt((w30 + (w184 - w30) * 31.5 / 154) / 61.5), rel=1e-12)  # 0.231882
    # VIX6M's pillar is Cboe's 184 d target: exact there, and 182 d interpolates between 93 d and 184 d
    six = pricing.var_swap_vol(np.array([184, 182]) / 365, 15, 20, 25, 30)
    assert six[0] == 0.30
    hand = np.sqrt((0.25**2 * 93 + (0.30**2 * 184 - 0.25**2 * 93) * 89 / 91) / 182)  # 0.299485
    assert six[1] == pytest.approx(hand, rel=1e-12)
    v9 = pricing.var_swap_vol(np.array([5, 20]) / 365, 15, 20, np.nan, np.nan)  # VIX9D 15 @ 9 d
    assert v9[0] == pytest.approx(0.15, rel=1e-12)  # below 9 d: flat at the VIX9D vol
    w9 = 0.15**2 * 9
    assert v9[1] == pytest.approx(np.sqrt((w9 + (w30 - w9) * 11 / 21) / 20), rel=1e-12)  # 0.190394


def _panel(n: int = 600) -> pd.DataFrame:
    """Deterministic VIX family with ln VIX3M = 0.3 + 0.9 ln VIX exactly."""
    t = np.arange(n)
    vix = 18 + 6 * np.sin(t / 9) + t / 100
    vix3m = np.exp(0.3 + 0.9 * np.log(vix))
    return pd.DataFrame(
        {"vix": vix, "vix9d": 0.9 * vix, "vix3m": vix3m, "vix6m": 1.05 * vix3m},
        pd.bdate_range("2005-01-03", periods=n),
    )


def test_fill_term_structure_backfills_vix3m():
    mkt = _panel()
    mkt.iloc[:300, mkt.columns.get_loc("vix3m")] = np.nan  # VIX3M only on the second half
    mkt.iloc[10, mkt.columns.get_loc("vix")] = np.nan  # no regressor on row 10: stays NaN
    mkt.iloc[400, mkt.columns.get_loc("vix3m")] = np.nan  # gap after the first valid date: not filled
    mkt.iloc[[20, 450], mkt.columns.get_loc("vix")] = 0.0  # bad zero prints: not a regressor, not in the fit
    mkt.attrs["warnings"] = ["from load_market"]
    out = pricing.fill_term_structure(mkt)
    expected = _panel()["vix3m"].where(~mkt.index.isin(mkt.index[[10, 20, 400]]))
    np.testing.assert_allclose(out["vix3m"], expected, rtol=0, atol=1e-9)
    assert out["vix3m"].iloc[[10, 20, 400]].isna().all()
    assert out["ts_filled"].dtype == bool
    assert out["ts_filled"].tolist() == [i < 300 and i not in (10, 20) for i in range(600)]
    assert out.attrs["warnings"] == ["from load_market"]
    pd.testing.assert_frame_equal(out[["vix", "vix9d", "vix6m"]], mkt[["vix", "vix9d", "vix6m"]])
    assert mkt["vix3m"].iloc[:300].isna().all() and "ts_filled" not in mkt  # input untouched


def test_fill_term_structure_chain_and_threshold():
    mkt = _panel()
    ln_vix = np.log(mkt["vix"])
    vix3m = np.exp(0.3 + 0.9 * ln_vix)
    vix6m = np.exp(0.1 + 0.95 * np.log(vix3m))  # = exp(0.1 + 0.95 (0.3 + 0.9 ln vix))
    vix9d = np.exp(-0.2 + 1.1 * ln_vix)
    mkt["vix6m"], mkt["vix9d"] = vix6m, vix9d
    mkt.iloc[:300, mkt.columns.get_loc("vix3m")] = np.nan
    mkt.iloc[:340, mkt.columns.get_loc("vix6m")] = np.nan  # rows < 300 need the filled vix3m
    mkt.iloc[:350, mkt.columns.get_loc("vix9d")] = np.nan  # exactly 250 overlap rows: enough to fill
    mkt.iloc[320, mkt.columns.get_loc("vix")] = np.nan  # vix3m is native here, vix is not
    out = pricing.fill_term_structure(mkt)
    np.testing.assert_allclose(out["vix3m"], vix3m, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["vix6m"], vix6m, rtol=0, atol=1e-9)  # row 320 too: regressed on vix3m
    np.testing.assert_allclose(out["vix9d"], vix9d.where(mkt.index != mkt.index[320]), rtol=0, atol=1e-9)
    assert out["ts_filled"].tolist() == [i < 350 for i in range(600)]  # union over the three targets
    assert out.attrs["warnings"] == []
    again = pricing.fill_term_structure(out)  # nothing left to fill, and the flags survive
    pd.testing.assert_frame_equal(again, out)


@pytest.mark.parametrize(
    "blank, warned",
    [({"vix9d": 351}, ["vix9d"]), ({"vix9d": 600}, ["vix9d"]), ({"vix3m": 600, "vix6m": 300}, ["vix3m", "vix6m"])],
    ids=["short-overlap", "all-missing", "regressor-missing"],
)
def test_fill_term_structure_short_overlap_warns(blank, warned):
    mkt = _panel()
    for col, n in blank.items():  # NaN on the first n rows
        mkt.iloc[:n, mkt.columns.get_loc(col)] = np.nan
    out = pricing.fill_term_structure(mkt)
    pd.testing.assert_frame_equal(out.drop(columns="ts_filled"), mkt)  # nothing filled
    assert not out["ts_filled"].any()
    assert [w.split(":")[0] for w in out.attrs["warnings"]] == warned


def test_quote_spread_rule():
    # T = 0 prices at intrinsic, which pins the mids exactly: 10, 0.2, 0; the last put is far OTM
    K = np.array([110.0, 100.2, 100.0, 50.0])
    mid, bid, ask = pricing.quote(100.0, K, np.array([0, 0, 0, 30 / 365]), 0.2, 120, 0.05, 0.0, "P")
    np.testing.assert_allclose(mid[:3], [10, 0.2, 0], rtol=0, atol=1e-12)
    assert 0 < mid[3] < 0.001
    np.testing.assert_allclose(ask - mid, [0.5, 0.05, 0.05, 0.05], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bid, [9.5, 0.15, 0, 0], rtol=0, atol=1e-12)
    assert (bid >= 0).all() and bid[2] == 0 and bid[3] == 0
    mid0, bid0, ask0 = pricing.quote(100.0, K, np.array([0, 0, 0, 30 / 365]), 0.2, 120, 0.05, 0.0, "P", h=0)
    assert (bid0 == mid0).all() and (ask0 == mid0).all() and (mid0 == mid).all()  # h = 0: mid fills


def test_quote_and_strike_pass_keywords_through():
    K, kw = np.array([3900.0, 4200]), {"m": 0.7, "nu": 1.1, "z_cap": 2.5}
    mid, _, ask = pricing.quote(S, K, T, VS, 130, R, Q, "C", h=0.1, **kw)
    vol = pricing.smile_vol(S, K, T, VS, 130, R, Q, **kw)
    np.testing.assert_allclose(mid, pricing.bs_price(S, K, T, vol, R, Q, "C"), rtol=1e-14)
    np.testing.assert_allclose(ask - mid, np.maximum(0.1 * mid, 0.05), rtol=1e-12)
    assert (mid > 0.5).all()  # so h = 0.1, not the 0.05 floor, sets the half-spread
    assert (mid != pricing.quote(S, K, T, VS, 130, R, Q, "C")[0]).all()
    Kd = pricing.strike_for_delta(S, T, VS, 130, R, Q, 0.05, **kw)
    vol_d = pricing.smile_vol(S, Kd, T, VS, 130, R, Q, **kw)
    assert pricing.bs_delta(S, Kd, T, vol_d, R, Q, "P") == pytest.approx(-0.05, abs=1e-6)
    assert Kd != pytest.approx(pricing.strike_for_delta(S, T, VS, 130, R, Q, 0.05), rel=1e-4)
    # the cap alone: a 20% OTM 1-month put sits at z ~ 4.2, past both caps
    far = [pricing.quote(S, 0.80 * S, T, VS, 130, R, Q, z_cap=z)[0] for z in (2.5, 4.0)]
    assert far[0] < 0.5 * far[1]
    for z_cap in (2.5, 4.0):  # the 0.001-delta put strike lies past the cap; the solver still hits the delta
        Kz = pricing.strike_for_delta(S, T, VS, 130, R, Q, 0.001, z_cap=z_cap)
        vol_z = pricing.smile_vol(S, Kz, T, VS, 130, R, Q, z_cap=z_cap)
        assert pricing.bs_delta(S, Kz, T, vol_z, R, Q, "P") == pytest.approx(-0.001, rel=1e-6)


def test_strike_for_delta():
    K = pricing.strike_for_delta(S, T, VS, 130, R, Q, 0.05)
    assert isinstance(K, float) and K < F
    vol = pricing.smile_vol(S, K, T, VS, 130, R, Q)
    assert pricing.bs_delta(S, K, T, vol, R, Q, "P") == pytest.approx(-0.05, abs=1e-6)
    assert pricing.strike_for_delta(S, T, VS, 130, R, Q, -0.05) == pytest.approx(K, rel=1e-12)  # sign ignored
    Kc = pricing.strike_for_delta(S, T, VS, 130, R, Q, 0.25, "C")
    vol_c = pricing.smile_vol(S, Kc, T, VS, 130, R, Q)
    assert Kc > F and pricing.bs_delta(S, Kc, T, vol_c, R, Q, "C") == pytest.approx(0.25, abs=1e-6)
    with pytest.raises(ValueError):
        pricing.strike_for_delta(S, T, VS, 130, R, Q, 1.5)
    with pytest.raises(ValueError):
        pricing.strike_for_delta(S, T, np.nan, 130, R, Q, 0.05)


def test_vectorized_matches_scalar():
    s = np.array([4000.0, 3500, 2800])
    k = np.array([3800.0, 3600, 2000])
    t = np.array([30, 61, 182]) / 365
    vs = np.array([0.15, 0.25, 0.60])
    skew = np.array([115.0, 130, 150])
    vec = pricing.quote(s, k, t, vs, skew, R, Q)
    ts = pricing.var_swap_vol(t, 0.9 * vs * 100, vs * 100, 1.1 * vs * 100, np.nan)
    for i in range(3):
        one = pricing.quote(s[i], k[i], t[i], vs[i], skew[i], R, Q)
        for a, b in zip(vec, one):
            assert a.shape == (3,) and a[i] == pytest.approx(float(b), rel=1e-14)
        assert ts[i] == pytest.approx(float(pricing.var_swap_vol(t[i], 90 * vs[i], 100 * vs[i], 110 * vs[i], np.nan)))
    series = pricing.smile_vol(pd.Series(s, index=[7, 8, 9]), k, t, vs, skew, R, Q)  # used positionally
    assert isinstance(series, np.ndarray)
    np.testing.assert_array_equal(series, pricing.smile_vol(s, k, t, vs, skew, R, Q))

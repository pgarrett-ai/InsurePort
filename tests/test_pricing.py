"""Tests for insureport.pricing: Black-Scholes-Merton prices and deltas (Hull's textbook values,
dividend-yield hand values, put-call parity, expiry and zero vol, NaN propagation), the SKEW
slope and the smile factor A(c), the smile shape (flat at SKEW 100, linear in z on the put wing
with no flat cap by default, the z_cap knob, the call-wing floor, the nu scaling, flat at expiry),
the closed-form floored (and capped) factor A_f and variance-swap consistency of the smile, both
by numerical integration, the monotonicity slope cap c_max(nu, s) (brute-force hand checks on both
branches, rising in s, the cached table vs the direct solve, the fixed point the smile uses, nu not a
pure scale where it binds), Lee's bound on the far wing (the slope cap that keeps its root, the
call wing, the put worth about half its strike past it), the deep wing's known convexity breach,
put prices non-decreasing in strike over a tenor x vol x SKEW x m x nu grid (strike and
standardized-moneyness grids, and the two slice-3 review counterexamples), a far-OTM put
non-decreasing in SKEW where c_max binds, wing IV falling with tenor in contango, the 2026-10-08
XSP quotes as a loose anchor, the variance-swap term structure (total-variance interpolation, flat
extrapolation, NaN pillars, the 9 d, 184 d and 365 d pillars), the regression back-fill of VIX3M /
VIX6M / VIX9D / VIX1Y (chain order, zero prints, fits on native rows only, the 250-row overlap
threshold, warnings), the bid/ask spread rule (h = 0 mid fills), keyword pass-through in quote /
strike_for_delta, strike_for_delta (small put deltas on the linear wing), and elementwise
vectorization.

Why: every option sleeve, replication and backtest is priced through this module, so its
formulas are pinned here to hand-checkable literals. Pure functions on synthetic inputs, no I/O.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
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
    c = 0.25  # SKEW 130, m = 0.5: 0.5 * 30 / 10 / 6, below the slope cap (0.513 at this 30 d tenor)
    s0 = float(pricing.smile_vol(S, F, T, VS, 130, R, Q, m=0.5))  # K = F: z = 0, the ATM vol sigma0
    # A_f(0.25, inf) = A(c) 1.138798 + floor 8.4e-7 (binds below z = -4) = 1.138799; no cap term
    assert s0 == pytest.approx(VS / np.sqrt(1.138799), rel=1e-6)
    K = F * np.exp(-np.array([2.0, 5.0, -2.0, -10.0]) * s0 * np.sqrt(T))  # z = +2, +5, -2, -10
    hand = np.array([s0 * (1 + 2 * c), s0 * (1 + 5 * c), s0 * (1 - c), 0.5 * s0])  # z = 5: still linear
    np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, 130, R, Q, m=0.5), hand, rtol=1e-12)
    # the z_cap knob still flattens the wing: z_cap = 3 gives sigma0 from A_f(0.25, 3) = A(c) + cap -3.471e-4
    # [2c(3Q - phi) + c^2(8Q - 3 phi), Q = N(-3), phi = phi(3)] + floor = 1.138452, and z = 5 prices as z = 3
    s3 = float(pricing.smile_vol(S, F, T, VS, 130, R, Q, m=0.5, z_cap=3.0))
    assert s3 == pytest.approx(VS / np.sqrt(1.138452), rel=1e-6)
    K5 = F * np.exp(-5 * s3 * np.sqrt(T))
    assert pricing.smile_vol(S, K5, T, VS, 130, R, Q, m=0.5, z_cap=3.0) == pytest.approx(s3 * (1 + 3 * c), rel=1e-12)
    # nu scales the whole smile, floor included (c = 0.25 < the cap at nu 0.8, 0.582, so the slope cap does
    # not bind); z is still standardised by sigma0, not nu sigma0
    np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, 130, R, Q, m=0.5, nu=0.8), 0.8 * hand, rtol=1e-12)
    assert K[1] < K[0] < F < K[2] < K[3]  # z > 0 is the OTM-put side
    # expired: c_max(nu, 0) = 0, so the smile is flat at nu vs; a missing T is not expiry and stays NaN
    T0 = np.array([0, -0.1])
    np.testing.assert_allclose(pricing.smile_vol(S, 3000.0, T0, VS, 130, R, Q, m=0.5, nu=1.1), 1.1 * VS, rtol=1e-14)
    assert np.isnan(pricing.smile_vol(S, F, np.nan, VS, 130, R, Q))
    for bad in (0.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="nu"):
            pricing.smile_vol(S, F, T, VS, 130, R, Q, nu=bad)


@pytest.mark.parametrize("z_cap", [2.5, 3.0, 4.0, np.inf])
@pytest.mark.parametrize("c", [0.1, 0.3, 0.6])
def test_floored_smile_factor_closed_form(c, z_cap):
    # A_f(c, b) = E[max(1 + c min(z+, b) - (c/2) z-, 0.5)^2], directly (smile_vol clips c at c_max)
    def f(z: float) -> float:
        return max(1 + c * min(max(z, 0), z_cap) - c / 2 * max(-z, 0), 0.5) ** 2 * norm.pdf(z)

    ev, _ = quad(f, -12, 12, points=[-1 / c, 0, min(z_cap, 12)], epsabs=1e-14, epsrel=1e-12)
    assert pricing._floored_smile_factor(c, z_cap) == pytest.approx(ev, rel=1e-9)


@pytest.mark.parametrize("t", [30 / 365, 0.5])
@pytest.mark.parametrize("nu", [1.0, 1.2])
@pytest.mark.parametrize("z_cap", [np.inf, 3.0])
@pytest.mark.parametrize("c", [0.1, 0.3, 0.6])
def test_smile_is_variance_swap_consistent(c, z_cap, nu, t):
    skew = 100 + 60 * c  # with m = 1, c = (SKEW - 100) / 60
    kw = {"m": 1, "z_cap": z_cap, "nu": nu}
    f = S * np.exp((R - Q) * t)
    s0 = float(pricing.smile_vol(S, f, t, VS, skew, R, Q, **kw)) / nu  # K = F: z = 0, nu sigma0
    # 0.6 is above the cap (0.513 / 0.46) at 30 d, below it (0.714 / 0.650) at 0.5 y
    ce = min(c, float(pricing._slope_cap(t, VS, nu, z_cap)))
    assert t == 30 / 365 or ce == c

    def vol(z: float) -> float:  # the module's smile at standardized moneyness z, floor and cap included
        return float(pricing.smile_vol(S, f * np.exp(-z * s0 * np.sqrt(t)), t, VS, skew, R, Q, **kw))

    ev, _ = quad(lambda z: vol(z) ** 2 * norm.pdf(z), -12, 12, points=[-1 / ce, 0, min(z_cap, 12)],
                 epsabs=1e-14, epsrel=1e-12)
    assert ev == pytest.approx((nu * VS) ** 2, rel=1e-9)  # stricter than the brief's 1e-8; nu scales vs


def _worst_d2(nu: float, s: float, c: float) -> float:
    """Brute force: the largest d2 = z/(nu(1 + c z)) - nu s (1 + c z)/2 on the linear put wing z >= 0."""
    def d2(z: float) -> float:
        return z / (nu * (1 + c * z)) - nu * s * (1 + c * z) / 2

    res = minimize_scalar(lambda u: -d2(np.expm1(u)), bounds=(0, 25), method="bounded", options={"xatol": 1e-12})
    return max(-res.fun, d2(0.0))


@pytest.mark.parametrize("nu, s, c_max", [(1.2, 0.05, 0.463218), (2.0, 0.75, 1.284086)], ids=["interior", "atm"])
def test_max_slope_hand_check(nu, s, c_max):
    # dP/dK = e^{-rT} [N(-d2) - nu c phi(d2)] >= 0 on the linear wing iff the Mills ratio N(-d2)/phi(d2) >= nu c
    # at the worst (largest) d2; c_max solves that with equality. nu 1.2, s 0.05 (b = nu s = 0.06) peaks inside
    # the wing; nu 2, s 0.75 (b = 1.5, a b > 2) at the money
    def mills(x: float) -> float:
        return norm.sf(x) / norm.pdf(x)  # N(-x)/phi(x), not the module's erfcx form (d2 stays small here)

    hand = brentq(lambda c: mills(_worst_d2(nu, s, c)) - nu * c, 0.05, 5, xtol=1e-14)
    assert float(pricing._max_slope(nu, s)) == pytest.approx(hand, abs=1e-6)
    assert hand == pytest.approx(c_max, abs=1e-6)


def test_max_slope_rising_in_s_and_table():
    # the -sigma sqrt(T)/2 term pulls d2 down, so more total vol s allows a steeper wing: c_max RISES with s
    # (~(2 nu s)^(1/3) / nu for small s, 0 at s = 0: the smile flattens as T -> 0)
    s_grid = np.geomspace(0.003, 1.5, 60)
    for n in (0.5, 1.0, 2.0):
        assert (np.diff(pricing._max_slope(n, s_grid)) > 0).all()
    assert pricing._max_slope(1.0, 0.0) == 0.0
    # the cached table (used by smile_vol) agrees with the direct solve, and never overshoots it
    rng = np.random.default_rng(3)
    n, s = rng.uniform(0.5, 2.0, 20), np.exp(rng.uniform(np.log(0.003), np.log(1.5), 20))
    np.testing.assert_allclose(pricing._c_max(n, s), pricing._max_slope(n, s), rtol=0, atol=1e-4)
    assert (pricing._c_max(n, s) <= pricing._max_slope(n, s)).all()


def test_steep_slope_capped_at_the_fixed_point():
    # SKEW 183, m 0.7: c = 0.968 is above the cap at 30 d. The smile uses c* = c_max(nu, sigma0(c*) sqrt T),
    # the slope whose own sigma0 makes it exactly safe (sigma0 falls as c rises, c_max rises with s)
    for nu in (0.8, 1.0, 1.3):
        cap = float(pricing._slope_cap(T, VS, nu, np.inf))
        s0 = float(pricing.smile_vol(S, F, T, VS, 183, R, Q, m=0.7, nu=nu)) / nu
        assert s0 == pytest.approx(VS / np.sqrt(pricing._floored_smile_factor(cap, np.inf)), rel=1e-12)
        exact = float(pricing._max_slope(nu, s0 * np.sqrt(T)))
        assert cap <= exact and cap == pytest.approx(exact, rel=1e-4) and cap < 0.968
        K = F * np.exp(-np.array([1.0, 5.0, -1.0]) * s0 * np.sqrt(T))  # z = 1, 5, -1
        hand = nu * s0 * np.array([1 + cap, 1 + 5 * cap, 1 - cap / 2])
        np.testing.assert_allclose(pricing.smile_vol(S, K, T, VS, 183, R, Q, m=0.7, nu=nu), hand, rtol=1e-12)
    # the cap is per tenor: tighter at 7 d than at 30 d, looser at 1 y
    caps = pricing._slope_cap(np.array([7, 30, 365]) / 365, VS, 1.0, np.inf)
    assert (np.diff(caps) > 0).all()
    # where it binds nu is not a pure scale: c*(nu 1.2) < c*(nu 1), so the ATM vol rises by more than 1.2x
    # (sigma0 = vs / sqrt(A_f(c*))) and the 25% OTM vol by less (61 d, SKEW 183, m 0.7)
    t, f = 61 / 365, S * np.exp((R - Q) * 61 / 365)
    a1, a2 = (pricing._floored_smile_factor(pricing._slope_cap(t, VS, n, np.inf), np.inf) for n in (1.0, 1.2))
    v1, v2 = (pricing.smile_vol(S, np.array([f, 0.75 * S]), t, VS, 183, R, Q, m=0.7, nu=n) for n in (1.0, 1.2))
    atm, otm = v2 / v1
    assert atm == pytest.approx(1.2 * np.sqrt(a1 / a2), rel=1e-12) and atm > 1.2 + 1e-3
    assert 1 < otm < 1.2


def test_lee_bound_guards_the_extreme_tail():
    # far from binding at z <= 12 for ordinary inputs (vs 0.15, SKEW 149, m 0.3: still linear at z = 12) ...
    for t in (1 / 365, 1.0):
        f, s0 = S * np.exp((R - Q) * t), float(pricing.smile_vol(S, S * np.exp((R - Q) * t), t, 0.15, 149, R, Q, m=0.3))
        c = float(pricing.smile_vol(S, f * np.exp(-s0 * np.sqrt(t)), t, 0.15, 149, R, Q, m=0.3)) / s0 - 1  # z = 1
        z12 = pricing.smile_vol(S, f * np.exp(-12 * s0 * np.sqrt(t)), t, 0.15, 149, R, Q, m=0.3)
        assert z12 == pytest.approx(s0 * (1 + 12 * c), rel=1e-12)
    # ... but with vs 0.8, SKEW 200, m 1, nu 1.5 at 7 d (c = 0.526, sigma0 sqrt T = 0.094) the linear wing
    # at k = ln(K/F) = -3 would ask for sigma = nu sigma0 (1 + c z) = 18.12 > Lee's sqrt(2 |k| / T) = 17.69
    # (the far crossing is at k = -2.84): clipped to the bound
    t, kw = 7 / 365, {"m": 1.0, "nu": 1.5}
    f = S * np.exp((R - Q) * t)
    s0 = float(pricing.smile_vol(S, f, t, 0.8, 200, R, Q, **kw)) / 1.5
    c = float(pricing.smile_vol(S, f * np.exp(-s0 * np.sqrt(t)), t, 0.8, 200, R, Q, **kw)) / (1.5 * s0) - 1  # z = 1
    linear, bound = 1.5 * s0 * (1 + c * 3 / (s0 * np.sqrt(t))), np.sqrt(2 * 3 / t)
    assert linear == pytest.approx(18.12, abs=0.01) and bound == pytest.approx(17.69, abs=0.01) and linear > bound
    K = f * np.exp(-3.0)
    sig = float(pricing.smile_vol(S, K, t, 0.8, 200, R, Q, **kw))
    assert sig == pytest.approx((1 - 1e-6) * bound, rel=1e-12) and sig < bound
    put = float(pricing.bs_price(S, K, t, sig, R, Q, "P"))  # ~ K e^{-rT} / 2 on the bound (d2 ~ 0)
    assert put < K * np.exp(-R * t)
    # (any BSM put is below its discounted strike; what the clip changes is the vol and the price)
    assert put < float(pricing.bs_price(S, K, t, linear, R, Q, "P"))
    k = np.array([-3.0, -6.0, -10.0, 3.0])  # Lee: sigma^2 T <= 2 |k|; the call wing (k > 0) needs no clip
    assert (pricing.smile_vol(S, f * np.exp(k), t, 0.8, 200, R, Q, **kw) ** 2 * t <= 2 * np.abs(k)).all()


def test_lee_root_kept_by_the_slope_cap():
    # vs 0.45 at 13 months (VIX1Y in a crash), SKEW 183, m 0.7, nu 1.3: the monotonicity cap alone allows
    # c = 0.968 and 2 nu^2 c s = 1.09 > 1, where the linear wing never meets Lee's bound (sigma^2 T / |k| was
    # 5.9 at k = -3, 17 at k = -10). _slope_cap also caps c at (1 - 1e-6)^2 / (2 nu^2 vs sqrt T) = 0.628
    t, kw = 400 / 365, {"m": 0.7, "nu": 1.3}
    f = S * np.exp((R - Q) * t)
    lee = (1 - 1e-6) ** 2 / (2 * 1.3**2 * 0.45 * np.sqrt(t))
    assert float(pricing._slope_cap(t, 0.45, 1.3, np.inf)) == pytest.approx(lee, rel=1e-12)
    s_lee = 0.45 / np.sqrt(pricing._floored_smile_factor(lee, np.inf)) * np.sqrt(t)  # sigma0 sqrt T at c = lee
    assert lee < pricing.skew_slope(183, 0.7) and lee < pricing._c_max(1.3, s_lee)  # the Lee cap is the one binding
    k = np.array([-3.0, -6.0, -10.0])
    sig = pricing.smile_vol(S, f * np.exp(k), t, 0.45, 183, R, Q, **kw)
    np.testing.assert_allclose(sig**2 * t / -k, 2 * (1 - 1e-6) ** 2, rtol=1e-12)  # on Lee's segment
    # past the far root d2 ~ 0: the put is worth ~ 1/2 - 1/(2 sqrt(pi |k|)) of its discounted strike (0.41 here)
    put = pricing.bs_price(S, f * np.exp(-10.0), t, sig[2], R, Q, "P") / (f * np.exp(-10.0 - R * t))
    assert 0.35 < put < 0.5


def test_deep_linear_wing_breaks_convexity():
    # the smile_vol LIMIT, pinned so it stays visible: at vs 0.15, SKEW 183, m 0.3 (c = 0.415), 30 d, the
    # wing is concave in K from z ~ 25 (K/S < 0.39) and the 0.25 / 0.30 / 0.35 S put butterfly costs -0.0067
    p25, p30, p35 = pricing.quote(1000.0, np.array([250.0, 300, 350]), 30 / 365, 0.15, 183, 0.03, 0.015, h=0, m=0.3)[0]
    assert p25 - 2 * p30 + p35 == pytest.approx(-0.0067, abs=1e-4)


_GRID_T = np.array([1 / 365, 7 / 365, 30 / 365, 60 / 365, 0.5, 1.0])[:, None, None, None]
_GRID_VS = np.array([0.10, 0.15, 0.30])[None, :, None, None]
_GRID_SKEW = np.array([110.0, 130, 150, 183])[None, None, :, None]


@pytest.mark.parametrize("z_cap", [np.inf, 3.0])
@pytest.mark.parametrize("nu", [0.8, 1.0, 1.3])
@pytest.mark.parametrize("m", [0.3, 0.5, 0.7])
def test_put_mids_non_decreasing_in_strike(m, nu, z_cap):
    # T x vs x SKEW broadcast in one call; SKEW 183 is the 2025 high, nu either side of 1, z_cap the knob
    K = S * np.arange(60, 201) / 200  # 0.30 S ... 1.00 S, step 0.005 S
    mid = pricing.quote(S, K, _GRID_T, _GRID_VS, _GRID_SKEW, R, Q, "P", m=m, nu=nu, z_cap=z_cap)[0]
    assert mid.shape == (6, 3, 4, 141)
    assert (np.diff(mid, axis=-1) >= -1e-9).all()
    # that grid holds only a few points of the 1 d and 7 d wing, so also step in standardized moneyness,
    # z from 15 down to -2 (K rising), across the linear wing (and Lee's bound where it binds) at every tenor
    z = np.linspace(15, -2, 1001)
    Kz = S * np.exp((R - Q) * _GRID_T) * np.exp(-z * _GRID_VS * np.sqrt(_GRID_T))
    mid = pricing.quote(S, Kz, _GRID_T, _GRID_VS, _GRID_SKEW, R, Q, "P", m=m, nu=nu, z_cap=z_cap)[0]
    assert (np.diff(mid, axis=-1) >= -1e-9).all()


def test_far_put_non_decreasing_in_skew():
    # a higher SKEW never cheapens the 25% OTM 2-month put (Universa preset), also past the SKEW where
    # c_max binds at m 0.7 and the slope stops steepening (the cap does not depend on SKEW)
    mid = pricing.quote(S, 0.75 * S, 61 / 365, VS, np.arange(120, 201), R, Q, m=0.7)[0]
    assert (np.diff(mid) >= -1e-9).all()


def test_review_counterexamples_fixed():
    # slice-3 review, r = q = 0: the uncapped wing gave P(0.30 S) = 64.5 > P(0.96 S) = 17.5 at 7 d
    p30, p96 = pricing.quote(S, np.array([0.30, 0.96]) * S, 7 / 365, 0.15, 160, 0, 0, "P", m=0.7)[0]
    assert p30 < p96
    # ... and a 25% OTM put still decays to 0 as T -> 0+ (the slope cap goes to 0 with s), through the
    # linear wing rather than a flat one: about 0.04 points of 4000 at 1 d, nothing at 30 s
    mid = pricing.quote(4000.0, 3000.0, np.array([1 / 365, 1e-6]), 0.15, 150, 0, 0, "P", m=0.5)[0]
    assert mid[0] < 0.1 and mid[1] < 1e-9


def test_wing_iv_falls_with_tenor_in_contango():
    # on the Cboe delayed XSP chain at the 2026-10-08 close, at fixed K/S 0.70 the market IV falls
    # 0.49 -> 0.44 -> 0.39 -> 0.37 over 29 -> 84 d although vs rises (contango); a linear wing in z falls too
    # (z shrinks as sqrt T grows), where the flat wing rose with vs. The sign, not the level. With that day's
    # own term structure (var_swap_vol of the pillars: 0.154 / 0.177 / 0.180) the model IV falls
    # 0.457 -> 0.368 -> 0.355
    days = np.array([29, 71, 84])
    vs = pricing.var_swap_vol(days / 365, 12.21, 15.41, 18.08, 20.04)
    iv = pricing.smile_vol(S, 0.70 * S, days / 365, vs, 149, R, Q, m=0.3)
    assert (np.diff(iv) < 0).all()
    # DEVIATION from the brief: its vs 0.154 / 0.181 / 0.200 (the VIX / VIX3M / VIX6M levels placed at
    # 29 / 71 / 84 d, a 28% forward vol from 71 to 84 d) give 0.4569 / 0.3717 / 0.3738, so only the fall from
    # 29 d is asserted there (the 71 -> 84 d step is not pinned either way)
    iv = pricing.smile_vol(S, 0.70 * S, days / 365, np.array([0.154, 0.181, 0.200]), 149, R, Q, m=0.3)
    assert iv[1] < iv[0] and iv[2] < iv[0]


def test_xsp_quotes_anchor_loosely():
    """The 2026-10-08 close of the Cboe delayed XSP chain: S 776.54, VIX9D / VIX / VIX3M /
    VIX6M 12.21 / 15.41 / 18.08 / 20.04, SKEW 149.19, r = ln(1.04043), q = 1.2%, m 0.3, nu 1. Market
    mid at K/S 0.70 (% of spot) over model mid, measured: 1.51 at 29 d, 1.49 at 43 d, 1.34 at 71 d,
    1.25 at 84 d (model IV 0.457 / 0.414 / 0.368 / 0.355 vs the market's 0.478 / 0.434 / 0.383 / 0.366;
    the flat z_cap 3 wing at m 0.3 gave ~64,000x, 314x, 11.6x, 5.8x). One calm day: asserted within
    3x (10x at 29 d), not pinned."""
    S0, r, q = 776.54, np.log(1.04043), 0.012
    days = np.array([29, 43, 71, 84])
    market = np.array([0.013, 0.032, 0.081, 0.104]) / 100
    vs = pricing.var_swap_vol(days / 365, 12.21, 15.41, 18.08, 20.04)
    model = pricing.quote(S0, 0.70 * S0, days / 365, vs, 149.19, r, q, "P", m=0.3, nu=1.0)[0] / S0
    ratio = market / model
    print("market / model at K/S 0.70:", dict(zip(days.tolist(), ratio.round(2).tolist())))
    assert 0.1 < ratio[0] < 10 and ((1 / 3 < ratio[1:]) & (ratio[1:] < 3)).all()


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
    # without vix1y (None, the default) the four-pillar results above hold, and past 184 d the vol stays
    # flat at VIX6M (test_var_swap_one_year_pillar has the 365 d pillar)
    assert pricing.var_swap_vol(300 / 365, 15, 20, 25, 30) == 0.30


def test_var_swap_one_year_pillar():
    # VIX1Y at Cboe's 365 d target: exact there, total variance linear in T between 184 d and 365 d,
    # flat vol past it (the 13-15 month TAIL-style puts)
    v = pricing.var_swap_vol(np.array([365, 250, 400]) / 365, 15, 20, 25, 30, vix1y=32)
    assert pricing.PILLARS_DAYS == (9, 30, 93, 184, 365)
    assert v[0] == 0.32
    hand = np.sqrt((0.30**2 * 184 + (0.32**2 * 365 - 0.30**2 * 184) * (250 - 184) / (365 - 184)) / 250)
    assert v[1] == pytest.approx(hand, rel=1e-12) and v[1] == pytest.approx(0.310808, abs=1e-6)
    assert v[2] == pytest.approx(0.32, rel=1e-12)
    # without VIX1Y the 250 d and 400 d tenors fall back to flat VIX6M, as before
    np.testing.assert_allclose(pricing.var_swap_vol(np.array([250, 400]) / 365, 15, 20, 25, 30), 0.30, rtol=1e-12)


def _panel(n: int = 600) -> pd.DataFrame:
    """Deterministic VIX family with ln VIX3M = 0.3 + 0.9 ln VIX exactly."""
    t = np.arange(n)
    vix = 18 + 6 * np.sin(t / 9) + t / 100
    vix3m = np.exp(0.3 + 0.9 * np.log(vix))
    return pd.DataFrame(
        {"vix": vix, "vix9d": 0.9 * vix, "vix3m": vix3m, "vix6m": 1.05 * vix3m, "vix1y": 1.1 * vix3m},
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
    vix1y = np.exp(0.05 + 0.98 * np.log(vix6m))
    mkt["vix6m"], mkt["vix9d"], mkt["vix1y"] = vix6m, vix9d, vix1y
    mkt.iloc[:300, mkt.columns.get_loc("vix3m")] = np.nan
    mkt.iloc[:340, mkt.columns.get_loc("vix6m")] = np.nan  # rows < 300 need the filled vix3m
    mkt.iloc[:350, mkt.columns.get_loc("vix9d")] = np.nan  # exactly 250 overlap rows: enough to fill
    mkt.iloc[:345, mkt.columns.get_loc("vix1y")] = np.nan  # from the (twice filled) vix6m below row 340
    mkt.iloc[320, mkt.columns.get_loc("vix")] = np.nan  # vix3m is native here, vix is not
    mkt.iloc[342, mkt.columns.get_loc("vix3m")] = np.nan  # a vix3m gap (not filled) where vix6m is native
    out = pricing.fill_term_structure(mkt)
    # row 342: vix1y is filled from vix6m, which a fit on vix3m (NaN there) could not do
    np.testing.assert_allclose(out["vix3m"], vix3m.where(mkt.index != mkt.index[342]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["vix6m"], vix6m, rtol=0, atol=1e-9)  # row 320 too: regressed on vix3m
    np.testing.assert_allclose(out["vix9d"], vix9d.where(mkt.index != mkt.index[320]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["vix1y"], vix1y, rtol=0, atol=1e-9)
    assert out["ts_filled"].tolist() == [i < 350 for i in range(600)]  # union over the four targets
    assert out.attrs["warnings"] == []
    again = pricing.fill_term_structure(out)  # nothing left to fill, and the flags survive
    pd.testing.assert_frame_equal(again, out)


def test_fill_term_structure_fits_on_native_rows_only():
    # VIX1Y (2007) starts before VIX6M (2008): rows where the regressor is itself back-filled fill the
    # target but stay out of its fit. Here vix1y sits on its line in vix6m on native rows and 30% above it
    # on rows 290-299, where vix6m is regression-filled; a fit on those rows would miss the line below 290
    mkt, exact = _panel(), _panel()
    mkt.iloc[:300, mkt.columns.get_loc("vix6m")] = np.nan
    mkt.iloc[:290, mkt.columns.get_loc("vix1y")] = np.nan
    mkt.iloc[290:300, mkt.columns.get_loc("vix1y")] *= 1.3
    out = pricing.fill_term_structure(mkt)
    np.testing.assert_allclose(out["vix6m"], exact["vix6m"], rtol=1e-12)
    np.testing.assert_allclose(out["vix1y"].iloc[:290], exact["vix1y"].iloc[:290], rtol=1e-9)


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
    # the linear wing keeps far strikes' delta alive: at 61 d, vs 0.15, SKEW 183 the 0.005 (dao_ch9) and
    # 0.001 put strikes lie past ln(K/F) = -1.5 (|delta| 0.0053 there), inside the [-8, 1] bracket (m 0.3)
    t = 61 / 365
    for d in (0.005, 0.001):
        Kd = pricing.strike_for_delta(S, t, 0.15, 183, R, Q, d, m=0.3)
        assert Kd < S * np.exp((R - Q) * t - 1.5)
        vol_d = pricing.smile_vol(S, Kd, t, 0.15, 183, R, Q, m=0.3)
        assert pricing.bs_delta(S, Kd, t, vol_d, R, Q, "P") == pytest.approx(-d, rel=1e-6)


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

"""Synthetic SPX option pricing: Black-Scholes-Merton on a VIX / SKEW volatility surface.

Pure numpy/scipy/pandas, no I/O. Every function except `strike_for_delta` (scalar) and
`fill_term_structure` (DataFrame) is elementwise with numpy broadcasting over scalars, arrays and
pandas Series; pandas inputs are used positionally (np.asarray), never index-aligned, and the
result is an ndarray. NaN inputs give NaN outputs, except: bs_price / bs_delta return intrinsic
(delta 0 / +-1) when T <= 0 or sigma <= 0 even if r, q or the other of T / sigma is NaN;
smile_vol at T <= 0 returns nu vs (flat) whatever S, K, r and q are; var_swap_vol skips NaN pillars.
The smile knobs m, beta and nu are finite scalars, beta >= 0, nu > 0 (skew_slope raises ValueError on any
other beta, smile_vol on any other nu), and z_cap is a scalar in [0, inf] (default inf: the put wing stays
linear out to Lee's bound).

Units: S, K and prices in index points; T in years; sigma, vs, r, q decimals; the vol indices
(vix9d, vix, vix3m, vix6m, vix1y) and skew in index points, as stored in the market panel.
Rates: callers pass the continuously compounded r = ln(1 + rf) and the panel's q; this module
does not convert.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import erfcx, ndtr
from scipy.stats import norm

SLOPE_REF: float = 0.263  # observed SPX smile slope: IV/vs = 0.968 + 0.263 z (docs/research/gap-3-q9 A, 2026-10-08)
GAMMA_REF: float = 4.919  # gamma = (SKEW - 100)/10 on that day, SKEW 149.19
SKEW_BETA: float = 0.5  # slope elasticity to gamma: 0 = constant wing, 1 = proportional (Gram-Charlier); ASSUMPTION
SKEW_MULT: float = 1.0  # pure scale on the anchored slope (skew_slope); not fitted: at m 1 the Cboe PPUT replica
# fails dCAGR out of sample (figures/validation_table.md)
HALF_SPREAD_MIN: float = 0.05  # SPX index points (the minimum tick)
PILLARS_DAYS = (9, 30, 93, 184, 365)  # VIX9D, VIX, VIX3M, VIX6M, VIX1Y (Cboe's target tenors)
Z_CAP: float = np.inf  # standardized moneyness past which the put wing goes flat: never, by default
_LEE = 1 - 1e-6  # smile_vol keeps sigma^2 T at most _LEE^2 * 2 |k| on the far wing (Lee's moment bound)
_MIN_OVERLAP = 250  # rows needed to fit a term-structure back-fill


def _is_call(cp: str) -> bool:
    """True for "C", False for "P"; anything else raises ValueError."""
    if cp not in ("C", "P"):
        raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")
    return cp == "C"


def _d1(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T); junk where T or sigma <= 0 (callers mask it)."""
    with np.errstate(all="ignore"):
        return (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def bs_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, r: ArrayLike, q: ArrayLike, cp: str = "P"
) -> np.ndarray:
    """BSM price with continuous dividend yield q, in the units of S.

    d1 = [ln(S/K) + (r - q + sigma^2/2) T]/(sigma sqrt T), d2 = d1 - sigma sqrt T;
    call = S e^{-qT} N(d1) - K e^{-rT} N(d2), put = K e^{-rT} N(-d2) - S e^{-qT} N(-d1).
    T <= 0 or sigma <= 0 gives plain (undiscounted) intrinsic max(S-K, 0) / max(K-S, 0).
    """
    call = _is_call(cp)
    S, K, T, sigma, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, sigma, r, q))
    d1 = _d1(S, K, T, sigma, r, q)
    with np.errstate(all="ignore"):
        d2 = d1 - sigma * np.sqrt(T)
    s_pv, k_pv = S * np.exp(-q * T), K * np.exp(-r * T)
    if call:
        live, intrinsic = s_pv * norm.cdf(d1) - k_pv * norm.cdf(d2), np.maximum(S - K, 0.0)
    else:
        live, intrinsic = k_pv * norm.cdf(-d2) - s_pv * norm.cdf(-d1), np.maximum(K - S, 0.0)
    return np.where((T <= 0) | (sigma <= 0), intrinsic, live)  # NaN T or sigma stays NaN, not intrinsic


def bs_delta(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, r: ArrayLike, q: ArrayLike, cp: str = "P"
) -> np.ndarray:
    """BSM spot delta: call e^{-qT} N(d1), put -e^{-qT} N(-d1).

    T <= 0 or sigma <= 0 gives 1 / -1 when the call / put is in the money, else 0 (0 at S == K).
    """
    call = _is_call(cp)
    S, K, T, sigma, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, sigma, r, q))
    d1 = _d1(S, K, T, sigma, r, q)
    dead = (T <= 0) | (sigma <= 0)
    if call:
        return np.where(dead, np.heaviside(S - K, 0.0), np.exp(-q * T) * norm.cdf(d1))
    return np.where(dead, -np.heaviside(K - S, 0.0), -np.exp(-q * T) * norm.cdf(-d1))


def var_swap_vol(
    T: ArrayLike, vix9d: ArrayLike, vix: ArrayLike, vix3m: ArrayLike, vix6m: ArrayLike, vix1y: ArrayLike | None = None
) -> np.ndarray:
    """Variance-swap vol (decimal) at tenor T (years) from the VIX term structure (index points).

    Pillars T_i = PILLARS_DAYS/365 with vols (vix9d, vix, vix3m, vix6m, vix1y)/100 and total
    variance w_i = sigma_i^2 T_i. Between the nearest available pillars w(T) is linear in T and the
    result is sqrt(w(T)/T); outside the available range it is the nearest pillar's vol (flat vol).
    NaN pillars are skipped per element (vix1y None = all NaN, the four-pillar result); one
    available pillar is used flat; none gives NaN.
    """
    vix1y = np.nan if vix1y is None else vix1y
    T, *vols = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (T, vix9d, vix, vix3m, vix6m, vix1y)))
    vol = np.stack(vols, axis=-1) / 100  # (..., 5) pillar vols
    P = np.asarray(PILLARS_DAYS) / 365
    ok = ~np.isnan(vol)
    below, above = ok & (P <= T[..., None]), ok & (P >= T[..., None])
    i_lo = np.asarray(len(P) - 1 - np.argmax(below[..., ::-1], axis=-1))  # last available pillar <= T
    i_hi = np.asarray(np.argmax(above, axis=-1))  # first available pillar >= T
    v_lo = np.take_along_axis(vol, i_lo[..., None], axis=-1)[..., 0]
    v_hi = np.take_along_axis(vol, i_hi[..., None], axis=-1)[..., 0]
    t_lo, t_hi = P[i_lo], P[i_hi]
    with np.errstate(all="ignore"):  # junk only where a pillar side is missing, and select drops it
        frac = np.where(t_hi > t_lo, (T - t_lo) / (t_hi - t_lo), 0.0)
        inside = np.sqrt((v_lo**2 * t_lo + frac * (v_hi**2 * t_hi - v_lo**2 * t_lo)) / T)
    has_lo, has_hi = below.any(axis=-1), above.any(axis=-1)
    return np.select([has_lo & has_hi, has_lo, has_hi], [inside, v_lo, v_hi], np.nan)


def skew_slope(skew: ArrayLike, m: float = SKEW_MULT, beta: float = SKEW_BETA) -> np.ndarray:
    """Put-wing slope per unit of standardized moneyness, anchored to an observed chain and damped in SKEW:
    c = m SLOPE_REF (gamma / GAMMA_REF)^beta, gamma = max(SKEW - 100, 0)/10; c = 0 at gamma = 0 for every
    beta >= 0 (no 0^0 = 1), NaN for a NaN SKEW.

    Cboe defines SKEW = 100 - 10 S with S the 30-day risk-neutral skewness, so gamma = -S for negative
    skew. The first-order Gram-Charlier smile (Backus-Foresi-Wu 2004) has c = gamma/6, which makes the
    whole smile slope proportional to SKEW; beta = 1 is that rule (with m_old gamma/6 at m = m_old
    GAMMA_REF / (6 SLOPE_REF)). c is also the slope of IV/vs in z_vs = -ln(K/F)/(vs sqrt T) (smile_vol:
    IV/vs = nu (sigma0/vs + c z_vs)), so SLOPE_REF = 0.263 is the 2026-10-08 Cboe XSP chain's fit at its
    SKEW 149.19 (GAMMA_REF 4.919; docs/research/gap-3-q9 A), reproduced exactly there at m = 1.
    Why damped: under c = 0.3 gamma/6 the 30% OTM ~80-day put was 0.083% of spot at SKEW 149 (the chain:
    0.104%), 0.044% at SKEW 142 and 0.004% at the 2018-19 median SKEW 126, ~25x below plausible market
    levels, so the budget presets bought 10-25x a real desk's notional; deep-wing richness in standardized
    moneyness is empirically far more stable than SKEW. At beta = 0.5 the 1990-2026 average SKEW (~122,
    gamma ~2.2) gives c ~0.176, the level the realized-payoff test of docs/research/gap-4-q10 found
    roughly fair, while SKEW 149.19 keeps the chain's slope.
    ASSUMPTION: one chain cannot estimate the SKEW sensitivity, so beta is a disclosed modelling choice and
    a sensitivity knob, swept over {0, 0.5, 1} in the case study; m scales the anchored slope (the app's
    m x {0.5 .. 1.5} sweep). Raises ValueError unless beta is finite and >= 0 (a negative beta cheapens the
    wing as SKEW rises).
    """
    if not (np.isfinite(beta) and beta >= 0):
        raise ValueError(f"beta must be finite and >= 0, got {beta}")
    g = np.maximum(np.asarray(skew, dtype=float) - 100, 0) / 10
    # ponytail: one chain anchors SLOPE_REF and beta is assumed; regress chain slopes on SKEW across dates to fit
    # beta once chains.py holds a panel of chains
    return m * SLOPE_REF * np.where(g > 0, (g / GAMMA_REF) ** beta, g)  # NaN > 0 is False: NaN passes through


def smile_factor(c: ArrayLike) -> np.ndarray:
    """A(c) = 1 + c/sqrt(2 pi) + (5/8) c^2 = E[(1 + c z+ - (c/2) z-)^2] for z ~ N(0, 1).

    (E[z+] = E[z-] = 1/sqrt(2 pi), E[(z+)^2] = E[(z-)^2] = 1/2, z+ z- = 0.) The smile's mean
    variance without its 0.5 floor (and any put-wing cap); smile_vol normalises with the floor-
    and cap-aware version below.
    """
    c = np.asarray(c, dtype=float)
    return 1 + c / np.sqrt(2 * np.pi) + 5 / 8 * c**2


def _floored_smile_factor(c: ArrayLike, z_cap: ArrayLike) -> np.ndarray:
    """A_f(c, b) = E[max(1 + c min(z+, b) - (c/2) z-, 0.5)^2] for z ~ N(0, 1), b = z_cap in [0, inf].

    Split the integral at a = -1/c, 0 and b. With the truncated-normal moments
    int_l^u phi = Phi(u) - Phi(l), int_l^u z phi = phi(l) - phi(u),
    int_l^u z^2 phi = Phi(u) - Phi(l) + l phi(l) - u phi(u), A_f = A(c) + floor + cap, where
    - floor (z < a): 0.25 replaces (1 + c z/2)^2, so
      floor = 0.25 Phi(a) - [Phi(a) - c phi(a) + (c^2/4)(Phi(a) - a phi(a))], 0 at c = 0;
    - cap (z > b): (1 + c b)^2 replaces (1 + c z)^2, so with Q = 1 - Phi(b),
      cap = (1 + c b)^2 Q - [Q + 2c phi(b) + c^2 (Q + b phi(b))]
          = 2c (b Q - phi(b)) + c^2 ((b^2 - 1) Q - b phi(b))  (< 0: the cap removes variance),
      and cap = 0 at b = inf (no flat wing: A_f = A(c) + floor).
    A_f(0, b) = 1. The pieces do not overlap because a < 0 <= b.
    """
    # ndtr and an explicit phi, not scipy.stats.norm: smile_vol calls this 7 times per call (_slope_cap)
    c = np.asarray(c, dtype=float)
    with np.errstate(all="ignore"):  # inf * 0 = NaN at b = inf (b Q) and c = 0 (a = -inf), dropped by the wheres
        tail, pdf_b = ndtr(-z_cap), np.exp(-np.square(z_cap) / 2) / np.sqrt(2 * np.pi)
        cap = np.where(np.isinf(z_cap), 0.0, 2 * c * (z_cap * tail - pdf_b) + c**2 * ((z_cap**2 - 1) * tail - z_cap * pdf_b))
        a = -1 / c
        cdf_a, pdf_a = ndtr(a), np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
        lost = cdf_a - c * pdf_a + c**2 / 4 * (cdf_a - a * pdf_a)
        return smile_factor(c) + np.where(c > 0, 0.25 * cdf_a - lost, 0.0) + cap


def _mills(x: ArrayLike) -> np.ndarray:
    """Mills ratio R(x) = N(-x)/phi(x) = sqrt(pi/2) erfcx(x/sqrt 2), stable in the far tail."""
    return np.sqrt(np.pi / 2) * erfcx(np.asarray(x, dtype=float) / np.sqrt(2))


def _max_slope(nu: ArrayLike, s: ArrayLike) -> np.ndarray:
    """c_max(nu, s): the steepest linear put-wing slope c at which put prices never fall as K rises,
    for a smile with s = sigma0 sqrt(T). Elementwise; the direct solve behind _c_max's table.

    On the wing z > 0, sigma = nu sigma0 (1 + c z), so d sigma/dK = -nu c/(K sqrt T), and with vega
    K e^{-rT} phi(d2) sqrt T, dP/dK = e^{-rT} [N(-d2) - nu c phi(d2)]: >= 0 iff the Mills ratio
    R(d2) >= nu c. With y = 1 + c z >= 1 the full d2 = z/(nu y) - nu s y/2 = (1 - 1/y)/(nu c) - nu s y/2
    peaks at y* = sqrt(2/(c s))/nu; with a = nu c, b = nu s the worst case is
    x = 1/a - sqrt(2b/a) when a b <= 2 (y* >= 1), else the ATM x = -b/2 (continuous at a b = 2).
    R falls, so the wing is monotone at every z iff R(x(a, b)) >= a: nu c_max = a_max(nu s), one
    function of b, found by bisection on [0, R(-b/2)] (the gap is > 0 near 0 and <= 0 at R(-b/2)
    since x >= -b/2), the safe end returned. a_max(0) = 0 and a_max RISES with b, ~(2b)^(1/3) for
    small b (more total vol pulls d2 down): c_max(1, s) = 0.186, 0.517, 1.255 at s = 0.003, 0.05, 0.4.
    Past a finite z_cap dP/dK = e^{-rT} N(-d2) > 0 and the capped wing's worst d2 is no larger, so
    c_max(nu, s) is safe for any z_cap. On the call wing d2 <= -b/2 (for a b <= 4), so
    R(d2) >= R(-b/2) >= a > a/2, the call-wing slope.
    """
    nu, s = np.broadcast_arrays(np.asarray(nu, dtype=float), np.asarray(s, dtype=float))
    b = nu * s
    lo, hi = np.zeros_like(b), _mills(-b / 2)
    for _ in range(60):  # hi - lo halves each step: below 1e-15 of hi after ~50
        a = (lo + hi) / 2
        ok = _mills(np.where(a * b <= 2, 1 / a - np.sqrt(2 * b / a), -b / 2)) >= a
        lo, hi = np.where(ok, a, lo), np.where(ok, hi, a)
    return np.where(b > 0, lo, 0.0) / nu  # b = 0: R(1/a) < a for all a > 0; float noise would leave ~1e-8


# ponytail: c_max per row from a cached table of a_max(b), 4001 points uniform in b^(1/3) over b = nu s in
# [0, 4], linearly interpolated (the exact solve per row is ~50x slower). a_max is convex there, so the
# chords overshoot it by < 2e-6 relative; the table is lowered by 1e-5 to stay on the safe side. Past b = 4
# (nu sigma0 sqrt T > 4, never in practice) it holds a_max(4), which is safe since a_max rises.
_SLOPE_U = np.linspace(0.0, 4 ** (1 / 3), 4001)
_SLOPE_A = _max_slope(1.0, _SLOPE_U**3) * (1 - 1e-5)


def _c_max(nu: ArrayLike, s: ArrayLike) -> np.ndarray:
    """c_max(nu, s) (see _max_slope) from the cached table, elementwise: a_max(nu s) / nu."""
    return np.interp(np.cbrt(nu * np.asarray(s, dtype=float)), _SLOPE_U, _SLOPE_A) / nu


def _slope_cap(T: ArrayLike, vs: ArrayLike, nu: float, z_cap: float) -> np.ndarray:
    """The put-wing slope cap smile_vol applies at tenor T and variance-swap vol vs: the fixed point
    c* = g(c*), g(c) = c_max(nu, sigma0(c) sqrt T), sigma0(c) = vs / sqrt(A_f(c, z_cap)), and at most
    (1 - 1e-6)^2 / (2 nu^2 vs sqrt T), which keeps Lee's far root real in smile_vol wherever
    A_f >= 1 (sigma0 <= vs): z_cap = inf or z_cap >~ 0.5.

    There sigma0 falls as c rises and c_max rises with s, so g falls and has one fixed point; iterating
    from c = 0 alternates around it (|g'| < 0.4) and an even number of steps ends at or below it,
    where c <= g(c), i.e. safe at its own sigma0. With a smaller finite z_cap A_f < 1 and sigma0
    rises with c, so g rises and the iterates climb from 0 toward c*, each below it: safe as well.
    Six steps land within 4e-8 of c* at vs 0.15, 30 d and 4e-6 (relative) at 1 y (z_cap = inf).
    Any slope below c* is then safe too, so the Lee cap keeps puts
    monotone; neither cap depends on SKEW, so prices stay non-decreasing in SKEW where one binds.
    The Lee cap binds only where nu vs sqrt T >~ 0.42-0.47 (at 1 y: from vs 0.45 at nu 1, 0.34 at
    nu 1.3). 0 at T <= 0.
    """
    s = np.asarray(vs, dtype=float) * np.sqrt(np.maximum(T, 0.0))
    c = np.zeros_like(s)
    for _ in range(6):
        c = _c_max(nu, s / np.sqrt(_floored_smile_factor(c, z_cap)))
    with np.errstate(divide="ignore"):  # s = 0: no Lee cap (c is 0 there anyway)
        return np.minimum(c, _LEE**2 / (2 * nu**2 * s))


def smile_vol(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vs: ArrayLike,
    skew: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    *,
    m: float = SKEW_MULT,
    nu: float = 1.0,
    z_cap: float = Z_CAP,
    beta: float = SKEW_BETA,
) -> np.ndarray:
    """Implied vol (decimal) at strike K and tenor T, given the variance-swap vol vs (decimal).

    F = S e^{(r-q)T}, k = ln(K/F), z = -k/(sigma0 sqrt T) (z > 0 for OTM puts),
    sigma = nu sigma0 max(1 + c min(z+, z_cap) - (c/2) z-, 0.5), clipped on the far put wing to
    Lee's bound (below). nu > 0 is the vol-risk-premium knob (1 = price at the variance-swap level);
    z_cap in [0, inf] defaults to Z_CAP = inf (a linear put wing); a finite z_cap flattens the wing
    past z_cap (a sensitivity knob). T <= 0 gives nu vs (the slope cap is 0 there; bs_price ignores
    the vol anyway); a NaN T gives NaN.

    Slope: c = min(skew_slope(skew, m, beta), _slope_cap(T, vs, nu, z_cap)): the monotonicity cap
    c_max(nu, s) of _max_slope at s = sigma0 sqrt T (a fixed point, as sigma0 depends on c) and
    Lee's root cap, so puts never fall as K rises. The cap tightens as T -> 0, so a far-OTM put at
    fixed K/F still decays to 0 at expiry; where it binds the whole smile flattens (ATM and call wing
    too), and nu moves the slope as well as the level.
    DEVIATION from the brief (c_max(nu, s) with s = sigma0 sqrt T as given): sigma0 depends on c,
    so the cap is the fixed point above; it does not depend on SKEW (prices stay monotone in SKEW).

    sigma0 = vs / sqrt(A_f(c, z_cap)), A_f the floor- (and cap-) aware smile factor
    (_floored_smile_factor), so E_N[sigma(z)^2] = (nu vs)^2 exactly wherever Lee's bound is slack.

    Lee's bound: sigma^2 T <= 2 |k| in the far wing (Lee 2004, the moment formula). The linear total
    vol nu (s + c |k|) meets (1 - 1e-6) sqrt(2 |k|) at the roots in sqrt|k| of
    nu c x^2 - (1 - 1e-6) sqrt2 x + nu s = 0, and sigma is clipped to (1 - 1e-6) sqrt(2 |k| / T) past
    the far root only; _slope_cap keeps that root real for z_cap = inf (or >~ 0.5). Below that a
    finite z_cap can leave it NaN and no clip applies, which is harmless: the wing is flat past
    z_cap. Far from binding at z <= 12 for ordinary inputs, it guards the extreme tail, where d2 ~ 0
    and a put is worth about half its discounted strike. No call-side clip: the call wing is at most
    nu sigma0, so sigma^2 T <= (nu s)^2 <= 2 k for every k >= (nu s)^2 / 2.
    DEVIATION from the brief (sigma <= sqrt(2 |k| / T) for every k < 0, and symmetrically for calls):
    that bound goes to 0 at the money, where the linear wing sits above it, so a literal clip would
    zero the ATM vol; Lee's formula is a statement about |k| -> inf, so only the far crossing is used.

    Consequences: (1) d sigma / d ln K = -nu c / sqrt(T) on the put wing (0 past a finite z_cap)
    and -nu c / (2 sqrt T) for K > F (until the floor), so at fixed K/F the wing IV falls with tenor
    in contango unless the forward variance between the tenors is steep, as on the Cboe delayed XSP
    chain at the 2026-10-08 close (IV/vs = 0.968 + 0.263 z out to z ~ 12, R^2 0.998);
    (2) ASSUMPTION: the call wing has half the put slope (checked later by the CLL replication's
    short calls); (3) put prices never fall as K rises (vertical spreads are arbitrage-free), on
    Lee's segment too (there d2 ~ 0 and R(0) > nu c).
    LIMIT: the SKEW sensitivity of the slope is an assumption (skew_slope: anchored to one chain, damped by
    beta = SKEW_BETA 0.5); beta in {0, 0.5, 1} is a sensitivity knob the case study must sweep.
    LIMIT: convexity (no butterfly arbitrage) is not enforced. The deep linear wing has negative
    density once c z is large (test_deep_linear_wing_breaks_convexity pins a case), and dP/dK drops
    where |d sigma / dK| jumps up as K rises: at Lee's far root and at a finite z_cap.
    """
    # ponytail: linear-in-z smile (one slope parameter, linear wings, 0.5 floor);
    # fit SVI per date when real option data is present (chains adapter, slice 9).
    # ponytail: one put-wing slope out to Lee's bound (the 2026-10-08 chain was linear in z to z ~ 12), so the
    # deep wing breaks convexity (LIMIT above); a Durrleman check, or a second, flatter far-wing slope, once
    # real chains show the wing bending or far-tail butterflies get priced
    # ponytail: slope capped at c*, so SKEW spikes stop steepening the wing (most at short tenors); same upgrade
    # ponytail: one slope cap for every z_cap (conservative at a finite z_cap: tighter than the capped wing's own
    # c_max(nu, z_cap), 0.650 at nu 1, z_cap 3, at short tenors); take the max of the two if the z_cap sweep
    # needs to isolate wing shape
    if not (np.isfinite(nu) and nu > 0):
        raise ValueError(f"nu must be finite and > 0, got {nu}")
    S, K, T, vs, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, vs, r, q))
    c = np.minimum(skew_slope(skew, m, beta), _slope_cap(T, vs, nu, z_cap))
    s0 = vs / np.sqrt(_floored_smile_factor(c, z_cap))
    with np.errstate(all="ignore"):  # a NaN T or vs fails both tests and stays NaN; T <= 0 has c = 0, no clip
        k = np.log(K / (S * np.exp((r - q) * T)))
        z = np.where((T <= 0) | (s0 <= 0), 0.0, -k / (s0 * np.sqrt(T)))
        sigma = nu * s0 * np.maximum(1 + c * np.minimum(np.maximum(z, 0), z_cap) - c / 2 * np.maximum(-z, 0), 0.5)
        far = (_LEE * np.sqrt(2) + np.sqrt(2 * _LEE**2 - 4 * nu**2 * c * s0 * np.sqrt(T))) / (2 * nu * c)  # inf at c = 0
        return np.where(-k > far**2, np.minimum(sigma, _LEE * np.sqrt(-2 * k / T)), sigma)


def quote(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vs: ArrayLike,
    skew: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: str = "P",
    *,
    m: float = SKEW_MULT,
    nu: float = 1.0,
    z_cap: float = Z_CAP,
    h: float = 0.05,
    beta: float = SKEW_BETA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mid, bid, ask) in index points: mid = BSM at smile_vol (m, nu, z_cap, beta passed through),
    half = max(h mid, HALF_SPREAD_MIN), bid = max(mid - half, 0), ask = mid + half.

    h = 0 means mid fills (half = 0, bid = ask = mid), as slice 5 replication uses. DEVIATION: the
    brief floors every half-spread at HALF_SPREAD_MIN, which would charge a tick at h = 0 too.
    """
    mid = bs_price(S, K, T, smile_vol(S, K, T, vs, skew, r, q, m=m, nu=nu, z_cap=z_cap, beta=beta), r, q, cp)
    half = np.maximum(h * mid, HALF_SPREAD_MIN) if h > 0 else np.zeros_like(mid)
    return mid, np.maximum(mid - half, 0.0), mid + half


def strike_for_delta(
    S: float,
    T: float,
    vs: float,
    skew: float,
    r: float,
    q: float,
    delta: float,
    cp: str = "P",
    *,
    m: float = SKEW_MULT,
    nu: float = 1.0,
    z_cap: float = Z_CAP,
    beta: float = SKEW_BETA,
) -> float:
    """Scalar strike K with |bs_delta(K, smile_vol(K))| = |delta| (the sign of delta is ignored;
    m, nu, z_cap, beta are passed to smile_vol).

    Solved with brentq on k = ln(K/F) over [-8, 1], F = S e^{(r-q)T}. A put's |delta| is at most
    e^{-qT} N(-sqrt(2 |k|)) at any vol (d1 = |k|/w + w/2 >= sqrt(2 |k|), w = sigma sqrt T), 3.2e-5 at
    k = -8, so every put target above that is bracketed whatever the wing. Raises ValueError when
    the target is not bracketed (e.g. |delta| >= e^{-qT}) or an input is NaN.
    """
    F = S * np.exp((r - q) * T)

    def gap(k: float) -> float:
        K = F * np.exp(k)
        vol = smile_vol(S, K, T, vs, skew, r, q, m=m, nu=nu, z_cap=z_cap, beta=beta)
        return abs(float(bs_delta(S, K, T, vol, r, q, cp))) - abs(delta)

    if not gap(-8.0) * gap(1.0) <= 0:  # same sign, or NaN
        raise ValueError(f"no strike with |delta| = {abs(delta)} for ln(K/F) in [-8, 1]")
    # ponytail: assumes T > 0 and vs, nu > 0 (strikes are picked when a leg opens); at expiry or
    # zero vol delta is a 0/1 step and this returns K ~ S. Guard it if a caller ever picks there.
    return float(F * np.exp(brentq(gap, -8.0, 1.0)))


def fill_term_structure(mkt: pd.DataFrame) -> pd.DataFrame:
    """Copy of the market panel with vix3m, vix6m, vix9d, vix1y back-filled before their first valid date.

    In order: vix3m from vix, vix6m from the (filled) vix3m, vix9d from vix, vix1y (Cboe, from
    2007-01-03) from the (filled) vix6m, each by OLS
    ln target = a + b ln regressor on the rows where both are positive in the input (a zero or
    negative print counts as missing; a back-filled regressor fills but never fits, so 2007 VIX1Y
    is not fitted on regression-filled VIX6M). Only rows before the target's first valid value whose
    regressor is positive (filled or not) are filled; they are True in the added boolean column
    `ts_filled` (flags already in the input are kept, so a second call changes nothing). A target
    with leading NaNs but fewer than 250 overlap rows (all-NaN target or regressor included) is left
    as is and noted in df.attrs["warnings"] (appended to any warnings already there).
    vix9d, vix3m, vix6m and vix1y must be columns (load_market always provides them); an absent
    one raises KeyError.
    """
    # ponytail: one static log-log fit over the whole overlap; a regime-dependent fit (e.g. by VIX
    # level or decade) if back-filled years drive replication errors.
    df = mkt.copy()
    filled = df.get("ts_filled", pd.Series(False, index=df.index)).eq(True)
    warnings = list(mkt.attrs.get("warnings", []))
    for y, x in (("vix3m", "vix"), ("vix6m", "vix3m"), ("vix9d", "vix"), ("vix1y", "vix6m")):
        lead = df[y].ffill().isna()  # ffill leaves exactly the leading NaNs
        if not lead.any():
            continue
        both = (mkt[y] > 0) & (mkt[x] > 0)  # native rows only; NaN compares False
        if both.sum() < _MIN_OVERLAP:
            warnings.append(f"{y}: only {both.sum()} rows overlap {x} (< {_MIN_OVERLAP}); not back-filled")
            continue
        todo = lead & (df[x] > 0)
        b, a = np.polyfit(np.log(df.loc[both, x]), np.log(df.loc[both, y]), 1)
        df.loc[todo, y] = np.exp(a + b * np.log(df.loc[todo, x]))
        filled |= todo
    df["ts_filled"] = filled
    df.attrs["warnings"] = warnings
    return df

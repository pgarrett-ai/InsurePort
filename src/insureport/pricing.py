"""Synthetic SPX option pricing: Black-Scholes-Merton on a VIX / SKEW volatility surface.

Pure numpy/scipy/pandas, no I/O. Every function except `strike_for_delta` (scalar) and
`fill_term_structure` (DataFrame) is elementwise with numpy broadcasting over scalars, arrays and
pandas Series; pandas inputs are used positionally (np.asarray), never index-aligned, and the
result is an ndarray. NaN inputs give NaN outputs, except: bs_price / bs_delta return intrinsic
(delta 0 / +-1) when T <= 0 or sigma <= 0 even if r, q or the other of T / sigma is NaN;
smile_vol at T <= 0 returns nu sigma0 whatever S, K, r and q are; var_swap_vol skips NaN pillars.

Units: S, K and prices in index points; T in years; sigma, vs, r, q decimals; the vol indices
(vix9d, vix, vix3m, vix6m) and skew in index points, as stored in the market panel.
Rates: callers pass the continuously compounded r = ln(1 + rf) and the panel's q; this module
does not convert.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm

SKEW_MULT: float = 0.5  # placeholder prior (spec: 0.3-0.7); slice 5 calibrates it on PPUT and overwrites it
HALF_SPREAD_MIN: float = 0.05  # SPX index points (the minimum tick)
PILLARS_DAYS = (9, 30, 93, 182)  # VIX9D, VIX, VIX3M, VIX6M
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


def var_swap_vol(T: ArrayLike, vix9d: ArrayLike, vix: ArrayLike, vix3m: ArrayLike, vix6m: ArrayLike) -> np.ndarray:
    """Variance-swap vol (decimal) at tenor T (years) from the VIX term structure (index points).

    Pillars T_i = PILLARS_DAYS/365 with vols (vix9d, vix, vix3m, vix6m)/100 and total variance
    w_i = sigma_i^2 T_i. Between the nearest available pillars w(T) is linear in T and the result
    is sqrt(w(T)/T); outside the available range it is the nearest pillar's vol (flat vol). NaN
    pillars are skipped per element; one available pillar is used flat; none gives NaN.
    """
    T, *vols = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (T, vix9d, vix, vix3m, vix6m)))
    vol = np.stack(vols, axis=-1) / 100  # (..., 4) pillar vols
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


def skew_slope(skew: ArrayLike, m: float = SKEW_MULT) -> np.ndarray:
    """Put-wing slope per unit of standardized moneyness: c = m * gamma / 6, gamma = max(SKEW - 100, 0)/10.

    Cboe defines SKEW = 100 - 10 S with S the 30-day risk-neutral skewness, so gamma = -S for
    negative skew; c = gamma/6 is the first-order Gram-Charlier smile slope (Backus-Foresi-Wu
    2004) and m scales it (calibrated on PPUT in slice 5).
    """
    return m * np.maximum(np.asarray(skew, dtype=float) - 100, 0) / 10 / 6


def smile_factor(c: ArrayLike) -> np.ndarray:
    """A(c) = 1 + c/sqrt(2 pi) + (5/8) c^2 = E[(1 + c z+ - (c/2) z-)^2] for z ~ N(0, 1).

    (E[z+] = E[z-] = 1/sqrt(2 pi), E[(z+)^2] = E[(z-)^2] = 1/2, z+ z- = 0.) The smile's mean
    variance without its 0.5 floor; smile_vol normalises with the floor-aware version below.
    """
    c = np.asarray(c, dtype=float)
    return 1 + c / np.sqrt(2 * np.pi) + 5 / 8 * c**2


def _floored_smile_factor(c: ArrayLike) -> np.ndarray:
    """A_f(c) = E[max(1 + c z+ - (c/2) z-, 0.5)^2] for z ~ N(0, 1): A(c) with the 0.5 floor applied.

    The floor binds for z < a = -1/c, where it replaces (1 + c z/2)^2 by 0.25, so
    A_f(c) = A(c) + 0.25 Phi(a) - [Phi(a) - c phi(a) + (c^2/4)(Phi(a) - a phi(a))]; A_f(0) = 1.
    """
    c = np.asarray(c, dtype=float)
    with np.errstate(all="ignore"):  # c = 0: a = -inf and a phi(a) = NaN, dropped by the where
        a = -1 / c
        lost = norm.cdf(a) - c * norm.pdf(a) + c**2 / 4 * (norm.cdf(a) - a * norm.pdf(a))
        return smile_factor(c) + np.where(c > 0, 0.25 * norm.cdf(a) - lost, 0.0)


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
) -> np.ndarray:
    """Implied vol (decimal) at strike K and tenor T, given the variance-swap vol vs (decimal).

    c = skew_slope(skew, m), F = S e^{(r-q)T}, k = ln(K/F), z = -k/(sigma0 sqrt T) (z > 0 for
    OTM puts), sigma = nu sigma0 max(1 + c z+ - (c/2) z-, 0.5). nu is the vol-risk-premium knob
    (1 = price at the variance-swap level). T <= 0 gives the ATM vol nu sigma0 (bs_price ignores
    the vol there anyway); a NaN T gives NaN.

    sigma0 = vs / sqrt(A_f(c)), A_f the floor-aware smile factor (_floored_smile_factor), so
    E_N[sigma(z)^2] = (nu vs)^2 exactly. DEVIATION from the spec's sigma0 = vs / sqrt(A(c)): A(c)
    ignores the floor, which binds for z < -1/c and would lift the smile's variance above the
    variance swap's (+1.3e-5 relative at c = 0.3, +0.32% at c = 0.6). The switch moves sigma0 by
    -3.7e-7 relative at c = 0.25, -0.16% at c = 0.6, and nothing measurable at c <= 0.1.

    Consequences: (1) d sigma / d ln K = -nu c / sqrt(T) for K < F and -nu c / (2 sqrt T) for
    K > F (until the floor), independent of sigma0, so the skew in log-moneyness scales as
    1/sqrt(T); (2) ASSUMPTION: the call wing has half the put slope (checked later by the CLL
    replication's short calls).

    LIMIT (the brief's formula, kept as specified): the put wing is unbounded, so
    dP/dK = e^{-rT} [N(-d2) - nu c phi(d2)] turns negative once N(-d2)/phi(d2) < nu c. Short-dated
    far-OTM put prices can then fall as K rises (static arbitrage; 7 d, vs 0.15, SKEW 160, m 0.7,
    r = q = 0: P(0.30 S) = 64.5 > P(0.96 S) = 17.5), and at fixed K/F a put does not decay to 0 as
    T -> 0+, because d2 -> 1/(nu c) - nu c ln(F/K)/2 (S 4000, K 3000, vs 0.15, SKEW 150, m 0.5:
    mid 1.76 at 1 d, 1.128 in the limit, then intrinsic 0 at T = 0).
    """
    # ponytail: first-order Gram-Charlier smile (one slope parameter, linear wings, 0.5 floor);
    # fit SVI per date when real option data is present (chains adapter, slice 9). Ceiling: the
    # uncapped put wing above (arbitrage and phantom value on short-dated far-OTM puts). Upgrade:
    # flatten the wing past z_cap, the largest z with z/(nu(1 + c z)) <= x*, N(-x*)/phi(x*) = nu c,
    # and add the cap's truncated moments to A_f; needs a spec decision before slices 4-5 rely on it.
    S, K, T, vs, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, vs, r, q))
    c = skew_slope(skew, m)
    s0 = vs / np.sqrt(_floored_smile_factor(c))
    with np.errstate(all="ignore"):  # a NaN T or vs fails both tests and stays NaN
        z = np.where((T <= 0) | (s0 <= 0), 0.0, -np.log(K / (S * np.exp((r - q) * T))) / (s0 * np.sqrt(T)))
    return nu * s0 * np.maximum(1 + c * np.maximum(z, 0) - c / 2 * np.maximum(-z, 0), 0.5)


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
    h: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mid, bid, ask) in index points: mid = BSM at smile_vol, half = max(h mid, HALF_SPREAD_MIN),
    bid = max(mid - half, 0), ask = mid + half.

    h = 0 means mid fills (half = 0, bid = ask = mid), as slice 5 replication uses. DEVIATION: the
    brief floors every half-spread at HALF_SPREAD_MIN, which would charge a tick at h = 0 too.
    """
    mid = bs_price(S, K, T, smile_vol(S, K, T, vs, skew, r, q, m=m, nu=nu), r, q, cp)
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
) -> float:
    """Scalar strike K with |bs_delta(K, smile_vol(K))| = |delta| (the sign of delta is ignored).

    Solved with brentq on k = ln(K/F) over [-1.5, 1.0], F = S e^{(r-q)T}. Raises ValueError when
    the target is not bracketed there (e.g. |delta| >= e^{-qT}) or an input is NaN.
    """
    F = S * np.exp((r - q) * T)

    def gap(k: float) -> float:
        K = F * np.exp(k)
        vol = smile_vol(S, K, T, vs, skew, r, q, m=m, nu=nu)
        return abs(float(bs_delta(S, K, T, vol, r, q, cp))) - abs(delta)

    if not gap(-1.5) * gap(1.0) <= 0:  # same sign, or NaN
        raise ValueError(f"no strike with |delta| = {abs(delta)} for ln(K/F) in [-1.5, 1.0]")
    # ponytail: assumes T > 0 and vs, nu > 0 (strikes are picked when a leg opens); at expiry or
    # zero vol delta is a 0/1 step and this returns K ~ S. Guard it if a caller ever picks there.
    return float(F * np.exp(brentq(gap, -1.5, 1.0)))


def fill_term_structure(mkt: pd.DataFrame) -> pd.DataFrame:
    """Copy of the market panel with vix3m, vix6m, vix9d back-filled before their first valid date.

    In order: vix3m from vix, vix6m from the (filled) vix3m, vix9d from vix, each by OLS
    ln target = a + b ln regressor on the rows where both are positive (a zero or negative print
    counts as missing). Only rows before the target's first valid value whose regressor is
    positive are filled; they are True in the added boolean column `ts_filled` (flags already in
    the input are kept, so a second call changes nothing). A target with leading NaNs but fewer
    than 250 overlap rows (all-NaN target or regressor included) is left as is and noted in
    df.attrs["warnings"] (appended to any warnings already there).
    vix9d, vix3m and vix6m must be columns (load_market always provides them); an absent one
    raises KeyError.
    """
    # ponytail: one static log-log fit over the whole overlap; a regime-dependent fit (e.g. by VIX
    # level or decade) if back-filled years drive replication errors.
    df = mkt.copy()
    filled = df.get("ts_filled", pd.Series(False, index=df.index)).eq(True)
    warnings = list(mkt.attrs.get("warnings", []))
    for y, x in (("vix3m", "vix"), ("vix6m", "vix3m"), ("vix9d", "vix")):
        lead = df[y].ffill().isna()  # ffill leaves exactly the leading NaNs
        if not lead.any():
            continue
        both = (df[y] > 0) & (df[x] > 0)  # NaN compares False
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

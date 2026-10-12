"""Spitznagel's regime ("dice") engine, Kelly sizing, the book-style annual bootstrap and a
stationary block bootstrap.

Regimes are bins of the S&P 500 calendar-year total return: the book's four (Safe Haven Part
Three, after the Universa "Safe Haven Investing" series) are <= -15% (the crash bin), -15..0,
0..15 and > 15%, crash bin first. A regime model is a probability p_j per regime and a return
per regime: the core's c_j, the haven's h_j and, for an allocation w, the portfolio's
P_j(w) = (1 - w) c_j + w h_j (book mode, annual reset) or, in estimated mode, the mean
calendar-year return of a backtest NAV column in that regime's years. The objective is the
geometric growth g = exp(sum_j p_j ln(1 + P_j)) - 1, not the arithmetic a = sum_j p_j P_j.
G(w) = sum_j p_j ln(1 + c_j + w (h_j - c_j)) is concave in w, so Kelly sizing is a root of G'.
dice_growth, dice_arith, cost_effective and kelly_w's slope skip regimes with p_j = 0 (an
estimated-mode regime with no years has a NaN row and p_j = 0); kelly_w's feasibility bound
keeps them (a calm sample's crash bin still ruins), and breakeven_crash_freq /
crash_frequency_sweep reassign the mass, so a zero-mass crash regime gets weight (_crash_split).

The bootstraps resample history and pair every w on the same paths: book_bootstrap draws whole
years i.i.d. (the book's method), bootstrap draws Politis-Romano blocks of rebalance-period
returns. Returns are decimal simple returns >= -1. Pure numpy/pandas/scipy; the only I/O is
reading the packaged sp500_annual.csv (S&P 500 total return and CPI inflation by year).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from insureport import metrics

# the book's four bins: crash < -15%, -15..0, 0..15, > 15 (right-closed, see regime_of)
REGIME_EDGES: tuple[float, ...] = (-np.inf, -0.15, 0.0, 0.15, np.inf)


def _label(lo: float, hi: float) -> str:
    """Bin label in percent: "< -15%", "-15%..0%", "> 15%"."""
    if lo == -np.inf:
        return f"< {hi * 100:g}%"
    return f"> {lo * 100:g}%" if hi == np.inf else f"{lo * 100:g}%..{hi * 100:g}%"


REGIME_LABELS = tuple(_label(a, b) for a, b in zip(REGIME_EDGES, REGIME_EDGES[1:]))  # "< -15%" ... "> 15%"
# Safe Haven Part Three cartoons, per bin (crash bin first); the insurance crash payoff X defaults to
# (1 - p)/p for a zero mean, the store of value's "CPI" to CPI + REAL_RATE (book_haven). Addendum D5:
# the Universa series' store of value earns CPI + 2% real (the default); the book's, 7% nominal
BOOK_HAVENS = {
    "insurance": ("X", -1.0, -1.0, -1.0),
    "alpha": (0.20, 0.10, 0.05, 0.05),
    "store_of_value": ("CPI", "CPI", "CPI", "CPI"),
    "store_of_value_book": (0.07, 0.07, 0.07, 0.07),
}
REAL_RATE = 0.02
SP500_CSV = Path(__file__).with_name("sp500_annual.csv")
_CAGR_Q = {"cagr_p5": 5, "cagr_p25": 25, "cagr_p50": 50, "cagr_p75": 75, "cagr_p95": 95}


def _annual_table() -> pd.DataFrame:
    """The packaged CSV (total_return, cpi) indexed by year (int); `#` lines (the source note) skipped."""
    return pd.read_csv(SP500_CSV, comment="#", index_col="year")


def load_sp500_annual() -> pd.Series:
    """S&P 500 calendar-year total return (decimal) from the packaged CSV, indexed by year (int),
    1928 to the last complete year."""
    return _annual_table()["total_return"]


def regime_of(ret: np.ndarray | float, edges: tuple[float, ...] = REGIME_EDGES) -> np.ndarray:
    """Regime index per return (0 = the lowest bin). Bins are right-closed, (edges[j], edges[j+1]]:
    -0.15 is in bin 0 (the crash bin), 0.0 in bin 1, 0.15 in bin 2. Values outside finite outer
    edges fall in the first / last bin. Raises ValueError on NaN (drop missing years first)."""
    ret = np.asarray(ret, dtype=float)
    if np.isnan(ret).any():
        raise ValueError("regime_of: NaN return; drop missing years first")
    return np.searchsorted(np.asarray(edges, dtype=float)[1:-1], ret, side="left")


def regime_table(spx_annual: pd.Series, edges: tuple[float, ...] = REGIME_EDGES) -> pd.DataFrame:
    """One row per regime (index 0..J-1, crash first): label, lo, hi (the bin edges), count (years
    in the bin), freq (count / years) and mean_ret (mean return of those years, NaN when none).
    NaN years are dropped."""
    r = spx_annual.dropna()
    j = regime_of(r.to_numpy(), edges)
    n_bins = len(edges) - 1
    count = np.bincount(j, minlength=n_bins)
    lo, hi = np.asarray(edges[:-1], dtype=float), np.asarray(edges[1:], dtype=float)
    return pd.DataFrame(
        {
            "label": [_label(a, b) for a, b in zip(lo, hi)],
            "lo": lo,
            "hi": hi,
            "count": count,
            "freq": count / count.sum(),
            "mean_ret": r.groupby(j).mean().reindex(range(n_bins)).to_numpy(),
        },
        index=pd.RangeIndex(n_bins, name="regime"),
    )


def book_haven(name: str, p_crash: float, insurance_payoff: float | None = None) -> np.ndarray:
    """Per-bin returns of a BOOK_HAVENS cartoon (crash bin first).

    The insurance crash payoff X is insurance_payoff, or (1 - p_crash) / p_crash: the payoff
    that makes p X - (1 - p) 100% = 0, i.e. a zero stand-alone mean (+900% at p = 0.1, Universa
    Part One; about +1000%, the "tenbagger", at p = 1/11, Part Three). Raises ValueError when the
    default X needs p_crash outside (0, 1]. "CPI" (store_of_value) is the mean of CPI_y + REAL_RATE
    over the packaged table's years y (1928-2025) in the bin, binned by regime_of(total_return); NaN
    for a bin with no year. That per-bin mean is a cartoon, not Part One's series: on 1997-2016 at a
    10% weight it costs -9.4 bp of CAGR a year, where Part One's year-by-year CPI_y + 2% gives -18.4 bp
    (it reports -17.9; test_part_one_cartoon_on_1997_2016).
    """
    table = BOOK_HAVENS[name]
    if "X" in table and insurance_payoff is None:
        if not 0 < p_crash <= 1:
            raise ValueError(f"p_crash must be in (0, 1] for the zero-mean insurance payoff, got {p_crash}")
        insurance_payoff = (1 - p_crash) / p_crash
    if "CPI" in table:
        a = _annual_table()[["total_return", "cpi"]].dropna()
        sov = (a["cpi"] + REAL_RATE).groupby(regime_of(a["total_return"].to_numpy())).mean()
        return sov.reindex(range(len(table))).to_numpy(float)
    return np.array([insurance_payoff if v == "X" else v for v in table], dtype=float)


def book_portfolio_returns(core_j: np.ndarray, haven_j: np.ndarray, w: np.ndarray | float) -> np.ndarray:
    """P_j(w) = (1 - w) c_j + w h_j, J x n_w (a scalar w is a one-point grid): annual reset to w."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    return np.outer(core_j, 1 - w) + np.outer(haven_j, w)


def implied_haven(core: np.ndarray | pd.Series, port: np.ndarray | pd.Series, w: float) -> np.ndarray | pd.Series:
    """The inverse of book_portfolio_returns: the haven return h with (1 - w) core + w h = port, i.e.
    (port - (1 - w) core) / w, elementwise (w > 0). On the engine's NAV columns 0 and w (calendar-year or
    rebalance-period returns) it is the haven as the portfolio held it: refilled at every rebalance (a
    -100% year does not zero the next), exact for carve-out rebalancing at those period ends at tc 0;
    under relever funding the sleeve plus the core exposure its financing keeps, less that financing
    (about core + sleeve - (rf + 50 bp)), so that mixing it back at w reproduces the levered portfolio."""
    return (port - (1 - w) * core) / w


def regime_portfolio_returns(
    nav: pd.DataFrame, spx_annual: pd.Series, edges: tuple[float, ...] = REGIME_EDGES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimated mode: (p, P, n) from the calendar-year returns of each NAV column.

    The years are metrics.full_year_returns(nav) that spx_annual has (non NaN). Each year goes to the
    regime of the S&P 500 return of the same year; n_j counts them, p_j = n_j / sum n,
    P (J x n_w) is the mean NAV-column return per regime, a NaN row where n_j = 0. Raises
    ValueError when no full year is left.
    """
    a = metrics.full_year_returns(nav)
    spx = spx_annual.dropna()
    a = a[a.index.isin(spx.index)]
    if a.empty:
        raise ValueError("no full calendar year of nav has an S&P 500 annual return")
    j = regime_of(spx.loc[a.index].to_numpy(), edges)
    n_bins = len(edges) - 1
    n = np.bincount(j, minlength=n_bins)
    return n / n.sum(), a.groupby(j).mean().reindex(range(n_bins)).to_numpy(), n


def _live(p: np.ndarray, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
    """p and each per-regime array restricted to the regimes with p_j != 0."""
    p = np.asarray(p, dtype=float)
    keep = p != 0  # NaN p stays: dice_* give NaN, cost_effective False, kelly_w raises
    return (p[keep], *(np.asarray(x, dtype=float)[keep] for x in xs))


def dice_growth(p: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Geometric growth per roll exp(sum_j p_j ln(1 + P_j)) - 1, per column of P (J or J x n_w).

    -1 (ruin) when 1 + P_j <= 0 in a regime with p_j > 0. The objective of the book: the die of
    Safe Haven ch. 2 averages +3.33% (dice_arith) but compounds at -1.53%.
    """
    p, P = _live(p, P)
    with np.errstate(divide="ignore"):
        return np.exp(p @ np.log(np.maximum(1 + P, 0.0))) - 1


def dice_arith(p: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Arithmetic mean per roll sum_j p_j P_j, per column of P (the ensemble average)."""
    p, P = _live(p, P)
    return p @ P


def kelly_w(p: np.ndarray, core_j: np.ndarray, haven_j: np.ndarray, w_max: float = 1.0) -> float:
    """The growth-optimal (Kelly) haven weight: argmax over [0, w_max] of G(w).

    G'(w) = sum_j p_j d_j / (1 + c_j + w d_j), d_j = h_j - c_j, decreasing in w, summed over
    the regimes with p_j > 0. Feasibility: 1 + c_j + w d_j > 0 in every regime with finite
    returns, p_j = 0 included (a calm sample's crash bin can still come), so w < (1 + c_j) / -d_j
    wherever d_j < 0 (a haven losing 100% caps w below 1); w_max_feasible = min(w_max, that bound
    less 1e-9 relative). w* = 0 if G'(0) <= 0; w_max_feasible if G' >= 0 there; else the brentq
    root of G' on (0, w_max_feasible). Two outcomes: w* = p/a - (1 - p)/b (Kelly 1956).
    """
    c_all, h_all = np.asarray(core_j, dtype=float), np.asarray(haven_j, dtype=float)
    d_all = h_all - c_all
    neg = d_all < 0  # NaN rows (estimated mode, no years) compare False and stay out
    bound = np.min((1 + c_all[neg]) / -d_all[neg]) if neg.any() else np.inf
    p, c, h = _live(p, c_all, h_all)
    d = h - c

    def slope(w: float) -> float:
        return float(p @ (d / (1 + c + w * d)))

    if slope(0.0) <= 0:
        return 0.0
    hi = min(w_max, bound * (1 - 1e-9))  # the bound is open (ruin on it): stop just short
    return float(hi) if slope(hi) >= 0 else float(brentq(slope, 0.0, hi))


def cost_effective(p: np.ndarray, core_j: np.ndarray, haven_j: np.ndarray) -> bool:
    """Spitznagel's test: a little of the haven raises growth, G'(0) = sum_j p_j (h_j - c_j)/(1 + c_j) > 0."""
    p, c, h = _live(p, core_j, haven_j)
    return bool(p @ ((h - c) / (1 + c)) > 0)


def _crash_split(p: np.ndarray, crash: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Within-group weights (wc, wn), each summing to 1: p's proportions inside the crash regimes
    and inside the others. A group with no mass in p is split equally, so a calm sample's single
    crash regime still gets weight 1. Raises ValueError unless crash names some, but not all, of
    the regimes 0..J-1 (an empty or out-of-range set would read as a 100% or 0% breakeven)."""
    p = np.asarray(p, dtype=float)
    if not set() < set(crash) < set(range(len(p))):
        raise ValueError(f"crash must name some but not all of regimes 0..{len(p) - 1}, got {crash}")
    is_crash = np.isin(np.arange(len(p)), crash)

    def share(mask: np.ndarray) -> np.ndarray:
        q = np.where(mask, p, 0.0)
        return q / q.sum() if q.sum() > 0 else mask / mask.sum()

    return share(is_crash), share(~is_crash)


def breakeven_crash_freq(
    p: np.ndarray, core_j: np.ndarray, haven_j: np.ndarray, crash: tuple[int, ...] = (0,)
) -> float:
    """Crash frequency pi* at which a sliver of the haven stops raising growth (G'(0; pi*) = 0).

    The crash regimes get total mass pi, the others 1 - pi, each group split in its proportions
    in p (equally when p gives it no mass). With x_j = (h_j - c_j)/(1 + c_j) and A / B the
    weighted means of x over the crash / other regimes, G'(0; pi) = pi A + (1 - pi) B, so
    pi* = B / (B - A). The haven is cost-effective for pi > pi* when A > B (it pays off in
    crashes) and for pi < pi* when A < B. NaN when B = A or pi* is outside [0, 1]; ValueError
    when crash is empty, out of range or every regime (_crash_split).
    """
    wc, wn = _crash_split(p, crash)
    c, h = np.asarray(core_j, dtype=float), np.asarray(haven_j, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (h - c) / (1 + c)
        A, B = (wt[wt > 0] @ x[wt > 0] for wt in (wc, wn))  # zero-weight (maybe NaN) rows stay out
        pi = B / (B - A)
    return float(pi) if 0 <= pi <= 1 else np.nan


def crash_frequency_sweep(
    p: np.ndarray,
    core_j: np.ndarray,
    haven_j: np.ndarray,
    w: np.ndarray,
    crash: tuple[int, ...] = (0,),
    grid: np.ndarray = np.linspace(0, 0.3, 31),
) -> pd.DataFrame:
    """dice_growth of the book-mode portfolio when crashes come with frequency pi, for each pi
    of grid (rows, index "pi") and each w (columns labelled by w rounded to 12 decimals).
    The crash mass is reassigned as in breakeven_crash_freq (a sample with no crash year still
    sweeps); at pi = the crash mass of p a row is dice_growth(p, book_portfolio_returns(core_j,
    haven_j, w))."""
    P = book_portfolio_returns(core_j, haven_j, w)
    wc, wn = _crash_split(p, crash)
    rows = [dice_growth(pi * wc + (1 - pi) * wn, P) for pi in grid]
    cols = pd.Index(np.round(np.atleast_1d(np.asarray(w, dtype=float)), 12), name="w")
    return pd.DataFrame(rows, index=pd.Index(np.asarray(grid, dtype=float), name="pi"), columns=cols)


def _cagr_cols(g: np.ndarray, g0: np.ndarray) -> dict:
    """CAGR percentiles over paths and p_beat = P(g > g0), paths paired with the w = 0 CAGR g0."""
    return {**dict(zip(_CAGR_Q, np.percentile(g, list(_CAGR_Q.values())))), "p_beat": float(np.mean(g > g0))}


def book_bootstrap(
    core_annual: pd.Series,
    haven_annual: pd.Series,
    w: np.ndarray,
    horizon: int = 25,
    n_paths: int = 10_000,
    seed: int = 0,
    rebalance: str = "annual",
    mode: str = "joint",
) -> pd.DataFrame:
    """The book's bootstrap: n_paths paths of `horizon` calendar years drawn i.i.d. with replacement.

    History = the years both annual series have. mode "joint" draws a year and takes its core
    and haven returns together (the crash link); "reshuffle" draws the haven's year separately
    (a true insurance payoff's gain must vanish); "no_crash" draws only years with a core
    return > -15% (the cost when no crash comes). Wealth per $1: rebalance "annual" ->
    prod_y [(1 - w)(1 + c_y) + w (1 + h_y)], "none" -> (1 - w) prod(1 + c) + w prod(1 + h).
    The same draws serve every w (paired). Rows = w (rounded to 12 decimals): cagr_p5 ... cagr_p95
    (CAGR = wealth^(1/horizon) - 1), tw_p50 (median terminal wealth), p_beat = P(CAGR_w > CAGR_0)
    and d_cagr_p50 = median(CAGR_w - CAGR_0), CAGR_0 being the core alone on the same path.
    """
    if mode not in ("joint", "reshuffle", "no_crash"):
        raise ValueError(f"mode must be 'joint', 'reshuffle' or 'no_crash', got {mode!r}")
    if rebalance not in ("annual", "none"):
        raise ValueError(f"rebalance must be 'annual' or 'none', got {rebalance!r}")
    both = pd.concat([core_annual, haven_annual], axis=1, join="inner").dropna()
    c, h = both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float)
    if mode == "no_crash":
        c, h = c[c > REGIME_EDGES[1]], h[c > REGIME_EDGES[1]]
    if not len(c):
        raise ValueError("no common year with both returns" + (" above -15% (no_crash)" if mode == "no_crash" else ""))
    # ponytail: annual i.i.d. draws as in the book; a moving-block option can come later
    rng = np.random.default_rng(seed)
    yc = rng.integers(0, len(c), (n_paths, horizon))
    yh = rng.integers(0, len(c), (n_paths, horizon)) if mode == "reshuffle" else yc
    gc, gh = 1 + c[yc], 1 + h[yh]
    tc, th = gc.prod(axis=1), gh.prod(axis=1)
    g0 = tc ** (1 / horizon) - 1
    rows = {}
    for x in np.atleast_1d(np.asarray(w, dtype=float)):
        tw = ((1 - x) * gc + x * gh).prod(axis=1) if rebalance == "annual" else (1 - x) * tc + x * th
        g = tw ** (1 / horizon) - 1
        rows[round(x, 12)] = {**_cagr_cols(g, g0), "tw_p50": np.median(tw), "d_cagr_p50": np.median(g - g0)}
    out = pd.DataFrame.from_dict(rows, orient="index").rename_axis("w")
    return out[[*_CAGR_Q, "tw_p50", "p_beat", "d_cagr_p50"]]


def stationary_bootstrap_idx(n: int, n_paths: int, horizon: int, mean_block: float, seed: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap: n_paths x horizon indices into a history of length n.

    I[:, 0] ~ U{0..n-1}; for h >= 1, with probability 1 / mean_block a fresh uniform index (a new
    block), otherwise I[:, h] = (I[:, h-1] + 1) mod n (the block continues, wrapping around).
    Block lengths are geometric with mean mean_block; mean_block = inf never restarts.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_paths, horizon))
    fresh = rng.random((n_paths, horizon)) < 1 / mean_block
    for t in range(1, horizon):
        idx[:, t] = np.where(fresh[:, t], idx[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap(
    rc: pd.Series,
    rs: pd.Series,
    w: np.ndarray,
    periods_per_year: int,
    horizons: tuple[int, ...] = (5, 10, 20),
    n_paths: int = 2000,
    mean_block: float = 6,
    seed: int = 0,
    dd_cap: float = 0.30,
) -> pd.DataFrame:
    """Stationary block bootstrap of the (1 - w) core / w sleeve portfolio, paired across w.

    rc, rs: core and sleeve returns per rebalance period (e.g. backtest.period_returns), on the
    periods both have. One index path set (stationary_bootstrap_idx, longest horizon; shorter
    horizons use its first H periods) serves every w and horizon. Per path, wealth after h
    periods W_h = prod_{k <= h} [(1 - w)(1 + rc[I_k]) + w (1 + rs[I_k])] (rebalanced to w every
    period), H = horizon * periods_per_year and ann = periods_per_year / H:
    cagr = W_H^ann - 1 (p5 ... p95), p_beat = P(cagr_w > cagr_0) on the same paths,
    ensemble_growth = mean(W_H)^ann - 1 and median_growth = median(W_H)^ann - 1 (the ergodicity
    panel: the ensemble average flatters what a typical path compounds), max drawdown from the
    running peak of W (starting at 1): maxdd_p50 its median, maxdd_p95 the depth only 5% of
    paths exceed (5th percentile of the negative values), p_dd_breach = P(maxdd < -dd_cap).
    Rows = (horizon, w) MultiIndex, w rounded to 12 decimals.
    """
    # ponytail: period resolution; misses intra-period drawdowns (daily-path bootstrap if verdicts hinge on it)
    both = pd.concat([rc, rs], axis=1, join="inner").dropna()
    c, s = both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float)
    if not len(c):
        raise ValueError("no common period with both returns (check that rc and rs share index labels)")
    H = [h * periods_per_year for h in horizons]
    idx = stationary_bootstrap_idx(len(c), n_paths, max(H), mean_block, seed)
    gc, gs = 1 + c[idx], 1 + s[idx]
    wealth0 = np.cumprod(gc, axis=1)
    rows, keys = [], []
    for x in np.atleast_1d(np.asarray(w, dtype=float)):
        wealth = np.cumprod((1 - x) * gc + x * gs, axis=1)
        ratio = np.maximum.accumulate(wealth, axis=1)  # running peak (from 1), then W / peak, in place
        np.maximum(ratio, 1.0, out=ratio)
        np.divide(wealth, ratio, out=ratio)
        for years, n in zip(horizons, H):
            tw, ann, dd = wealth[:, n - 1], periods_per_year / n, ratio[:, :n].min(axis=1) - 1
            rows.append(
                {
                    **_cagr_cols(tw**ann - 1, wealth0[:, n - 1] ** ann - 1),
                    "ensemble_growth": tw.mean() ** ann - 1,
                    "median_growth": np.median(tw) ** ann - 1,
                    "maxdd_p50": np.median(dd),
                    "maxdd_p95": np.percentile(dd, 5),
                    "p_dd_breach": float(np.mean(dd < -dd_cap)),
                }
            )
            keys.append((years, round(x, 12)))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["horizon", "w"])).sort_index()

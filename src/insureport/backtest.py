"""Two-asset backtest engine vectorized over the allocation grid, the safe-haven frontier and its
robustness post-processing.

For every w of a grid at once the portfolio holds (1 - w) in the core and w in the sleeve,
drifts with the two daily return series and goes back to (1 - w, w) on calendar dates (an
option sleeve's `event` dates) or when the sleeve share leaves a relative band, paying
proportional costs on the flows. The frontier table holds the metrics.py statistics per w;
best_allocation is the empirical Kelly choice under a drawdown cap; the cash-matched functions
compare a sleeve with holding less equity (the T-bill frontier, AQR's test).

Pure numpy/pandas, no I/O. Returns are daily simple returns on a DatetimeIndex; NAV frames have
one column per w, labelled by the w value; frontier tables are indexed by w.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insureport import metrics


@dataclass(frozen=True)
class Policy:
    """Rebalancing and cost rules.

    band: also rebalance when the sleeve share s/v strays from w by more than band * w
    (relative; None = calendar dates only). tc_core / tc_sleeve: proportional cost on the core /
    sleeve flows.
    """

    band: float | None = None
    tc_core: float = 0.0002
    tc_sleeve: float = 0.0002


def simulate(
    core: pd.Series, sleeve: pd.Series, w: np.ndarray | float, rebalance: pd.DatetimeIndex, policy: Policy = Policy()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NAV of the (1 - w) core / w sleeve portfolio for every w of a 1-D grid (a scalar w is a
    one-point grid), and its trading stats.

    The engine runs on the dates where both core and sleeve have a (non-NaN) return. A return on
    a date the other series lacks compounds into that series' next common date, so nothing is
    lost when, e.g., a core with a missing print meets a sleeve on the full ^GSPC calendar; a
    rebalance date the engine does not see moves to the next common date.
    NAV is 1.0 on the first date (its returns are not applied), held as c = 1 - w, s = w. Each
    later day c and s grow by their returns; on a rebalance date, or when
    |s/v - w| > band * w (never at w = 0, whose s stays 0), the cost
    tc_core |(1 - w) v - c| + tc_sleeve |w v - s| comes out of v and the legs reset to
    (1 - w) v, w v. A sleeve return of -1 empties the sleeve until the next rebalance refills it.

    Returns (nav, stats): nav is dates x w, its columns labelled by w rounded to 12 decimals
    (np.linspace(0, 1, 21)[6] is labelled 0.3, not 0.30000000000000004); stats, indexed the
    same, holds turnover (sum of the sleeve flows |w v - s| / v per year) and cost_drag (sum of
    costs / v per year), with v the pre-cost value on the day and years = calendar days /
    365.25. A wiped-out NAV (v = 0) trades nothing. Raises ValueError when core and sleeve
    share fewer than 2 dates.
    """
    core, sleeve = core.dropna(), sleeve.dropna()
    dates = core.index.intersection(sleeve.index)
    if len(dates) < 2:
        raise ValueError(f"core and sleeve share {len(dates)} dates; need at least 2")
    rc, rs = (period_returns(x, dates).iloc[:, 0].reindex(dates).to_numpy(float) for x in (core, sleeve))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    wc = 1 - w
    band_w = np.full(len(w), np.inf) if policy.band is None else policy.band * w
    pos = dates.searchsorted(rebalance)
    due = np.zeros(len(dates), dtype=bool)
    due[pos[pos < len(dates)]] = True
    nav = np.empty((len(dates), len(w)))
    nav[0] = 1.0
    c, s = wc.copy(), w.copy()
    flows, costs = np.zeros(len(w)), np.zeros(len(w))
    # ponytail: daily loop; numba if the grid grows
    with np.errstate(divide="ignore", invalid="ignore"):  # v = 0: s/v is NaN and never trips the band
        for t in range(1, len(dates)):
            c *= 1 + rc[t]
            s *= 1 + rs[t]
            v = c + s
            hit = due[t] | (np.abs(s / v - w) > band_w)
            if hit.any():
                ds = np.where(hit, w * v - s, 0.0)
                cost = policy.tc_core * np.abs(np.where(hit, wc * v - c, 0.0)) + policy.tc_sleeve * np.abs(ds)
                per_v = np.where(v > 0, 1 / v, 0.0)
                flows += np.abs(ds) * per_v
                costs += cost * per_v
                v = v - cost
                c = np.where(hit, wc * v, c)
                s = np.where(hit, w * v, s)
            nav[t] = v
    cols = pd.Index(np.round(w, 12), name="w")  # labels only; the engine used the exact w
    years = (dates[-1] - dates[0]).days / 365.25
    stats = pd.DataFrame({"turnover": flows / years, "cost_drag": costs / years}, index=cols)
    return pd.DataFrame(nav, dates, cols), stats


def frontier(
    core: pd.Series,
    sleeve: pd.Series,
    w: np.ndarray,
    rebalance: pd.DatetimeIndex,
    policy: Policy = Policy(),
    crashes: dict[str, tuple[str, str]] = metrics.CRASHES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frontier table (indexed by w) and the NAV frame of simulate().

    Columns, all from metrics.py on the NAV columns: cagr, ann_vol (of the daily NAV returns),
    arith_mean (mean calendar-year return), vol_tax (vol_tax's tax), max_dd, calmar
    (cagr / |max_dd|, NaN at max_dd 0), tuw, longest_dd_days (longest_recovery_days),
    worst_year (lowest calendar-year return), turnover and cost_drag (simulate's stats), then
    "crash:<name>" = nav(last date <= end) / nav(first date >= start) - 1 for each crash window
    inside the sample, as metrics.crash_returns computes it.
    """
    nav, stats = simulate(core, sleeve, w, rebalance, policy)
    annual = metrics.annual_returns(nav)
    g, dd = metrics.cagr(nav), metrics.max_drawdown(nav)
    table = pd.DataFrame(
        {
            "cagr": g,
            "ann_vol": metrics.ann_vol(nav.pct_change().iloc[1:]),
            "arith_mean": annual.mean(),
            "vol_tax": metrics.vol_tax(nav)["tax"],
            "max_dd": dd,
            "calmar": g / -dd.where(dd < 0),
            "tuw": metrics.time_under_water(nav),
            "longest_dd_days": [metrics.longest_recovery_days(nav[x]) for x in nav],
            "worst_year": annual.min(),
            "turnover": stats["turnover"],
            "cost_drag": stats["cost_drag"],
        }
    )
    # metrics.crash_returns' window rule on every column at once (per column it costs ~0.3 s at 41 w)
    for name, (start, end) in crashes.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        window = nav.loc[start:end]
        if nav.index[0] <= start and end <= nav.index[-1] and len(window):
            table[f"crash:{name}"] = window.iloc[-1] / window.iloc[0] - 1
    return table, nav


def best_allocation(table: pd.DataFrame, dd_cap: float | None) -> float | None:
    """Empirical Kelly: the w with the highest cagr among rows with max_dd >= -dd_cap.

    dd_cap None = no cap. Ties go to the smaller w. None when no row is feasible.
    """
    ok = table if dd_cap is None else table[table["max_dd"] >= -dd_cap]
    if ok.empty:
        return None
    return float(ok["cagr"].sort_index().idxmax())


def _cash_interp(cash_table: pd.DataFrame, by: str, x: float, y: str) -> float:
    """Column y ("w" = the index) of the cash frontier at by == x, linear in `by` after sorting
    on it; NaN when x is outside by's range (ends included)."""
    cash = cash_table.assign(w=cash_table.index).sort_values(by)
    return float(np.interp(x, cash[by].to_numpy(float), cash[y].to_numpy(float), left=np.nan, right=np.nan))


def dd_matched_cash_gap(table: pd.DataFrame, cash_table: pd.DataFrame, w: float) -> float:
    """cagr(w) minus the T-bill frontier's CAGR at the same max_dd (linear in max_dd).

    cash_table is a frontier table for the T-bill sleeve (max_dd monotone in w). w is a grid
    value or its label in table's index (rounded to 12 decimals as simulate labels it; KeyError
    when off the grid). NaN when max_dd(w) is outside the cash frontier's max_dd range.
    """
    row = table.loc[float(np.round(w, 12))]
    return float(row["cagr"] - _cash_interp(cash_table, "max_dd", row["max_dd"], "cagr"))


def cagr_matched_cash_dd(table: pd.DataFrame, cash_table: pd.DataFrame, w: float) -> dict:
    """Israelov's "divested twin": the T-bill-frontier allocation with the same CAGR as w.

    {"w_cash": cash w with cagr == cagr(w), "maxdd_cash": its max_dd (both linear in cagr
    between neighbouring grid points), "maxdd_gap": max_dd(w) - maxdd_cash} (a positive gap:
    the sleeve falls less for the same growth). w is a grid value or its label in table's index
    (rounded to 12 decimals as simulate labels it; KeyError when off the grid). All NaN when
    cagr(w) is outside the cash frontier's CAGR range.

    Cash CAGR is humped in w when the core's arithmetic excess return over T-bills lies between
    0 and its variance (common in weak-decade subsamples), so one CAGR can have two cash twins.
    The twin is taken past the CAGR peak (the deleverage side: less equity, shallower max_dd,
    the harder comparison for the sleeve), and before the peak only when that side cannot
    reach cagr(w) (a CAGR below the w = 1 end's).
    """
    # ponytail: one CAGR hump at most (a rebalanced two-asset mix is ~concave in w); noisier frontiers need a root scan
    row = table.loc[float(np.round(w, 12))]
    cash = cash_table.sort_index()
    peak = cash["cagr"].idxmax()
    branch = cash.loc[peak:]
    if row["cagr"] < branch["cagr"].min():
        branch = cash.loc[:peak]
    maxdd_cash = _cash_interp(branch, "cagr", row["cagr"], "max_dd")
    return {
        "w_cash": _cash_interp(branch, "cagr", row["cagr"], "w"),
        "maxdd_cash": maxdd_cash,
        "maxdd_gap": float(row["max_dd"] - maxdd_cash),
    }


def rolling_excess(nav: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    """Per column: trailing CAGR over 252 * years daily returns minus the w = 0 column's.

    The CAGR of each window is metrics.cagr's (calendar-day annualization); rows before the
    window is full are NaN. nav needs a 0.0 column.
    """
    n = 252 * years
    days = nav.index.to_series().diff(n).dt.days
    g = (nav / nav.shift(n)).pow(365.25 / days, axis=0) - 1
    return g.sub(g[0.0], axis=0)


def period_returns(r: pd.Series | pd.DataFrame, period_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Compound daily returns into the periods ending on each given date.

    A row belongs to the first period end on or after its date (so rows before the first end
    belong to the first period); rows after the last end form a trailing period. Each period is
    labelled by its last row's date; ends with no rows give no period. NaNs inside a period are
    skipped (count as 0%); only a period whose returns are all NaN (e.g. a column that starts
    later) is NaN. Per column.
    """
    r = pd.DataFrame(r)
    period = pd.DatetimeIndex(period_ends).sort_values().searchsorted(r.index)
    label = r.index.to_series().groupby(period).transform("last").to_numpy()
    return (1 + r).groupby(label).prod(min_count=1) - 1

"""Portfolio diagnostics through Spitznagel's lens (the objective is the geometric growth rate):
CAGR vs the arithmetic mean, the volatility tax, drawdowns, betas, crash participation,
concentration and the index-put hedge notional.

Pure pandas/numpy, no I/O. `r` is a daily simple-return Series, `nav` a level Series on a
DatetimeIndex; functions marked "per column" also take a DataFrame (one portfolio per column)
and return one value per column. Years are 365.25 calendar days.
"""

import numpy as np
import pandas as pd

from insureport.portfolio import Portfolio

# Harvey et al. (2019)'s eight S&P 500 drawdowns over 15% (1985-2018), COVID, the 2022 grind and the 2025 tariff
# shock (addendum D14)
CRASHES: dict[str, tuple[str, str]] = {
    "1987 Black Monday": ("1987-08-25", "1987-12-04"),
    "1990 Gulf War": ("1990-07-16", "1990-10-11"),
    "1998 LTCM": ("1998-07-17", "1998-10-08"),
    "2000-02 Dot-com": ("2000-03-24", "2002-10-09"),
    "2007-09 GFC": ("2007-10-09", "2009-03-09"),
    "2010 Flash crash": ("2010-04-23", "2010-07-02"),
    "2011 Debt ceiling": ("2011-04-29", "2011-10-03"),
    "2018 Q4": ("2018-09-20", "2018-12-24"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 Rates": ("2022-01-03", "2022-10-12"),
    "2025 Tariff shock": ("2025-02-19", "2025-04-08"),
}
# drawdown speed (D14: puts win fast crashes, lose slow grinds): "fast" when the window spans at most 6 months
CRASH_SPEED: dict[str, str] = {
    k: "fast" if (pd.Timestamp(e) - pd.Timestamp(s)).days <= 183 else "slow" for k, (s, e) in CRASHES.items()
}


def nav_from_returns(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """NAV_t = prod_{s <= t} (1 + r_s) with the first return ignored, so NAV is 1.0 on the first date. Per column."""
    growth = 1 + r
    growth.iloc[0] = 1.0
    return growth.cumprod()


def cagr(nav: pd.Series | pd.DataFrame) -> float | pd.Series:
    """(NAV_last / NAV_first) ** (365.25 / calendar days between them) - 1. Per column."""
    days = (nav.index[-1] - nav.index[0]).days
    return (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1


def ann_vol(r: pd.Series) -> float:
    """std(daily returns, ddof 1) * sqrt(252)."""
    return r.std() * np.sqrt(252)


def annual_returns(nav: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calendar-year returns NAV(last row of year Y) / NAV(last row of Y - 1) - 1, indexed by year.

    The first (partial) year is measured from the first NAV, and dropped when it is that NAV
    alone (a NAV starting on its year's last row has no first-year return); the last year may
    be partial too. Per column.
    """
    year_end = nav.groupby(nav.index.year).last()
    prev = year_end.shift(1)
    prev.iloc[0] = nav.iloc[0]
    out = year_end / prev - 1
    return out.iloc[1:] if (nav.index.year == nav.index[0].year).sum() == 1 else out


def full_year_returns(nav: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """annual_returns(nav) for full calendar years only: the first year is dropped (the NAV starts inside
    it, so its return is measured from the first NAV), the last when the NAV ends before that year's last
    business day (Mon-Fri, so a calendar month-end NAV ending on a weekend Dec 31 keeps it). Per column."""
    a = annual_returns(nav)
    first, last = nav.index[0], nav.index[-1]
    return a[(a.index > first.year) & ((a.index < last.year) | ((last + pd.offsets.BDay()).year > last.year))]


def vol_tax(nav: pd.Series | pd.DataFrame, full_years: bool = False) -> dict:
    """Volatility tax on the calendar-year returns R_1..R_n of `nav` (annual_returns; full_year_returns
    when full_years, so a partial first or last year cannot pull the mean below the CAGR).

    arith = mean(R), geo = (prod(1 + R))^(1/n) - 1, tax = arith - geo, approx = var(R, ddof=0) / 2
    (the second-order estimate of the tax). Values are Series for a DataFrame; NaN with no year.
    """
    # ponytail: full_years False (the frontier's arith_mean / vol_tax) counts partial first/last years as years
    a = full_year_returns(nav) if full_years else annual_returns(nav)
    arith = a.mean()
    geo = (1 + a).prod() ** (1 / len(a)) - 1 if len(a) else arith
    return {"arith": arith, "geo": geo, "tax": arith - geo, "approx": a.var(ddof=0) / 2}


def drawdown(nav: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """NAV_t / max_{s <= t} NAV_s - 1 (0 at a running peak, negative below it). Per column."""
    return nav / nav.cummax() - 1


def max_drawdown(nav: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Most negative drawdown (0 if NAV never falls). Per column."""
    return drawdown(nav).min()


def time_under_water(nav: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Fraction of rows strictly below the running peak. Per column."""
    return (nav < nav.cummax()).mean()


def longest_recovery_days(nav: pd.Series) -> int:
    """Longest calendar-day span from a peak to the first later date at or above it.

    Only spans with at least one row strictly below the peak count (0 if NAV never falls); a
    drawdown still open at the end runs to the last date.
    """
    under = nav < nav.cummax()
    if not under.any():
        return 0
    at_peak = nav.index.to_series().where(~under)  # dates at a running peak, NaT elsewhere
    span = at_peak.bfill().fillna(nav.index[-1]) - at_peak.ffill()
    return int(span[under].max().days)


def window_return(level: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """level(last date <= end) / level(first date >= start) - 1; NaN unless [start, end] lies inside level's dates."""
    level = level.dropna()
    if level.empty or start < level.index[0] or end > level.index[-1]:
        return np.nan
    window = level.loc[start:end]
    if window.empty:  # a sparse (e.g. monthly) level with no row inside the window
        return np.nan
    return float(window.iloc[-1] / window.iloc[0] - 1)


def crash_returns(
    nav: pd.Series, crashes: dict[str, tuple[str, str]] = CRASHES, bench: pd.Series | None = None
) -> pd.DataFrame:
    """NAV return over each crash window that lies wholly inside the sample.

    Rows = crash names (windows cut by the sample edges are omitted); columns start, end (the
    window dates), ret = nav(last date <= end) / nav(first date >= start) - 1, and bench_ret,
    the same for the `bench` level series (NaN without a bench or when it does not cover the window).
    """
    rows = {}
    for name, (s, e) in crashes.items():
        start, end = pd.Timestamp(s), pd.Timestamp(e)
        ret = window_return(nav, start, end)
        if not np.isnan(ret):
            rows[name] = [start, end, ret, np.nan if bench is None else window_return(bench, start, end)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=["start", "end", "ret", "bench_ret"])


def beta(r: pd.Series, m: pd.Series) -> float:
    """OLS slope cov(r, m) / var(m) of daily returns, on the dates both have."""
    both = pd.concat([r, m], axis=1, join="inner").dropna()
    c = np.cov(both.iloc[:, 0], both.iloc[:, 1])
    return float(c[0, 1] / c[1, 1])


def downside_beta(r: pd.Series, m: pd.Series) -> float:
    """OLS slope of r on m using only the days where m < 0."""
    return beta(r, m[m < 0])


def concentration(weights: dict[str, float]) -> dict:
    """Concentration of the non-CASH weights, renormalized to sum to 1.

    hhi = sum w_i^2, eff_n = 1 / hhi, top1 = max w_i, top5 = sum of the 5 largest. All NaN when
    nothing but CASH is held.
    """
    w = sorted((v for k, v in weights.items() if k != "CASH"), reverse=True)
    total = sum(w)
    if not total > 0:
        return dict.fromkeys(("hhi", "eff_n", "top1", "top5"), np.nan)
    w = [v / total for v in w]
    hhi = sum(v * v for v in w)
    return {"hhi": hhi, "eff_n": 1 / hhi, "top1": w[0], "top5": sum(w[:5])}


def hedge_notional(value: float, beta_down: float, spx_level: float) -> dict:
    """SPX put notional that hedges the portfolio's downside beta.

    notional = value * max(beta_down, 0); spx_contracts = notional / (100 * SPX) ($100
    multiplier); xsp_contracts = notional / (10 * SPX) (XSP is SPX / 10 with a $100 multiplier).
    A negative downside beta (bonds, gold) needs no index puts: notional 0.
    """
    # ponytail: fractional contracts; rounding and minimum size are the UI's job
    notional = value * max(beta_down, 0.0)
    return {
        "notional": notional,
        "spx_contracts": notional / (100 * spx_level),
        "xsp_contracts": notional / (10 * spx_level),
    }


def diagnose(core_r: pd.Series, spxtr_r: pd.Series, p: Portfolio, mkt: pd.DataFrame) -> dict:
    """Every diagnostic of the core return series in one flat dict.

    Keys: start, end, years (calendar days / 365.25), cagr, arith / vol_tax / vol_tax_approx
    (vol_tax's arith / tax / approx over the full calendar years; NaN with none), ann_vol, max_dd, tuw,
    longest_recovery_days, beta, downside_beta (vs spxtr_r), hhi, eff_n, top1, top5 (concentration of
    p.weights), spx_cagr,
    spx_max_dd (S&P 500 total return over the core's sample), crashes (crash_returns with the
    S&P 500 total-return NAV as bench) and hedge (hedge_notional on p.value, or $1,000,000 when
    it is None, at the last mkt.spx). core_r's first row is the 0 placeholder core_returns emits,
    so it is left out of ann_vol and the betas (nav_from_returns ignores it too).

    spxtr_r is the S&P 500 total return's pct_change, so its level exists from the row before
    its first valid return. core_r is first trimmed to that span (and its first row reset to
    the 0 placeholder) so every core and S&P statistic covers the same window; `start` reports
    it. Raises ValueError if fewer than two core dates fall inside it.
    """
    valid = spxtr_r.dropna()
    if valid.empty:
        raise ValueError("spxtr_r has no returns")
    first = spxtr_r.index[max(spxtr_r.index.get_loc(valid.index[0]) - 1, 0)]
    core_r = core_r.loc[first : valid.index[-1]].copy()
    if len(core_r) < 2:
        raise ValueError(f"core_r has fewer than two dates in spxtr_r's span {first:%F}..{valid.index[-1]:%F}")
    core_r.iloc[0] = 0.0
    nav = nav_from_returns(core_r)
    start, end = nav.index[0], nav.index[-1]
    spx_nav = nav_from_returns(spxtr_r.loc[start:end])
    daily = core_r.iloc[1:]
    # DEVIATION (slice 11, an existing public API's output changed): arith / vol_tax / vol_tax_approx now cover the
    # full calendar years only (vol_tax full_years=True; they spanned the partial first and last years before), so a
    # partial year cannot pull the mean below the CAGR on the Diagnostics page
    vt = vol_tax(nav, full_years=True)
    beta_down = downside_beta(daily, spxtr_r)
    return {
        "start": start,
        "end": end,
        "years": (end - start).days / 365.25,
        "cagr": cagr(nav),
        "arith": vt["arith"],
        "vol_tax": vt["tax"],
        "vol_tax_approx": vt["approx"],
        "ann_vol": ann_vol(daily),
        "max_dd": max_drawdown(nav),
        "tuw": time_under_water(nav),
        "longest_recovery_days": longest_recovery_days(nav),
        "beta": beta(daily, spxtr_r),
        "downside_beta": beta_down,
        **concentration(p.weights),
        "spx_cagr": cagr(spx_nav),
        "spx_max_dd": max_drawdown(spx_nav),
        "crashes": crash_returns(nav, bench=spx_nav),
        "hedge": hedge_notional(p.value if p.value is not None else 1_000_000, beta_down, mkt["spx"].dropna().iloc[-1]),
    }

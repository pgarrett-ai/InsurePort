"""Pricer validation: replicate Cboe's PPUT, PPUT3M and CLL strategy indices with synthetic option
sleeves, measure how closely they track the real index levels, and calibrate pricing.SKEW_MULT on PPUT.

Pure numpy/pandas + sleeves.py, no I/O. Each replica is sleeves.option_sleeve in overlay mode
(coverage 1: one option per unit of index) holding the S&P 500 total-return index, rolled on
Cboe's calendar (third Friday, or the preceding trading day) and filled at mid (h = 0: Cboe fills at
an intraday VWAP and marks at mid; the strategies' default h = 0.05 is a separate, conservative
choice). The real levels are the panel columns pput, cll and pput3m (load_market has no pput3m:
pass data.cboe("PPUT3M") to validate, or add the column).

ASSUMPTION: the specs follow Cboe's published PPUT / PPUT3M / CLL methodologies as read in the
project research (PPUT: monthly put 5% OTM; PPUT3M: quarterly put 10% OTM; CLL: quarterly put 5%
OTM, monthly call 10% OTM sold; puts on the Mar/Jun/Sep/Dec cycle; CLL's cross-roll not modelled,
see CLL_SPEC).
Known error sources, disclosed rather than modelled:
- strikes: Cboe takes the SPX value before 11:00 ET on the roll day and a listed strike (PPUT the
  first below 0.95 S, PPUT3M the first at or above 0.90 S, CLL put at or below 0.95 S, call at or
  above 1.10 S); the replica uses the close and continuous strikes (within one strike increment);
- fills and settlement: Cboe fills at an 11:30 ET VWAP and settles expiring options at the AM
  Special Opening Quotation; the replica trades and settles at the close (tracking splits roll
  days from the rest for this reason);
- the real levels: load_market forward-fills Cboe gaps, so tracking drops unchanged prints, and
  Cboe's CLL file steps -19.8% on 2009-10-16 with the S&P 500 down 0.8% (SPLICES, neutralised in
  validate).
Index levels before LAUNCH are Cboe back-tests, not live index values.
"""

import numpy as np
import pandas as pd

from insureport import metrics, pricing, sleeves
from insureport.sleeves import Leg, OptionSpec

PPUT_SPEC = OptionSpec(legs=(Leg("P", otm=0.05, tenor_months=1, hold_months=1),), mode="overlay", coverage=1.0)
PPUT3M_SPEC = OptionSpec(
    legs=(Leg("P", otm=0.10, tenor_months=3, hold_months=3, every_months=3),), mode="overlay", coverage=1.0
)
# ponytail: no CLL cross-roll (Cboe re-strikes the put at 0.95 S when 1.10 S falls below the standing put
# strike, a ~13.6% drop): on the 1990-2026 closes it fires only on 2002-07-19, 2008-10-17 and 2008-11-21,
# Cboe's 2008 levels show none, and modelled it raised CLL TE from 2008-08-26 (m 0.3) from 3.18% to 4.62%
CLL_SPEC = OptionSpec(
    legs=(
        Leg("P", otm=0.05, tenor_months=3, hold_months=3, every_months=3),
        Leg("C", otm=0.10, tenor_months=1, hold_months=1, qty=-1.0),
    ),
    mode="overlay",
    coverage=1.0,
)
SPECS = {"PPUT": PPUT_SPEC, "PPUT3M": PPUT3M_SPEC, "CLL": CLL_SPEC}
TOLERANCE = {
    "PPUT": {"te": 0.015, "dcagr": 0.005, "corr": 0.90},
    "PPUT3M": {"te": 0.020, "dcagr": 0.0075, "corr": 0.85},
    "CLL": {"te": 0.025, "dcagr": 0.010, "corr": 0.85},
}
WINDOWS = {"PPUT": ("2016-01-01", None), "PPUT3M": ("2016-01-01", None), "CLL": ("2008-08-26", None)}  # OOS
TRAIN_END = "2015-12-31"
LAUNCH = {"PPUT": "2015-08-03", "CLL": "2008-09-04", "PPUT3M": "2021-08-02"}  # earlier levels are Cboe back-tests
SPLICES = {"CLL": "2009-10-16"}  # Cboe file breaks: the level steps with no market move behind it


def _roll_days(index: pd.DatetimeIndex, which: str) -> pd.DatetimeIndex:
    """Days `which` trades options on, inside a trading-day index: every monthly roll
    (sleeves.roll_dates, third Fridays snapped back over holidays) for PPUT and CLL (monthly
    calls), only the Mar/Jun/Sep/Dec ones for PPUT3M."""
    rolls = sleeves.roll_dates(index)
    return rolls[rolls.month % 3 == 0] if which == "PPUT3M" else rolls


def replicate(
    mkt: pd.DataFrame,
    which: str,
    *,
    m: float = pricing.SKEW_MULT,
    nu: float = 1.0,
    h: float = 0.0,
    z_cap: float = pricing.Z_CAP,
) -> pd.Series:
    """Synthetic level of index `which` on mkt.index, scaled to equal the real index (column
    which.lower()) on the first date where both exist; NaN before the replica starts.

    The replica is sleeves.option_sleeve(SPECS[which]) with its default underlying (spxtr) and
    m, nu, h, z_cap passed on, on mkt with its term structure filled (pricing.fill_term_structure,
    unless mkt has ts_filled). PPUT starts on mkt's first row (unhedged until its first roll).
    PPUT3M and CLL must buy their quarterly put on the Mar/Jun/Sep/Dec cycle, and option_sleeve
    counts every_months from its first roll, so their panel starts on the first roll in one of
    those months whose row has usable pricing inputs (the rule of sleeves._inputs). Raises
    ValueError for an unknown target, a panel without the target's column, or no date where both
    levels exist.
    """
    if which not in SPECS:
        raise ValueError(f"unknown target {which!r}; expected one of {list(SPECS)}")
    col, kw = which.lower(), {"m": m, "nu": nu, "h": h, "z_cap": z_cap}
    if col not in mkt:
        raise ValueError(f"mkt has no {col!r} column to scale the {which} replica to")
    panel = mkt if "ts_filled" in mkt else pricing.fill_term_structure(mkt)
    if which != "PPUT":
        usable = pd.Series(sleeves._inputs(panel)["eff"] == np.arange(len(panel)), panel.index)
        quarterly = _roll_days(panel.index, "PPUT3M")
        quarterly = quarterly[usable[quarterly].to_numpy()]
        if quarterly.empty:
            raise ValueError(f"no Mar/Jun/Sep/Dec roll with pricing inputs to start the {which} replica on")
        panel = panel.loc[quarterly[0] :]
    r = sleeves.option_sleeve(panel, SPECS[which], **kw)["ret"]
    r.attrs = {}  # drop the leg ledger pandas would copy into every derived object
    nav = metrics.nav_from_returns(r).reindex(mkt.index)
    actual = mkt[col]
    both = nav.notna() & actual.notna()
    if not both.any():
        raise ValueError(f"the {which} replica and mkt[{col!r}] share no date")
    first = both.idxmax()
    return (nav * (actual[first] / nav[first])).rename(f"{col}_synth")


def tracking(
    synth: pd.Series, actual: pd.Series, bench: pd.Series, *, rolls: pd.DatetimeIndex | None = None
) -> dict:
    """How closely the level series `synth` tracks `actual`, on the dates where all three levels exist.

    An actual level equal to the one before it is a stale print (load_market forward-fills Cboe
    gaps up to 5 rows) and is dropped, so the move lands on the next real print for both series.
    Monthly: the last common date of each month gives the month-end levels; returns run between
    consecutive month-ends, so the first month (no month-end before it) is dropped and n_months
    counts returns, month-ends minus one (23 on a Jan 2018 .. Dec 2019 panel of 24 months; validate
    starts from the month before its window, so there it equals the window's full months):
    te = std(synth - actual returns, ddof 1) sqrt(12); dcagr =
    cagr(synth) - cagr(actual) and dmaxdd = max_drawdown(synth) - max_drawdown(actual)
    (metrics.py, on the month-end levels); corr = corr(synth - bench, actual - bench), the
    correlation of the two hedge P&Ls.
    Daily (needs daily levels and the sorted roll dates): te_roll / te_nonroll = std(synth - actual
    daily returns) sqrt(252) on returns whose span (previous common date, date] holds a roll / on
    the others, NaN without rolls. Settlement at the AM SOQ and the VWAP fill land on roll days,
    which the close-to-close replica cannot see.
    """
    actual = actual.where(actual.diff().ne(0))
    lv = pd.concat([synth, actual, bench], axis=1, keys=["s", "a", "b"]).dropna()
    # ponytail: a trailing partial month (data ending mid-month) counts as a month; noise at 100+ months
    month_end = lv.groupby(lv.index.to_period("M")).tail(1)
    ret = month_end.pct_change().iloc[1:]
    out = {
        "te": float((ret["s"] - ret["a"]).std() * np.sqrt(12)),
        "dcagr": float(metrics.cagr(month_end["s"]) - metrics.cagr(month_end["a"])),
        "corr": float((ret["s"] - ret["b"]).corr(ret["a"] - ret["b"])),
        "dmaxdd": float(metrics.max_drawdown(month_end["s"]) - metrics.max_drawdown(month_end["a"])),
        "n_months": len(ret),
        "te_roll": np.nan,
        "te_nonroll": np.nan,
    }
    if rolls is not None:
        daily = lv.pct_change().iloc[1:]
        gap = daily["s"] - daily["a"]
        n = rolls.searchsorted(lv.index, side="right")  # rolls on or before each date
        on = n[1:] > n[:-1]
        out["te_roll"] = float(gap[on].std() * np.sqrt(252))
        out["te_nonroll"] = float(gap[~on].std() * np.sqrt(252))
    return out


def validate(
    mkt: pd.DataFrame,
    *,
    m: float = pricing.SKEW_MULT,
    nu: float = 1.0,
    h: float = 0.0,
    z_cap: float = pricing.Z_CAP,
    windows: dict[str, tuple[str, str | None]] = WINDOWS,
    pput3m: pd.Series | None = None,
) -> pd.DataFrame:
    """The validation table: one row per target in `windows` (target -> (start, end), end None =
    open), indexed by target.

    pput3m (the real PPUT3M level, e.g. data.cboe("PPUT3M")) is reindexed on mkt.index (no
    forward fill: filled rows would be stale prints, dropped anyway); a target whose real series is
    not in the panel (PPUT3M without pput3m) gets no row. Real levels equal to the one before are stale prints
    and dropped first (as in tracking), so the window dates are real prints. A SPLICES date in the
    panel is neutralised: the real levels before it are rescaled so that its return (from the last
    real print before it) equals the replica's. Each replica runs on the whole panel (m, nu, h,
    z_cap passed on) and is cut to the window, starting from the last common date before `start`
    so the window's first full month counts; tracking() on the daily levels with _roll_days().
    Columns: window_start / window_end (first / last common date used), te, dcagr, corr, dmaxdd,
    n_months, te_roll, te_nonroll, then te_ok (te <= TOLERANCE te), dcagr_ok (|dcagr| <= TOLERANCE
    dcagr), corr_ok (corr >= TOLERANCE corr) and passed (all three). Raises ValueError when a
    window holds fewer than 3 common month-ends (no tracking error to measure).
    """
    if pput3m is not None:
        mkt = mkt.assign(pput3m=pput3m.reindex(mkt.index))
    rows = {}
    for which, (start, end) in windows.items():
        col = which.lower()
        if col not in mkt:
            continue
        synth = replicate(mkt, which, m=m, nu=nu, h=h, z_cap=z_cap)
        actual = mkt[col].where(mkt[col].diff().ne(0))  # stale prints out before the window is cut (see tracking)
        if which in SPLICES and (d := pd.Timestamp(SPLICES[which])) in actual.index[1:]:
            p = actual[:d].iloc[:-1].last_valid_index()  # the last real print before the splice
            actual = actual.where(actual.index >= d, actual * (actual[d] / actual[p]) / (synth[d] / synth[p]))
        lv = pd.concat([synth, actual, mkt["spxtr"]], axis=1).dropna().loc[:end]
        lv = lv.iloc[max(lv.index.searchsorted(pd.Timestamp(start)) - 1, 0) :]
        if lv.index.to_period("M").nunique() < 3:
            raise ValueError(f"{which}: window {start}..{end} has fewer than 3 common month-ends")
        t = tracking(lv.iloc[:, 0], lv.iloc[:, 1], lv.iloc[:, 2], rolls=_roll_days(mkt.index, which))
        tol = TOLERANCE[which]
        ok = {"te_ok": t["te"] <= tol["te"], "dcagr_ok": abs(t["dcagr"]) <= tol["dcagr"],
              "corr_ok": t["corr"] >= tol["corr"]}
        rows[which] = {"window_start": lv.index[0], "window_end": lv.index[-1], **t, **ok, "passed": all(ok.values())}
    return pd.DataFrame.from_dict(rows, orient="index")


def calibrate_skew(
    mkt: pd.DataFrame,
    m_grid: np.ndarray = np.round(np.arange(0.2, 1.01, 0.05), 2),
    train_end: str = TRAIN_END,
    h: float = 0.0,
) -> pd.DataFrame:
    """PPUT replication on mkt up to train_end for each m of m_grid, indexed by m.

    The panel is cut at train_end before anything runs (the term-structure back-fill included),
    so nothing later leaks in. Columns: te, dcagr, corr (tracking over the whole training panel),
    cap_share (share of training days with SKEW and VIX whose m gamma / 6 is above the pricer's
    slope cap for a 30-day option at that day's VIX, pricing._slope_cap at nu 1 and Z_CAP, where m
    stops moving prices: the m objective is identified by the other days; the cap tightens as T
    falls, so later in each PPUT month it binds more often) and best: the lowest te among the m
    with |dcagr| <= TOLERANCE["PPUT"]["dcagr"] (the lowest te overall when none qualifies), ties to
    the smaller |dcagr|.
    Why not te alone: m sets the premium level, which te (a demeaned std) barely sees. On PPUT
    1990-2015 te falls steadily but only 0.19 pp across the grid (2.08% at m 0.2 to 1.89% at 1.0)
    while dcagr falls 3 pp (0.0% to -3.0%/yr), so te alone picks the grid edge m 1 with a price
    level six times outside the CAGR tolerance. With te monotone in m this rule picks the largest
    m inside the tolerance: the tolerance, not the fit, sets m (m 0.3, using 0.39 of the 0.50 pp
    budget). argmin |dcagr| would give m 0.2.
    """
    train = mkt.loc[:train_end]
    days = train[["skew", "vix"]].dropna()
    c_max = pricing._slope_cap(30 / 365, days["vix"] / 100, 1.0, pricing.Z_CAP)
    rows = {}
    for m in m_grid:
        t = tracking(replicate(train, "PPUT", m=m, h=h), train["pput"], train["spxtr"])
        cap = float((pricing.skew_slope(days["skew"], m) > c_max).mean())
        rows[float(m)] = {"te": t["te"], "dcagr": t["dcagr"], "corr": t["corr"], "cap_share": cap}
    out = pd.DataFrame.from_dict(rows, orient="index").rename_axis("m")
    ok = out["dcagr"].abs() <= TOLERANCE["PPUT"]["dcagr"]
    cand = out[ok] if ok.any() else out
    out["best"] = out.index == cand.assign(a=cand["dcagr"].abs()).sort_values(["te", "a"]).index[0]
    return out

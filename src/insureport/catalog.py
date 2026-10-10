"""The safe-haven catalog: candidate presets, the shortlist, evaluation and ranking.

CANDIDATES holds the presets (Candidate; edit any parameter with dataclasses.replace), each citing
its research source in `note`. build_sleeve turns a candidate into a daily sleeve return series
(per unit of sleeve capital, the engine's convention) and its rebalance calendar; suggest shortlists
candidates from metrics.diagnose output; evaluate runs every test on one candidate (the frontier and
its empirical-Kelly w*, the drawdown-capped w_cap, the AQR deleverage twins, the block and book
bootstraps, rolling windows, crash payoffs, the calm-year bleed, the critic's stand-alone view, the
robustness re-runs) and assigns the verdict; rank orders the rows.

Sizing (addendum section 8): every candidate is judged at the same rule, its CAGR-maximizing weight
w* (best_allocation with no cap: Kelly, the single-haven rule), so a haven that cuts drawdowns enough
to meet the cap is not forced to a larger weight than one that cannot. The cap only sets w_cap, where
the AQR twins are compared. DEFERRED: the addendum's catalog comparison mode, equal-tail sizing (each
haven at the smallest w reaching a common 5th-percentile 25-year CAGR, "cannot reach target" otherwise).

Denominator discipline (addendum D13): sleeve P&L is reported in portfolio points (the hedged minus
the unhedged portfolio), never as a return on premium. Universa / Spitznagel figures carry
UNIVERSA_LABEL (D15). The TAIL proxy and the Universa / TAIL known-answer checks live here rather than
in replicate.py: they are catalog presets read through build_sleeve and evaluate, while replicate.py
validates the pricer against Cboe's indices. Benchmarks on every evaluate row (D7): the S&P 500 TR and
60/40 over the row's sample, and the divested T-bill twin.

Pure numpy/pandas over the other modules; the only I/O is scenarios' packaged annual table.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from insureport import backtest, metrics, portfolio, pricing, replicate, scenarios, sleeves
from insureport.backtest import Policy
from insureport.portfolio import cash_returns
from insureport.sleeves import Leg, OptionSpec

ARCHETYPES = ("insurance", "alpha", "store_of_value", "baseline")
KINDS = ("option", "index", "etf", "series", "composite")
REBALANCE = ("events", "monthly", "quarterly", "annual")
H_DEFAULT = 0.04  # half-spread / mid: an 8%-of-mid spread on deep-OTM 2-3 month puts (addendum section 12)
# ponytail: the near-money overlays (prot_put, collar) share H_DEFAULT; pass pricing_kw h if their spread matters
W_UNIVERSA = 0.0333  # the 96.67 / 3.33 model portfolio
UNIVERSA_LABEL = "self-reported, hypothetical, window-dependent"
CASH_GRID = np.linspace(0, 1, 41)  # the T-bill frontier behind the AQR twins
RELEVER_MAX_W = 1 / 3  # relever runs L = 1 / (1 - w) <= 1.5, the design's "hold more equity" range
OFFSETS = (5, 10, 15)  # roll-offset sweep, business days
SHOCK, SHOCK_VIX = -0.20, (40.0, 55.0, 70.0)  # shock_pnl: the S&P 500 month, every VIX pillar at its end
CALM_ROWS = 252  # shock_pnl's pre-shock state: the medians of mkt's last year, the same recent market as its spot
BLEED_WARN = 0.026  # calm-year bleed warning, %/yr of NAV (Brown's gross-decay estimate; addendum section 3)
TAIL_BOND, TAIL_BUDGET, TAIL_FEE = 0.90, 0.01, 0.006  # TAIL proxy: bond share of the non-put capital, puts / NAV a month, fee
# Cambria TAIL calendar-year NAV returns (data-and-implementation Q7; addendum section 4)
TAIL_CALENDAR = {2018: 0.0233, 2019: -0.1399, 2020: 0.0698, 2021: -0.1281, 2022: -0.1315, 2023: -0.1298,
                 2024: -0.0998, 2025: 0.0574}
# Universa's 96.67 / 3.33 portfolio vs the S&P 500 [universa-practice Q2; 3, 4]: (check, window start (the
# close before it), end, Universa portfolio, S&P 500, drag: the row compares the sleeve's contribution in
# portfolio points, portfolio - 0.9667 S&P 500)
UNIVERSA_CHECKS = (
    ("March 2020", "2020-02-28", "2020-03-31", 0.004, -0.124, False),
    ("Q1 2020", "2019-12-31", "2020-03-31", -0.063, -0.196, False),
    ("2019 sleeve drag (portfolio points)", "2018-12-31", "2019-12-31", -0.0169, 0.315, True),
    ("Mar-Dec 2008", "2008-02-29", "2008-12-31", 0.099, -0.307, False),
)
LAUNCH = {k.lower(): v for k, v in replicate.LAUNCH.items()} | {"vxth": "2011-07-29"}  # by index source
TIERS = {"cost-effective (robust)": 0, "cost-effective (fragile)": 1, "not cost-effective": 2,
         "insufficient history": 3}


@dataclass(frozen=True)
class Candidate:
    """One safe haven to test.

    archetype: the book's "insurance" | "alpha" | "store_of_value", or "baseline" (a reference row).
    kind: "option" (a synthetic SPX option sleeve from `spec`), "index" (a Cboe index level), "etf"
    (adjusted closes), "series" (T-bills, the 10-year proxy, trend) or "composite" (built from
    several). source: the panel column / px column / series the sleeve reads. w_grid: the
    allocations tried (fractions of the total portfolio; evaluate adds 0). rebalance: the portfolio
    calendar, "events" (the option sleeve's own roll dates) | "monthly" | "quarterly" | "annual".
    model_priced: option prices come from pricing.py. note: provenance and caveats for the UI.
    """

    key: str
    name: str
    archetype: str
    kind: str
    spec: OptionSpec | None = None
    source: str | None = None
    w_grid: tuple[float, ...] = ()
    rebalance: str = "events"
    model_priced: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        """Raise ValueError on an archetype, kind, rebalance or w_grid the catalog cannot run (the app edits presets)."""
        if self.archetype not in ARCHETYPES or self.kind not in KINDS or self.rebalance not in REBALANCE:
            raise ValueError(f"{self.key}: archetype {self.archetype!r}, kind {self.kind!r} or rebalance "
                             f"{self.rebalance!r} not in {ARCHETYPES}, {KINDS}, {REBALANCE}")
        if not all(0 <= w <= 1 for w in self.w_grid):
            raise ValueError(f"{self.key}: w_grid values must be in [0, 1], got {self.w_grid}")


_FINE = tuple(sorted({*np.round(np.arange(1, 25) * 0.0025, 4).tolist(), W_UNIVERSA}))  # 0.25% .. 6%, and 3.33%
_UNIT = tuple(np.round(np.arange(1, 21) * 0.05, 2).tolist())  # 5% .. 100%
# month-start tranches of the third monthly expiry (the third Friday two months on, ~11 weeks: Rule B's
# "11-12 weeks"), each sold at the second roll, 14-20 days before expiry (the first roll with <= 35 days left)
_RULE_B = {"tenor_months": 2, "hold_months": 2}
# 6% of the sleeve's year-end NAV a month = 0.20% of portfolio NAV at w = 3.33%. The year-end rebalance harvests
# the live tranches into T-bills (harvest_at_anchor), so the year's budget is cash: a base of Q4-marked puts that
# decayed before they were sold ran the fixture's sleeve dry by June 2019 and wiped it out (-100%) in July.
# ponytail: the harvest sells the December tranche a month early, so January holds one tranche, not two
UNIVERSA_SPEC = OptionSpec((Leg("P", otm=0.30, **_RULE_B),), premium_frac=0.06, budget_base="anchor",
                           roll_when="month_start", harvest_at_anchor=True)
# TAIL proxy: one premium sleeve whose non-put capital earns TAIL_BOND of the DGS10 bond proxy plus T-bills (the carry
# build_sleeve passes), spending TAIL_BUDGET of its NAV at every roll (anchors None), the ladder never rescaled
TAIL_SPEC = OptionSpec(
    # ponytail: the monthly expiry 14 months out stands in for TAIL's quarterly one 13-15 months out (same mean
    # tenor, monthly rather than quarterly rungs); rolls on third Fridays, so 4 months (~120 days) and 1 month left
    (Leg("P", otm=0.08, tenor_months=14, hold_months=10, qty=0.75),
     Leg("P", otm=0.08, tenor_months=14, hold_months=13, qty=0.25)),
    premium_frac=TAIL_BUDGET, budget_base="anchor",
)

CANDIDATES: dict[str, Candidate] = {c.key: c for c in (
    Candidate(
        "universa_model", "Universa model 96.67/3.33", "insurance", "option", UNIVERSA_SPEC, w_grid=_FINE,
        rebalance="annual", model_priced=True,
        note="Addendum section 3 / D2 (universa-practice Q2-Q3; Rule B, dao-of-capital-taleb Q3.4): 96.67% S&P 500 TR "
             "+ 3.33% T-bill-backed sleeve, rebalanced at year-end; on each month's first trading day a tranche of 30% "
             "OTM SPX puts on the third monthly expiry (~11 weeks), sold at the roll with <= 35 days left (about 2 "
             "tranches live); 0.20% of NAV a month gross on the last year-end NAV (6% of the sleeve); the year-end "
             "rebalance sells the live tranches into T-bills, so the sleeve restarts T-bill-backed; no profit-taking "
             f"trigger (opt-in monetize_mult); fills at mid -/+ half an 8%-of-mid spread. Universa figures: "
             f"{UNIVERSA_LABEL}.",
    ),
    Candidate(
        "universa_2023", "Universa 2023 98/2 (display)", "insurance", "option", UNIVERSA_SPEC, w_grid=(0.02,),
        rebalance="quarterly", model_priced=True,
        note="Display preset (universa-practice design 1; gap-7-q2 F5): 98/2 rebalanced quarterly with the "
             "universa_model sleeve; quarterly harvesting is not comparable with the annual 3.33% series. Universa "
             f"figures: {UNIVERSA_LABEL}.",
    ),
    Candidate(
        "dao_ch9", "Dao of Capital ch. 9 99.5/0.5", "insurance", "option",
        # ponytail: one brentq per roll (~4 s a 1990-> build, x6 with evaluate's rebuilds); vectorise
        # strike_for_delta over the scheduled opens (sleeves' ponytail) if the app needs it faster. The book's K/S
        # 0.70 fallback is not wired: strike_for_delta brackets |delta| 0.005 at any vol and fails only on NaN
        # inputs, and rolls skip rows without usable inputs
        OptionSpec((Leg("P", delta=-0.005, tenor_months=2, hold_months=1),)),
        w_grid=tuple(np.round(np.arange(1, 9) * 0.0025, 4).tolist()), model_priced=True,
        note="The Dao of Capital ch. 9 prototype (addendum section 3 alternatives; dao-of-capital-taleb design 5): "
             "99.5/0.5, 0.5% of NAV a month gross on 2-month SPX puts at |delta| 0.005, each sold a month later, the "
             "split reset at every monthly roll, no leverage. The delta strike rests on the model's far wing, which "
             "PPUT does not validate (K/S about 0.75 at the median 1990-2026 roll, 0.63-0.83 for 80% of them).",
    ),
    Candidate(
        "put_spread", "Put spread 10/25", "insurance", "option",
        replace(UNIVERSA_SPEC, legs=(Leg("P", otm=0.10, **_RULE_B), Leg("P", otm=0.25, qty=-1.0, **_RULE_B))),
        w_grid=_FINE, rebalance="annual", model_priced=True,
        note="Long 10% / short 25% OTM SPX put spread on universa_model's calendar and budget: monthly tranches of "
             "the third monthly expiry sold at <= 35 days left, 6% of the sleeve a month on the year-end NAV, "
             "harvested into T-bills at year-end (critics-and-literature design 8-9).",
    ),
    Candidate(
        "prot_put", "Protective put 5% OTM", "insurance", "option", OptionSpec((Leg("P", otm=0.05),), mode="overlay"),
        w_grid=_UNIT, model_priced=True,
        note="Protective put, PPUT-like (critics-and-literature T1, T12): 5% OTM 1-month SPX puts held to expiry on "
             "downside beta x the core's value, financed by selling core; third-Friday rolls.",
    ),
    Candidate(
        "collar", "Collar 95-110", "insurance", "option",
        OptionSpec((Leg("P", otm=0.05, tenor_months=3, hold_months=3, every_months=3), Leg("C", otm=0.10, qty=-1.0)),
                   mode="overlay"),
        w_grid=_UNIT, model_priced=True,
        note="95-110 collar on Cboe CLL's structure (addendum section 5): a 3-month 95% put bought every third month, "
             "a 1-month 110% call sold monthly, on downside beta x the core's value.",
    ),
    Candidate(
        "pput_idx", "Cboe PPUT index", "insurance", "index", source="pput", w_grid=_UNIT, rebalance="annual",
        note="Cboe S&P 500 5% Put Protection Index levels (real quotes; data-and-implementation Q6.1); back-tested "
             "before its 2015-08-03 launch.",
    ),
    Candidate(
        "pput3m_idx", "Cboe PPUT3M index", "insurance", "index", source="pput3m", w_grid=_UNIT, rebalance="annual",
        note="Cboe 3-month 10% OTM Put Protection Index (addendum section 5; pass data.cboe('PPUT3M') as "
             "px['pput3m']); back-tested before its 2021-08-02 launch.",
    ),
    Candidate(
        "cll_idx", "Cboe CLL index", "insurance", "index", source="cll", w_grid=_UNIT, rebalance="annual",
        note="Cboe 95-110 Collar Index from 2009-10-16 only (the file's -19.8% splice; addendum section 13); "
             "2020-02-28 / 2020-03-02 likely bad put marks, kept and flagged; back-tested before its 2008-09-04 "
             "launch.",
    ),
    Candidate(
        "vxth_idx", "Cboe VXTH index", "insurance", "index", source="vxth", w_grid=_UNIT, rebalance="annual",
        note="Cboe VIX Tail Hedge Index (S&P 500 plus VIX calls; a catalog candidate, not a pricer target, its w(F) "
             "schedule unverified: data-and-implementation Q6.5); back-tested before its 2011-07-29 launch.",
    ),
    Candidate(
        "tail", "Cambria TAIL ETF", "insurance", "etf", source="TAIL", w_grid=_UNIT, rebalance="annual",
        note="Cambria Tail Risk ETF (2017 on): too short a history (insufficient history); duration-contaminated, "
             "about 90% 10-year Treasuries plus puts (addendum section 4).",
    ),
    Candidate(
        "tail_proxy", "TAIL proxy (filings)", "insurance", "composite", TAIL_SPEC, source="DGS10", w_grid=_UNIT,
        rebalance="annual", model_priced=True,
        note="Addendum section 4 / D9 (gap-8-q7 F14-F15): each month 1% of NAV on ~8% OTM SPX puts 14 months out "
             "(TAIL buys the quarterly expiry 13-15 months out), 75% sold at the first roll <= 135 days left, the rest "
             "<= 45; the non-put capital 90% constant-maturity 10-year Treasury (par-bond proxy on FRED DGS10), 10% "
             "T-bills, so the bond is ~85-90% of NAV and drifts with the put marks; 0.60%/yr fee accrued daily. "
             "DEVIATION from the brief (backtest.simulate of the bonds and a put sleeve reset on its rolls, which "
             "rescaled the whole ladder every month): one put sleeve whose cash earns the bond mix. Priced off the "
             "vix1y pillar; duration-contaminated.",
    ),
    Candidate(
        "tbills", "T-bills (hold less equity)", "store_of_value", "series", source="rf", w_grid=_UNIT,
        rebalance="annual",
        note="3-month T-bills (FRED DTB3): AQR's 'hold less equity' baseline (critics-and-literature design 5); "
             "store of value, mechanical.",
    ),
    Candidate(
        "ust10", "10-year Treasury proxy", "store_of_value", "series", source="DGS10", w_grid=_UNIT,
        rebalance="annual",
        note="Constant-maturity 10-year par-bond proxy on FRED DGS10 (1962 on); store of value, statistical: it fell "
             "with stocks in 2022 (addendum section 14).",
    ),
    Candidate(
        "tlt", "TLT (20+ year Treasuries)", "store_of_value", "etf", source="TLT", w_grid=_UNIT, rebalance="annual",
        note="iShares 20+ Year Treasury Bond ETF (2002 on); store of value, statistical (addendum section 14).",
    ),
    Candidate(
        "ief", "IEF (7-10 year Treasuries)", "store_of_value", "etf", source="IEF", w_grid=_UNIT, rebalance="annual",
        note="iShares 7-10 Year Treasury Bond ETF (2002 on); store of value, statistical (addendum section 14).",
    ),
    Candidate(
        "gold", "Gold (GLD)", "insurance", "etf", source="GLD", w_grid=_UNIT, rebalance="annual",
        note="SPDR Gold Shares (2004 on): insurance (candidate), statistical; the book's index files gold under "
             "insurance (addendum D12). It worked mainly in the 1970s.",
    ),
    Candidate(
        "trend", "S&P 500 trend (12-month TSMOM)", "alpha", "series", source="spxtr", w_grid=_UNIT,
        rebalance="annual",
        note="S&P 500 12-month time-series momentum, long / flat, monthly signal (addendum section 14): alpha, "
             "statistical.",
    ),
    Candidate(
        "dbmf", "DBMF (managed futures)", "alpha", "etf", source="DBMF", w_grid=_UNIT, rebalance="annual",
        note="iMGP DBi Managed Futures Strategy ETF (2019 on), CTA replication: alpha, statistical (addendum section "
             "14); too short a history (insufficient history).",
    ),
    Candidate(
        "sixty_forty", "60/40 reference", "baseline", "composite", source="AGG", w_grid=(0.4,), rebalance="annual",
        note="Reference row (addendum D7): the core 60% / US Aggregate bonds 40% (iShares AGG from 2003, the DGS10 "
             "par-bond proxy before), rebalanced annually; with an S&P 500 TR core it is the classic 60/40.",
    ),
)}
OPTION_PICKS = ("universa_model", "dao_ch9", "put_spread", "pput_idx", "collar")  # suggest: downside beta >= 0.7


def rebalance_calendar(
    index: pd.DatetimeIndex, how: str, events: pd.DatetimeIndex | None = None
) -> pd.DatetimeIndex:
    """Portfolio rebalance dates: the last index date of each year ("annual"), quarter ("quarterly")
    or month ("monthly"), or `events` unchanged ("events"; empty when None: buy and hold). A trailing
    partial period (the index ends before its period's last weekday) has no rebalance date, so a
    panel ending mid-year neither rebalances nor harvests option legs (harvest_at_anchor) on its last
    row. Raises ValueError for any other `how`."""
    # ponytail: weekdays, not the exchange calendar: an index ending on a holiday-shortened period's last
    # trading day before its last weekday loses that one final rebalance
    if how == "events":
        return pd.DatetimeIndex([]) if events is None else events
    freq = {"monthly": "M", "quarterly": "Q", "annual": "Y"}.get(how)
    if freq is None:
        raise ValueError(f"rebalance must be one of {REBALANCE}, got {how!r}")
    idx = pd.DatetimeIndex(index)
    ends = idx[~idx.to_period(freq).duplicated(keep="last")]
    if len(idx) and np.datetime64(idx[-1].date()) < np.busday_offset(idx[-1].to_period(freq).end_time.date(), 0,
                                                                       roll="backward"):
        ends = ends[:-1]
    return ends


def _series(src: str, mkt: pd.DataFrame, px: pd.DataFrame | None) -> pd.Series:
    """Column `src` of the market panel, else of px; ValueError naming it when neither has it."""
    if src in mkt:
        return mkt[src]
    if px is not None and src in px:
        return px[src]
    raise ValueError(f"no data for source {src!r}: pass it as a column of px")


def _anchors(c: Candidate, index: pd.DatetimeIndex) -> pd.DatetimeIndex | None:
    """Budget anchors for c's spec on index: its rebalance calendar; None (every roll is an anchor) for
    "events" candidates and the TAIL proxy (TAIL_BUDGET of its current NAV every month)."""
    return None if c.rebalance == "events" or c.kind == "composite" else rebalance_calendar(index, c.rebalance)


def _bonds(source: str, mkt: pd.DataFrame, px: pd.DataFrame | None, start: pd.Timestamp) -> pd.Series:
    """Daily returns of the bond ETF `source` (adjusted closes) from `start`, the DGS10 par-bond proxy's
    before the ETF's first price, read only when start is before it (FRED dates the ETF lacks later are left
    out: the next ETF return already holds their move). ValueError when the ETF has no price."""
    etf = _series(source, mkt, px).dropna()
    if etf.empty:  # an all-NaN column (a failed fetch framed with the others)
        raise ValueError(f"no data for source {source!r}")
    r = etf.pct_change()
    if start >= etf.index[0]:
        return r.fillna(0.0).loc[start:]  # the first return is a placeholder the engine never applies
    return r.combine_first(sleeves.bond_proxy(_series("DGS10", mkt, px)).loc[: etf.index[0]]).loc[start:]


def build_sleeve(
    c: Candidate, mkt: pd.DataFrame, px: pd.DataFrame | None, core_r: pd.Series, beta_down: float, *,
    pricing_kw: dict | None = None, chain: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """The candidate's daily sleeve returns (per unit of sleeve capital) and its rebalance calendar.

    Everything is cut to the core's span (core_r's first to last date). mkt is the market panel
    (data.load_market); px holds the other sources by column: ETF adjusted closes (TAIL, TLT, IEF, GLD,
    DBMF, AGG), Cboe levels missing from the panel (pput3m), FRED DGS10 yields in percent.
    - option, and the composite with a spec (tail_proxy): sleeves.option_sleeve on the panel (term
      structure filled on the whole panel first) from the core's first date, h = H_DEFAULT unless
      pricing_kw (m, nu, h, z_cap, offset_bdays, passed on) says otherwise, chain passed on, anchors
      _anchors (the rebalance calendar; None for "events" and the composite); overlay specs hold the core
      (underlying core_r) and options on coverage = beta_down (ValueError unless > 0). The composite's
      cash earns TAIL_BOND x the par-bond proxy on its source's yields (DGS10, carried onto the panel's
      dates) + the rest T-bills (option_sleeve's carry), less TAIL_FEE accrued on calendar days / 365. The
      returned series carries attrs premium_paid and coverage (the sleeve's columns, coverage marked daily;
      strip them before arithmetic).
      Calendar: rebalance_calendar on the panel, with the sleeve's event dates for "events".
    - index: the level's pct change (first row 0), from replicate.SPLICES' date for CLL.
    - etf: adjusted-close pct change (first row 0) on the ETF's own dates.
    - series: source "rf" T-bills, "spxtr" 12-month TSMOM (both on the whole panel, then cut), else a
      yield in percent through sleeves.bond_proxy.
    - composite without a spec (sixty_forty): px[source] returns, the DGS10 proxy's before the ETF's start
      (DGS10 needed only when the core starts before it).
    Calendar for every other kind: rebalance_calendar on the panel (empty for "events").
    """
    first, last = core_r.index[0], core_r.index[-1]
    kw = {"h": H_DEFAULT} | (pricing_kw or {})
    if c.spec is not None and "ts_filled" not in mkt:
        mkt = pricing.fill_term_structure(mkt)
    span = mkt.loc[first:last]
    if c.spec is not None:
        overlay = c.spec.mode == "overlay"
        spec = replace(c.spec, coverage=beta_down) if overlay else c.spec
        carry = None
        if c.kind == "composite":
            y = _series(c.source, mkt, px).dropna().reindex(span.index, method="ffill")
            carry = TAIL_BOND * sleeves.bond_proxy(y) + (1 - TAIL_BOND) * sleeves.tbill_sleeve(span)
        df = sleeves.option_sleeve(span, spec, underlying=core_r if overlay else None, chain=chain,
                                   anchors=_anchors(c, span.index), carry=carry, **kw)
        r = df["ret"].copy()
        r.attrs = {}
        if carry is not None:
            r -= TAIL_FEE * span.index.to_series().diff().dt.days.fillna(0) / 365
        r.attrs = {"premium_paid": df["premium_paid"], "coverage": df["coverage"]}
        return r, rebalance_calendar(span.index, c.rebalance, span.index[df["event"].to_numpy()])
    if c.kind in ("index", "etf"):  # a level: its first return (NaN) is the 0 placeholder
        lvl = _series(c.source, mkt, px).dropna()
        r = lvl.loc[replicate.SPLICES.get(c.source.upper()) :].pct_change().fillna(0.0)
    elif c.source == "rf":
        r = sleeves.tbill_sleeve(mkt)
    elif c.source == "spxtr":
        r = sleeves.tsmom_sleeve(mkt["spxtr"], mkt["rf"])
    elif c.kind == "series":
        r = sleeves.bond_proxy(_series(c.source, mkt, px))
    else:
        r = _bonds(c.source, mkt, px, first)
    return r.loc[first:last], rebalance_calendar(span.index, c.rebalance)


def suggest(diag: dict) -> list[tuple[str, str]]:
    """Shortlist (candidate key, reason) from metrics.diagnose output (downside_beta, eff_n, top1), in
    priority order, each key once with its reasons joined by "; ": tbills (the baseline); downside beta
    >= 0.7 the index-put presets OPTION_PICKS; downside beta < 0.4 trend and ust10, the alternatives
    without index beta; then always ust10 (2022 shown), gold, trend, tail and dbmf (short history) and
    tail_proxy (TAIL's structure with a longer history). Every index-put key suggested (OPTION_PICKS,
    tail, tail_proxy) gets the basis-risk warning when downside beta < 0.4 and "index puts hedge beta,
    not single-name risk" when eff_n < 5 or top1 > 0.25."""
    b, eff_n, top1 = (diag.get(k, np.nan) for k in ("downside_beta", "eff_n", "top1"))
    out = [("tbills", "baseline: hold less equity (AQR's test); a haven must beat this divested mix")]
    if b >= 0.7:
        out += [(k, f"downside beta {b:.2f} >= 0.7: index puts track this portfolio's crashes") for k in OPTION_PICKS]
    risk = f"downside beta {b:.2f} < 0.4: index puts carry basis risk here"
    if b < 0.4:
        out += [("trend", f"{risk}; trend needs no index beta (alpha)"),
                ("ust10", f"{risk}; Treasuries hedge rate-led crashes (store of value)")]
    out += [
        ("ust10", "store of value, statistical: shown with 2022, when bonds fell with stocks"),
        ("gold", "insurance (candidate), statistical: the book files gold under insurance"),
        ("trend", "alpha archetype: S&P 500 time-series momentum"),
        ("tail", "short history (2017 on): reported as insufficient history; duration-contaminated"),
        ("dbmf", "short history (2019 on): reported as insufficient history"),
        ("tail_proxy", "TAIL's structure rebuilt from its filings, testable over a longer history"),
    ]
    puts = [k for k in (*OPTION_PICKS, "tail", "tail_proxy") if k in {k for k, _ in out}]
    if b < 0.4:
        out += [(k, risk) for k in puts]
    if eff_n < 5 or top1 > 0.25:
        conc = f"index puts hedge beta, not single-name risk (effective N {eff_n:.1f}, top holding {top1:.0%})"
        out += [(k, conc) for k in puts]
    reasons: dict[str, list[str]] = {}
    for k, why in out:
        reasons.setdefault(k, []).append(why)
    return [(k, "; ".join(v)) for k, v in reasons.items()]


def _frontier(
    core_r: pd.Series, sleeve_r: pd.Series, grid: np.ndarray, rebalance: pd.DatetimeIndex, policy: Policy,
    funding: str, rf: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """backtest.frontier over the grid (crash windows metrics.CRASHES, read at call time); "relever": one
    run per w on the core levered to L = 1 / (1 - w) (portfolio.lever, rf + 50 bp), so (1 - w) L = 1 keeps
    equity at 100% and the sleeve is financed on top."""
    if funding == "carve_out":
        return backtest.frontier(core_r, sleeve_r, grid, rebalance, policy, metrics.CRASHES)
    runs = [backtest.frontier(portfolio.lever(core_r, rf, 1 / (1 - w)), sleeve_r, np.array([w]), rebalance, policy,
                              metrics.CRASHES) for w in grid]
    return pd.concat([t for t, _ in runs]), pd.concat([n for _, n in runs], axis=1)


def _sleeve_share(nav_w: pd.Series, sleeve_r: pd.Series, rebalance: pd.DatetimeIndex, w: float) -> pd.Series:
    """The sleeve's share of portfolio NAV at weight w on nav_w's dates: w on the first date and on
    rebalance dates (after the reset; dates off nav_w's calendar move to the next one, as in
    backtest.simulate), w nav_w[r] G / nav_w in between, G the sleeve's growth since r, the last of them."""
    # ponytail: ignores band rebalances and costs (Policy.band None, tc ~ 2 bp); exact for calendar rebalancing at tc 0
    d = nav_w.index
    g = 1 + backtest.period_returns(sleeve_r, d).iloc[:, 0].reindex(d).fillna(0.0)
    due = np.zeros(len(d), dtype=bool)
    pos = d.searchsorted(rebalance)
    due[pos[pos < len(d)]] = True
    due[0] = True
    g[due] = 1.0
    return w * nav_w.where(due).ffill() * g.groupby(due.cumsum()).cumprod() / nav_w


def calm_year_bleed(nav_w: pd.Series, nav_0: pd.Series, spx_annual: pd.Series) -> float:
    """Mean of the calendar-year return difference ret_w - ret_0 over the full calendar years
    (scenarios.full_year_returns) whose S&P 500 return (spx_annual, by year) is above -15% (outside the
    crash bin): negative = the hedge's cost when no crash comes, in portfolio points. NaN without one."""
    a = scenarios.full_year_returns(pd.concat([nav_w, nav_0], axis=1, keys=["w", "u"]).dropna())
    calm = spx_annual.reindex(a.index) > scenarios.REGIME_EDGES[1]
    return float((a["w"] - a["u"])[calm.to_numpy()].mean())


def critic_view(sleeve_r: pd.Series, spxtr_r: pd.Series, rf: pd.Series) -> dict:
    """The sleeve stand-alone, the critics' (AQR's) view: cagr and max_dd of its NAV, ann_vol, sharpe
    (mean / std of the daily excess return over cash_returns(rf) x sqrt 252) and the daily OLS of its
    excess return on the S&P 500's (spxtr_r) on common dates: beta and alpha (intercept x 252). The first
    row is the placeholder every sleeve starts with and is skipped."""
    s = sleeve_r.copy()
    s.attrs = {}
    nav = metrics.nav_from_returns(s)
    cash = cash_returns(rf.reindex(s.index).ffill())
    ex = (s - cash).iloc[1:]
    both = pd.concat([ex, spxtr_r.reindex(ex.index) - cash.iloc[1:]], axis=1).dropna()
    y, x = both.iloc[:, 0], both.iloc[:, 1]
    beta = metrics.beta(y, x)
    with np.errstate(invalid="ignore", divide="ignore"):  # a T-bill sleeve has no excess return: NaN Sharpe
        sharpe = float(ex.mean() / ex.std() * np.sqrt(252))
    return {
        "cagr": float(metrics.cagr(nav)),
        "ann_vol": float(metrics.ann_vol(s.iloc[1:])),
        "max_dd": float(metrics.max_drawdown(nav)),
        "sharpe": sharpe,
        "alpha": float((y.mean() - beta * x.mean()) * 252),
        "beta": beta,
    }


def shock_pnl(
    c: Candidate, mkt: pd.DataFrame, w: float, *, equity: float | None = None, pricing_kw: dict | None = None,
) -> dict[float, float]:
    """Portfolio P&L over a one-month S&P 500 fall of SHOCK (-20%) for a put preset (any candidate with a
    spec) at weight w, by the VIX level reached (SHOCK_VIX): equity x SHOCK + w x the sleeve's return from
    one roll to the next, equity the S&P 500 weight (1 - w by default, the carve-out; 1 under relever
    funding). The addendum (section 3) asks for this next to every put preset.

    The sleeve is the candidate's spec on a synthetic panel of 300 business days from 2001-03-01: S at
    mkt's last S&P 500 close, every VIX pillar, SKEW, rf and q at their medians over mkt's last CALM_ROWS
    rows (today's calm market, as S is: a whole-history median mixes 1990s SKEW, near 120, with today's
    spot, and the far wing's price, hence the notional the budget buys, moves several-fold with SKEW);
    between its last two rolls (March / April 2002 for month-start rolls: a steady-state ladder by then,
    two months clear of the year-end anchor's harvest; anchors as build_sleeve sets them) S falls
    geometrically to (1 + SHOCK) S and every pillar moves linearly to the SHOCK_VIX value. An overlay
    holds that index (its spec's coverage); the TAIL proxy's cash earns T-bills. Pricing as build_sleeve
    (h = H_DEFAULT unless pricing_kw). Raises ValueError when c has no spec.
    """
    # ponytail: a flat post-shock term structure (real crashes invert it), SKEW held at its calm median, the TAIL
    # proxy's Treasury held flat through the shock and its fee left out
    if c.spec is None:
        raise ValueError(f"{c.key}: shock_pnl needs an option spec")
    idx = pd.bdate_range("2001-03-01", periods=300)
    i0, i1 = idx.get_indexer(sleeves.roll_dates(idx, c.spec.roll_every_months, when=c.spec.roll_when)[-2:])
    vols = sleeves._VOLS  # ponytail: private name; renaming it touches test_replicate, outside this slice
    calm = mkt.reindex(columns=[*vols, "skew", "rf", "q"]).iloc[-CALM_ROWS:].median()
    k = np.clip((np.arange(len(idx)) - i0) / (i1 - i0), 0.0, 1.0)  # 0 up to the shock's start, 1 from its end
    S = float(mkt["spx"].dropna().iloc[-1]) * (1 + SHOCK) ** k
    eq = 1 - w if equity is None else equity
    out = {}
    for v in SHOCK_VIX:
        panel = pd.DataFrame({"spx": S, "spxtr": S, "skew": calm["skew"], "rf": calm["rf"], "q": calm["q"],
                              **{p: calm[p] + (v - calm[p]) * k for p in vols}}, index=idx)
        df = sleeves.option_sleeve(panel, c.spec, anchors=_anchors(c, idx), **({"h": H_DEFAULT} | (pricing_kw or {})))
        month = (1 + df["ret"].iloc[i0 + 1 : i1 + 1]).prod() - 1  # from the roll's NAV_post (a restart's unit capital)
        out[v] = float(eq * SHOCK + w * month)
    return out


def universa_checks(nav_3_33: pd.Series, spxtr: pd.Series) -> pd.DataFrame:
    """Known-answer comparisons with Universa's reported 96.67 / 3.33 portfolio (UNIVERSA_LABEL; informational,
    never fit targets): one row per UNIVERSA_CHECKS window, columns check, universa_reported, ours (the
    window return of nav_3_33, metrics.window_return from the close before the window; for the drag row
    its 2019 return minus 0.9667 x the S&P 500's, the sleeve's contribution in portfolio points), gap =
    ours - universa_reported, spx_reported, spx_ours (spxtr's window return) and label (UNIVERSA_LABEL,
    D15). NaN where the series do not cover a window."""
    rows = []
    for check, a, b, port, spx, drag in UNIVERSA_CHECKS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ours, spx_ours = metrics.window_return(nav_3_33, a, b), metrics.window_return(spxtr, a, b)
        ours -= (1 - W_UNIVERSA) * spx_ours if drag else 0.0
        rows.append({"check": check, "universa_reported": port, "ours": ours, "gap": ours - port,
                     "spx_reported": spx, "spx_ours": spx_ours, "label": UNIVERSA_LABEL})
    return pd.DataFrame(rows)


def tail_checks(proxy_r: pd.Series) -> pd.DataFrame:
    """The TAIL proxy's calendar-year returns against Cambria TAIL's reported NAV returns (TAIL_CALENDAR;
    informational): columns year, tail_reported, ours, gap = ours - tail_reported; NaN for a year the
    proxy does not cover in full (scenarios.full_year_returns)."""
    ours = scenarios.full_year_returns(metrics.nav_from_returns(proxy_r.dropna())).reindex(list(TAIL_CALENDAR))
    return pd.DataFrame([{"year": y, "tail_reported": v, "ours": ours[y], "gap": ours[y] - v}
                         for y, v in TAIL_CALENDAR.items()])


def _ex_best_crash_year(ann: pd.DataFrame) -> bool | None:
    """Geometric dCAGR (h over u) still > 0 without the best of 2008 / 2020 for the hedge (largest h - u);
    None when neither year is in the full-year table."""
    years = [y for y in (2008, 2020) if y in ann.index]
    if not years:
        return None
    rest = ann.drop(max(years, key=lambda y: ann.at[y, "h"] - ann.at[y, "u"]))
    geo = (1 + rest).prod() ** (1 / len(rest)) - 1
    return bool(geo["h"] > geo["u"])


def _dice_beats(pair: pd.DataFrame, spx_annual: pd.Series, edge: float) -> bool | None:
    """Estimated-mode dice with the crash bin's edge at `edge` (bins edge / 0 / 15%): regime frequencies
    from all of spx_annual's years, per-regime mean full-year returns of pair's columns u and h; True when
    h compounds faster. None (n/a) when a regime with mass has no full year in the sample: its mean is
    unknown (a post-2009 sample has no year at or below -20%). Needs a full year (evaluate runs it on 10+)."""
    edges = (-np.inf, edge, 0.0, 0.15, np.inf)
    _, P, n = scenarios.regime_portfolio_returns(pair, spx_annual, edges)
    freq = scenarios.regime_table(spx_annual, edges)["freq"].to_numpy()
    if (n[freq > 0] == 0).any():
        return None
    g = scenarios.dice_growth(freq, P)
    return bool(g[1] > g[0])


def _sold_before_expiry(spec: OptionSpec) -> bool:
    """True when some leg closes on a roll before its expiry (month-start rolls, or held fewer months
    than its tenor), where the roll-offset sweep moves timing only."""
    every = spec.roll_every_months
    return spec.roll_when == "month_start" or any(
        math.ceil((L.hold_months or L.tenor_months) / every) * every < L.tenor_months for L in spec.legs)


def evaluate(
    c: Candidate, core_r: pd.Series, sleeve_r: pd.Series, rebalance: pd.DatetimeIndex, mkt: pd.DataFrame, *,
    dd_cap: float = 0.30, cash_table: pd.DataFrame | None = None, policy: Policy = Policy(), funding: str = "carve_out",
    bootstrap_paths: int = 2000, seed: int = 0, px: pd.DataFrame | None = None, beta_down: float | None = None,
    pricing_kw: dict | None = None, chain: pd.DataFrame | None = None,
) -> dict:
    """Every test of one candidate, as one flat dict (rank's row).

    The frontier (backtest.frontier, crash windows metrics.CRASHES) on the grid {0} + c.w_grid, funding
    "carve_out" ((1 - w) core + w sleeve) or "relever" (the core levered to 1 / (1 - w) by
    portfolio.lever at rf + 50 bp, so equity stays at 100% and the sleeve is financed on top; w <=
    RELEVER_MAX_W, larger grid weights dropped and noted, ValueError when no positive one is left; the
    row is labelled levered with gross_exposure 1 + w*). w_star = best_allocation with no cap (the
    CAGR maximum, Kelly: the one sizing rule for every candidate, addendum section 8); w_cap =
    best_allocation under dd_cap (NaN and noted when no grid weight meets it). The sample is the
    engine's common dates.
    Keys (bp = x 1e4; "u" = w 0, "h" = w*):
    - candidate, archetype, kind, model_priced, funding, sample_start, sample_end, years (calendar days
      / 365.25), n_crashes_in_sample (CRASHES windows inside the sample), w_star, w_cap, gross_exposure;
    - cagr_unhedged, cagr_star, d_cagr_bp, arith_cost_bp (d mean calendar-year return, the x axis of
      the cost-effectiveness plane), maxdd_unhedged, maxdd_star, calmar_star, worst_year_star;
    - the D7 benchmarks over the sample: bench_spxtr_cagr / _maxdd (S&P 500 TR, mkt spxtr) and
      bench_6040_cagr / _maxdd (60% S&P 500 TR / 40% AGG, the DGS10 proxy before AGG, rebalanced at
      year-end; NaN and noted unless px has AGG, and DGS10 when the sample starts before AGG);
    - the AQR twins against the T-bill frontier (cash_table, else computed on the sample over CASH_GRID,
      carve-out, same calendar and policy), at w_cap (w* when it is NaN, noted): d_cagr_vs_dd_matched_cash_bp
      (backtest.dd_matched_cash_gap), dd_gap_vs_cagr_matched_cash (backtest.cagr_matched_cash_dd's
      maxdd_gap, the divested twin), beats_deleverage (that weight > 0 and the first > 0);
    - p_beat_10y: P(10-year CAGR h > u) in scenarios.bootstrap of the two NAVs' rebalance-period returns
      (rebalanced to h at each period end, as the engine is), bootstrap_paths paths, seed;
      d_cagr_p50_bp / d_cagr_nocrash_bp: median 25-year dCAGR of scenarios.book_bootstrap (joint /
      no_crash) on the full calendar-year returns of u and h; rolling10_win_rate: share of rolling
      10-year windows (backtest.rolling_excess) where h beats u; p_beat_10y and rolling10_win_rate are
      NaN at w* = 0 (nothing tested, not "never beats");
    - sleeve scores at w_scores (w*, or the smallest grid weight when w* = 0, noted), in portfolio
      points: crash_payoff_mean / _fast / _slow (mean crash-window return h - u, all / by
      metrics.CRASH_SPEED), calm_year_bleed, avg_premium_yr (sum of premium_paid x the sleeve's share of
      NAV, per year; option sleeves, from build_sleeve's attrs, else NaN), avg_coverage (mean daily long
      option notional / portfolio NAV for option sleeves: build_sleeve's daily coverage x the sleeve's
      share; else the sleeve's capital share, its notional),
      bleed_notional (calm_year_bleed / avg_coverage), attribution attr_crash_bp / attr_noncrash_bp (the
      sum of the year differences h - u in crash-bin / other years over the number of years: they add up
      to the mean difference) and d_beta (daily beta to the S&P 500, h minus u);
    - the put presets' own scores, at w_preset = W_UNIVERSA when the grid has it (the Universa presets), else
      w_scores: shock20_pnl (shock_pnl at SHOCK_VIX[1] for every candidate with a spec, equity 1 under
      relever; all three in notes; NaN otherwise) and, for premium option presets, the calm-year bleed
      in notes, warned above BLEED_WARN;
    - critic_sleeve_cagr, critic_sleeve_sharpe, critic_alpha, critic_beta: critic_view of the sleeve;
    - verdict: "insufficient history" if years < 10 or no crash window; else the base case (addendum
      section 15, D6) is d_cagr_p50 > 0 and d_cagr_bp > 0; when it passes, the robustness checks run:
      rolling10 (win rate >= 0.5; n/a with no full 10-year window), since2015 (dCAGR from 2015-01-01 > 0),
      ex_best_crash_year, crash_edge_-10% / -20% (estimated-mode dice, _dice_beats; n/a when a regime has
      no year in the sample), and for option / composite specs dCAGR
      > 0 again on rebuilt sleeves (build_sleeve with px, beta_down, pricing_kw, chain): roll_offsets
      (OFFSETS; only when legs are sold before expiry), mid_fills (h 0) and wing_z_cap (z_cap 3, or inf
      when the base is 3; model-priced only). "cost-effective (robust)" when every check passes (n/a
      ones skipped), "cost-effective (fragile)" when one fails, else "not cost-effective"; robustness
      names each check's pass / fail / n/a ("not run" otherwise); notes collects the run's caveats.
    Plus _table (the frontier) and _nav (its NAV frame), which rank drops. beta_down None: the core's
    downside beta to the S&P 500 (only overlay rebuilds use it). Raises ValueError for an unknown funding,
    or for the TAIL proxy without its source (DGS10) in px (its rebuilds need it).
    """
    if funding not in ("carve_out", "relever"):
        raise ValueError(f"funding must be 'carve_out' or 'relever', got {funding!r}")
    if c.kind == "composite" and c.spec is not None:
        _series(c.source, mkt, px)  # the robustness rebuilds read it: fail before the frontier, not after
    extra = dict(sleeve_r.attrs)
    sleeve_r = sleeve_r.copy()
    sleeve_r.attrs = {}
    grid = np.array(sorted({0.0, *c.w_grid}))
    notes = []
    if funding == "relever":
        cut, grid = grid[grid > RELEVER_MAX_W], grid[grid <= RELEVER_MAX_W]
        if not (grid > 0).any():
            raise ValueError(f"{c.key}: relever needs a grid weight in (0, 1/3] (L = 1 / (1 - w) <= 1.5), got "
                             f"{c.w_grid}")
        if len(cut):
            notes.append("relever: grid weights above 1/3 dropped (" + ", ".join(f"{w:g}" for w in cut) + ")")
    table, nav = _frontier(core_r, sleeve_r, grid, rebalance, policy, funding, mkt["rf"])
    start, end = nav.index[0], nav.index[-1]
    years = (end - start).days / 365.25
    w_star, w_cap = backtest.best_allocation(table, None), backtest.best_allocation(table, dd_cap)
    w_twin = w_star if w_cap is None else w_cap
    if w_cap is None:
        w_cap = np.nan
        notes.append(f"no w meets the {dd_cap:.0%} drawdown cap: w_cap n/a, the AQR twins at w*")
    positive = table.index[table.index > 0]
    w_s = w_star if w_star > 0 or positive.empty else float(positive.min())
    if w_s != w_star:
        notes.append(f"w* = 0: sleeve scores at w = {w_s:g}, the smallest grid weight")
    if funding == "relever":
        notes.append(f"levered: the core at 1 / (1 - w) (100% equity) plus the sleeve (and any equity it holds), "
                     f"financed at rf + 50 bp; gross exposure {1 + w_star:.3g}x")
    if cash_table is None:
        cash_table, _ = backtest.frontier(core_r.loc[start:end], sleeves.tbill_sleeve(mkt).loc[start:end], CASH_GRID,
                                          rebalance, policy, metrics.CRASHES)
    spx_annual, spx_r = scenarios.load_sp500_annual(), mkt["spxtr"].pct_change()
    crash = [k for k in table.columns if k.startswith("crash:")]
    h, u = table.loc[w_star], table.loc[0.0]
    d_cagr = float(h["cagr"] - u["cagr"])
    dd_gap = backtest.dd_matched_cash_gap(table, cash_table, w_twin)
    spx = spx_r.loc[start:end].fillna(0.0)
    spx_nav = metrics.nav_from_returns(spx)
    bench = {"bench_spxtr_cagr": float(metrics.cagr(spx_nav)), "bench_spxtr_maxdd": float(metrics.max_drawdown(spx_nav)),
             "bench_6040_cagr": np.nan, "bench_6040_maxdd": np.nan}
    try:
        b60, _ = backtest.simulate(spx, _bonds("AGG", mkt, px, start).loc[:end], 0.4,
                                   rebalance_calendar(nav.index, "annual"), policy)
        bench |= {"bench_6040_cagr": float(metrics.cagr(b60[0.4])), "bench_6040_maxdd": float(metrics.max_drawdown(b60[0.4]))}
    except ValueError:  # no AGG (or DGS10 before it) in px, or no overlap with the sample
        notes.append("60/40 benchmark not run: pass AGG in px (and DGS10 when the sample starts before AGG)")

    pair = pd.DataFrame({"u": nav[0.0], "h": nav[w_star]})
    ann = scenarios.full_year_returns(pair)
    p_beat = win10 = np.nan  # w* = 0 tests nothing: n/a, not "never beats"
    if w_star > 0:
        per = backtest.period_returns(pair.pct_change().iloc[1:], rebalance)
        boot = scenarios.bootstrap(per["u"], per["h"], np.array([0.0, 1.0]), max(1, round(len(per) / years)),
                                   horizons=(10,), n_paths=bootstrap_paths, seed=seed, dd_cap=dd_cap)
        p_beat = float(boot.at[(10, 1.0), "p_beat"])
        ex10 = backtest.rolling_excess(nav[[0.0, w_star]], 10)[w_star].dropna()
        win10 = float((ex10 > 0).mean()) if len(ex10) else np.nan  # 10 calendar years can be < 2520 rows

    def book(mode: str) -> float:
        """Median 25-year dCAGR (h over u) of the book bootstrap in `mode`; NaN when it has no year to draw."""
        try:
            out = scenarios.book_bootstrap(ann["u"], ann["h"], np.array([0.0, 1.0]), n_paths=bootstrap_paths, seed=seed,
                                           mode=mode)
        except ValueError:  # no full year (no year above -15% for no_crash)
            return np.nan
        return float(out.at[1.0, "d_cagr_p50"])

    d50 = book("joint")

    # sleeve scores at w_s, in portfolio points
    pair_s = pd.DataFrame({"u": nav[0.0], "h": nav[w_s]})
    payoff = table.loc[w_s, crash] - table.loc[0.0, crash]
    speed = np.array([metrics.CRASH_SPEED.get(k.removeprefix("crash:")) for k in crash], dtype=object)
    bleed = calm_year_bleed(nav[w_s], nav[0.0], spx_annual)
    share = _sleeve_share(nav[w_s], sleeve_r, rebalance, w_s)
    prem, cov = np.nan, float(share.mean())  # no option attrs: the sleeve's capital is its notional
    if "premium_paid" in extra:
        prem = float((extra["premium_paid"].reindex(nav.index).fillna(0.0) * share).sum() / years)
        cov = float((extra["coverage"].reindex(nav.index).ffill().fillna(0.0) * share).mean())
    a = scenarios.full_year_returns(pair_s)
    diff = (a["h"] - a["u"])[a.index.isin(spx_annual.index)]
    in_crash = (spx_annual.reindex(diff.index) <= scenarios.REGIME_EDGES[1]).to_numpy()
    w_p = W_UNIVERSA if W_UNIVERSA in table.index else w_s  # BLEED_WARN and the addendum's shock are at 3.33%
    shock20 = np.nan
    if c.spec is not None:
        shock = shock_pnl(c, mkt, w_p, equity=1.0 if funding == "relever" else None, pricing_kw=pricing_kw)
        shock20 = shock[SHOCK_VIX[1]]
        notes.append(f"-20% month at w = {w_p:g}: P&L " + " / ".join(f"{p:+.1%}" for p in shock.values())
                     + " with VIX " + " / ".join(f"{v:g}" for v in shock) + " after it")
    if c.kind == "option" and c.spec.mode == "premium":
        bleed_p = calm_year_bleed(nav[w_p], nav[0.0], spx_annual)
        notes.append(f"calm-year bleed {-bleed_p:.1%}/yr of NAV at w = {w_p:g}"
                     + (f", above the {BLEED_WARN:.1%} warning" if -bleed_p > BLEED_WARN else ""))
    critic = critic_view(sleeve_r, spx_r, mkt["rf"])
    launch = LAUNCH.get(c.source) if c.kind == "index" else None
    if launch and start < pd.Timestamp(launch):
        notes.append(f"Cboe back-test levels before the {launch} launch")
    bad = [d for d in replicate.BAD_MARKS.get(str(c.source).upper(), ()) if start <= pd.Timestamp(d) <= end]
    if c.kind == "index" and bad:
        notes.append(f"likely bad Cboe marks kept (flagged): {', '.join(bad)}")

    checks: dict[str, bool | None] = {}
    if years < 10 or not crash:
        verdict = "insufficient history"
    elif d50 > 0 and d_cagr > 0:
        since = pair.loc["2015-01-01":]
        checks = {
            "rolling10": None if np.isnan(win10) else bool(win10 >= 0.5),
            "since2015": None if len(since) < 2 else bool(metrics.cagr(since["h"]) > metrics.cagr(since["u"])),
            "ex_best_crash_year": _ex_best_crash_year(ann),
            "crash_edge_-10%": _dice_beats(pair, spx_annual, -0.10),
            "crash_edge_-20%": _dice_beats(pair, spx_annual, -0.20),
        }
        if c.spec is not None:
            beta = beta_down if beta_down is not None else metrics.downside_beta(core_r.iloc[1:], spx_r)

            def rebuilt(**over: float) -> bool:
                s, cal = build_sleeve(c, mkt, px, core_r, beta, pricing_kw=(pricing_kw or {}) | over, chain=chain)
                s.attrs = {}
                t, _ = _frontier(core_r, s, np.array([0.0, w_star]), cal, policy, funding, mkt["rf"])
                return bool(t.at[w_star, "cagr"] > t.at[0.0, "cagr"])

            checks["roll_offsets"] = (all(rebuilt(offset_bdays=k) for k in OFFSETS) if _sold_before_expiry(c.spec)
                                      else None)
            if checks["roll_offsets"] is None:
                notes.append("roll-offset sweep skipped: legs held to expiry (an offset would change coverage and "
                             "tenor)")
            checks["mid_fills"] = rebuilt(h=0.0)
            if c.model_priced:
                z = (pricing_kw or {}).get("z_cap", pricing.Z_CAP)
                checks["wing_z_cap"] = rebuilt(z_cap=3.0 if np.isinf(z) else np.inf)
        failed = [k for k, v in checks.items() if v is False]
        verdict = "cost-effective (fragile)" if failed else "cost-effective (robust)"
        if failed:
            notes.append(f"fragile: fails {', '.join(failed)}")
    else:
        verdict = "not cost-effective"

    return {
        "candidate": c.key, "archetype": c.archetype, "kind": c.kind, "model_priced": c.model_priced,
        "funding": funding, "sample_start": start, "sample_end": end, "years": years, "n_crashes_in_sample": len(crash),
        "w_star": w_star, "w_cap": w_cap, "gross_exposure": 1 + w_star if funding == "relever" else 1.0,
        "cagr_unhedged": float(u["cagr"]), "cagr_star": float(h["cagr"]), "d_cagr_bp": d_cagr * 1e4,
        "arith_cost_bp": float(h["arith_mean"] - u["arith_mean"]) * 1e4, "maxdd_unhedged": float(u["max_dd"]),
        "maxdd_star": float(h["max_dd"]), "calmar_star": float(h["calmar"]), "worst_year_star": float(h["worst_year"]),
        **bench, "d_cagr_vs_dd_matched_cash_bp": dd_gap * 1e4,
        "dd_gap_vs_cagr_matched_cash": backtest.cagr_matched_cash_dd(table, cash_table, w_twin)["maxdd_gap"],
        "beats_deleverage": bool(w_twin > 0 and dd_gap > 0), "p_beat_10y": p_beat,
        "d_cagr_p50_bp": d50 * 1e4, "d_cagr_nocrash_bp": book("no_crash") * 1e4, "rolling10_win_rate": win10,
        "w_scores": w_s, "crash_payoff_mean": float(payoff.mean()) if crash else np.nan,
        "crash_payoff_fast": float(payoff[speed == "fast"].mean()) if (speed == "fast").any() else np.nan,
        "crash_payoff_slow": float(payoff[speed == "slow"].mean()) if (speed == "slow").any() else np.nan,
        "calm_year_bleed": bleed, "avg_premium_yr": prem, "avg_coverage": cov,
        "bleed_notional": bleed / cov if cov > 0 else np.nan,
        "attr_crash_bp": float(diff[in_crash].sum() / len(diff)) * 1e4 if len(diff) else np.nan,
        "attr_noncrash_bp": float(diff[~in_crash].sum() / len(diff)) * 1e4 if len(diff) else np.nan,
        "d_beta": metrics.beta(nav[w_s].pct_change(), spx_r) - metrics.beta(nav[0.0].pct_change(), spx_r),
        "shock20_pnl": shock20,
        "critic_sleeve_cagr": critic["cagr"], "critic_sleeve_sharpe": critic["sharpe"], "critic_alpha": critic["alpha"],
        "critic_beta": critic["beta"], "verdict": verdict,
        "robustness": ", ".join(f"{k} {'n/a' if v is None else 'pass' if v else 'fail'}" for k, v in checks.items())
        or "not run",
        "notes": "; ".join(notes), "_table": table, "_nav": nav,
    }


def rank(rows: list[dict]) -> pd.DataFrame:
    """evaluate rows as a table without the "_" keys, best first: by verdict tier (TIERS: robust, fragile,
    not cost-effective, insufficient history), then d_cagr_bp descending; empty for no rows."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    order = df.assign(tier=df["verdict"].map(TIERS)).sort_values(["tier", "d_cagr_bp"], ascending=[True, False],
                                                                  kind="stable")
    return df.loc[order.index].reset_index(drop=True)

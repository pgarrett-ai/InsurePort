"""Safe-haven sleeves as daily return series per unit of sleeve capital.

The two-asset engine (backtest.py) combines (1 - w) core + w sleeve, so every safe haven is a
daily simple-return series on the market panel's index. Option sleeves (option_sleeve) roll
synthetic SPX option packages priced by pricing.py in one of two sizing modes:
- premium: the sleeve is cash; each roll spends premium_frac of the cash on the package
  (Universa-style tail hedges, put spreads); a sleeve wiped out restarts from unit capital, so
  the engine's event-day rebalance refills it (the "monthly premium" behaviour);
- overlay: the sleeve is the underlying plus options on coverage x its value, the premium
  financed by selling stock (protective put, collar, Cboe PPUT / CLL replication).
Given a real end-of-day chain (chains.py), option_sleeve opens listed SPX contracts and prices
them from its quotes, the model filling the leg-days without one.
Also T-bills (tbill_sleeve, AQR's "hold less equity"), a constant-maturity par-bond proxy
(bond_proxy) and SPX time-series momentum (tsmom_sleeve).

Pure numpy/pandas + pricing.py and the chains.py lookups, no I/O. Panel columns as in
data.load_market: spx, spxtr and the vol indices in index points (plus spx_open, needed only by
specs that strike or settle at the open), rf and q decimals. Cash accrues rf_{t-1} * calendar days / 360
(portfolio.cash_returns); options are priced with r = ln(1 + rf).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from insureport import chains, pricing
from insureport.portfolio import cash_returns

WIPED_OUT = 1e-12  # a sleeve left with this fraction of its NAV at the previous event restarts from unit capital
_VOLS = ("vix9d", "vix", "vix3m", "vix6m", "vix1y")  # pricing.var_swap_vol pillars, in order


@dataclass(frozen=True)
class Leg:
    """One option leg of a package, per unit of package notional.

    cp "P" | "C". Strike when the leg opens, off the reference level S_ref (ref "close": the row's
    close, "open": its open, spx_open, the pre-11:00 level Cboe strikes off): otm -> K = S_ref (1 - otm)
    for puts, S_ref (1 + otm) for calls; delta -> the |delta| strike from pricing.strike_for_delta at
    S_ref (exactly one of otm / delta). strike_rule then snaps K to the grid (index points):
    "continuous" keeps it, "floor" takes the largest grid multiple strictly below it (PPUT's "first
    listed strike below 95%"), "ceil" the smallest at or above it (the CLL family, PPUT3M).
    expiry = third Friday of (open month + tenor_months) (option_sleeve shifts it with offset
    rolls); the leg is closed ceil(hold_months / roll_every_months) rolls after it opens (None =
    tenor_months, i.e. held to expiry). qty: + long / - short per unit of notional.
    every_months = N: opened only at rolls whose ordinal is a multiple of N.
    """

    cp: str
    otm: float | None = None
    delta: float | None = None
    tenor_months: int = 1
    hold_months: int | None = None
    qty: float = 1.0
    every_months: int = 1
    strike_rule: str = "continuous"
    grid: float = 5.0
    ref: str = "close"


@dataclass(frozen=True)
class OptionSpec:
    """A rolled option strategy.

    mode "premium": each roll spends premium_frac of a budget base on the package: budget_base
    "cash", the sleeve's cash; "anchor" (addendum D3), the sleeve NAV at the last anchor date
    (option_sleeve's anchors), capped at the cash; "overlay": the sleeve holds the underlying and options
    on coverage x its value. monetize_mult: close and re-strike every leg once the long legs are worth
    monetize_mult x what they cost. roll_every_months: roll on every N-th monthly roll date, those of
    roll_dates(when=roll_when): "third_friday" (Cboe's calendar) or "month_start" (the first trading day
    of the month, Universa's Rule B; expiries stay third Fridays). settle: the S&P 500 level an expiring
    leg pays intrinsic against, on its expiry trading day: "close" (PM settlement), or "open" (spx_open,
    the proxy for Cboe's AM Special Opening Quotation). harvest_at_anchor: sell every held leg into cash
    at the close of each anchor row (the portfolio rebalance), so the anchor budget base is all T-bills.
    """

    legs: tuple[Leg, ...]
    mode: str = "premium"
    premium_frac: float = 1.0
    coverage: float = 1.0
    monetize_mult: float | None = None
    roll_every_months: int = 1
    settle: str = "close"
    budget_base: str = "cash"
    roll_when: str = "third_friday"
    harvest_at_anchor: bool = False


def third_friday(year: int, month: int) -> pd.Timestamp:
    """Third Friday of the month; month may run past 12 (third_friday(2019, 13) is 2020-01-17)."""
    first = pd.Timestamp(year + (month - 1) // 12, (month - 1) % 12 + 1, 1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def roll_dates(
    index: pd.DatetimeIndex, every_months: int = 1, offset_bdays: int = 0, when: str = "third_friday"
) -> pd.DatetimeIndex:
    """Monthly roll dates inside a sorted trading-day index.

    when "third_friday": each month's third Friday snaps to the last index date on or before it (Good
    Friday 2019-04-19 -> 2019-04-18). A month gets no roll when that date is not in the same month (no
    index dates up to its Friday) or when the index ends before the Friday (the roll has not
    happened yet). when "month_start": the first index date of each month (the index's own first date
    counts, so a panel starting mid-month rolls on its first day). Then every every_months-th roll is
    kept (0, N, 2N, ...) and each is moved offset_bdays index positions later (the roll-offset sweep);
    rolls pushed past the end drop out. Raises ValueError for any other `when`.
    """
    idx = pd.DatetimeIndex(index)
    if when not in ("third_friday", "month_start"):
        raise ValueError(f"when must be 'third_friday' or 'month_start', got {when!r}")
    if idx.empty:
        return idx
    ym = idx.year * 12 + idx.month
    if when == "month_start":
        pos = np.flatnonzero(np.r_[True, np.diff(ym) != 0])
    else:
        fri = pd.date_range(idx[0].to_period("M").start_time, idx[-1], freq="WOM-3FRI")
        pos = idx.searchsorted(fri, side="right") - 1
        pos = pos[(pos >= 0) & (ym[np.maximum(pos, 0)] == fri.year * 12 + fri.month)]
    pos = pos[::every_months] + offset_bdays
    return idx[pos[(pos >= 0) & (pos < len(idx))]]


def _check(spec: OptionSpec) -> None:
    """Raise ValueError on a spec option_sleeve cannot run (app users edit these parameters)."""
    if spec.mode not in ("premium", "overlay"):
        raise ValueError(f"mode must be 'premium' or 'overlay', got {spec.mode!r}")
    if spec.mode == "premium" and not 0 < spec.premium_frac <= 1:
        raise ValueError(f"premium_frac must be in (0, 1], got {spec.premium_frac}")
    if spec.mode == "overlay" and not spec.coverage > 0:
        raise ValueError(f"coverage must be > 0, got {spec.coverage}")
    if spec.monetize_mult is not None and not spec.monetize_mult > 1:
        raise ValueError(f"monetize_mult must be > 1, got {spec.monetize_mult}")
    if not spec.legs or spec.roll_every_months < 1:
        raise ValueError("need at least one leg and roll_every_months >= 1")
    if spec.settle not in ("close", "open"):
        raise ValueError(f"settle must be 'close' or 'open', got {spec.settle!r}")
    if spec.budget_base not in ("cash", "anchor"):  # roll_dates checks roll_when
        raise ValueError(f"budget_base must be 'cash' or 'anchor', got {spec.budget_base!r}")
    for L in spec.legs:
        if L.cp not in ("P", "C") or (L.otm is None) == (L.delta is None):
            raise ValueError(f"{L}: cp must be 'P' or 'C' and exactly one of otm / delta set")
        if (L.otm is not None and not -1 < L.otm < 1) or (L.delta is not None and not 0 < abs(L.delta) < 1):
            raise ValueError(f"{L}: need -1 < otm < 1 / 0 < |delta| < 1")
        if L.tenor_months < 1 or L.every_months < 1 or (L.hold_months is not None and L.hold_months < 1):
            raise ValueError(f"{L}: tenor_months, hold_months and every_months must be >= 1")
        if L.strike_rule not in ("continuous", "floor", "ceil") or L.ref not in ("close", "open") or not L.grid > 0:
            raise ValueError(f"{L}: need strike_rule continuous / floor / ceil, ref close / open and grid > 0")
        held = math.ceil((L.hold_months or L.tenor_months) / spec.roll_every_months) * spec.roll_every_months
        if held > L.tenor_months:
            raise ValueError(f"{L}: closed at the roll {held} months out, after its {L.tenor_months}-month expiry")


def _snap(K: float, rule: str, grid: float) -> float:
    """K on the strike grid: "floor" = the largest multiple of grid strictly below K, "ceil" = the
    smallest at or above, "continuous" = K itself. K / grid is rounded to 9 decimals first, so a
    target on the grid up to float noise counts as on it (2700 x 1.1 = 2970.0000000000005 -> ceil 2970)."""
    # ponytail: fixed 5-pt grid; real listings are 5/10/25 pts by level
    q = round(K / grid, 9)
    if rule == "floor":
        return grid * (math.ceil(q) - 1)
    return grid * math.ceil(q) if rule == "ceil" else K


def _inputs(mkt: pd.DataFrame, settle: str = "close") -> dict[str, np.ndarray]:
    """Per-row pricing inputs as arrays (r = ln(1 + rf); spx forward-filled), the row dates (`day`)
    and `eff`: the last row <= i whose inputs are usable (spx, skew, rf, q finite and at least one
    VIX pillar), -1 before the first; `stale` marks rows that borrow an earlier row's inputs.
    `open`: spx_open, the row's spx where it is missing or <= 0 (or absent from mkt); `settle`: the
    level expiring legs settle at, open or spx per the argument.
    DEVIATION from the brief ("any pricing input NaN" is a fallback): a row missing only some VIX
    pillars is usable, var_swap_vol interpolating over the rest, and is not counted.
    """
    # ponytail: partial-pillar days are not flagged; count them separately if the UI must disclose them
    px = {c: mkt[c].to_numpy(float) for c in ("spx", "skew", "q", *_VOLS)}
    px["r"] = np.log1p(mkt["rf"].to_numpy(float))
    ok = np.isfinite(np.stack([px["spx"], px["skew"], px["r"], px["q"]])).all(0)
    ok &= np.isfinite(np.stack([px[c] for c in _VOLS])).any(0)
    i = np.arange(len(mkt))
    px["eff"] = np.maximum.accumulate(np.where(ok, i, -1))
    px["stale"] = px["eff"] != i
    px["spx"] = mkt["spx"].ffill().to_numpy(float)  # S is the row's own, carried only where spx itself is missing
    o = mkt.get("spx_open", mkt["spx"]).to_numpy(float)
    # ponytail: a missing open falls back to the close per row, silently (load_market's warnings count missing
    # opens); count the roll / settle rows hit in attrs if a partial ^GSPC Open refresh ever bites
    px["open"] = np.where(o > 0, o, px["spx"])  # NaN compares False: no open print, the row's close
    px["settle"] = px["open"] if settle == "open" else px["spx"]
    px["day"] = mkt.index.to_numpy().astype("datetime64[D]")
    return px


def _settle_row(day: np.ndarray, expiry: np.ndarray) -> np.ndarray:
    """Row on which a leg expiring on `expiry` settles (at its close or open, per OptionSpec.settle): the
    last trading day on or before it (a Good Friday expiry settles on Thursday); len(day) when expiry,
    rolled back off a weekend (a pre-2015 Saturday expiration settles on the Friday), is past the data."""
    e = np.busday_offset(np.asarray(expiry, "datetime64[D]"), 0, roll="backward")
    return np.where(e <= day[-1], np.searchsorted(day, e, side="right") - 1, len(day))


def _quotes(
    px: dict[str, np.ndarray], day: np.ndarray, K: np.ndarray, expiry: np.ndarray, call: np.ndarray,
    *, m: float, nu: float, h: float, z_cap: float, beta: float, chain: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(mid, bid, ask, fallback) in index points, elementwise: option (K, expiry, call) on panel row `day`.

    The model branch: pricing.quote (m, nu, h, z_cap, beta passed on) at the row's S, T = (expiry -
    date)/365 and the vols, skew, r, q of the last usable row (_inputs `eff`: the row itself
    unless its inputs are unusable), vs = var_swap_vol(T, VIX pillars). From the settle row
    (_settle_row) on, T = 0: the mid is the intrinsic value at the settle row's px["settle"] (its
    close, or its open for AM settlement), frozen.
    chain (canonical, chains.py): rows before the settle row take the "SPX" quote listed for
    (date, expiry, K, cp) where there is one (chains.quotes, the vectorised quote_at); fallback (bool)
    marks the rows before the settle row priced by the model because none is listed. Without a chain,
    fallback marks the rows priced off carried inputs (_inputs `stale`).
    """
    # ponytail: the open stands in for the SOQ (the opening prints of all 500 stocks); EOD data has no SOQ
    settle = _settle_row(px["day"], expiry)
    d = np.minimum(day, settle)
    row = px["eff"][d]
    T = np.where(day >= settle, 0.0, (expiry - px["day"][d]).astype(float) / 365)
    S = np.where(day >= settle, px["settle"][d], px["spx"][d])
    vs = pricing.var_swap_vol(T, *(px[c][row] for c in _VOLS))
    out = np.empty((3, len(day)))
    for cp in "PC":
        sel = call == (cp == "C")
        if sel.any():
            r = row[sel]
            out[:, sel] = pricing.quote(
                S[sel], K[sel], T[sel], vs[sel], px["skew"][r], px["r"][r], px["q"][r], cp,
                m=m, nu=nu, h=h, z_cap=z_cap, beta=beta,
            )
    if chain is None:
        return out[0], out[1], out[2], px["stale"][day]
    live = day < settle
    q = chains.quotes(chain, px["day"][day], expiry, K, np.where(call, "C", "P"))
    got = live & ~np.isnan(q[0])
    out[:, got] = q[:, got]
    return out[0], out[1], out[2], live & ~got


def _open_legs(
    px: dict[str, np.ndarray], opens: list[tuple[int, Leg, int]], offset: int, *, m: float, nu: float, h: float,
    z_cap: float, beta: float, chain: pd.DataFrame | None = None,
) -> list[dict]:
    """New legs from (open row, Leg, scheduled close row; len(panel) = past the data).

    Strike from the open row's S_ref (Leg.ref: its close or open) by otm, or pricing.strike_for_delta
    (delta, at S_ref and the last usable vols, skew, r, q), snapped by Leg.strike_rule. Expiry = third
    Friday of (M + tenor_months), M the month of the row `offset` rows before the open (the un-shifted
    roll); with offset != 0 it then moves like the rolls: snapped to the last row on or before it and
    shifted `offset` rows (past the data, `offset` business days). With a chain, K and expiry then
    snap to chains.select_contract(chain, open date, target, K, cp) (source "chain"), target = the
    un-shifted third Friday's trading day (the last row on or before it: a holiday Friday's AM monthly
    is listed on the Thursday; past the data, the day before the Friday), so an offset leg is still the
    listed monthly. The leg keeps the model's K and expiry (source "model") when nothing is listed, or,
    DEVIATION from the brief (its select_contract result taken as is), when the listing found is one
    before the target that settles before the leg's scheduled close (a chain short of far expirations;
    it would sit dead at intrinsic until its roll). Each leg carries its quotes `mid`, `bid`,
    `ask` and `fb` (_quotes' fallback flags) (arrays, one entry per row from its open to its close or
    the last row, all legs in ONE _quotes call) and unit_cost = ask (long) / bid (short) on the open row.
    """
    if not opens:
        return []
    day = px["day"]
    n = len(day)
    legs = []
    for t, L, close in opens:
        d = pd.Timestamp(day[min(max(t - offset, 0), n - 1)])
        expiry = third_friday(d.year, d.month + L.tenor_months).to_datetime64().astype("datetime64[D]")
        # the chain's listed monthly: on the Friday's trading day; past the data from the Thursday, the listing
        # of a holiday Friday (Good Friday, Juneteenth) the panel cannot show.
        # ponytail: past the data a rootless chain (ORATS / IvyDB, all "SPX") then opens a 2022+ Thursday weekly
        # instead of the Friday monthly, one day short; prefer a Friday listing there if that ever matters
        target = day[_settle_row(day, expiry)] if expiry <= day[-1] else expiry - np.timedelta64(1, "D")
        if offset:
            j = int(np.searchsorted(day, expiry, side="right")) - 1 + offset
            expiry = (day[j] if expiry <= day[-1] and 0 <= j < n
                      else (pd.Timestamp(expiry) + pd.offsets.BDay(offset)).to_datetime64().astype("datetime64[D]"))
        S, row = px["open" if L.ref == "open" else "spx"][t], px["eff"][t]
        if L.otm is not None:
            K = S * (1 - L.otm if L.cp == "P" else 1 + L.otm)
        else:
            # ponytail: one scalar brentq per open (~30 ms, ~15 s for a monthly 1990-> sleeve); if delta
            # legs reach the app or sweeps, bisect all scheduled opens at once (their strikes are path-independent)
            T = (expiry - day[t]).astype(float) / 365
            vs = float(pricing.var_swap_vol(T, *(px[c][row] for c in _VOLS)))
            args = (S, T, vs, px["skew"][row], px["r"][row], px["q"][row])
            K = pricing.strike_for_delta(*args, L.delta, L.cp, m=m, nu=nu, z_cap=z_cap, beta=beta)
        K, source = _snap(K, L.strike_rule, L.grid), "model"
        # ponytail: one root, SPX monthlies (AM-settled); SPXW weeklies / PM dailies in a chain are ignored. Give
        # Leg a root if a strategy needs weeklies (then settle those at the close)
        hit = None if chain is None else chains.select_contract(chain, day[t], target, K, L.cp)
        e = None if hit is None else hit[0].to_datetime64().astype("datetime64[D]")
        if e is not None and _settle_row(day, e) >= min(close, _settle_row(day, target)):
            expiry, K, source = e, hit[1], "chain"
        settle = int(_settle_row(px["day"], expiry))
        legs.append({"leg": L, "open": t, "close": close, "K": float(K), "expiry": expiry, "settle": settle,
                     "n": 0.0, "closed_on": None, "reason": None, "source": source})
    lens = [min(g["close"], n - 1) - g["open"] + 1 for g in legs]
    day = np.concatenate([np.arange(g["open"], g["open"] + k) for g, k in zip(legs, lens)])
    K = np.repeat([g["K"] for g in legs], lens)
    expiry = np.repeat(np.array([g["expiry"] for g in legs]), lens)
    call = np.repeat([g["leg"].cp == "C" for g in legs], lens)
    cuts = np.cumsum(lens)[:-1]
    quotes = (np.split(a, cuts) for a in _quotes(px, day, K, expiry, call, m=m, nu=nu, h=h, z_cap=z_cap,
                                                 beta=beta, chain=chain))
    for g, mid, bid, ask, fb in zip(legs, *quotes):
        g["mid"], g["bid"], g["ask"], g["fb"] = mid, bid, ask, fb
        g["unit_cost"] = float(ask[0] if g["leg"].qty > 0 else bid[0])
    return legs


def _cashed_in(held: list[dict], val: np.ndarray, mult: float) -> np.ndarray:
    """Rows of val (n_j mid_j, rows x held legs) where the long legs are worth >= mult x cost,
    cost = sum_j n_j unit_cost_j (the net debit paid for them), or the long legs' cost alone when
    that is a net credit (<= 0)."""
    net = sum(g["n"] * g["unit_cost"] for g in held)
    cost = net if net > 0 else sum(g["n"] * g["unit_cost"] for g in held if g["n"] > 0)
    long = np.array([g["n"] > 0 for g in held], dtype=bool)
    return val[:, long].sum(1) >= mult * cost if cost > 0 else np.zeros(len(val), dtype=bool)


def option_sleeve(
    mkt: pd.DataFrame,
    spec: OptionSpec,
    *,
    underlying: pd.Series | None = None,
    m: float = pricing.SKEW_MULT,
    nu: float = 1.0,
    h: float = 0.05,
    offset_bdays: int = 0,
    z_cap: float = pricing.Z_CAP,
    chain: pd.DataFrame | None = None,
    anchors: pd.DatetimeIndex | None = None,
    carry: pd.Series | None = None,
    beta: float = pricing.SKEW_BETA,
) -> pd.DataFrame:
    """Daily returns of a rolled synthetic-SPX-option sleeve, per unit of sleeve capital.

    Calendar: rolls = roll_dates(mkt.index, spec.roll_every_months, offset_bdays, spec.roll_when),
    dropping any before the first row with usable pricing inputs; leg i opens at rolls whose ordinal is a
    multiple of its every_months, expires on the third Friday of (open month + tenor_months) and
    closes ceil(hold_months / roll_every_months) rolls later (_check rejects a close after expiry).
    DEVIATION from the brief ("the event in month M + hold_months"): the same roll whenever every
    month has one; a month missing from the data pushes the close one roll later.
    DEVIATION from the brief (third-Friday expiries): with offset_bdays the open month is read on
    the un-shifted calendar and the expiry moves offset_bdays rows like the rolls (_open_legs), so
    the offset sweep changes roll timing only, not tenor or coverage. Under a chain the leg is the
    listed monthly of that un-shifted calendar instead, so the tenor stays but coverage does not:
    offset_bdays > 0 leaves it settled at intrinsic for offset_bdays rows before its roll, < 0 closes
    it at the bid / ask |offset_bdays| rows before it expires.
    Before the first roll the sleeve is all cash (premium) or all underlying (overlay).

    State: cash C (grows by 1 + rf_{t-1} days/360, or by 1 + `carry`, daily returns on mkt's dates (NaN
    0), when given: the TAIL proxy's cash is a Treasury), stock A (overlay; grows with `underlying`,
    default spxtr.pct_change()), legs n_j (signed index units). NAV = A + C + sum n_j mid_j.
    Quotes from pricing.quote via _quotes (m, nu, h, z_cap, beta passed on; m, nu, z_cap, beta also to
    strike_for_delta): marks at mid; opens at the ask (long) / bid (short); closes before expiry
    at the bid (long) / ask (short); from the last trading day on or before expiry the leg is
    worth intrinsic on that day's close (spec.settle "close") or open ("open": spx_open, the
    row's close where it is missing), either way before the roll's new legs open at the close.
    Strikes per Leg (ref, strike_rule, grid). A spec that settles or strikes at the open needs an
    spx_open column with at least one value > 0 (an all-missing open, e.g. a failed ^GSPC Open
    fetch, would silently strike and settle every roll at the close).
    At a roll on row t: (1) close the legs due, proceeds to C, then monetize the legs held over
    the roll if they qualify (below); (2) NAV_pre, ret_t = NAV_pre / NAV_{t-1} - 1;
    (3) NAV_pre <= WIPED_OUT x NAV_post of the previous event (start: 1): ret_t = -1 exactly and
    A = 0, C = 1 (restart from unit capital). The state is then rescaled to NAV_pre = 1 (returns
    are scale-free), so the test is relative and a sleeve that bleeds for decades cannot
    underflow; attrs n stays on the scale of the NAV compounded from ret (reset to 1 by a restart);
    (4) open the package (the scheduled legs plus any monetized ones), u = sum qty_j fill_j:
      premium: B = premium_frac C (budget_base "cash"), or min(premium_frac base, C) ("anchor"; base =
      the NAV at the close of the last anchor row: the first row on or after each `anchors` date; the
      start, NAV 1, before the first; anchors None = every event, so base = NAV_pre; a restart from unit
      capital re-anchors), u > 0 required, N = B / u, C -= B (no cash left: B = N = 0,
      and the zero-size legs are left out of attrs);
      overlay: A += C, C = 0, A <- A / (1 + coverage u / S), N = coverage A / S, B = N u;
    n_j = qty_j N, premium_paid_t = B / NAV_pre, coverage_t = S sum_{held, n_j > 0} n_j / NAV_pre
    (every long leg held after the open, S the row's own). New legs count at cost, so
    NAV_post = NAV_pre and the spread lands in ret_{t+1} = NAV_{t+1} / NAV_post - 1.
    Monetize (monetize_mult = M set): on the first day with sum_{long} n_j mid_j >= M x cost over
    the legs checked, cost = sum_j n_j unit_cost_j (the net debit paid for them), close them
    (bid / ask) and open the same legs again that day (strike off that day's S, expiry from the
    month of the row offset_bdays earlier, i.e. the un-shifted calendar), each keeping its
    predecessor's roll; the day is an event like a roll. Between rolls
    every held leg is checked, on a roll the legs held over it. DEVIATION from the brief: a
    net-credit package (cost <= 0, e.g. a collar) is checked against its long legs' cost, since
    the brief's rule would fire at once.
    Harvest (spec.harvest_at_anchor): every anchor row is an event that closes all held legs (bid /
    ask, close_reason "harvest") and opens none (a roll on the same row opens its legs after), so the
    budget base is cash: the year-end rebalance sweeps a payoff into T-bills instead of carrying put
    marks that decay before they are sold (with them the cash cap ran the sleeve dry after a year-end
    spike, the fixture's 2018 Q4).

    Between events holdings are fixed and marked in one vectorized step. A row with unusable
    inputs (_inputs) prices every leg at the row's own S and date with the last usable vols, SKEW,
    r and q; each leg-day held or closed on it before settlement counts in fallback_quotes.
    DEVIATION from the brief (carry the previous day's mark): a carried mark would also carry
    yesterday's spot into settlements, new strikes, closes and overlay sizing. DEVIATION from the
    slice-4 brief (held leg-days; changed in slice 9, with or without a chain): a close priced off
    carried inputs also counts, as a close without a chain quote does below.

    chain (a canonical chains.py chain, e.g. chains.load_chain): real quotes. At every open
    (scheduled or monetize) the leg's strike (strike_rule applied) and the third Friday of its
    un-shifted calendar snap to chains.select_contract among the "SPX" rows (_open_legs); nothing
    listed, or (DEVIATION from the brief, which takes select_contract's pick as is) only a listing that
    would expire before the leg's scheduled close: the model's strike and expiry. Opens (ask long /
    bid short), marks (mid) and closes (bid long / ask short) before the settle row take that
    contract's chain quote on the day; a leg-day without one is priced by the
    model (same strike and expiry) and, when held at the close or closed that day, counts in
    fallback_quotes, which then counts these missing quotes only (a stale model input under a chain
    quote is not a fallback). Settlement stays intrinsic (spec.settle) from the last trading day on or
    before the listed expiration (a pre-2015 Saturday expiration settles on the Friday, a holiday
    Friday's Thursday listing on the Thursday).

    Returns a DataFrame on mkt.index: ret, event (bool, rolls, monetizations and harvests), premium_paid
    (B / NAV_pre on events, else 0), coverage (long option notional held / NAV at each close: S_t
    sum_{held, n_j > 0} n_j / NAV_t, NAV_post on events, 0 while no long leg is held or NAV <= 0),
    fallback_quotes (int). attrs["legs"]: one dict per leg
    opened (size n != 0), by open date: open_date, close_date (scheduled; None past the data),
    expiry, cp, K, n, unit_cost, closed_on (None if still open), close_reason ("roll" | "expiry" |
    "monetize" | "harvest" | None), settle_px (the level it settled at; None unless it reached its expiry in
    the data), source ("chain": a listed contract, "model": none found or no chain). pandas
    deep-copies attrs into every derived object (~50 ms per op on a 9,000-day sleeve), so strip it
    before heavy arithmetic: r = df["ret"].copy(); r.attrs = {}.
    Raises ValueError on an invalid spec or a premium package that costs <= 0.
    """
    _check(spec)
    needs_open = spec.settle == "open" or any(L.ref == "open" for L in spec.legs)
    if needs_open and ("spx_open" not in mkt or not (mkt["spx_open"] > 0).any()):
        raise ValueError("the spec settles or strikes at the open: mkt needs spx_open values")
    if "ts_filled" not in mkt:
        mkt = pricing.fill_term_structure(mkt)
    px, n, kw = _inputs(mkt, spec.settle), len(mkt), {"m": m, "nu": nu, "h": h, "z_cap": z_cap, "beta": beta,
                                                       "chain": chain}
    premium = spec.mode == "premium"
    ev = mkt.index.get_indexer(roll_dates(mkt.index, spec.roll_every_months, offset_bdays, spec.roll_when))
    ev = ev[px["eff"][ev] >= 0].tolist()
    steps = [math.ceil((L.hold_months or L.tenor_months) / spec.roll_every_months) for L in spec.legs]
    plan = [(t, L, ev[k + s] if k + s < len(ev) else n)
            for k, t in enumerate(ev) for L, s in zip(spec.legs, steps) if k % L.every_months == 0]
    legs = _open_legs(px, plan, offset_bdays, **kw)  # strikes are path-independent: price every scheduled leg up front
    due: dict[int, list[dict]] = {}
    for g in legs:
        due.setdefault(g["open"], []).append(g)

    r_und = mkt["spxtr"].ffill().pct_change() if underlying is None else underlying.reindex(mkt.index)
    g_und = 1 + r_und.fillna(0.0).to_numpy(float)
    r_cash = cash_returns(mkt["rf"].ffill().fillna(0.0)) if carry is None else carry.reindex(mkt.index).fillna(0.0)
    g_cash = 1 + r_cash.to_numpy(float)
    g_und[0] = g_cash[0] = 1.0  # row 0 is the start: NAV 1
    ret, paid, fb = np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    event, cover = np.zeros(n, dtype=bool), np.zeros(n)
    A, C = (0.0, 1.0) if premium else (1.0, 0.0)
    held: list[dict] = []
    pos, nav, lvl = -1, 1.0, 1.0  # holdings as of the close of row pos; nav = NAV the next return is measured from
    anchor = np.zeros(n, dtype=bool)  # budget_base "anchor": rows whose closing NAV becomes the budget base
    if anchors is not None:
        at = mkt.index.searchsorted(pd.DatetimeIndex(anchors))
        anchor[at[at < n]] = True
    budget = 1.0  # the anchor budget base, in the state's units (rescaled with it at every event)
    for stop in [*ev, n]:
        while True:
            # mark rows pos+1 .. stop-1 with fixed holdings, cut at a monetize trigger or a harvest
            days = np.arange(pos + 1, stop)
            val = (np.column_stack([g["n"] * g["mid"][days - g["open"]] for g in held]) if held
                   else np.zeros((len(days), 0)))
            navs = A * np.cumprod(g_und[days]) + C * np.cumprod(g_cash[days]) + val.sum(1)
            cut = len(days)
            if spec.monetize_mult is not None:
                hits = np.flatnonzero(_cashed_in(held, val, spec.monetize_mult))
                cut = hits[0] if len(hits) else cut
            if spec.harvest_at_anchor and held:
                hits = np.flatnonzero(anchor[days[:cut]])
                cut = hits[0] if len(hits) else cut
            days, navs = days[:cut], navs[:cut]
            long_n = sum(g["n"] for g in held if g["n"] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                cover[days] = np.where(navs > 0, px["spx"][days] * long_n / navs, 0.0)
            hit = np.flatnonzero(anchor[days])
            budget = navs[hit[-1]] if len(hit) else budget
            # ponytail: ret is a double. Under a flat or near-flat put wing a far-OTM put held to expiry at f ~ 1
            # marks below 1e-16 of the day before in the last ~4 trading days at VIX <= 15, ret rounds to -1, and a
            # gain or crash payoff in expiry week is lost to compounding (test_expiry_week_payoff_is_lost_to_return_
            # rounding). The default linear wing (the anchored slope) keeps such marks alive at VIX 12 for SKEW >=
            # 100.5; a finite z_cap or a flat smile still rounds ret to -1. Add a NAV column (or have the engine read
            # sleeve NAV levels) if hold-to-expiry premium presets are used.
            with np.errstate(divide="ignore", invalid="ignore"):
                base = np.r_[nav, navs[:-1]]
                ret[days] = np.where(base > 0, navs / base - 1, 0.0)  # no capital, no return
            nav = navs[-1] if cut else nav
            t = pos + 1 + cut  # the event row: a monetize trigger, a harvest, the roll at `stop`, or past the end
            if t >= n:
                break
            A *= np.prod(g_und[pos + 1 : t + 1])
            C *= np.prod(g_cash[pos + 1 : t + 1])
            rest = [g for g in held if g["close"] > t]  # between rolls every held leg; on a roll, those held over it
            harvest = spec.harvest_at_anchor and bool(anchor[t])
            monetize = t < stop and not harvest
            if not (monetize or harvest) and rest and spec.monetize_mult is not None:
                mark = np.array([[g["n"] * g["mid"][t - g["open"]] for g in rest]])
                monetize = bool(_cashed_in(rest, mark, spec.monetize_mult)[0])
            # ponytail: monetize-day rebalance flows are charged the normal bid haircut only on the option legs
            # ponytail: a leg past settlement sits at frozen intrinsic (no rf) until its roll: when a month is
            # missing from the data, or under a chain with offset_bdays > 0 (the listed monthly expires before the
            # shifted roll). Sweep it to cash on the settle row if gappy panels or offset chain sweeps matter.
            for g in held:
                if g["close"] <= t or monetize or harvest:
                    i, expired = t - g["open"], t >= g["settle"]
                    C += g["n"] * (g["mid"] if expired else g["bid"] if g["leg"].qty > 0 else g["ask"])[i]
                    g["closed_on"] = t
                    early = "harvest" if harvest else "monetize"
                    g["reason"] = early if g["close"] > t else "expiry" if expired else "roll"
            held = [g for g in held if g["closed_on"] is None]
            marks = sum(g["n"] * g["mid"][t - g["open"]] for g in held)
            nav_pre = A + C + marks
            ret[t], event[t] = (nav_pre / nav - 1 if nav > 0 else 0.0), True
            wiped = nav_pre <= WIPED_OUT
            if wiped:  # DEVIATION from the brief (pure premium only): overlay too, if its stock hits 0
                ret[t], A, C, lvl = -1.0, 0.0, 1.0, 1.0
                nav_pre = C + marks
            if wiped or anchors is None or anchor[t]:
                budget = nav_pre
            # rescale to NAV_pre = 1: the state is in units of the last event's NAV_post, so WIPED_OUT above is
            # relative to it; lvl = reported NAV per unit (attrs n)
            A, C, lvl, budget = A / nav_pre, C / nav_pre, lvl * nav_pre, budget / nav_pre
            for g in held:
                g["n"] /= nav_pre
            reopen = _open_legs(px, [(t, g["leg"], g["close"]) for g in rest], offset_bdays, **kw) if monetize else []
            legs += reopen
            new = reopen + due.get(t, [])
            if new:
                S = px["spx"][t]
                u = sum(g["leg"].qty * g["unit_cost"] for g in new)
                if premium:
                    if not u > 0:
                        raise ValueError(f"premium package costs {u:.4f} <= 0 on {mkt.index[t]:%Y-%m-%d}")
                    B = spec.premium_frac * C
                    if spec.budget_base == "anchor":
                        B = max(min(spec.premium_frac * budget, C), 0.0)
                    N, C = B / u, C - B
                else:
                    A, C = (A + C) / (1 + spec.coverage * u / S), 0.0
                    N = spec.coverage * A / S
                    B = N * u
                for g in new:
                    g["n"] = g["leg"].qty * N
                    g["n_out"] = g["n"] * lvl
                paid[t] = B
                held += new
            cover[t] = px["spx"][t] * sum(g["n"] for g in held if g["n"] > 0)
            nav, pos = 1.0, t
            if t == stop:
                break

    legs = [g for g in legs if g["n"]]  # drop N = 0 packages (premium cash all in legs held over the roll)
    for g in legs:  # leg-days priced by a fallback (carried inputs / no chain quote): held or closed, pre-settlement
        e = min(n if g["closed_on"] is None else g["closed_on"] + 1, g["settle"])
        fb[g["open"] : e] += g["fb"][: e - g["open"]]
    out = pd.DataFrame(
        {"ret": ret, "event": event, "premium_paid": paid, "coverage": cover, "fallback_quotes": fb},
        index=mkt.index,
    )
    date = mkt.index
    out.attrs["legs"] = [
        {
            "open_date": date[g["open"]],
            "close_date": date[g["close"]] if g["close"] < n else None,
            "expiry": pd.Timestamp(g["expiry"]),
            "cp": g["leg"].cp,
            "K": g["K"],
            "n": g["n_out"],
            "unit_cost": g["unit_cost"],
            "closed_on": None if g["closed_on"] is None else date[g["closed_on"]],
            "close_reason": g["reason"],
            "settle_px": (float(px["settle"][g["settle"]]) if g["settle"] < n
                          and (g["closed_on"] is None or g["closed_on"] >= g["settle"]) else None),
            "source": g["source"],
        }
        for g in sorted(legs, key=lambda g: g["open"])
    ]
    return out


def tbill_sleeve(mkt: pd.DataFrame) -> pd.Series:
    """T-bills (hold less equity, AQR's baseline): rf_{t-1} * calendar days / 360, first row 0."""
    return cash_returns(mkt["rf"])


def bond_proxy(yield_pct: pd.Series, maturity_years: float = 10.0) -> pd.Series:
    """Daily return of a constant-maturity par bond, the long-Treasury proxy before ETF history.

    Each day hold a par bond with coupon c = y_{t-1} (percent, semiannual, missing yields carried
    forward) and maturity maturity_years M, repriced at today's yield y_t, i = y_t / 200:
    P_t = (c/2) (1 - (1 + i)^{-2M}) / i + 100 (1 + i)^{-2M}  (annuity factor 2M at i = 0),
    ret_t = P_t / 100 - 1 + (c/100) * calendar days / 365 (coupon carry). First row 0.
    5% -> 6% at M = 10 gives P = 92.5613.
    """
    # ponytail: constant maturity, no roll-down or convexity of an aging bond; fine for a 10-y proxy
    y = yield_pct.ffill()
    c, i = y.shift(1), y / 200
    disc = (1 + i) ** (-2 * maturity_years)
    annuity = ((1 - disc) / i).where(i != 0, 2 * maturity_years)  # NaN i stays NaN
    days = y.index.to_series().diff().dt.days
    out = (c / 2 * annuity + 100 * disc) / 100 - 1 + c / 100 * days / 365
    out.iloc[0] = 0.0
    return out


def tsmom_sleeve(spxtr: pd.Series, rf: pd.Series, lookback: int = 252) -> pd.Series:
    """S&P 500 time-series momentum, long/flat, rebalanced monthly.

    At each month end (last index date of a month) the signal is the trailing `lookback`-row
    excess return TR_t / TR_{t-L} - cash_t / cash_{t-L}, cash = cumulative T-bill
    (1 + rf_daily); position for the next month = 1 if positive, else 0 (also 0 before the
    first full lookback). Daily return = rf_daily + pos (r_spx - rf_daily), rf_daily =
    rf_{t-1} days / 360, r_spx = TR pct change (first row 0).
    """
    # ponytail: long/flat; add short leg if trend results hinge on it
    tr = spxtr.ffill()
    rfd = cash_returns(rf.reindex(tr.index).ffill()).fillna(0.0)
    cash = (1 + rfd).cumprod()
    excess = tr / tr.shift(lookback) - cash / cash.shift(lookback)
    month_end = ~tr.index.to_period("M").duplicated(keep="last")
    pos = (excess > 0).astype(float).where(month_end).ffill().shift(1).fillna(0.0)
    return rfd + pos * (tr.pct_change().fillna(0.0) - rfd)

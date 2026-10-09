"""Tests for insureport.sleeves: third Fridays and the roll calendar (Good Friday snap, offsets,
quarterly rolls, months without data, data ending before the roll), the option sleeve in premium
mode (full budget wiped out and refilled each roll, the output columns, the spread cost on the day
after the open, coverage, half budget halving the NAV over 50 rolls with no spurious wipe-out, a
crash paying intrinsic at expiry, a Good Friday expiry settling on Thursday, monetization closing
at the bid and re-striking (also under offset rolls), against the net debit of a spread, the long
legs' cost of a net-credit collar and on a roll day), overlay mode (the put financed by selling
stock at coverage 1 and 0.5, premium paid, a custom underlying, a wiped-out stock restarting),
short legs (opened at the bid, bought back at the ask, the negative cash swept into stock), a
CLL-like collar (quarterly put, monthly short call), a three-tranche ladder spending a fraction of
its cash, coverage counting every long leg held and 0 while none is, the offset and roll-frequency
pass-through (expiries move with offset rolls), rolls before the first usable pricing row skipped,
a NaN input day on a quiet day and on a crash expiry (model inputs carried, spot not) and a
partial-pillar day (not a fallback), the expiry-week return-rounding limit, cash accrual and the
rates, dividends, VIX pillars (VIX1Y on a 13-month put) and smile knobs reaching the pricer, the
term-structure back-fill, a delta-targeted strike, z_cap reaching the quotes and the delta strike,
spec validation, the T-bill sleeve, the par-bond proxy, long/flat time-series momentum and the
build time of a Universa-like sleeve on a 9,000-day panel.

Why: every option-based safe haven in the backtests is one of these return series, so the
accounting (what is paid, when, at which quote) is pinned to values recomputed here from
pricing.quote on small synthetic panels. Pure functions on synthetic inputs, no I/O.
"""

import time

import numpy as np
import pandas as pd
import pytest

from insureport import metrics, pricing, sleeves
from insureport.sleeves import Leg, OptionSpec

S0 = 4000.0
PUT25 = Leg("P", otm=0.25)  # 25% OTM 1-month put held to expiry


def _panel(periods: int = 260, start: str = "2019-01-01", **cols) -> pd.DataFrame:
    """Flat synthetic panel on business days (no holidays): S 4000, every VIX pillar 20, SKEW 130,
    rf = q = 0, spxtr = spx; keyword columns override."""
    df = pd.DataFrame({"spx": S0, "vix": 20.0, "skew": 130.0, "rf": 0.0, "q": 0.0} | cols,
                      index=pd.bdate_range(start, periods=periods))
    for c in ("vix9d", "vix3m", "vix6m", "vix1y"):
        df[c] = df["vix"]
    df["spxtr"] = df["spx"]
    return df


def _quote(leg: dict, t: pd.Timestamp, S: float = S0, vols: tuple = (20.0,) * 5, rf: float = 0.0, q: float = 0.0,
           **kw) -> tuple:
    """(mid, bid, ask) of an attrs["legs"] entry on date t: SKEW 130, VIX pillars `vols` (vix9d, vix,
    vix3m, vix6m[, vix1y]), r = ln(1 + rf), q; kw (m, nu, h, z_cap) passed to pricing.quote."""
    T = (leg["expiry"] - t).days / 365
    return pricing.quote(S, leg["K"], T, pricing.var_swap_vol(T, *vols), 130.0, np.log1p(rf), q, leg["cp"], **kw)


def _nav(df: pd.DataFrame) -> pd.Series:
    """Sleeve NAV (1.0 on the first row; on an event day it is NAV_pre)."""
    return metrics.nav_from_returns(df["ret"])


def _live(legs: list[dict], d: pd.Timestamp) -> list[dict]:
    """Legs held at the close of date d."""
    return [g for g in legs if g["open_date"] <= d and (g["closed_on"] is None or g["closed_on"] > d)]


def test_third_friday_and_roll_dates():
    assert sleeves.third_friday(2019, 4) == pd.Timestamp("2019-04-19")
    assert sleeves.third_friday(2019, 13) == pd.Timestamp("2020-01-17")  # months past 12 roll into the next year
    idx = pd.bdate_range("2019-03-01", "2019-06-28").drop(pd.Timestamp("2019-04-19"))  # Good Friday
    ev = sleeves.roll_dates(idx)
    assert ev.tolist() == pd.to_datetime(["2019-03-15", "2019-04-18", "2019-05-17", "2019-06-21"]).tolist()
    shifted = sleeves.roll_dates(idx, offset_bdays=5)
    assert (idx.get_indexer(shifted) == idx.get_indexer(ev) + 5).all()
    assert sleeves.roll_dates(idx, every_months=3).tolist() == [ev[0], ev[3]]
    # no April dates: no April roll (never snapped back into March); data ending before June's
    # third Friday: no June roll yet
    gap = idx[(idx.month != 4) & (idx < "2019-06-20")]
    assert sleeves.roll_dates(gap).tolist() == [ev[0], ev[2]]


def test_premium_full_budget_is_wiped_out_and_refilled_each_roll():
    mkt = _panel()
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)))
    ev = sleeves.roll_dates(mkt.index)
    assert df.index.equals(mkt.index) and df.index[df["event"]].equals(ev)
    assert list(df.columns) == ["ret", "event", "premium_paid", "coverage", "fallback_quotes"]
    assert df["event"].dtype == bool and df["fallback_quotes"].dtype.kind == "i"
    assert df.at[ev[0], "ret"] == 0.0 and (df.loc[ev[1:], "ret"] == -1.0).all()
    assert (df.loc[ev, "premium_paid"] == 1.0).all() and (df.loc[~df["event"], "premium_paid"] == 0).all()
    legs = df.attrs["legs"]
    assert [g["open_date"] for g in legs] == ev.tolist()
    assert [g["close_reason"] for g in legs] == ["expiry"] * (len(ev) - 1) + [None]
    for g in legs:  # the spread is paid on the day after the open: n = 1 / ask, NAV = n * mid
        t = g["open_date"]
        nxt = mkt.index[mkt.index.get_loc(t) + 1]
        ask = _quote(g, t)[2]
        assert g["K"] == 0.75 * S0 and g["unit_cost"] == pytest.approx(ask, rel=1e-12)
        assert df.at[t, "coverage"] == pytest.approx(S0 / ask, rel=1e-12)  # N S / NAV_post, N = 1 / ask
        assert df.at[nxt, "ret"] == pytest.approx(_quote(g, nxt)[0] / ask - 1, abs=1e-9)


def test_premium_half_budget_halves_nav_each_roll():
    # 50 rolls: NAV_pre falls to 0.5^49 ~ 1.8e-15, below WIPED_OUT, but never by 1 - 1e-12 since the last roll
    df = sleeves.option_sleeve(_panel(periods=1100), OptionSpec((PUT25,), premium_frac=0.5))
    pre = _nav(df)[df["event"]].to_numpy()  # NAV_pre on each roll
    assert pre[-1] < sleeves.WIPED_OUT and (df["ret"] > -1).all()
    np.testing.assert_allclose(pre[1:] / pre[:-1], 0.5, rtol=1e-9)


def test_crash_put_pays_intrinsic_at_expiry():
    mkt = _panel(periods=60)
    mkt.loc[mkt.index > "2019-01-25", ["spx", "spxtr"]] = 0.6 * S0  # after the 01-18 open, through expiry
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)))
    first = df.attrs["legs"][0]
    expiry = pd.Timestamp("2019-02-15")
    assert first["open_date"] == pd.Timestamp("2019-01-18") and first["K"] == 0.75 * S0
    assert first["expiry"] == expiry and first["closed_on"] == expiry and first["close_reason"] == "expiry"
    pre = _nav(df)[expiry]
    assert pre == pytest.approx(0.0 + first["n"] * 0.15 * S0, rel=1e-12)  # C = 0 at f = 1, rf = 0
    cost = first["n"] * first["unit_cost"]  # the whole unit NAV
    assert cost == pytest.approx(1.0, rel=1e-12)
    assert pre / cost == pytest.approx(0.15 * S0 / first["unit_cost"], rel=1e-12)


def test_good_friday_expiry_settles_on_thursday():
    mkt = _panel(periods=60, start="2019-03-01").drop(pd.Timestamp("2019-04-19"))  # Good Friday
    mkt.loc[mkt.index >= "2019-04-01", ["spx", "spxtr"]] = 2800.0
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)))
    g, thu = df.attrs["legs"][0], pd.Timestamp("2019-04-18")
    assert g["expiry"] == pd.Timestamp("2019-04-19") and g["closed_on"] == thu and g["close_reason"] == "expiry"
    assert _nav(df)[thu] == pytest.approx(g["n"] * (3000 - 2800), rel=1e-12)  # intrinsic on Thursday's close, C = 0


def test_monetize_sells_at_the_bid_and_restrikes():
    mkt = _panel(periods=60)
    hit = pd.Timestamp("2019-01-31")
    mkt.loc[mkt.index >= hit, ["spx", "spxtr"]] = 3200.0
    spec = OptionSpec((Leg("P", otm=0.25, tenor_months=2, hold_months=1),), monetize_mult=2.0)
    df = sleeves.option_sleeve(mkt, spec)
    assert df.index[df["event"]].tolist() == pd.to_datetime(["2019-01-18", "2019-01-31", "2019-02-15", "2019-03-15"]).tolist()
    old, new = [g for g in df.attrs["legs"] if hit in (g["open_date"], g["closed_on"])]
    assert old["open_date"] == pd.Timestamp("2019-01-18") and old["close_reason"] == "monetize"
    assert _quote(old, hit, 3200.0)[0] >= 2 * old["unit_cost"] > _quote(old, hit - pd.Timedelta(days=1))[0]
    assert _nav(df)[hit] == pytest.approx(old["n"] * _quote(old, hit, 3200.0)[1], rel=1e-12)  # C = 0: the bid proceeds
    # fresh package: strike off that day's S and expiry from that month, closed at the next scheduled roll
    assert new["K"] == 0.75 * 3200.0 and new["expiry"] == pd.Timestamp("2019-03-15")
    assert new["close_date"] == old["close_date"] == new["closed_on"] == pd.Timestamp("2019-02-15")
    # offset rolls (02-01, 03-01, 03-29): a 02-12 re-strike reads its month 10 rows earlier (01-29, January),
    # so it expires on March's shifted roll, after its close
    mkt.loc[mkt.index >= hit, ["spx", "spxtr"]] = S0
    mkt.loc[mkt.index >= "2019-02-12", ["spx", "spxtr"]] = 3200.0
    new = next(g for g in sleeves.option_sleeve(mkt, spec, offset_bdays=10).attrs["legs"]
               if g["open_date"] == pd.Timestamp("2019-02-12"))
    assert new["expiry"] == pd.Timestamp("2019-03-29") > new["close_date"] == pd.Timestamp("2019-03-01")


def test_monetize_put_spread_against_the_net_debit():
    vols = (35.0,) * 4
    mkt = _panel(periods=60, vix=35.0)
    k = np.maximum(np.arange(60) - mkt.index.get_loc(pd.Timestamp("2019-01-18")), 0)
    mkt["spx"] = S0 * 0.995**k  # a 0.5%-a-day slide from the 01-18 open
    spec = OptionSpec((Leg("P", otm=0.10, tenor_months=2, hold_months=1),
                       Leg("P", otm=0.20, tenor_months=2, hold_months=1, qty=-1.0)), monetize_mult=2.0)
    df = sleeves.option_sleeve(mkt, spec)
    long, short = df.attrs["legs"][:2]
    assert long["n"] * long["unit_cost"] + short["n"] * short["unit_cost"] == pytest.approx(1.0, rel=1e-12)  # net debit
    days = mkt.index[(mkt.index > long["open_date"]) & (mkt.index < pd.Timestamp("2019-02-15"))]
    mark = np.array([long["n"] * _quote(long, d, mkt.at[d, "spx"], vols)[0] for d in days])
    hit = days[np.argmax(mark >= 2.0)]  # the first day the long leg is worth 2x the net debit
    assert mark.max() < 2 * long["n"] * long["unit_cost"]  # 2x the long leg's own cost is never reached
    assert df.index[df["event"]][1] == hit and long["closed_on"] == short["closed_on"] == hit
    assert long["close_reason"] == short["close_reason"] == "monetize"


def test_monetize_net_credit_collar_against_the_long_legs_cost():
    mkt = _panel(periods=60)
    k = np.maximum(np.arange(60) - mkt.index.get_loc(pd.Timestamp("2019-01-18")), 0)
    mkt["spx"] = S0 * 0.99**k  # a 1%-a-day slide from the 01-18 open
    spec = OptionSpec((Leg("P", otm=0.05), Leg("C", otm=0.01, qty=-1.0)), mode="overlay", monetize_mult=3.0)
    df = sleeves.option_sleeve(mkt, spec)
    put, call = df.attrs["legs"][:2]
    assert put["unit_cost"] < call["unit_cost"]  # u = ask_put - bid_call < 0: a net credit
    days = mkt.index[(mkt.index > put["open_date"]) & (mkt.index < pd.Timestamp("2019-02-15"))]
    hit = days[np.argmax([_quote(put, d, mkt.at[d, "spx"])[0] >= 3 * put["unit_cost"] for d in days])]
    assert df.index[df["event"]][1] == hit and put["closed_on"] == call["closed_on"] == hit
    assert put["close_reason"] == call["close_reason"] == "monetize"


def test_roll_day_monetizes_the_legs_held_over_it():
    mkt = _panel(periods=120)
    roll = pd.Timestamp("2019-03-15")
    mkt.loc[mkt.index >= roll, "spx"] = 0.7 * S0  # the crash lands on the third tranche's roll
    spec = OptionSpec((Leg("P", otm=0.25, tenor_months=3),), premium_frac=0.3, monetize_mult=3.0)
    df = sleeves.option_sleeve(mkt, spec)
    legs = df.attrs["legs"]
    assert df.index[df["event"]].equals(sleeves.roll_dates(mkt.index))  # harvested on the roll, not the day after
    cashed = [g for g in legs if g["close_reason"] == "monetize"]
    assert [g["open_date"] for g in cashed] == pd.to_datetime(["2019-01-18", "2019-02-15"]).tolist()
    assert all(g["closed_on"] == roll for g in cashed)
    bids = sum(g["n"] * _quote(g, roll, 0.7 * S0)[1] for g in cashed)
    assert _nav(df)[roll] == pytest.approx(0.7**2 + bids, rel=1e-12)  # cash after two 30% budgets + the bid proceeds
    # one package: the scheduled tranche plus the two re-opened ones, which keep their rolls; B = 0.3 C = 0.3 NAV_pre
    new = [g for g in legs if g["open_date"] == roll]
    assert [g["close_date"] for g in new] == [None, *(g["close_date"] for g in cashed)]  # None: past the data
    assert all(g["K"] == 0.75 * (0.7 * S0) and g["n"] == new[0]["n"] for g in new)
    assert df.at[roll, "premium_paid"] == pytest.approx(0.3, rel=1e-12)


def test_overlay_finances_the_put_by_selling_stock():
    mkt = _panel()
    for cov in (1.0, 0.5):
        df = sleeves.option_sleeve(mkt, OptionSpec((Leg("P", otm=0.05),), mode="overlay", coverage=cov))
        nav, ev, legs = _nav(df), df.index[df["event"]], df.attrs["legs"]
        for a, b, g in zip(ev[:-1], ev[1:], legs):  # the put expires worthless, so NAV_pre shrinks by the financing
            assert nav[b] / nav[a] == pytest.approx(1 / (1 + cov * _quote(g, a)[2] / S0), abs=1e-9)
        for a, g in zip(ev, legs):  # u = ask / S: A = NAV_pre / (1 + cov u), N = cov A / S, premium cov A u / NAV
            u = _quote(g, a)[2] / S0
            assert g["n"] == pytest.approx(cov * nav[a] / (1 + cov * u) / S0, rel=1e-12)
            assert df.at[a, "coverage"] == pytest.approx(cov / (1 + cov * u), abs=1e-12)
            assert df.at[a, "premium_paid"] == pytest.approx(cov * u / (1 + cov * u), abs=1e-12)
        assert (df.loc[: ev[0], "coverage"].iloc[:-1] == 0).all()
    spec = OptionSpec((Leg("P", otm=0.05),), mode="overlay")
    # an underlying that loses everything empties the stock; the put then expires worthless and the next
    # roll restarts from unit capital
    crash = pd.Series(0.0, mkt.index)
    crash.iloc[mkt.index.get_loc(ev[0]) + 3] = -1.0
    dead = sleeves.option_sleeve(mkt, spec, underlying=crash)
    u = _quote(dead.attrs["legs"][1], ev[1])[2] / S0
    assert dead.at[ev[1], "ret"] == -1.0 and dead.at[ev[1], "coverage"] == pytest.approx(1 / (1 + u), abs=1e-12)
    # an underlying earning 1% a day grows the stock between rolls; the put still marks on flat spx
    up = sleeves.option_sleeve(mkt, spec, underlying=pd.Series(0.01, mkt.index))
    nav, g = _nav(up), up.attrs["legs"][1]
    t = g["open_date"]
    stock = nav[t] / (1 + _quote(g, t)[2] / S0)  # A after selling stock for the put
    assert g["n"] == pytest.approx(stock / S0, rel=1e-12)
    for k, d in enumerate(mkt.index[(mkt.index > t) & (mkt.index < ev[2])], start=1):
        assert nav[d] == pytest.approx(stock * 1.01**k + g["n"] * _quote(g, d)[0], rel=1e-12)


def test_short_leg_opens_at_the_bid_and_is_bought_back_at_the_ask():
    mkt = _panel()
    spec = OptionSpec((Leg("P", otm=0.05), Leg("C", otm=0.10, tenor_months=2, hold_months=1, qty=-1.0)), mode="overlay")
    df = sleeves.option_sleeve(mkt, spec)
    (a, b), (put, call) = df.index[df["event"]][:2], df.attrs["legs"][:2]
    bid = _quote(call, a)[1]
    assert put["cp"] == "P" and call["unit_cost"] == pytest.approx(bid, rel=1e-12)
    stock = 1 / (1 + (_quote(put, a)[2] - bid) / S0)  # A after financing u = ask_put - bid_call
    assert call["n"] == -put["n"] == pytest.approx(-stock / S0, rel=1e-12)
    assert call["closed_on"] == b and call["close_reason"] == "roll"  # a month before its expiry
    assert _nav(df)[b] == pytest.approx(stock + call["n"] * _quote(call, b)[2], rel=1e-12)  # the put expired worthless
    # the buy-back left C < 0; it is swept into the stock before financing the next package
    put2, call2 = df.attrs["legs"][2:4]
    u = (_quote(put2, b)[2] - _quote(call2, b)[1]) / S0
    assert put2["n"] == pytest.approx(_nav(df)[b] / (1 + u) / S0, rel=1e-12)


def test_collar_quarterly_put_and_monthly_short_call():
    mkt = _panel()
    spec = OptionSpec(
        (Leg("P", otm=0.05, tenor_months=3, hold_months=3, every_months=3), Leg("C", otm=0.10, qty=-1.0)),
        mode="overlay",
    )
    df = sleeves.option_sleeve(mkt, spec)
    ev, legs = df.index[df["event"]], df.attrs["legs"]
    puts, calls = [g for g in legs if g["cp"] == "P"], [g for g in legs if g["cp"] == "C"]
    assert [g["open_date"] for g in puts] == ev[::3].tolist()
    assert [g["closed_on"] for g in puts[:-1]] == ev[3::3].tolist()
    assert [g["open_date"] for g in calls] == ev.tolist()
    assert [g["closed_on"] for g in calls[:-1]] == ev[1:].tolist()
    assert calls[0]["n"] == -puts[0]["n"] < 0 and calls[0]["K"] == pytest.approx(1.1 * S0, rel=1e-12)
    for d in mkt.index:
        live = [g["cp"] for g in _live(legs, d)]
        assert live.count("P") <= 1 and live.count("C") <= 1
    # a call-only roll: u is the new call's bid alone, the held put stays out of the package
    a = ev[1]
    bid = _quote(calls[1], a)[1]
    assert calls[1]["unit_cost"] == pytest.approx(bid, rel=1e-12)
    stock_and_cash = _nav(df)[a] - puts[0]["n"] * _quote(puts[0], a)[0]
    assert calls[1]["n"] == pytest.approx(-stock_and_cash / (1 - bid / S0) / S0, rel=1e-12)


def test_ladder_holds_three_tranches():
    df = sleeves.option_sleeve(_panel(), OptionSpec((Leg("P", otm=0.25, tenor_months=3),), premium_frac=0.3))
    ev, legs = df.index[df["event"]], df.attrs["legs"]
    assert [len(_live(legs, e)) for e in ev] == [1, 2] + [3] * (len(ev) - 2)
    assert [sum(g["closed_on"] == e for g in legs) for e in ev] == [0, 0, 0] + [1] * (len(ev) - 3)
    for e in ev:  # coverage counts every tranche held, not the newest one
        assert df.at[e, "coverage"] == pytest.approx(S0 * sum(g["n"] for g in _live(legs, e)) / _nav(df)[e], rel=1e-12)
    # the budget is 30% of CASH, not NAV: worthless expiries and rf = 0 leave 0.7^k cash before roll k
    budget = df.loc[ev, "premium_paid"] * _nav(df)[ev]
    np.testing.assert_allclose(budget, 0.3 * 0.7 ** np.arange(len(ev)), rtol=1e-12)


def test_coverage_counts_the_put_held_over_a_roll_with_no_budget():
    df = sleeves.option_sleeve(_panel(), OptionSpec((Leg("P", otm=0.10, tenor_months=2),)))  # f = 1, to expiry
    ev, legs = df.index[df["event"]], df.attrs["legs"]
    # every other roll all the NAV is in the put bought the roll before: nothing opens (no zero-size leg),
    # and coverage is the held put's notional, n S / (n mid) = S / mid
    assert [g["open_date"] for g in legs] == ev[::2].tolist()
    for a, g in zip(ev[1::2], legs):
        assert df.at[a, "premium_paid"] == 0 and df.at[a, "coverage"] == pytest.approx(S0 / _quote(g, a)[0], rel=1e-12)


def test_coverage_is_zero_while_no_legs_are_open():
    df = sleeves.option_sleeve(_panel(), OptionSpec((Leg("P", otm=0.25, every_months=2),)))  # every other roll
    ev, legs = df.index[df["event"]], df.attrs["legs"]
    assert [g["open_date"] for g in legs] == ev[::2].tolist()
    for g in legs:  # f = 1 on a restarted unit sleeve: N = 1 / ask
        assert df.at[g["open_date"], "coverage"] == pytest.approx(S0 / _quote(g, g["open_date"])[2], rel=1e-12)
    for a, b in zip(ev[1::2], ev[2::2]):  # the put expired on a and nothing is open until b
        assert (df.loc[a:b, "coverage"].iloc[:-1] == 0).all() and df.at[a, "premium_paid"] == 0


def test_offset_and_roll_frequency_reach_the_calendar():
    mkt = _panel()
    spec = OptionSpec((Leg("P", otm=0.05),), mode="overlay")  # PPUT-like: tenor = hold = 1
    for k in (5, 10, 15):
        df = sleeves.option_sleeve(mkt, spec, offset_bdays=k)
        ev, legs = df.index[df["event"]], df.attrs["legs"]
        assert ev.equals(sleeves.roll_dates(mkt.index, offset_bdays=k))
        # the expiries move with the rolls: each leg is held to its expiry, the next roll
        assert [g["expiry"] for g in legs[:-1]] == [g["closed_on"] for g in legs[:-1]] == ev[1:].tolist()
        assert {g["close_reason"] for g in legs[:-1]} == {"expiry"}
    two = sleeves.option_sleeve(mkt, OptionSpec((Leg("P", otm=0.25, tenor_months=2),), roll_every_months=2))
    ev = two.index[two["event"]]
    assert ev.equals(sleeves.roll_dates(mkt.index, every_months=2))
    assert [g["closed_on"] for g in two.attrs["legs"][:-1]] == ev[1:].tolist()


def test_rolls_before_the_first_usable_inputs_are_skipped():
    mkt = _panel()
    mkt.loc[mkt.index < "2019-02-20", [*sleeves._VOLS, "skew"]] = np.nan
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)))
    # the 01-18 and 02-15 rolls have no model inputs at or before them (none borrowed from later rows)
    assert df.index[df["event"]][0] == pd.Timestamp("2019-03-15") == df.attrs["legs"][0]["open_date"]
    assert (df.loc[:"2019-03-14", "ret"] == 0).all() and (df["fallback_quotes"] == 0).all()  # all cash, rf = 0


def test_nan_input_day_carries_the_model_inputs_not_the_spot():
    mkt = _panel(periods=60)
    bad, roll = pd.Timestamp("2019-01-29"), pd.Timestamp("2019-02-15")  # mid-segment, and the first expiry's roll
    mkt.loc[[bad, roll], "skew"] = np.nan
    mkt.loc["2019-02-05", "vix9d"] = np.nan  # a partial-pillar day: var_swap_vol uses the rest, not a fallback
    mkt.loc[bad, "spx"] = 3800.0
    # in the money before the roll (a worthless mark would round the day's ret to -1 and lose the NAV),
    # then down again on the NaN roll day: yesterday's spot would settle at 200, today's at 600
    mkt.loc[mkt.index >= "2019-02-11", "spx"] = 0.7 * S0
    mkt.loc[mkt.index >= roll, "spx"] = 0.6 * S0
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)))
    first, second = df.attrs["legs"][:2]
    nav = _nav(df)
    assert nav[bad] == pytest.approx(first["n"] * _quote(first, bad, 3800.0)[0], rel=1e-12)  # today's S, SKEW carried
    assert nav[roll] == pytest.approx(first["n"] * 0.15 * S0, rel=1e-12)  # settled at the roll day's own S (C = 0)
    assert second["K"] == 0.75 * (0.6 * S0)
    assert second["unit_cost"] == pytest.approx(_quote(second, roll, 0.6 * S0)[2], rel=1e-12)
    assert df["fallback_quotes"].tolist() == [int(d in (bad, roll)) for d in mkt.index]  # the new leg on the roll


def test_expiry_week_payoff_is_lost_to_return_rounding():
    # pins the ret-rounding limit (ponytail in option_sleeve): under a flat put wing (finite z_cap) a far-OTM
    # put held to expiry at f = 1 and VIX 12 marks below 1e-16 of the day before in expiry week, so ret rounds
    # to -1 and the crash payoff that follows never reaches the compounded NAV
    mkt = _panel(periods=40, vix=12.0)
    mkt.loc[mkt.index >= "2019-02-14", ["spx", "spxtr"]] = 0.7 * S0
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), m=0.5, z_cap=3.0)
    g = df.attrs["legs"][0]
    assert df.at[pd.Timestamp("2019-02-13"), "ret"] == -1.0  # the mark fell below 1e-16 of the day before
    assert g["closed_on"] == pd.Timestamp("2019-02-15") and g["n"] * (0.75 - 0.7) * S0 > 1000  # settled > 1000x
    assert _nav(df)["2019-02-15"] == 0.0
    # the default linear wing keeps the mark alive into expiry (-30% on 02-13), so the payoff lands: ~600x
    nav = _nav(sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), m=0.5))
    assert nav["2019-02-13"] > 0 and nav["2019-02-15"] > 100


def test_rates_dividends_and_pillars_reach_cash_and_the_pricer():
    vols = (16.0, 20.0, 22.0, 24.0, 26.0)  # vix9d, vix, vix3m, vix6m, vix1y
    mkt = _panel(rf=0.05, q=0.02)
    mkt["vix9d"], mkt["vix3m"], mkt["vix6m"], mkt["vix1y"] = vols[0], vols[2], vols[3], vols[4]
    mkt["spxtr"] = S0 * 1.0001 ** np.arange(len(mkt))
    kw = {"m": 0.3, "nu": 1.1, "h": 0.0}
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,), premium_frac=0.5), **kw)
    (a, b), g = df.index[df["event"]][:2], df.attrs["legs"][0]
    cash = (1 + 0.05 * mkt.index.to_series().diff().dt.days.fillna(0) / 360).cumprod()  # T-bills, actual/360
    nav = _nav(df)
    np.testing.assert_allclose(nav[:a], cash[:a], rtol=1e-12)  # all cash before the first roll
    rates = {"vols": vols, "rf": 0.05, "q": 0.02} | kw
    assert g["unit_cost"] == pytest.approx(_quote(g, a, **rates)[2], rel=1e-12)
    assert g["n"] == pytest.approx(0.5 * cash[a] / g["unit_cost"], rel=1e-12)
    seg = mkt.index[(mkt.index >= a) & (mkt.index < b)]  # half the cash still accrues, plus n mid
    np.testing.assert_allclose(nav[seg], [0.5 * cash[d] + g["n"] * _quote(g, d, **rates)[0] for d in seg], rtol=1e-12)
    # overlay with the default underlying holds spxtr before the first roll
    ov = sleeves.option_sleeve(mkt, OptionSpec((Leg("P", otm=0.05),), mode="overlay"))
    np.testing.assert_allclose(_nav(ov)[:a], mkt["spxtr"][:a] / S0, rtol=1e-12)
    # a 13-month (TAIL-style) put is priced off the 365 d VIX1Y pillar, not flat VIX6M past 184 d
    y = sleeves.option_sleeve(mkt, OptionSpec((Leg("P", otm=0.25, tenor_months=13, hold_months=1),)), **kw)
    g = y.attrs["legs"][0]
    assert (g["expiry"] - a).days > 365
    assert g["unit_cost"] == pytest.approx(_quote(g, a, **rates)[2], rel=1e-12)
    assert g["unit_cost"] > 1.05 * _quote(g, a, **(rates | {"vols": vols[:4]}))[2]


def test_term_structure_is_back_filled_before_pricing():
    mkt = _panel(periods=300)
    mkt["vix"] = mkt["vix3m"] = mkt["vix6m"] = 20 + 2 * np.sin(np.arange(300) / 10)
    mkt["vix9d"] = (0.8 * mkt["vix"]).where(mkt.index > "2019-01-28")  # missing over the first leg's first days
    spec = OptionSpec((PUT25,))
    np.testing.assert_array_equal(sleeves.option_sleeve(mkt, spec)["ret"],
                                  sleeves.option_sleeve(pricing.fill_term_structure(mkt), spec)["ret"])


def test_delta_leg_strike():
    df = sleeves.option_sleeve(_panel(periods=30), OptionSpec((Leg("P", delta=0.05),)))
    g = df.attrs["legs"][0]
    T = (g["expiry"] - g["open_date"]).days / 365
    assert g["K"] == pytest.approx(pricing.strike_for_delta(S0, T, 0.2, 130.0, 0.0, 0.0, 0.05, "P"), rel=1e-12)


def test_z_cap_reaches_quotes_and_delta_strikes():
    mkt = _panel(periods=30)
    g = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), z_cap=2.5).attrs["legs"][0]  # z ~ 5: beyond either cap
    assert g["unit_cost"] == pytest.approx(_quote(g, g["open_date"], z_cap=2.5)[2], rel=1e-12)
    assert g["unit_cost"] != pytest.approx(_quote(g, g["open_date"])[2], rel=1e-3)
    d = sleeves.option_sleeve(mkt, OptionSpec((Leg("P", delta=0.01),)), z_cap=2.5).attrs["legs"][0]
    T = (d["expiry"] - d["open_date"]).days / 365
    K = pricing.strike_for_delta(S0, T, 0.2, 130.0, 0.0, 0.0, 0.01, "P", z_cap=2.5)
    assert d["K"] == pytest.approx(K, rel=1e-12)


@pytest.mark.parametrize(
    "spec",
    [
        OptionSpec((PUT25,), premium_frac=0.0),
        OptionSpec((PUT25,), premium_frac=1.5),
        OptionSpec((PUT25,), mode="hedge"),
        OptionSpec((Leg("P"),)),  # neither otm nor delta
        OptionSpec((Leg("P", otm=0.25, delta=0.05),)),  # both
        OptionSpec(()),
        OptionSpec((Leg("X", otm=0.25),)),
        OptionSpec((Leg("P", otm=0.25, tenor_months=0),)),
        OptionSpec((PUT25,), monetize_mult=1.0),
        OptionSpec((Leg("P", otm=0.05, qty=-1.0),)),  # a premium package must cost > 0
        OptionSpec((Leg("P", otm=0.25, hold_months=2),)),  # held past its expiry
        OptionSpec((Leg("P", otm=0.05),), mode="overlay", roll_every_months=3),  # a 1-month put, quarterly rolls
    ],
)
def test_invalid_spec_raises(spec):
    with pytest.raises(ValueError):
        sleeves.option_sleeve(_panel(periods=30), spec)


def test_tbill_sleeve_accrues_actual_360():
    r = sleeves.tbill_sleeve(_panel(periods=4, start="2019-01-02", rf=0.05))  # Wed, Thu, Fri, Mon
    np.testing.assert_allclose(r, [0, 0.05 / 360, 0.05 / 360, 0.05 * 3 / 360], rtol=0, atol=1e-15)


def test_bond_proxy_carry_and_repricing():
    idx = pd.bdate_range("2019-01-02", periods=4)  # Wed, Thu, Fri, Mon
    flat = sleeves.bond_proxy(pd.Series(5.0, idx))
    np.testing.assert_allclose(flat, [0, 0.05 / 365, 0.05 / 365, 0.05 * 3 / 365], rtol=0, atol=1e-12)
    jump = sleeves.bond_proxy(pd.Series([5.0, 5.0, 6.0, 6.0], idx))
    # 5% par bond, 20 semiannual periods, repriced at 6%: 2.5 (1 - 1.03^-20)/0.03 + 100/1.03^20 = 92.5613
    assert jump.iloc[2] == pytest.approx(0.925613 - 1 + 0.05 / 365, abs=1e-6)
    assert (sleeves.bond_proxy(pd.Series(0.0, idx)) == 0).all()  # zero yield: annuity factor 2M, P = 100


def test_tsmom_long_then_flat():
    idx = pd.bdate_range("2019-01-01", periods=600)  # to 2021-04-19
    tr = pd.Series(100 * np.cumprod(np.where(np.arange(600) < 300, 1.001, 0.999)), idx)  # up 300 days, then down
    rf = pd.Series(np.where(idx < "2020-10-01", 0.10, 0.0), idx)  # a step while flat exposes the one-day lag
    r = sleeves.tsmom_sleeve(tr, rf)
    # the first full 252-row lookback ends 2019-12-19, so the 2019-12-31 month end is the first signal (long).
    # With 10% T-bills the 12-month excess turns negative on 2020-06-11 (TR alone: only on 2020-08-18), so the
    # 2020-06-30 signal is flat
    long = (idx >= "2020-01-01") & (idx <= "2020-06-30")
    np.testing.assert_allclose(r[long], tr.pct_change()[long], rtol=0, atol=1e-15)
    rfd = rf.shift(1) * idx.to_series().diff().dt.days / 360
    np.testing.assert_allclose(r[~long][1:], rfd[~long][1:], rtol=0, atol=1e-15)
    assert r.iloc[0] == 0 and r["2020-10-01"] == pytest.approx(0.10 / 360, abs=1e-15) and r["2020-10-02"] == 0


def test_universa_sleeve_builds_fast():
    rng = np.random.default_rng(7)
    n = 9000
    spx = 350 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    vix = 18 * np.exp(0.3 * np.sin(np.arange(n) / 150) + rng.normal(0, 0.08, n))
    mkt = pd.DataFrame(
        {"spx": spx, "spxtr": spx * np.exp(0.02 * np.arange(n) / 252), "vix": vix, "vix9d": 0.9 * vix,
         "vix3m": 1.05 * vix, "vix6m": 1.1 * vix, "vix1y": 1.15 * vix, "skew": 115 + 20 * rng.random(n),
         "rf": 0.03, "q": 0.02},
        index=pd.bdate_range("1990-01-02", periods=n),
    )
    spec = OptionSpec((Leg("P", otm=0.25, tenor_months=2, hold_months=1),), monetize_mult=5.0)
    t0 = time.perf_counter()
    df = sleeves.option_sleeve(mkt, spec)
    took = time.perf_counter() - t0
    print(f"Universa-like sleeve on {n} days: {took:.2f} s, {df['event'].sum()} events")
    assert took < 5  # target < 2 s; slack for slow machines
    assert df["event"].sum() > len(sleeves.roll_dates(mkt.index))  # some monetizations were timed too
    assert np.isfinite(df["ret"]).all() and (df["ret"] >= -1).all()

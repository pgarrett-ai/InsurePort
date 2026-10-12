"""Tests for insureport.replicate: tracking metrics (identical series, hand values on three
monthly returns sampled at month-end, TE without the months of bad marks, the roll-day /
non-roll-day split, stale prints dropped), Cboe's roll calendar on the Juneteenth 2026 holiday, the
PPUT / CLL / PPUT3M replicas on the 2018-2019 fixture panel (aligned, scaled to the real index on
the first common date, the quarterly put on the Mar/Jun/Sep/Dec cycle), the slice-5 ("naive")
conventions reproduced, the Cboe conventions of all three replicas (listed strikes off the open,
settlement at the open, which alone moves roll days only; no open values, no Cboe replica), the
1990-2015 fill cost per option per roll (both bounds of the window; an ask FILL_ADD over a non-zero
mid), the smile and convention knobs reaching the replica and the table, validate passing conventions
and fill on and calibrate_skew running on replicate's Cboe defaults, the validation table (its pass flags, the run
settings, the CLL bad-mark flag), a neutralised splice and too-short windows, a regression pin of
PPUT tracking at the pricer's defaults (SKEW_MULT 1 on the anchored slope, SKEW_BETA 0.5; TE bound only,
slice 5's |dCAGR| <= 1%/yr bound breached and recorded there), the calibration grid (its default m grid;
smallest |dCAGR|, ties to the lower TE; slope-cap share; nothing after train_end used) and unknown
targets and settings.

Why: the replicas are the pricer's only check against traded-option strategy indices, and the
pricer feeds every option sleeve. Committed fixture CSV plus synthetic series, no I/O.
"""

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insureport import metrics, pricing, replicate, sleeves

FIXTURE = Path(__file__).parent / "fixtures" / "market_2018_2019.csv"
MKT = pd.read_csv(FIXTURE, index_col="date", parse_dates=True)
FIRST_QUARTERLY_ROLL = pd.Timestamp("2018-03-16")  # third Friday of March 2018


def test_tracking_of_a_series_against_itself():
    x, b = MKT["pput"], MKT["spxtr"]  # PPUT - S&P is a non-constant hedge P&L
    t = replicate.tracking(x, x, b, rolls=sleeves.roll_dates(MKT.index))
    assert (t["te"], t["dcagr"], t["dmaxdd"], t["te_roll"], t["te_nonroll"]) == (0, 0, 0, 0, 0)
    assert t["corr"] == pytest.approx(1, abs=1e-12)
    # n_months counts monthly returns: 24 month-ends minus one, since January 2018 has no
    # month-end before it (validate starts a month early, so its n_months = the window's full months)
    assert t["n_months"] == 23
    assert np.isnan(replicate.tracking(x, x, b)["te_roll"])  # no roll dates, no split


def test_tracking_hand_values():
    idx = pd.to_datetime(["2019-01-31", "2019-02-28", "2019-03-29", "2019-04-30"])
    synth = pd.Series([100.0, 102, 101, 104], idx)
    actual = pd.Series([100.0, 101, 102, 103], idx)
    bench = pd.Series([100.0, 103, 100, 105], idx)
    mid = pd.to_datetime(["2019-01-15", "2019-02-15", "2019-03-15", "2019-04-15"])  # off-level mid-month rows
    synth, actual, bench = (pd.concat([x, pd.Series(v, mid)]).sort_index()
                            for x, v in ((synth, 150.0), (actual, 50.0), (bench, 120.0)))
    t = replicate.tracking(synth, actual, bench)  # month-end rows only: the mid-month ones change nothing
    # synth - actual monthly: 0.02 - 0.01, 101/102 - 102/101, 104/101 - 103/102 = 0.01, -0.0197049, 0.0198990
    assert t["te"] == pytest.approx(0.0713981, abs=1e-7)  # std(ddof 1) * sqrt(12)
    assert t["dcagr"] == pytest.approx(1.04 ** (365.25 / 89) - 1.03 ** (365.25 / 89), rel=1e-12)  # 0.045665
    assert t["dmaxdd"] == pytest.approx(101 / 102 - 1, rel=1e-12)  # actual never falls
    # hedge P&Ls s - b = (-0.01, 0.0193223, -0.0202970), a - b = (-0.02, 0.0390272, -0.0401961);
    # corr(synth, actual) without the bench would be -0.2301
    assert t["corr"] == pytest.approx(0.9999858, abs=1e-6)
    assert t["n_months"] == 3 and t["te_ex_bad"] == t["te"]  # no bad marks
    # a bad mark in March drops the March return (Feb-end to Mar-end) from te_ex_bad: std(0.01, 0.0198990) sqrt(12)
    bad = replicate.tracking(synth, actual, bench, bad=pd.DatetimeIndex(["2019-03-15"]))
    assert bad["te_ex_bad"] == pytest.approx(0.0242476, abs=1e-7) and bad["te"] == t["te"]


def test_tracking_splits_roll_days_from_the_rest():
    a, b = MKT["pput"], MKT["spxtr"]
    rolls = sleeves.roll_dates(MKT.index)
    synth = a * (1 + 0.001 * MKT.index.isin(rolls)).cumprod()  # off only on roll days
    t = replicate.tracking(synth, a, b, rolls=rolls)
    assert t["te_nonroll"] == pytest.approx(0, abs=1e-12)
    # roll-day gap (1 + r_a) 1.001 - 1 - r_a = 0.001 (1 + r_a), annualized with sqrt(252)
    assert t["te_roll"] == pytest.approx((0.001 * (1 + a.pct_change()[rolls])).std() * np.sqrt(252), rel=1e-9)


def test_tracking_drops_stale_prints():
    a, b, rolls = MKT["pput"], MKT["spxtr"], sleeves.roll_dates(MKT.index)
    synth = a * (1 + 0.001 * MKT.index.isin(rolls)).cumprod()
    days = pd.to_datetime(["2019-05-31", "2019-06-21"])  # a month-end and a roll day
    stale = a.copy()
    stale[days] = a.shift(1)[days]  # forward-filled, as load_market does over a Cboe gap
    t = replicate.tracking(synth, stale, b, rolls=rolls)
    assert t == replicate.tracking(synth, a.drop(days), b, rolls=rolls)  # same as a missing print
    # the move of the missing roll-day print lands on 2019-06-24, still counted as a roll return
    assert t["te_nonroll"] == pytest.approx(0, abs=1e-12)


def test_roll_dates_snap_juneteenth_to_thursday():
    idx = pd.bdate_range("2026-06-01", "2026-06-30").drop(pd.Timestamp("2026-06-19"))  # Juneteenth, a Friday
    assert sleeves.roll_dates(idx).tolist() == [pd.Timestamp("2026-06-18")]


@pytest.mark.parametrize("which", ["PPUT", "CLL", "PPUT3M"])
def test_replicate_is_aligned_and_scaled_to_the_real_index(which):
    pput3m = (MKT["spxtr"] / 60).where(MKT.index >= "2018-06-01")  # a stand-in PPUT3M from June 2018
    mkt = MKT.assign(pput3m=pput3m)
    s, actual = replicate.replicate(mkt, which), mkt[which.lower()]
    first = (s.notna() & actual.notna()).idxmax()
    assert s.index.equals(MKT.index)
    assert s[first] == pytest.approx(actual[first], rel=1e-12) and s[first:].notna().all()
    # quarterly puts open on the Mar/Jun/Sep/Dec cycle: those replicas start on March 2018's roll
    assert s.first_valid_index() == (MKT.index[0] if which == "PPUT" else FIRST_QUARTERLY_ROLL)
    assert first == {"PPUT": MKT.index[0], "CLL": FIRST_QUARTERLY_ROLL, "PPUT3M": pd.Timestamp("2018-06-01")}[which]


def test_knobs_reach_the_replica_and_the_table():
    kw = {"m": 0.7, "nu": 1.2, "h": 0.05, "z_cap": 2.5, "beta": 1.0}
    for conv in ("cboe", "naive"):  # the fixture is after 2015: the cboe fill is the mid, no add-on
        nav = metrics.nav_from_returns(sleeves.option_sleeve(MKT, replicate.SPECS[conv]["PPUT"], **kw)["ret"])
        assert replicate.replicate(MKT, "PPUT", conventions=conv, **kw).to_numpy() == pytest.approx(
            (nav * MKT["pput"].iloc[0]).to_numpy(), rel=1e-12)
    w = {"PPUT": ("2018-01-01", None)}
    base = replicate.validate(MKT, windows=w).loc["PPUT", ["te", "dcagr"]]
    for k, v in (kw | {"conventions": "naive"}).items():  # each knob alone moves the table
        assert not np.allclose(replicate.validate(MKT, windows=w, **{k: v}).loc["PPUT", ["te", "dcagr"]], base,
                               rtol=1e-9, atol=0)


def test_naive_conventions_reproduce_slice_5():
    """conventions="naive" is the slice-5 replica (strikes 0.95 / 0.90 / 1.10 x the close, continuous,
    settled at the close); the default cboe fill is the mid after 2015. Levels pinned from the pre-5b code
    (slice-5 conventions on the slice-3c linear-wing pricer with the VIX1Y pillar) on this fixture at
    SKEW_MULT 0.3 under the old slope c = m gamma / 6, which is beta = 1 at m = 0.3 GAMMA_REF / (6 SLOPE_REF)
    (slice 3d): the same levels to float noise."""
    old = {"m": 0.3 * pricing.GAMMA_REF / (6 * pricing.SLOPE_REF), "beta": 1.0}
    mkt = MKT.assign(pput3m=(MKT["spxtr"] / 60).where(MKT.index >= "2018-06-01"))
    pins = {
        "PPUT": {"2018-02-09": 844.7875855550132, "2018-12-24": 785.3898217654562, "2019-12-31": 1030.005626387901},
        "CLL": {"2018-06-29": 794.9640789772856, "2018-12-24": 785.3062561400677, "2019-12-31": 1010.3601034011399},
        "PPUT3M": {"2018-06-29": 89.51158345449845, "2018-12-24": 84.75928959033531, "2019-12-31": 113.91488318342678},
    }
    for which, vals in pins.items():
        s = replicate.replicate(mkt, which, conventions="naive", **old)
        assert s[list(vals)].tolist() == pytest.approx(list(vals.values()), rel=1e-12)


@pytest.mark.parametrize("which,unsettled", [("PPUT", 1), ("PPUT3M", 1), ("CLL", 2)])
def test_cboe_conventions_strike_and_settle_at_the_open(which, unsettled):
    # PPUT: the largest 5-point strike strictly below 95% of the roll day's open; PPUT3M (90%) and CLL (put 95%,
    # call 110%): the smallest at or above; every expiring leg settles at its expiry day's open
    mkt = MKT if which == "PPUT" else MKT.loc[FIRST_QUARTERLY_ROLL:]
    legs = sleeves.option_sleeve(mkt, replicate.SPECS["cboe"][which], h=0.0).attrs["legs"]
    for g in legs:
        mult = 1.10 if g["cp"] == "C" else {"PPUT": 0.95, "PPUT3M": 0.90, "CLL": 0.95}[which]
        target = mult * MKT.at[g["open_date"], "spx_open"]
        lo, hi = (target - 5, target) if which == "PPUT" else (target, target + 5)
        assert g["K"] % 5 == 0 and lo <= g["K"] < hi
        assert g["settle_px"] in (None, MKT.loc[: g["expiry"], "spx_open"].iloc[-1])
    # the fixture's opens differ from its closes, so a close-settled leg fails the line above; the last put (and
    # CLL's last call) expire after the data
    assert sum(g["settle_px"] is not None for g in legs) == len(legs) - unsettled > 0
    # no open values (a failed ^GSPC Open fetch): the Cboe replica raises rather than strike and settle at the close
    with pytest.raises(ValueError, match="spx_open"):
        replicate.replicate(MKT.assign(spx_open=np.nan, pput3m=MKT["spxtr"] / 60), which)


def test_settling_at_the_open_moves_roll_days_only():
    # DEVIATION from the brief (cboe differs on roll days only; see the replicate module docstring): settling
    # at the open alone moves the roll days only, the expiring put's payoff changing NAV_pre and the same
    # package being re-sized from it (returns are scale-free). The Cboe PPUT spec against itself settled at
    # the close: same listed strikes off the open, so the marks between rolls match
    cboe = replicate.SPECS["cboe"]["PPUT"]
    a = sleeves.option_sleeve(MKT, cboe, h=0.0)["ret"]
    b = sleeves.option_sleeve(MKT, dataclasses.replace(cboe, settle="close"), h=0.0)["ret"]
    on = MKT.index.isin(sleeves.roll_dates(MKT.index))
    assert (a - b)[~on].abs().max() < 1e-12 and (a - b)[on].abs().max() > 1e-3  # Dec 2018: the put finished ITM


def _flat(start: str, end: str, S: float = 2000.0) -> pd.DataFrame:
    """A flat business-day panel from start to end: S, open, TR and the real levels at S, every VIX
    pillar 0 (each option is worth its intrinsic value, 0 here), SKEW 130, rf = q = 0."""
    idx = pd.bdate_range(start, end)
    df = pd.DataFrame({"spx": S, "spx_open": S, "spxtr": S, "skew": 130.0, "rf": 0.0, "q": 0.0, "pput": S, "cll": S},
                      index=idx)
    for c in sleeves._VOLS:
        df[c] = 0.0
    return df


@pytest.mark.parametrize("start,end,charged", [("2015-09-01", "2016-03-31", 4), ("1989-09-01", "1990-03-31", 3)])
@pytest.mark.parametrize("which,opened", [("PPUT", (1, 1, 1, 1, 1, 1, 1)), ("CLL", (0, 1, 1, 2, 1, 1, 2))])
def test_cboe_fill_costs_030_points_per_option_per_roll_1990_2015(which, opened, start, end, charged):
    """Each 1990-2015 roll lowers the level by FILL_ADD (0.30) points per option opened per index unit
    held (level / S units at zero option value); rolls before 1990 (fill = the 4 pm mark) and after 2015
    (the mid) cost nothing. opened = options opened on the Sep..Mar rolls, `charged` of them inside
    FILL_ADD_DATES (Sep..Dec 2015, Jan..Mar 1990); CLL starts on September's (scaled there, cost unseen)."""
    S, mkt = 2000.0, _flat(start, end)
    mid, cboe = (replicate.replicate(mkt, which, fill=f) for f in ("mid", "cboe"))
    assert (mid.dropna() == S).all() and replicate.FILL_ADD == 0.30  # options at 0: the mid replica is flat
    rolls = sleeves.roll_dates(mkt.index)
    inside = (rolls >= "1990-01-01") & (rolls <= "2015-12-31")
    assert len(rolls) == 7 and inside.sum() == charged
    paid = np.array(opened) * inside
    cut = (-cboe.diff()).reindex(rolls).fillna(0.0)  # CLL's first date has no level before it
    np.testing.assert_allclose(cut, 0.30 * paid * cboe[rolls] / S, rtol=1e-12, atol=1e-12)
    assert (cboe.diff().drop(rolls).dropna() == 0).all()  # no cost off the rolls
    assert mid.iloc[-1] - cboe.iloc[-1] == pytest.approx(0.30 * paid.sum(), rel=1e-3)  # about 1 unit per index


def test_cboe_fill_is_an_ask_030_points_over_a_nonzero_mid(monkeypatch):
    """On a single-leg overlay the add-on is exactly an ask of mid + FILL_ADD (replicate._fill_cost): with the
    fill window over the fixture, the cboe-fill PPUT equals a mid-fill PPUT whose quotes put the ask FILL_ADD
    over the mid, on every day but the rolls (the add-on is charged on the roll, the quoted ask the day after)."""
    monkeypatch.setattr(replicate, "FILL_ADD_DATES", ("2018-01-01", "2019-12-31"))
    cboe, mid = (replicate.replicate(MKT, "PPUT", fill=f) for f in ("cboe", "mid"))
    assert mid.iloc[-1] - cboe.iloc[-1] > 1  # the add-on is not negligible here
    quote = pricing.quote

    def wide(*a, **kw):
        mid = quote(*a, **kw)[0]
        return mid, mid - replicate.FILL_ADD, mid + replicate.FILL_ADD

    monkeypatch.setattr(pricing, "quote", wide)
    ask = replicate.replicate(MKT, "PPUT", fill="mid")
    off = ~MKT.index.isin(sleeves.roll_dates(MKT.index))
    np.testing.assert_allclose(cboe[off], ask[off], rtol=1e-12)


def test_validate_passes_conventions_and_fill_on_and_calibrate_skew_uses_the_cboe_defaults(monkeypatch):
    """The fixture is after 2015, where the cboe fill is the mid, so the calls are recorded: validate passes
    conventions and fill on; calibrate_skew leaves both at replicate's defaults, Cboe conventions and fill,
    the setting the PPUT validation of the committed SKEW_MULT runs under."""
    seen, real = [], replicate.replicate
    monkeypatch.setattr(replicate, "replicate", lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    runs = [("naive", "mid"), ("cboe", "cboe")]
    for conv, fill in runs:
        replicate.validate(MKT, windows={"PPUT": ("2018-01-01", None)}, conventions=conv, fill=fill)
    assert [(kw["conventions"], kw["fill"]) for kw in seen] == runs
    seen.clear()
    replicate.calibrate_skew(MKT, m_grid=[0.3, 0.5], train_end="2019-12-31")
    assert len(seen) == 2 and not any({"conventions", "fill"} & set(kw) for kw in seen)
    p = inspect.signature(real).parameters
    assert p["conventions"].default == "cboe" and p["fill"].default == "cboe"


def test_validate_table_and_flags():
    v = replicate.validate(MKT, windows={"PPUT": ("2018-01-01", None), "CLL": ("2018-01-01", None)})
    assert v.index.tolist() == ["PPUT", "CLL"]
    assert {"window_start", "window_end", "te", "dcagr", "corr", "dmaxdd", "n_months", "te_roll", "te_nonroll",
            "te_ex_bad", "bad_marks", "conventions", "fill", "z_cap", "beta", "te_ok", "dcagr_ok", "corr_ok",
            "passed"} <= set(v.columns)
    assert v[["te_roll", "te_nonroll"]].notna().all().all()
    assert (v["conventions"] == "cboe").all() and (v["fill"] == "cboe").all() and (v["z_cap"] == pricing.Z_CAP).all()
    assert (v["beta"] == pricing.SKEW_BETA).all()
    assert (v["bad_marks"] == "").all() and (v["te_ex_bad"] == v["te"]).all()  # the CLL bad marks are in 2020
    for which, row in v.iterrows():
        tol = replicate.TOLERANCE[which]
        assert row["te_ok"] == (row["te"] <= tol["te"]) and row["corr_ok"] == (row["corr"] >= tol["corr"])
        assert row["dcagr_ok"] == (abs(row["dcagr"]) <= tol["dcagr"])
        assert row["passed"] == (row["te_ok"] and row["dcagr_ok"] and row["corr_ok"])
        assert row["window_end"] == MKT.index[-1]
    # PPUT from its first row (Feb 2018 on); CLL from March 2018's quarterly roll (Apr 2018 on), whose
    # CLL print (2018-03-16) is stale, so the first real common date is the Monday after
    assert v["n_months"].tolist() == [23, 21] and v.at["CLL", "window_start"] == pd.Timestamp("2018-03-19")
    # PPUT3M is validated only when its series is passed in; its roll days are the quarterly ones
    assert "PPUT3M" not in replicate.validate(MKT, windows={"PPUT3M": ("2018-01-01", None)}).index
    pput3m = MKT["spxtr"] / 60
    with_3m = replicate.validate(MKT, windows={"PPUT3M": ("2018-06-01", None)}, pput3m=pput3m)
    assert with_3m.index.tolist() == ["PPUT3M"] and with_3m.at["PPUT3M", "window_start"] == pd.Timestamp("2018-05-31")
    assert (replicate._roll_days(MKT.index, "PPUT3M").month % 3 == 0).all()


@pytest.mark.parametrize("window", [("2030-01-01", None), ("2010-01-01", "2010-12-31"), ("2019-12-02", None)])
def test_validate_rejects_windows_without_three_month_ends(window):
    with pytest.raises(ValueError, match="fewer than 3"):
        replicate.validate(MKT, windows={"PPUT": window})


def test_validate_flags_bad_marks(monkeypatch):
    assert replicate.BAD_MARKS == {"CLL": ("2020-02-28", "2020-03-02")}  # addendum section 13
    w = {"PPUT": ("2018-01-01", None), "CLL": ("2018-01-01", None)}
    clean = replicate.validate(MKT, windows=w, conventions="naive", z_cap=3.0)
    monkeypatch.setitem(replicate.BAD_MARKS, "CLL", ("2019-02-28", "2019-03-04", "2021-01-04"))  # last: off the panel
    v = replicate.validate(MKT, windows=w, conventions="naive", z_cap=3.0)
    assert v.at["CLL", "bad_marks"] == "2019-02-28, 2019-03-04" and v.at["PPUT", "bad_marks"] == ""
    assert v.at["PPUT", "te_ex_bad"] == v.at["PPUT", "te"] and v.at["CLL", "te_ex_bad"] != v.at["CLL", "te"]
    assert v.at["CLL", "te"] == clean.at["CLL", "te"]  # flagged, not removed from te
    assert (v["conventions"] == "naive").all() and (v["z_cap"] == 3.0).all()


def test_validate_neutralises_a_splice(monkeypatch):
    w, d = {"CLL": ("2018-01-01", None)}, "2019-06-03"
    stepped = MKT.assign(cll=MKT["cll"] * np.where(MKT.index < d, 1.25, 1.0))  # a -20% step with no market move
    monkeypatch.setitem(replicate.SPLICES, "CLL", d)
    pd.testing.assert_frame_equal(replicate.validate(stepped, windows=w), replicate.validate(MKT, windows=w))
    monkeypatch.delitem(replicate.SPLICES, "CLL")  # unhandled, the step is a tracking error
    assert replicate.validate(stepped, windows=w).at["CLL", "te"] > replicate.validate(MKT, windows=w).at["CLL", "te"]


def test_pput_tracking_regression_pin():
    """PPUT on the 2018-2019 fixture at the committed defaults (SKEW_MULT 1 on the slope anchored to the observed
    market smile, SKEW_BETA 0.5; not fitted to PPUT), Cboe conventions and fill (h = 0): te = 0.01007,
    dcagr = -0.01378, corr = 0.98868 (23 months). Before slice 3d (c = 0.3 gamma / 6) te 0.01045, dcagr -0.00797,
    corr 0.98568; slice 5 (naive, m 0.3) had te 0.0112, dcagr +0.0092. The anchored slope is steeper than the
    old one below SKEW ~156 (0.191 vs 0.130 at the fixture's median SKEW 126), so the 5% OTM put costs more and
    the replica lags Cboe's PPUT by 0.58 pp/yr more. Loose bound: te within 2x the out-of-sample TE tolerance.
    Slice 5 also bounded |dcagr| <= 1%/yr here; the pinned -1.38% breaches it (live out-of-sample PPUT dCAGR:
    figures/validation_table.md), an open trade-off of the anchored slope, so only the pin holds dcagr."""
    t = replicate.tracking(replicate.replicate(MKT, "PPUT"), MKT["pput"], MKT["spxtr"])
    assert t["te"] <= 2 * replicate.TOLERANCE["PPUT"]["te"]
    assert (t["te"], t["dcagr"], t["corr"]) == pytest.approx((0.01007, -0.01378, 0.98868), abs=1e-5)


def test_calibrate_skew_marks_one_best(monkeypatch):
    """best = the m with the smallest |dcagr| on the training panel, ties to the lower te."""
    assert inspect.signature(replicate.calibrate_skew).parameters["m_grid"].default == (0.5, 0.75, 1.0, 1.25, 1.5)
    g = replicate.calibrate_skew(MKT, m_grid=[0.5, 2.0, 3.0], train_end="2019-12-31")
    assert g.index.tolist() == [0.5, 2.0, 3.0] and g["best"].sum() == 1
    assert {"te", "dcagr", "corr", "cap_share"} <= set(g.columns)
    # the slope cap binds where m 0.263 (gamma / 4.919)^0.5 > c*, c* the 30 d cap at the day's VIX (0.39 .. 0.64
    # on the fixture, median 0.46; SKEW 111 .. 159): never at m 0.5 (nor anywhere on the default grid, up to
    # m 1.5), 109 days at m 2, 386 at m 3
    assert g["cap_share"].tolist() == pytest.approx([0, 109 / len(MKT), 386 / len(MKT)])
    assert g.index[g["best"]].tolist() == [g["dcagr"].abs().idxmin()]
    monkeypatch.setattr(replicate, "replicate", lambda mkt, which, **kw: kw["m"])  # tracking sees m as the synth
    monkeypatch.setattr(replicate, "tracking", lambda m, a, b: {"te": {0.3: 0.02, 0.5: 0.01, 0.7: 0.03}[m],
                                                                "dcagr": {0.3: 0.002, 0.5: -0.002, 0.7: 0.003}[m],
                                                                "corr": 1.0})
    g = replicate.calibrate_skew(MKT, m_grid=[0.3, 0.5, 0.7], train_end="2019-12-31")
    assert g.index[g["best"]].tolist() == [0.5]  # |dcagr| tie between 0.3 and 0.5 goes to the lower te


def test_calibrate_skew_ignores_data_after_train_end():
    cut = MKT.index <= "2018-12-31"
    later = MKT.assign(vix=MKT["vix"].where(cut, 80.0), pput=MKT["pput"].where(cut, 1.0))
    a = replicate.calibrate_skew(MKT, m_grid=[0.3], train_end="2018-12-31")
    pd.testing.assert_frame_equal(a, replicate.calibrate_skew(later, m_grid=[0.3], train_end="2018-12-31"))


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="TAIL"):
        replicate.replicate(MKT, "TAIL")
    with pytest.raises(ValueError, match="pput3m"):  # no real PPUT3M series in the panel to scale to
        replicate.replicate(MKT, "PPUT3M")
    with pytest.raises(ValueError, match="conventions"):
        replicate.replicate(MKT, "PPUT", conventions="am")
    with pytest.raises(ValueError, match="fill"):
        replicate.replicate(MKT, "PPUT", fill="ask")

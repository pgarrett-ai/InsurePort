"""Tests for insureport.replicate: tracking metrics (identical series, hand values on three
monthly returns sampled at month-end, the roll-day / non-roll-day split, stale prints dropped),
Cboe's roll calendar on the Juneteenth 2026 holiday, the PPUT / CLL / PPUT3M replicas on the
2018-2019 fixture panel (aligned, scaled to the real index on the first common date, the quarterly
put on the Mar/Jun/Sep/Dec cycle), the smile and fill knobs reaching the replica and the table,
the validation table, its pass flags, a neutralised splice and too-short windows, a regression pin
of PPUT tracking with the committed pricing.SKEW_MULT, the calibration grid (lowest TE within the
PPUT CAGR tolerance, else lowest TE, ties to the smaller |dCAGR|; slope-cap share; nothing after
train_end used) and unknown targets.

Why: the replicas are the pricer's only check against traded-option strategy indices, and the
calibrated SKEW_MULT feeds every option sleeve. Committed fixture CSV plus synthetic series, no I/O.
"""

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
    assert t["n_months"] == 3


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
    kw = {"m": 0.7, "nu": 1.2, "h": 0.05, "z_cap": 2.5}
    nav = metrics.nav_from_returns(sleeves.option_sleeve(MKT, replicate.PPUT_SPEC, **kw)["ret"])
    assert replicate.replicate(MKT, "PPUT", **kw).to_numpy() == pytest.approx(
        (nav * MKT["pput"].iloc[0]).to_numpy(), rel=1e-12)
    w = {"PPUT": ("2018-01-01", None)}
    base = replicate.validate(MKT, windows=w).loc["PPUT", ["te", "dcagr"]]
    for k, v in kw.items():  # each knob alone moves the table
        assert not np.allclose(replicate.validate(MKT, windows=w, **{k: v}).loc["PPUT", ["te", "dcagr"]], base,
                               rtol=1e-9, atol=0)


def test_validate_table_and_flags():
    v = replicate.validate(MKT, windows={"PPUT": ("2018-01-01", None), "CLL": ("2018-01-01", None)})
    assert v.index.tolist() == ["PPUT", "CLL"]
    assert {"window_start", "window_end", "te", "dcagr", "corr", "dmaxdd", "n_months", "te_roll", "te_nonroll",
            "te_ok", "dcagr_ok", "corr_ok", "passed"} <= set(v.columns)
    assert v[["te_roll", "te_nonroll"]].notna().all().all()
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


def test_validate_neutralises_a_splice(monkeypatch):
    w, d = {"CLL": ("2018-01-01", None)}, "2019-06-03"
    stepped = MKT.assign(cll=MKT["cll"] * np.where(MKT.index < d, 1.25, 1.0))  # a -20% step with no market move
    monkeypatch.setitem(replicate.SPLICES, "CLL", d)
    pd.testing.assert_frame_equal(replicate.validate(stepped, windows=w), replicate.validate(MKT, windows=w))
    monkeypatch.delitem(replicate.SPLICES, "CLL")  # unhandled, the step is a tracking error
    assert replicate.validate(stepped, windows=w).at["CLL", "te"] > replicate.validate(MKT, windows=w).at["CLL", "te"]


def test_pput_tracking_regression_pin():
    """PPUT on the 2018-2019 fixture at SKEW_MULT = 0.3 (h = 0): te = 0.0112, dcagr = +0.0092,
    corr = 0.986 (23 months). Pinned loosely with the committed pricing.SKEW_MULT: te within 2x
    the out-of-sample TE tolerance, |dcagr| <= 1%/yr. The other selection rules miss the pin:
    dcagr +0.0149 at m 0.2 (argmin |dcagr|), -0.0295 at m 1.0 (argmin te alone)."""
    t = replicate.tracking(replicate.replicate(MKT, "PPUT"), MKT["pput"], MKT["spxtr"])
    assert t["te"] <= 2 * replicate.TOLERANCE["PPUT"]["te"] and abs(t["dcagr"]) <= 0.01


def test_calibrate_skew_marks_one_best(monkeypatch):
    """best = the lowest te among the m within the PPUT CAGR tolerance, the lowest te overall when
    no m is within it, ties to the smaller |dcagr|."""
    g = replicate.calibrate_skew(MKT, m_grid=[0.3, 0.5, 0.7], train_end="2019-12-31")
    assert g.index.tolist() == [0.3, 0.5, 0.7] and g["best"].sum() == 1
    assert {"te", "dcagr", "corr", "cap_share"} <= set(g.columns)
    # the slope cap binds above SKEW 100 + 60 c_max / m: 155.70 at m 0.7, passed on 2018-08-13 only (159.03)
    assert g["cap_share"].tolist() == pytest.approx([0, 0, 1 / len(MKT)])
    ok = g[g["dcagr"].abs() <= replicate.TOLERANCE["PPUT"]["dcagr"]]
    assert g["te"].idxmin() not in ok.index  # the fixture separates the two rules
    assert g.loc[g["best"], "te"].iloc[0] == ok["te"].min()
    monkeypatch.setitem(replicate.TOLERANCE, "PPUT", {**replicate.TOLERANCE["PPUT"], "dcagr": 0.0})
    g = replicate.calibrate_skew(MKT, m_grid=[0.3, 0.5, 0.7], train_end="2019-12-31")
    assert g.loc[g["best"], "te"].iloc[0] == g["te"].min() and g["best"].sum() == 1
    monkeypatch.setattr(replicate, "replicate", lambda mkt, which, m, h: m)  # tracking sees m as the synth
    monkeypatch.setattr(replicate, "tracking", lambda m, a, b: {"te": 0.01, "dcagr": {0.3: 0.004, 0.5: -0.002,
                                                                                        0.7: 0.003}[m], "corr": 1.0})
    g = replicate.calibrate_skew(MKT, m_grid=[0.3, 0.5, 0.7], train_end="2019-12-31")
    assert g.index[g["best"]].tolist() == [0.5]  # a three-way te tie


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

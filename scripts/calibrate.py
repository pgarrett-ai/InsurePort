"""Calibrate pricing.SKEW_MULT on Cboe PPUT (1990-2015) and validate the synthetic option pricer
out of sample against Cboe PPUT, PPUT3M and CLL (network required).

Usage, from the repo root: .venv/Scripts/python scripts/calibrate.py
Prints the m grid and the chosen m (with the share of training days where the slope cap binds),
the validation table with PASS / FAIL per target and the roll-day vs non-roll-day TE split for
z_cap in {inf, 2.5, 3, 4} (inf, the default linear wing pricing.Z_CAP, is the gate; the finite caps
are the flat-wing sensitivity rows), two informational rows (PPUT3M 2004-2015 inside the training
period, CLL from 2009-10-19, after Cboe's 2009-10-16 splice), the SKEW_MULT line to paste into pricing.py and a
markdown table for METHODOLOGY.md. A row whose window starts before its target's launch
(replicate.LAUNCH) is labelled as holding Cboe back-test levels. Writes
figures/calibration_pput, replication_pput and replication_cll as PNG (kaleido installed) or
HTML. Edits no source file.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from insureport import pricing
from insureport import replicate as rp
from insureport.data import cboe, load_market

FIGURES = Path(__file__).resolve().parents[1] / "figures"
Z_CAPS = (pricing.Z_CAP, 2.5, 3.0, 4.0)  # the default (inf, the gate) first, then the flat-wing sensitivity
PNG = importlib.util.find_spec("kaleido") is not None
COLS = ["window_start", "window_end", "n_months", "te", "dcagr", "corr", "dmaxdd", "te_roll", "te_nonroll", "verdict"]


def save(fig: go.Figure, name: str) -> None:
    """Write figures/<name>.png with kaleido, else figures/<name>.html."""
    FIGURES.mkdir(exist_ok=True)
    if PNG:
        path = FIGURES / f"{name}.png"
        fig.write_image(path, width=1100, height=550, scale=2)
    else:
        path = FIGURES / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
    print(f"wrote {path.relative_to(FIGURES.parent)}")


def mark(fig: go.Figure, date: str, text: str) -> None:
    """Dashed vertical line at `date` with a label on top (add_vline's own label breaks on dates)."""
    fig.add_vline(x=pd.Timestamp(date), line_dash="dot", line_color="gray")
    fig.add_annotation(x=pd.Timestamp(date), y=1.0, yref="paper", text=text, showarrow=False, xanchor="left",
                       yanchor="top", textangle=-90, font_size=11)


def level_figure(mkt: pd.DataFrame, which: str, m: float, marks: dict[str, str]) -> go.Figure:
    """Synthetic vs real index level from their first common date, log scale."""
    synth, actual = rp.replicate(mkt, which, m=m), mkt[which.lower()]
    first = (synth.notna() & actual.notna()).idxmax()
    fig = go.Figure()
    fig.add_scatter(x=actual.index[actual.index >= first], y=actual[first:], name=f"Cboe {which}")
    fig.add_scatter(x=synth.index[synth.index >= first], y=synth[first:], name=f"synthetic (m = {m:g}, mid fills)")
    fig.update_yaxes(type="log", title="index level (log)")
    fig.update_layout(title=f"{which}: synthetic replica vs Cboe index", template="plotly_white", legend_x=0.01)
    for date, text in marks.items():
        mark(fig, date, text)
    return fig


def verdicts(v: pd.DataFrame) -> pd.DataFrame:
    """The table with a PASS / FAIL column naming the failed checks."""
    fails = v[["te_ok", "dcagr_ok", "corr_ok"]].apply(lambda r: [c[:-3] for c, ok in r.items() if not ok], axis=1)
    return v.assign(verdict=["PASS" if not f else "FAIL (" + ", ".join(f) + ")" for f in fails])


def labelled(v: pd.DataFrame) -> pd.DataFrame:
    """The table with each row name (target first) followed by '(Cboe back-test before <launch>)' when
    its window starts before the target's launch."""
    names = {}
    for name, r in v.iterrows():
        launch = pd.Timestamp(rp.LAUNCH[name.split()[0]])
        names[name] = name + (f" (Cboe back-test before {launch:%Y-%m-%d})" if r["window_start"] < launch else "")
    return v.rename(index=names)


def show(v: pd.DataFrame) -> str:
    """Printable table: back-test periods labelled, dates as days, rates in %."""
    out = labelled(v)[COLS].copy()
    for c in ("window_start", "window_end"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    for c in ("te", "dcagr", "dmaxdd", "te_roll", "te_nonroll"):
        out[c] = (100 * out[c]).map("{:+.2f}%".format)
    out["corr"] = out["corr"].map("{:.3f}".format)
    return out.to_string()


mkt = load_market("1990-01-01")
print(f"panel {mkt.index[0]:%Y-%m-%d} .. {mkt.index[-1]:%Y-%m-%d}, {len(mkt)} rows")

grid = rp.calibrate_skew(mkt)
m = float(grid.index[grid["best"]][0])
print(f"\nSKEW_MULT grid, PPUT 1990 .. {rp.TRAIN_END} (h = 0, nu = 1, z_cap = {pricing.Z_CAP:g}; "
      f"Cboe back-test levels before PPUT's {rp.LAUNCH['PPUT']} launch):")
print(grid.to_string(float_format=lambda x: f"{x:.5f}"))
RULE = (f"lowest TE among m with |dCAGR| <= {rp.TOLERANCE['PPUT']['dcagr']:.2%} (ties to the smaller |dCAGR|); "
        "TE falls with m, so this is the largest m inside the CAGR tolerance")
print(f"chosen m = {m:g}: {RULE}; the slope cap binds on {grid.at[m, 'cap_share']:.2%} of training days at that m")
for rule, alt in (("lowest TE alone", grid["te"].idxmin()),
                  ("lowest |dCAGR|", grid["dcagr"].abs().idxmin())):
    print(f"  {rule} would pick m = {alt:g}: TE {grid.at[alt, 'te']:.2%}, dCAGR {grid.at[alt, 'dcagr']:+.2%}")

pput3m = cboe("PPUT3M")
for z in Z_CAPS:
    v = verdicts(rp.validate(mkt, m=m, z_cap=z, pput3m=pput3m))
    if z == pricing.Z_CAP:
        gate = v
    print(f"\nValidation, m = {m:g}, z_cap = {z:g}, out-of-sample windows {rp.WINDOWS}:")
    print(show(v))
in_sample = rp.validate(mkt, m=m, windows={"PPUT3M": ("2004-03-19", rp.TRAIN_END)}, pput3m=pput3m)
post_splice = rp.validate(mkt, m=m, windows={"CLL": ("2009-10-19", None)})
extra = verdicts(pd.concat([in_sample.rename(index={"PPUT3M": "PPUT3M 2004-2015 (in-sample, further OTM)"}),
                            post_splice.rename(index={"CLL": f"CLL after the {rp.SPLICES['CLL']} splice"})]))
print(f"\nInformational rows (z_cap = {pricing.Z_CAP:g}):")
print(show(extra))

print("\nPaste into src/insureport/pricing.py:")
print(f"SKEW_MULT: float = {m:g}  # calibrated on PPUT 1990-{rp.TRAIN_END[:4]} "
      "(scripts/calibrate.py; replicate.calibrate_skew)")

print(f"\nMarkdown for METHODOLOGY.md (z_cap = {pricing.Z_CAP:g}, h = 0 mid fills, m = {m:g}):\n")
print("| Target | Window | Months | TE | dCAGR | Hedge-P&L corr | dMaxDD | TE roll / non-roll days "
      "| Tolerance (TE, abs dCAGR, corr) | Result |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, r in labelled(pd.concat([gate, extra])).iterrows():
    tol = rp.TOLERANCE[name.split()[0]]
    print(f"| {name} | {r['window_start']:%Y-%m} .. {r['window_end']:%Y-%m} | {r['n_months']} | {r['te']:.2%} | "
          f"{r['dcagr']:+.2%} | {r['corr']:.3f} | {r['dmaxdd']:+.2%} | {r['te_roll']:.2%} / {r['te_nonroll']:.2%} | "
          f"{tol['te']:.1%}, {tol['dcagr']:.2%}, {tol['corr']:.2f} | {r['verdict']} |")
print(f"\nSKEW_MULT = {m:g}, calibrated on Cboe PPUT 1990-{rp.TRAIN_END[:4]} (back-test levels before "
      f"{rp.LAUNCH['PPUT']}): {RULE}.")

if not PNG:
    print("\nkaleido is not installed: figures are written as HTML, not PNG")
fig = go.Figure()
fig.add_scatter(x=grid.index, y=100 * grid["te"], name="TE (%/yr)", mode="lines+markers")
fig.add_scatter(x=grid.index, y=100 * grid["dcagr"], name="dCAGR, synthetic - Cboe (%/yr)", mode="lines+markers")
fig.add_vline(x=m, line_dash="dot", line_color="gray")
fig.update_layout(title=f"PPUT replication 1990-{rp.TRAIN_END[:4]} vs SKEW_MULT (chosen m = {m:g}; "
                        f"Cboe back-test levels before {rp.LAUNCH['PPUT']})",
                  xaxis_title="SKEW_MULT m", yaxis_title="%/yr", template="plotly_white")
save(fig, "calibration_pput")
pput_marks = {rp.TRAIN_END: "train | test", rp.LAUNCH["PPUT"]: "launch: earlier = Cboe back-test"}
save(level_figure(mkt, "PPUT", m, pput_marks), "replication_pput")
save(level_figure(mkt, "CLL", m, {rp.LAUNCH["CLL"]: "launch", rp.SPLICES["CLL"]: "Cboe CLL splice (-19.8%)",
                                  rp.TRAIN_END: "PPUT calibration ends"}), "replication_cll")

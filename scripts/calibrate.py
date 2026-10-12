"""Validate the synthetic option pricer out of sample against Cboe PPUT, PPUT3M and CLL, and show for
information how PPUT 1990-2015 would scale pricing.SKEW_MULT (network required).

Usage, from the repo root: .venv/Scripts/python scripts/calibrate.py
SKEW_MULT m is a scale on the put-wing slope anchored to the observed SPX smile (pricing.skew_slope:
c = m SLOPE_REF (gamma / GAMMA_REF)^SKEW_BETA), committed at 1 and not fitted (RULE below).
Prints the panel's load_market warnings (an spx_open with no data stops the run: sleeves.option_sleeve
refuses the Cboe conventions without it), the m grid and the committed m and beta (with the share of
training days where the slope cap binds), all under the Cboe replica conventions and fill
(replicate.SPECS["cboe"], fill "cboe"); a stale-open sensitivity (the share of 1990-2015 PPUT roll
days whose ^GSPC open is the prior close, and the PPUT replica's training TE / dCAGR at the committed m
with those opens masked, so those rolls strike and settle at their own close, next to the unmasked
ones: disclosed, not adopted); the validation table with PASS / FAIL per target, TE with and
without the CLL bad-mark months, the roll-day vs non-roll-day TE split and the conventions, fill and
z_cap of each row, for z_cap in {inf, 3} (inf, the default linear wing pricing.Z_CAP, is the gate)
plus the 2.5 / 4 flat-wing sensitivity; a before / after table (the
old slope c = 0.3 gamma / 6, i.e. beta 1 at m = 0.3 GAMMA_REF / (6 SLOPE_REF): the pre-5b replica, slice 5's
naive conventions and mid fills, and the pre-3d pricer under the Cboe conventions and fill; then the anchored
slope with naive / mid, Cboe / mid and Cboe / Cboe); two informational rows (PPUT3M 2004-2015 inside the
training period, CLL from 2009-10-19, after Cboe's 2009-10-16 splice); and a markdown table for
METHODOLOGY.md, also written to figures/validation_table.md.
A row whose window starts before its target's launch (replicate.LAUNCH) is labelled as holding Cboe
back-test levels. Writes figures/calibration_pput, replication_pput and
replication_cll as PNG (plotly's write_image, or kaleido 0.2.1's own PlotlyScope, which plotly >= 6
refuses in write_image, fed plotly's bundled plotly.js so its base64 arrays decode), as HTML only if
both fail. Edits no source file.
"""

from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go

from insureport import pricing
from insureport import replicate as rp
from insureport.data import cboe, load_market

FIGURES = Path(__file__).resolve().parents[1] / "figures"
Z_CAPS = (pricing.Z_CAP, 3.0, 2.5, 4.0)  # the gate (inf) and z_cap 3 make the final table; 2.5 and 4: sensitivity
COLS = ["conventions", "fill", "z_cap", "window_start", "window_end", "n_months", "te", "te_ex_bad", "dcagr", "corr",
        "dmaxdd", "te_roll", "te_nonroll", "bad_marks", "verdict"]
OLD = {"m": 0.3 * pricing.GAMMA_REF / (6 * pricing.SLOPE_REF), "beta": 1.0}  # c = 0.3 gamma / 6: the slope before 3d


def save(fig: go.Figure, name: str) -> None:
    """Write figures/<name>.png (write_image, else kaleido 0.2.1's PlotlyScope), or figures/<name>.html if both
    raise; the other format's file from an earlier run is removed."""
    FIGURES.mkdir(exist_ok=True)
    path, size = FIGURES / f"{name}.png", {"width": 1100, "height": 550, "scale": 2}
    try:
        try:
            fig.write_image(path, **size)
        except RuntimeError:  # plotly >= 6 wants kaleido >= 1; the installed 0.2.1 still renders through its own API,
            from kaleido.scopes.plotly import PlotlyScope  # given plotly's plotly.js (its own predates base64 arrays)

            js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
            path.write_bytes(PlotlyScope(plotlyjs=str(js)).transform(fig, format="png", **size))
    except Exception as e:  # noqa: BLE001 - any kaleido / Chromium failure falls back to HTML
        print(f"PNG export failed ({e}); writing HTML")
        path = FIGURES / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
    path.with_suffix(".html" if path.suffix == ".png" else ".png").unlink(missing_ok=True)
    print(f"wrote {path.relative_to(FIGURES.parent)}")


def mark(fig: go.Figure, date: str, text: str) -> None:
    """Dashed vertical line at `date` with a label on top (add_vline's own label breaks on dates)."""
    fig.add_vline(x=pd.Timestamp(date), line_dash="dot", line_color="gray")
    fig.add_annotation(x=pd.Timestamp(date), y=1.0, yref="paper", text=text, showarrow=False, xanchor="left",
                       yanchor="top", textangle=-90, font_size=11)


def level_figure(mkt: pd.DataFrame, which: str, m: float, marks: dict[str, str]) -> go.Figure:
    """Synthetic (Cboe conventions and fill, SKEW_BETA) vs real index level from their first common date, log scale."""
    synth, actual = rp.replicate(mkt, which, m=m), mkt[which.lower()]
    first = (synth.notna() & actual.notna()).idxmax()
    fig = go.Figure()
    fig.add_scatter(x=actual.index[actual.index >= first], y=actual[first:], name=f"Cboe {which}")
    fig.add_scatter(x=synth.index[synth.index >= first], y=synth[first:],
                    name=f"synthetic (m = {m:g}, beta = {pricing.SKEW_BETA:g}, Cboe conventions)")
    fig.update_yaxes(type="log", title="index level (log)")
    fig.update_layout(title=f"{which}: synthetic replica vs Cboe index", template="plotly_white",
                      legend={"orientation": "h", "y": -0.15})
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


def show(v: pd.DataFrame, cols: list[str] = COLS) -> str:
    """Printable table: back-test periods labelled, dates as days, rates in %."""
    out = labelled(v)[cols].copy()
    for c in ("window_start", "window_end"):
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    for c in ("te", "te_ex_bad", "dcagr", "dmaxdd", "te_roll", "te_nonroll"):
        out[c] = (100 * out[c]).map("{:+.3f}%".format)
    out["corr"] = out["corr"].map("{:.3f}".format)
    return out.to_string()


mkt = load_market("1990-01-01")
print(f"panel {mkt.index[0]:%Y-%m-%d} .. {mkt.index[-1]:%Y-%m-%d}, {len(mkt)} rows")
for w in mkt.attrs["warnings"]:
    print(f"  warning: {w}")

m, beta = pricing.SKEW_MULT, pricing.SKEW_BETA  # a scale on the anchored slope, not fitted to PPUT's level (RULE)
grid = rp.calibrate_skew(mkt, m_grid=sorted({*rp.M_GRID, m}))  # m on the grid: the rows below look it up
print(f"\nSKEW_MULT grid (a scale on the anchored slope, beta = {beta:g}), PPUT 1990 .. {rp.TRAIN_END} (Cboe "
      f"conventions and fill, h = 0, nu = 1, z_cap = {pricing.Z_CAP:g}; Cboe back-test levels before PPUT's "
      f"{rp.LAUNCH['PPUT']} launch) - for information:")
print(grid.to_string(float_format=lambda x: f"{x:.5f}"))
RULE = (f"m scales the put-wing slope anchored to the observed SPX smile, c = m {pricing.SLOPE_REF:g} "
        f"(gamma / {pricing.GAMMA_REF:g})^beta with gamma = (SKEW - 100) / 10: the slope of the fit IV/vs = 0.968 + "
        "0.263 z across four expiries at SKEW 149.19 (docs/research/gap-3-q9 A), reproduced exactly there at m = 1. "
        f"beta = {beta:g} damps the slope's SKEW sensitivity (0: a constant wing, 1: the proportional Gram-Charlier "
        "rule, as in the earlier proportional rule c = 0.3 gamma / 6); one chain cannot identify it, so beta is a "
        "disclosed assumption, swept over {0, 0.5, 1} in the case study. PPUT's level is NOT used to fit m: fitting "
        "it moved the old rule's m from 0.2 to 0.05 depending on fill and settlement conventions that closing data "
        "cannot identify, so the Cboe replicas validate the pricer and do not calibrate it")
print(f"committed m = {m:g}, beta = {beta:g}: {RULE}; the slope cap binds on {grid.at[m, 'cap_share']:.2%} of "
      "training days")
for rule, alt in (("lowest TE alone", grid["te"].idxmin()),
                  ("lowest |dCAGR| (calibrate_skew's best)", grid.index[grid["best"]][0])):
    print(f"  {rule} would pick m = {alt:g}: TE {grid.at[alt, 'te']:.2%}, dCAGR {grid.at[alt, 'dcagr']:+.2%}")

# Yahoo's ^GSPC open is mostly the prior close before ~2005 (data.load_market): on those rolls the "open" is
# the previous close. Masked, the rolls fall back to their own close (sleeves._inputs); reported, not adopted
stale = mkt["spx_open"] == mkt["spx"].shift()
rolls = rp._roll_days(mkt.index, "PPUT")
train, later = rolls[rolls <= rp.TRAIN_END], rolls[rolls > rp.TRAIN_END]
masked = rp.calibrate_skew(mkt.assign(spx_open=mkt["spx_open"].mask(stale)), m_grid=(m,))
STALE = (f"On {int(stale[train].sum())} of {len(train)} PPUT roll days 1990-{rp.TRAIN_END[:4]} Yahoo's ^GSPC open "
         "equals the previous close (a stale print), so those rolls strike and settle at the previous day's close "
         f"({int(stale[later].sum())} of the {len(later)} later ones). With those opens masked (the rolls strike and "
         f"settle at their own close) the PPUT replica at m = {m:g} has training TE {masked.at[m, 'te']:.2%}, dCAGR "
         f"{masked.at[m, 'dcagr']:+.2%} (unmasked: TE {grid.at[m, 'te']:.2%}, dCAGR {grid.at[m, 'dcagr']:+.2%}). m is "
         "not fitted, so masking moves no parameter; the stale opens are kept and disclosed rather than masked.")
print(f"\nStale-open sensitivity: {STALE}")

pput3m = cboe("PPUT3M")
sweep = pd.concat([verdicts(rp.validate(mkt, m=m, z_cap=z, pput3m=pput3m)) for z in Z_CAPS])
gate = sweep[sweep["z_cap"].isin(Z_CAPS[:2])]
print(f"\nValidation, m = {m:g}, beta = {beta:g}, out-of-sample windows {rp.WINDOWS} (gate: z_cap = {pricing.Z_CAP:g}; "
      f"te_ex_bad = TE without the months of {rp.BAD_MARKS}):")
print(show(gate))
print("\nFlat-wing sensitivity (z_cap 2.5, 4):")
print(show(sweep[~sweep["z_cap"].isin(Z_CAPS[:2])]))

before_after = pd.concat([
    *(verdicts(rp.validate(mkt, conventions=c, fill=f, pput3m=pput3m, **kw)).assign(slope=name)
      for c, f, kw, name in (("naive", "mid", OLD, "0.3 gamma/6"), ("cboe", "cboe", OLD, "0.3 gamma/6"),
                             ("naive", "mid", {}, "anchored"), ("cboe", "mid", {}, "anchored"))),
    sweep[sweep["z_cap"] == pricing.Z_CAP].assign(slope="anchored"),  # cboe / cboe: the gate rows above
])
print(f"\nBefore / after (z_cap = {pricing.Z_CAP:g}; the old slope 0.3 gamma/6 with naive / mid is the pre-5b replica, "
      f"slice-5 conventions on the slice-3c pricer, and with cboe / cboe the pre-3d pricer; anchored = m {m:g}, "
      f"beta {beta:g}):")
print(show(before_after, ["slope", *COLS]))

in_sample = rp.validate(mkt, m=m, windows={"PPUT3M": ("2004-03-19", rp.TRAIN_END)}, pput3m=pput3m)
post_splice = rp.validate(mkt, m=m, windows={"CLL": ("2009-10-19", None)})
extra = verdicts(pd.concat([in_sample.rename(index={"PPUT3M": "PPUT3M 2004-2015 (in-sample, further OTM)"}),
                            post_splice.rename(index={"CLL": f"CLL after the {rp.SPLICES['CLL']} splice"})]))
print(f"\nInformational rows (z_cap = {pricing.Z_CAP:g}):")
print(show(extra))

md = [f"Validation at SKEW_MULT = {m:g}, SKEW_BETA = {beta:g} ({RULE}; Cboe PPUT 1990-{rp.TRAIN_END[:4]} holds "
      f"back-test levels before {rp.LAUNCH['PPUT']}). Cboe conventions (listed strikes off the open, settlement at "
      "the open) and "
      f"fill (closing mid, plus {rp.FILL_ADD:.2f} index points per option per roll on 1990-2015 rolls), h = 0, "
      f"nu = 1. {STALE} TE ex bad months drops the months of the CLL bad marks {', '.join(rp.BAD_MARKS['CLL'])}; "
      "PASS / FAIL uses TE with them in. Window = the first and last month of monthly returns.",
      "",
      "| Target | Window | Months | z_cap | Conventions / fill | TE | TE ex bad months | dCAGR | Hedge-P&L corr "
      "| dMaxDD | TE roll / non-roll days | Tolerance (TE, abs dCAGR, corr) | Result |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, r in labelled(pd.concat([gate, extra])).iterrows():
    tol = rp.TOLERANCE[name.split()[0]]
    first = r["window_start"].to_period("M") + 1  # window_start is the anchor month-end; returns start a month later
    md.append(f"| {name} | {first} .. {r['window_end']:%Y-%m} | {r['n_months']} | {r['z_cap']:g} | "
              f"{r['conventions']} / {r['fill']} | {r['te']:.3%} | {r['te_ex_bad']:.3%} | {r['dcagr']:+.3%} | "
              f"{r['corr']:.3f} | {r['dmaxdd']:+.2%} | {r['te_roll']:.2%} / {r['te_nonroll']:.2%} | "
              f"{tol['te']:.1%}, {tol['dcagr']:.2%}, {tol['corr']:.2f} | {r['verdict']} |")
print("\nMarkdown for METHODOLOGY.md:\n")
print("\n".join(md))
FIGURES.mkdir(exist_ok=True)
(FIGURES / "validation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\nwrote figures/validation_table.md")

fig = go.Figure()
fig.add_scatter(x=grid.index, y=100 * grid["te"], name="TE (%/yr)", mode="lines+markers")
fig.add_scatter(x=grid.index, y=100 * grid["dcagr"], name="dCAGR, synthetic - Cboe (%/yr)", mode="lines+markers")
fig.add_vline(x=m, line_dash="dot", line_color="gray")
fig.update_layout(title=f"PPUT replication 1990-{rp.TRAIN_END[:4]} vs SKEW_MULT (committed m = {m:g}, beta = "
                        f"{beta:g})<br><sup>m scales the anchored slope; Cboe conventions and fill; Cboe back-test "
                        f"levels before {rp.LAUNCH['PPUT']}</sup>",
                  xaxis_title="SKEW_MULT m (scale on the anchored slope)", yaxis_title="%/yr", template="plotly_white")
save(fig, "calibration_pput")
pput_marks = {rp.TRAIN_END: "validation window starts (m not fitted)",
              rp.LAUNCH["PPUT"]: "launch: earlier = Cboe back-test"}
save(level_figure(mkt, "PPUT", m, pput_marks), "replication_pput")
save(level_figure(mkt, "CLL", m, {rp.LAUNCH["CLL"]: "launch", rp.SPLICES["CLL"]: "Cboe CLL splice (-19.8%)",
                                  rp.BAD_MARKS["CLL"][0]: "bad put marks"}),
     "replication_cll")

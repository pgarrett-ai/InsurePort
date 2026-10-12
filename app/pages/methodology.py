"""Methodology page: METHODOLOGY.md, the pricer's validation table against Cboe PPUT / PPUT3M / CLL with
the replication charts, the data coverage of the market panel and the disclosures."""

import numpy as np
import pandas as pd
import streamlit as st

import common
from insureport import charts, replicate

SOURCES = {"spx": "Yahoo ^GSPC close", "spx_open": "Yahoo ^GSPC open", "spxtr": "Yahoo ^SP500TR", "vix": "Cboe VIX",
           "vix9d": "Cboe VIX9D", "vix3m": "Yahoo ^VIX3M", "vix6m": "Cboe VIX6M", "vix1y": "Cboe VIX1Y",
           "skew": "Cboe SKEW", "rf": "FRED DTB3", "q": "derived: S&P 500 TR vs price, trailing year",
           "pput": "Cboe PPUT", "cll": "Cboe CLL", "vxth": "Cboe VXTH"}
PASS_FAIL = {"PASS": charts.STATUS["cost-effective (robust)"], "FAIL": charts.STATUS["not cost-effective"]}
DISCLOSURES = """
- **Model-priced options are not traded prices.** Every option result rests on the VIX / SKEW surface; the PPUT
  family validates only the near-money wing, the far wing rests on the SKEW slope assumption. The sidebar's m, nu
  and h and the Stress lab's sensitivity sweep show how much the answers move; a real end-of-day chain
  (data/chains/) replaces the model where it has quotes.
- **In-sample choices.** w* is picked on the same history it is scored on; the bootstrap, rolling windows, decades
  and roll offsets are the guardrails, not proof. A pass is never "proven".
- **Path dependence.** Roll timing alone moves option results (the roll-offset sweep).
- **Short histories** (TAIL 2017, DBMF 2019) are reported as insufficient history; the current holdings carry
  survivorship.
- **Not modelled:** taxes, margin, contract granularity, capacity, financing beyond rf + 50 bp.
- **Data:** Yahoo Finance (personal-use terms), Cboe and FRED, fetched fail-soft and cached; S&P and Cboe index
  names are trademarks of their owners. Universa / Spitznagel figures are self-reported and hypothetical.
- Not investment advice. Not affiliated with Universa Investments, Cboe or Cambria. GPL-3.0.
"""

st.title("Methodology")
path = common.ROOT / "METHODOLOGY.md"
if path.exists():
    st.markdown(path.read_text(encoding="utf-8"))
else:
    st.info("METHODOLOGY.md is not in this checkout yet: the full method write-up appears here once it is.")

st.header("Pricer validation")
s, mkt = common.settings(), common.market()
try:
    with st.spinner("Replicating Cboe PPUT, PPUT3M and CLL with the model pricer..."):
        v = common.validation_cached(s["m"], s["nu"], 0.0)  # Cboe's fill convention, h = 0 (the sidebar's banner)
except ValueError as e:  # e.g. an empty Cboe series after a failed first fetch
    st.warning(f"Validation not available: {e}")
else:
    shown = v[["window_start", "window_end", "n_months", "te", "dcagr", "corr", "dmaxdd", "te_roll", "te_nonroll"]]
    st.dataframe(shown.assign(result=np.where(v["passed"], "PASS", "FAIL")).style
                 .format({"window_start": "{:%Y-%m-%d}", "window_end": "{:%Y-%m-%d}", "te": "{:.2%}", "dcagr": "{:+.2%}",
                          "corr": "{:.3f}", "dmaxdd": "{:+.1%}", "te_roll": "{:.2%}", "te_nonroll": "{:.2%}"},
                         na_rep="n/a")
                 .map(lambda r: f"background-color: {PASS_FAIL[r]}33", subset=["result"]))
    st.caption("Out-of-sample windows; pass when tracking error, |dCAGR| and hedge-P&L correlation are all within "
               + "; ".join(f"{k}: TE <= {t['te']:.1%}, |dCAGR| <= {t['dcagr']:.2%}, corr >= {t['corr']:.2f}"
                           for k, t in replicate.TOLERANCE.items())
               + f". At m = {s['m']:g}, nu = {s['nu']:g}, h = 0, on Cboe's fill convention: the closing mid plus "
               f"{replicate.FILL_ADD:g} index points per option per roll through 2015, the mid after.")
for which in ("PPUT", "PPUT3M", "CLL"):
    actual = common.pput3m() if which == "PPUT3M" else mkt[which.lower()]
    try:
        with st.spinner(f"Replicating {which}..."):
            synth = common.replica_cached(which, s["m"], s["nu"])
    except ValueError as e:
        st.caption(f"{which}: no replica ({e})")
        continue
    actual = replicate.unsplice(actual, synth, which)  # Cboe's CLL file break, neutralised as the validation does
    st.plotly_chart(charts.replication_chart(synth, actual, replicate.WINDOWS[which][0], which),
                    key=f"replica:{which}")
common.note(f"Cboe index levels before each launch ({', '.join(f'{k} {d}' for k, d in replicate.LAUNCH.items())}) "
            "are Cboe back-tests, hatched in the charts. Shaded: the dates before each target's validation window. "
            + "; ".join(f"Cboe's {k} file steps on {d} with no market move behind it: the levels from then on are "
                        "rescaled so that day's return is the replica's, as in the validation"
                        for k, d in replicate.SPLICES.items()) + ".")

st.header("Data coverage")
first = mkt.attrs.get("first_dates", {})
coverage = pd.DataFrame({"source": [SOURCES.get(k, "") for k in first], "first date in the panel": list(first.values()),
                         "before it": ["regression-filled from the VIX term structure"
                                       if mkt[k].first_valid_index() != d else "" for k, d in first.items()]},
                        index=pd.Index(list(first), name="series"))
st.dataframe(coverage.style.format({"first date in the panel": lambda d: "none" if pd.isna(d) else f"{d:%Y-%m-%d}"}))
st.caption("First dates are those of the market panel, which starts in 1990: a series with older history shows its "
           "first 1990 date. Regression-filled: pricing.fill_term_structure back-filled it before that date.")
for w in mkt.attrs.get("warnings", []):
    st.caption(f"Panel note: {w}")

st.header("Disclosures")
st.markdown(DISCLOSURES)

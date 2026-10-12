"""Diagnostics page: the loaded portfolio through Spitznagel's lens (metrics.diagnose): CAGR vs the
arithmetic mean, the volatility tax, loss-based risk, betas, crash participation, concentration and the
index-put hedge notional."""

import streamlit as st

import common
from insureport import charts, metrics, portfolio

st.title("Diagnostics")
p, core = common.require_portfolio()
s, mkt = common.settings(), common.market()
spx_r = mkt["spxtr"].pct_change()
with st.spinner("Diagnosing the portfolio..."):
    d = metrics.diagnose(core, spx_r, p, mkt)
st.caption(f"Sample {d['start']:%Y-%m-%d} to {d['end']:%Y-%m-%d} ({d['years']:.1f} years), against the S&P 500 "
           "total return over the same days. Risk here is loss: drawdowns, time under water, crash participation.")

nav = metrics.nav_from_returns(core.loc[d["start"] : d["end"]])
years = metrics.full_year_returns(nav)  # diagnose's mean and tax are over these full calendar years
c = st.columns(4)
c[0].metric("CAGR", f"{d['cagr']:.2%}", f"{(d['cagr'] - d['spx_cagr']) * 1e4:+.0f} bp vs S&P 500")
c[1].metric("Arithmetic mean", "n/a" if years.empty else f"{d['arith']:.2%}",
            help="Mean full-calendar-year return: what an average year pays, not what the portfolio compounds.")
c[2].metric("Volatility tax", "n/a" if years.empty else f"{d['vol_tax']:.2%}")
c[3].metric("Max drawdown", f"{d['max_dd']:.1%}")
c = st.columns(4)
c[0].metric("Time under water", f"{d['tuw']:.0%}", help="Share of days below the running peak.")
c[1].metric("Longest recovery", f"{d['longest_recovery_days']:,} days")
c[2].metric("Beta", f"{d['beta']:.2f}")
c[3].metric("Downside beta", f"{d['downside_beta']:.2f}", help="Beta on the S&P 500's down days only.")
c = st.columns(4)
c[0].metric("S&P 500 CAGR", f"{d['spx_cagr']:.2%}")
c[1].metric("S&P 500 max drawdown", f"{d['spx_max_dd']:.1%}")
if years.empty:
    st.caption("No full calendar year in the sample: the arithmetic mean and the volatility tax need one.")
else:
    st.caption(f"Over the {len(years)} full calendar years {years.index[0]}-{years.index[-1]}: volatility tax = "
               f"arithmetic mean - geometric mean = {d['vol_tax']:.2%}/yr; the lognormal approximation sigma^2 / 2 "
               f"gives {d['vol_tax_approx']:.2%}/yr. The partial first and last years are left out, so the geometric "
               "mean is the CAGR over those years; the CAGR card covers every day of the sample.")

navs = {"Portfolio": nav, "S&P 500 TR": metrics.nav_from_returns(spx_r.loc[d["start"] : d["end"]])}
if s["leverage"]:  # the "hold more equity" comparison: the core levered at rf + 50 bp
    for L in (1.25, 1.5):
        try:
            levered = portfolio.lever(core.loc[d["start"] :], mkt["rf"], L)
            navs[f"Portfolio levered {L}x"] = metrics.nav_from_returns(levered)
        except ValueError as e:
            st.caption(f"{L}x not shown: {e}")
st.plotly_chart(charts.growth_chart(navs))
st.plotly_chart(charts.underwater_chart(nav))
if not years.empty:
    st.plotly_chart(charts.annual_bars(years, d["arith"], d["arith"] - d["vol_tax"]))

st.subheader("Crash participation")
crashes = d["crashes"]
if crashes.empty:
    st.info("No crash window (metrics.CRASHES) lies wholly inside the sample.")
else:
    shown = crashes.assign(participation=crashes["ret"] / crashes["bench_ret"])
    st.dataframe(shown.style.format({"start": "{:%Y-%m-%d}", "end": "{:%Y-%m-%d}", "ret": "{:.1%}",
                                     "bench_ret": "{:.1%}", "participation": "{:.0%}"}))
    st.plotly_chart(charts.crash_table_fig(crashes))

st.subheader("Concentration")
c = st.columns(4)
c[0].metric("HHI", f"{d['hhi']:.3f}")
c[1].metric("Effective N", f"{d['eff_n']:.1f}")
c[2].metric("Top holding", f"{d['top1']:.0%}")
c[3].metric("Top 5", f"{d['top5']:.0%}")
st.caption("Of the non-cash weights. Index puts hedge beta, not single-name risk: a concentrated book keeps its own "
           "crashes.")

st.subheader("Hedge notional")
hedge = d["hedge"]
c = st.columns(3)
c[0].metric("Hedge notional", f"${hedge['notional']:,.0f}")
c[1].metric("SPX contracts", f"{hedge['spx_contracts']:.2f}")
c[2].metric("XSP contracts", f"{hedge['xsp_contracts']:.1f}")
floored = " (floored at 0: a negative downside beta needs no index puts)" if d["downside_beta"] < 0 else ""
st.caption(f"notional = value x downside beta = ${p.value:,.0f} x {max(d['downside_beta'], 0.0):.2f}{floored}; one SPX "
           f"contract covers 100 x the index ({mkt['spx'].dropna().iloc[-1]:,.0f}), XSP a tenth of that. Fractional: "
           "round to whole contracts.")

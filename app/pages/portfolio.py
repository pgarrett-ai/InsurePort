"""Portfolio page: holdings from a sample, the editor or a CSV upload (weights or shares), fetched prices,
and the core return series built into session state (portfolio, core_r, proxied)."""

from dataclasses import replace

import pandas as pd
import streamlit as st

import common
from insureport import portfolio

st.title("Portfolio")
st.write("Holdings as weights (fractions or percent) or as share counts; a `CASH` row is cash. Edit the table, "
         "then **Build** to fetch prices and compute the portfolio's daily returns.")

presets = sorted(p.name for p in common.SAMPLES.glob("*.csv"))
preset = st.selectbox("Sample portfolio", presets, key="preset")
upload = st.file_uploader("...or upload a holdings CSV: a `ticker` column plus `weight` or `shares`, at most 1 MB",
                          type="csv")
try:
    text = common.read_upload(upload) if upload is not None else (common.SAMPLES / preset).read_text(encoding="utf-8")
    holdings = portfolio.parse_holdings(text)
except ValueError as e:
    st.error(f"Holdings not read: {e}")
    st.stop()
shares = "shares" in holdings
edited = st.data_editor(holdings, num_rows="dynamic", hide_index=True,
                        key=f"editor:{upload.name if upload is not None else preset}")

c1, c2, c3 = st.columns(3)
if shares:  # one key per unit: a dollar amount must not carry over as a weight (or back) when the preset changes
    cash = c1.number_input("Extra cash ($)", min_value=0.0, value=0.0, step=1_000.0, key="cash_usd")
else:
    cash = c1.number_input("Extra cash (weight, as a fraction like the table's)", min_value=0.0, value=0.0, step=0.01,
                           key="cash_w", help="Added to the holdings' weights, then all are rescaled to 100%: "
                                              "0.10 next to weights summing to 1 gives 0.10 / 1.10 = 9.1% cash.")
value = c2.number_input("Portfolio value ($, sizes the hedge notional)", min_value=1.0, value=1_000_000.0,
                        step=10_000.0, key="value", disabled=shares,
                        help="Shares mode values the holdings at their last prices instead.")
proxy = c3.toggle("Fill short histories with the S&P 500 TR", value=False, key="proxy",
                  help="Tickers listed after the S&P 500 TR's first date take its returns before their first price.")

if st.button("Build", type="primary", key="build"):
    try:  # the editor's rows pass the same validation as an upload
        h = portfolio.parse_holdings(edited.dropna(how="all").to_csv(index=False))
    except ValueError as e:
        st.error(f"Holdings not valid: {e}")
        st.stop()
    tickers = tuple(sorted(t for t in h["ticker"] if t != "CASH"))
    mkt = common.market()
    px = common.prices_cached(tickers) if tickers else pd.DataFrame()
    unknown = [t for t in tickers if t not in px.columns]
    if unknown:
        st.error(f"Unknown tickers (no price data): {', '.join(unknown)}. Fix or remove them and build again.")
        st.stop()
    try:
        p = portfolio.to_portfolio(h, px.ffill().iloc[-1] if len(px) else None, cash)
        p = p if p.value is not None else replace(p, value=value)
        with st.spinner("Computing the portfolio's daily returns..."):
            core, proxied = portfolio.core_returns(p, px, mkt["rf"], spxtr=mkt["spxtr"], proxy_fill=proxy)
    except (ValueError, KeyError) as e:
        st.error(f"Portfolio not built: {e}")
        st.stop()
    st.session_state.update(portfolio=p, core_r=core, proxied=proxied)
    for k in ("evaluated", "evaluated_on", "sensitivity"):  # results of the previous portfolio
        st.session_state.pop(k, None)

if "portfolio" in st.session_state:
    p, core = st.session_state["portfolio"], st.session_state["core_r"]
    st.subheader("Loaded portfolio")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample start", f"{core.index[0]:%Y-%m-%d}")
    c2.metric("Sample end", f"{core.index[-1]:%Y-%m-%d}")
    c3.metric("Value", f"${p.value:,.0f}")
    weights = pd.Series(p.weights, name="weight").rename_axis("ticker").sort_values(ascending=False).to_frame()
    st.dataframe(weights.style.format("{:.1%}"))
    if st.session_state["proxied"]:
        common.note(f"S&P 500 TR returns stand in for {', '.join(st.session_state['proxied'])} before their first "
                    "price (proxy fill): those years are back-filled, not the holdings' own history.")
    st.caption("Next: **Diagnostics** for its crash exposure, **Candidates** to test safe havens.")

"""Candidates page: the shortlist for this portfolio (catalog.suggest), per-candidate parameters, and
Evaluate -> the ranking (catalog.rank) with both scorecards, the three denominators of option sleeves,
the cost-effectiveness plane, payoff scatters and the Universa known-answer table."""

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import common
from insureport import catalog, charts, data, metrics, scenarios
from insureport.catalog import CANDIDATES, REBALANCE, Candidate

PCT = ["w_star", "cagr_unhedged", "cagr_star", "maxdd_unhedged", "maxdd_star", "p_beat_10y", "rolling10_win_rate",
       "calm_year_bleed", "crash_payoff_mean", "critic_sleeve_cagr", "critic_alpha", "dd_gap_vs_cagr_matched_cash",
       "avg_premium_yr", "sleeve CAGR (on sleeve capital)", "net return on premium", "premium spent (% of NAV / yr)"]
BP = ["d_cagr_bp", "d_cagr_p50_bp", "d_cagr_vs_dd_matched_cash_bp", "portfolio dCAGR (bp/yr)"]
PORTFOLIO_VIEW = ["candidate", "verdict", "w_star", "cagr_unhedged", "cagr_star", "d_cagr_bp", "maxdd_unhedged",
                  "maxdd_star", "p_beat_10y", "d_cagr_p50_bp", "rolling10_win_rate", "d_cagr_vs_dd_matched_cash_bp",
                  "beats_deleverage", "calm_year_bleed", "crash_payoff_mean", "robustness"]
CRITIC_VIEW = ["candidate", "verdict", "critic_sleeve_cagr", "critic_sleeve_sharpe", "critic_alpha", "critic_beta",
               "dd_gap_vs_cagr_matched_cash", "beats_deleverage", "avg_premium_yr"]


def styled(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Percent / bp formats, n/a for NaN, the verdict cell tinted with its status colour (next to its text)."""
    fmt = {c: "{:.2%}" for c in PCT if c in df} | {c: "{:+.0f}" for c in BP if c in df}
    fmt |= {c: "{:.2f}" for c in ("critic_sleeve_sharpe", "critic_beta") if c in df}
    out = df.style.format(fmt, na_rep="n/a")
    if "verdict" in df:
        out = out.map(lambda v: f"background-color: {charts.STATUS.get(v, charts.NEUTRAL)}33", subset=["verdict"])
    return out


def edit(c: Candidate, s: dict, beta_down: float) -> tuple[Candidate, float]:
    """The candidate as edited in its expander (legs' OTM / tenor / hold, premium per roll, monetize multiple,
    rebalance; the sidebar's rebalance overrides) and the overlay coverage (downside beta by default)."""
    cover = common.coverage(beta_down)
    with st.expander(f"{c.name} parameters"):
        st.caption(c.note)
        if s["rebalance"] == "preset":
            reb = st.selectbox("Rebalance", REBALANCE, index=REBALANCE.index(c.rebalance), key=f"{c.key}:rebalance")
        else:
            reb = s["rebalance"]
            st.caption(f"Rebalance: {reb} (the sidebar's setting overrides the preset's {c.rebalance}).")
        if c.spec is None:
            return replace(c, rebalance=reb), cover
        legs = []
        for i, L in enumerate(c.spec.legs):
            col = st.columns(3)
            what = f"Leg {i + 1}: {'long' if L.qty > 0 else 'short'} {'put' if L.cp == 'P' else 'call'}"
            if L.otm is None:
                col[0].caption(f"{what}, strike at |delta| {abs(L.delta):g}")
                otm = None
            else:
                otm = col[0].number_input(f"{what}, OTM (%)", 0.0, 90.0, L.otm * 100, 1.0, key=f"{c.key}:otm{i}") / 100
            tenor = col[1].number_input("Tenor (months)", 1, 24, L.tenor_months, key=f"{c.key}:tenor{i}")
            hold = col[2].number_input("Hold (months)", 1, 24, L.hold_months or L.tenor_months, key=f"{c.key}:hold{i}")
            legs.append(replace(L, otm=otm, tenor_months=int(tenor), hold_months=int(hold)))
        spec, col = replace(c.spec, legs=tuple(legs)), st.columns(2)
        if spec.mode == "premium":
            frac = col[0].number_input("Premium per roll (% of the budget base)", 0.1, 100.0, spec.premium_frac * 100,
                                       key=f"{c.key}:premium")
            spec = replace(spec, premium_frac=frac / 100)
        else:
            cover = col[0].number_input("Coverage (x the core's value)", 0.01, 3.0, cover, key=f"{c.key}:coverage",
                                        help="Defaults to the portfolio's downside beta.")
        mon = col[1].number_input("Monetize at x cost (0 = off; else > 1)", 0.0, 100.0, spec.monetize_mult or 0.0,
                                  key=f"{c.key}:monetize")
        return replace(c, spec=replace(spec, monetize_mult=mon or None), rebalance=reb), cover


st.title("Candidates")
p, core = common.require_portfolio()
s, mkt = common.settings(), common.market()
spx_r = mkt["spxtr"].pct_change()
with st.spinner("Diagnosing the portfolio..."):
    diag = metrics.diagnose(core, spx_r, p, mkt)

st.subheader("Shortlist")
short = catalog.suggest(diag)
st.dataframe(pd.DataFrame([(k, CANDIDATES[k].name, why) for k, why in short], columns=["key", "candidate", "why"]),
             hide_index=True)
picks = st.multiselect("Candidates to evaluate", list(CANDIDATES), default=[k for k, _ in short],
                       format_func=lambda k: CANDIDATES[k].name, key="picks")
edited = [edit(CANDIDATES[k], s, diag["downside_beta"]) for k in picks]

chain_dir = data.data_dir() / "chains"
files = sorted(f.name for f in chain_dir.glob("*") if f.suffix.lower() in (".csv", ".zip")) if chain_dir.is_dir() else []
chain = st.selectbox("Option prices", ["", *files], key="chain",
                     format_func=lambda f: f"real end-of-day chain: {f}" if f else "model-priced (VIX / SKEW surface)",
                     help=f"Vendor CSV / zip files in {chain_dir}; never uploaded through the browser (multi-GB).")
chain_path = str(chain_dir / chain) if chain else None

if st.button("Evaluate", type="primary", key="evaluate", disabled=not edited):
    done, bar = {}, st.progress(0.0)
    for i, (c, cover) in enumerate(edited):
        bar.progress(i / len(edited), text=f"Evaluating {c.name} ({i + 1} of {len(edited)})...")
        args = common.eval_args(c, s, cover, chain_path)
        try:
            row = common.evaluate_cached(core, *args)
        except (ValueError, KeyError) as e:
            st.warning(f"{c.name}: not evaluated ({e})")
            continue
        done[c.key] = {"args": args, "row": row}
    bar.empty()
    st.session_state.update(evaluated=done, evaluated_on=(common.fingerprint(core), dict(s)))

ev = common.evaluated(core)
if ev:
    table = catalog.rank([e["row"] for e in ev.values()])
    st.subheader("Ranking")
    st.caption("From the last Evaluate. w* is each candidate's CAGR-maximizing weight; bp = basis points a year.")
    st.markdown("**Portfolio view** (Spitznagel): does the whole portfolio compound faster with the haven?")
    st.dataframe(styled(table[PORTFOLIO_VIEW]), hide_index=True)
    st.markdown("**Critic view** (AQR): the sleeve on its own, and the divested T-bill twin with the same CAGR.")
    st.dataframe(styled(table[CRITIC_VIEW]), hide_index=True)

    specs = {k: common.candidate_from_tuple(e["args"].params) for k, e in ev.items()}
    opts = {k: c for k, c in specs.items() if c.spec is not None}  # option sleeves and the TAIL proxy
    if opts:
        st.markdown("**Three denominators** (option sleeves): the same P&L on the portfolio, on the sleeve's capital "
                    "and on the premium spent. Only the first is the objective; the premium is a small base.")
        rows = []
        for k, c in opts.items():
            a, row = ev[k]["args"], ev[k]["row"]
            rop = np.nan
            if c.kind == "option" and c.spec.mode == "premium":
                sleeve, cal = common.sleeve_cached(core, a.params, a.pricing, a.chain_path, a.cover)
                rop = catalog.return_on_premium(sleeve, mkt["rf"], cal)
            rows.append({"candidate": k, "portfolio dCAGR (bp/yr)": row["d_cagr_bp"],
                         "sleeve CAGR (on sleeve capital)": row["critic_sleeve_cagr"], "net return on premium": rop,
                         "premium spent (% of NAV / yr)": row["avg_premium_yr"]})
        st.dataframe(styled(pd.DataFrame(rows)), hide_index=True)
        st.caption("Sleeve CAGR: the critic's buy-and-hold sleeve, never refilled (one wipe-out reads -100%). Net "
                   "return on premium: option P&L per dollar of premium, the sleeve refilled at each rebalance as the "
                   "portfolio refills it (-100% = every premium dollar lost); n/a for overlays, whose sleeve holds the "
                   "core, and for the TAIL proxy, whose Treasury carry is not option P&L.")
    if table["model_priced"].any():
        common.note("model-priced")
    for k, c in specs.items():
        common.launch_note(c, ev[k]["row"]["sample_start"])

    st.plotly_chart(charts.ce_plane(table))
    st.subheader("Payoff by S&P 500 year")
    st.caption("Each haven's calendar-year return as the portfolio held it at w = w* (the smallest grid weight when "
               "w* = 0): implied from the portfolio's years at 0 and w, so the sleeve is refilled at every rebalance "
               "and a -100% year does not empty the next (under relever funding: the core plus the sleeve less its "
               "financing, the haven that mixed in at w gives the levered portfolio's year).")
    for tab, (k, e) in zip(st.tabs([CANDIDATES[k].name for k in ev]), ev.items()):
        with tab:
            nav, w = e["row"]["_nav"], e["row"]["w_scores"]
            years = metrics.full_year_returns(nav[[0.0, w]])
            spx = metrics.full_year_returns(mkt["spxtr"].reindex(nav.index).ffill())
            haven = scenarios.implied_haven(years[0.0], years[w], w)
            st.plotly_chart(charts.payoff_scatter(haven, spx, e["row"]["candidate"]), key=f"payoff:{k}")

    with st.expander("Notes and caveats per candidate"):
        for _, r in table.iterrows():
            st.markdown(f"**{r['candidate']}**: {r['notes'] or 'none'}")
    with st.expander("Universa known-answer checks (self-reported hypothetical)"):
        common.note("self-reported")
        if "universa_model" in ev:
            # Universa's model portfolio is 96.67% S&P 500 TR, so the check runs on that core, not the user's
            a = ev["universa_model"]["args"]
            spx_core = common.window(mkt["spxtr"].pct_change().fillna(0.0))
            try:
                with st.spinner("Running the 96.67 / 3.33 portfolio on the S&P 500 TR..."):
                    sl, cl = common.sleeve_cached(spx_core, a.params, a.pricing, a.chain_path, 1.0)
                    _, nav = common.frontier_cached(spx_core, sl, (0.0, catalog.W_UNIVERSA), cl, a.policy, "carve_out")
                checks = catalog.universa_checks(nav[catalog.W_UNIVERSA], mkt["spxtr"])
                st.dataframe(checks.style.format({c: "{:+.1%}" for c in checks if c not in ("check", "label")},
                                                 na_rep="n/a"), hide_index=True)
                st.caption("Ours: the evaluated universa_model sleeve at 3.33% on the S&P 500 TR over the sidebar's "
                           "dates, whatever the loaded portfolio; n/a where the dates miss a window.")
            except ValueError as e:
                st.caption(f"Not run: {e}")
        else:
            st.caption("Evaluate the Universa model (96.67 / 3.33) to compare it with Universa's reported figures.")

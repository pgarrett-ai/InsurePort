"""Stress lab page: one candidate under every test. Frontier (w*, the drawdown cap, the T-bill frontier and
the AQR twins), Bootstrap (block-bootstrap fans, the book's annual bootstrap), Dice (regime engine, Kelly,
breakeven crash frequency), Robustness (rolling windows, decades, roll offsets) and Sensitivity (pricing
assumptions, on demand).

The bootstraps and the estimated dice read the haven off the engine's portfolio NAVs at 0 and w
(scenarios.implied_haven), not off the sleeve compounded alone: the portfolio refills the sleeve at every
rebalance (a -100% year does not empty the next), and the funding setting carries through."""

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import common
from insureport import backtest, catalog, charts, metrics, scenarios
from insureport.catalog import CANDIDATES

OFFSETS = (0, *catalog.OFFSETS)  # roll-offset sweep, business days (0 = the base calendar)

st.title("Stress lab")
p, core = common.require_portfolio()
s, mkt = common.settings(), common.market()
ev = common.evaluated(core)
ranked = catalog.rank([e["row"] for e in ev.values()])["candidate"].tolist() if ev else []  # best first
keys = [*ranked, *(k for k in CANDIDATES if k not in ev)]
key = st.selectbox("Candidate", keys, index=keys.index(ranked[0] if ranked else "universa_model"),
                   key="stress_candidate",
                   format_func=lambda k: CANDIDATES[k].name + (" (as evaluated)" if k in ev else " (preset)"),
                   help="Evaluated candidates keep their Candidates-page parameters; the others run at their presets.")
if key in ev:
    args = ev[key]["args"]
else:
    c0 = CANDIDATES[key] if s["rebalance"] == "preset" else replace(CANDIDATES[key], rebalance=s["rebalance"])
    args = common.eval_args(c0, s, common.coverage(metrics.downside_beta(core.iloc[1:], mkt["spxtr"].pct_change())))
c = common.candidate_from_tuple(args.params)
try:
    with st.spinner(f"Evaluating {c.name}..."):
        row = common.evaluate_cached(core, *args)
        _, cal = common.sleeve_cached(core, args.params, args.pricing, args.chain_path, args.cover)
except (ValueError, KeyError) as e:
    st.error(f"{c.name} cannot run on this portfolio: {e}")
    st.stop()
table, nav, w_star = row["_table"], row["_nav"], row["w_star"]
w_s = row["w_scores"]  # w*, or the smallest grid weight when w* = 0
start, end = nav.index[0], nav.index[-1]
st.caption(f"{row['verdict']}: w* = {w_star:.2%}, dCAGR {row['d_cagr_bp']:+.0f} bp/yr over {start:%Y-%m-%d} to "
           f"{end:%Y-%m-%d} ({row['years']:.1f} years), {args.funding} funding.")
if c.model_priced:
    common.note("model-priced")
common.launch_note(c, start)

# ponytail: every tab but Sensitivity computes on each run (st.tabs renders all of them; the bootstraps and the roll
# offsets are cached, so a rerun is a cache read); st.fragment or a per-tab button if a first visit gets slow
frontier_tab, boot_tab, dice_tab, robust_tab, sens_tab = st.tabs(["Frontier", "Bootstrap", "Dice", "Robustness",
                                                                    "Sensitivity"])

with frontier_tab:
    st.plotly_chart(charts.frontier_chart(table, args.dd_cap, row["_cash"], w_star))
    col = st.columns(4)
    col[0].metric("w* (max CAGR)", f"{w_star:.2%}", f"{row['d_cagr_bp']:+.0f} bp/yr")
    col[1].metric(f"w_cap ({args.dd_cap:.0%} drawdown cap)", "n/a" if np.isnan(row["w_cap"]) else f"{row['w_cap']:.2%}")
    dd = row["d_cagr_vs_dd_matched_cash_bp"]
    col[2].metric("vs drawdown-matched T-bills", "n/a" if np.isnan(dd) else f"{dd:+.0f} bp/yr",
                  help="CAGR minus the T-bill frontier's CAGR at the same max drawdown (AQR's test); n/a when that "
                       "drawdown is outside the T-bill frontier's range.")
    gap = row["dd_gap_vs_cagr_matched_cash"]
    col[3].metric("Max-DD vs CAGR-matched T-bills", "n/a" if np.isnan(gap) else f"{gap:+.1%}",
                  help="Max drawdown minus that of the divested T-bill twin with the same CAGR (> 0: it falls less).")
    st.caption(f"The twins are measured at w_cap (w* when no weight meets the cap). Beats deleverage: "
               f"{'yes' if row['beats_deleverage'] else 'no'}.")

with boot_tab:
    bcal = cal if len(cal) >= 2 else catalog.rebalance_calendar(nav.index, "annual")  # buy-and-hold: yearly periods
    per = backtest.period_returns(nav[[0.0, w_s]].pct_change().iloc[1:], bcal).dropna()
    ppy = max(1, round(len(per) / row["years"]))  # ponytail: mean periods a year; uneven "events" calendars blur it
    with st.spinner("Block bootstrap (2,000 paths)..."):
        boot = common.bootstrap_cached(per[0.0], scenarios.implied_haven(per[0.0], per[w_s], w_s), tuple(table.index),
                                       ppy, (5, 10, 20), args.dd_cap)
    for years in (5, 10, 20):
        st.plotly_chart(charts.fan_chart(boot, years), key=f"fan:{years}")
    st.caption(f"Stationary block bootstrap of {len(per)} rebalance periods ({ppy} a year), paired across w: the "
               f"portfolio's periods at 0 and w = {w_s:.2%} give the haven's (refilled each period, {args.funding} "
               "funding included), mixed back as (1 - w) core + w haven at every w.")
    st.markdown("**The book's bootstrap**: 25 calendar years drawn with replacement, 10,000 paths, at "
                f"w = {w_s:.2%}.")
    a = metrics.full_year_returns(nav[[0.0, w_s]])
    haven_y = scenarios.implied_haven(a[0.0], a[w_s], w_s)
    rows = {}
    for mode in ("joint", "reshuffle", "no_crash"):
        try:
            b = scenarios.book_bootstrap(a[0.0], haven_y, np.array([0.0, w_s]), mode=mode)
            rows[mode] = b.loc[round(w_s, 12), ["cagr_p5", "cagr_p50", "cagr_p95", "p_beat", "d_cagr_p50"]]
        except ValueError as e:
            st.caption(f"{mode}: n/a ({e})")
    st.dataframe(pd.DataFrame(rows).T.rename_axis("mode").style.format("{:.2%}", na_rep="n/a"))
    st.caption("joint: core and haven from the same year; reshuffle: from different years (an insurance payoff's "
               "gain must vanish); no_crash: only years above -15%, the cost when no crash comes.")

with dice_tab:
    spx_annual = scenarios.load_sp500_annual()
    reg = scenarios.regime_table(spx_annual)
    st.plotly_chart(charts.regime_table_fig(reg))
    mode = st.radio("Haven returns per regime", ("book", "estimated"), horizontal=True, key="dice_mode",
                    format_func=lambda m: {"book": "book: a cartoon haven",
                                           "estimated": "estimated: this candidate's backtest"}[m])
    p_crash = float(reg.at[0, "freq"])
    if mode == "book":
        haven = st.selectbox("Cartoon haven", list(scenarios.BOOK_HAVENS), key="dice_haven")
        payoff = None
        if haven == "insurance":
            payoff = st.slider("Insurance payoff in a crash year (x the premium)", 0.0, 30.0,
                               round((1 - p_crash) / p_crash, 1), 0.5, key="dice_payoff",
                               help="Default: the zero-mean payoff (1 - p) / p; it loses 100% in every other year.")
        core_j, haven_j = reg["mean_ret"].to_numpy(float), scenarios.book_haven(haven, p_crash, payoff)
        ek = f"dice:book:{haven}:{payoff}"  # a fresh table per cartoon and payoff
    else:
        try:
            _, P, n = scenarios.regime_portfolio_returns(nav[[0.0, w_s]], spx_annual)
        except ValueError:  # no full calendar year in the sample: every regime is filled by hand
            P, n = np.full((len(reg), 2), np.nan), np.zeros(len(reg), dtype=int)
        core_j, haven_j = P[:, 0], scenarios.implied_haven(P[:, 0], P[:, 1], w_s)
        st.caption(f"Mean full-calendar-year return of the core and of the haven (as the portfolio held it at "
                   f"w = {w_s:.2%}) per regime over the sample "
                   f"({', '.join(f'{k}: {v} yr' for k, v in zip(reg['label'], n))}); fill empty regimes by hand.")
        ek = f"dice:estimated:{hash((args, common.fingerprint(core)))}"  # hand edits belong to this backtest only
    tbl = st.data_editor(pd.DataFrame({"regime": reg["label"], "p": reg["freq"], "core": core_j, "haven": haven_j}),
                         disabled=["regime"], hide_index=True, key=ek)
    pj, cj, hj = (tbl[k].to_numpy(float) for k in ("p", "core", "haven"))
    if np.isnan(np.r_[cj, hj][np.r_[pj, pj] > 0]).any():
        st.warning("A regime with a probability has no return: fill it in the table.")
    else:
        w = np.linspace(0, 1, 201)
        P = scenarios.book_portfolio_returns(cj, hj, w)
        k_w = scenarios.kelly_w(pj, cj, hj)
        st.plotly_chart(charts.dice_chart(w, scenarios.dice_growth(pj, P), scenarios.dice_arith(pj, P), k_w))
        pi = scenarios.breakeven_crash_freq(pj, cj, hj)
        col = st.columns(3)
        col[0].metric("Kelly w*", f"{k_w:.1%}")
        col[1].metric("Cost-effective (G'(0) > 0)", "yes" if scenarios.cost_effective(pj, cj, hj) else "no")
        col[2].metric("Breakeven crash frequency", "n/a" if np.isnan(pi) else f"{pi:.1%}", f"history {p_crash:.1%}",
                      delta_color="off")
        sweep = scenarios.crash_frequency_sweep(pj, cj, hj, np.array(sorted({0.0, round(k_w / 2, 4), round(k_w, 4)})))
        st.plotly_chart(charts.crash_freq_chart(sweep))
        st.caption("Each year resets to (1 - w) core + w haven; p sums to "
                   f"{pj.sum():.3f}. Regime frequencies: S&P 500 total return 1928 on (Damodaran).")

with robust_tab:
    for years in (5, 10):
        st.plotly_chart(charts.rolling_chart(backtest.rolling_excess(nav[[0.0, w_s]], years), w_s, years=years),
                        key=f"rolling:{years}")
    st.markdown("**Decades** (each from the previous decade's last close)")
    dec = backtest.subperiods(nav, w_s)
    names = {"cagr_0": "unhedged CAGR", "cagr_w": f"CAGR at w = {w_s:.2%}", "d_cagr": "dCAGR (bp/yr)"}
    st.dataframe(dec.set_axis([f"{y}s" for y in dec.index]).rename(columns=names)
                 .style.format("{:.2%}").format(lambda x: f"{x * 1e4:+.0f}", subset=["dCAGR (bp/yr)"]))
    st.markdown("**Roll offset** (path dependence: the same strategy rolled k business days later)")
    if c.spec is None:
        st.dataframe(pd.DataFrame({"dCAGR (bp/yr)": [np.nan]}, index=pd.Index(["n/a: no option rolls"], name="offset")))
    else:
        offs = {}
        with st.spinner("Re-running the sleeve at each roll offset..."):
            for k in OFFSETS:
                sl, cl = common.sleeve_cached(core, args.params, (*args.pricing[:3], k), args.chain_path, args.cover)
                tab, _ = common.frontier_cached(core, sl, (0.0, w_s), cl, args.policy, args.funding)
                offs[k] = (tab.at[w_s, "cagr"] - tab.at[0.0, "cagr"]) * 1e4
        st.dataframe(pd.Series(offs, name="dCAGR (bp/yr)").rename_axis("offset (business days)").to_frame()
                     .style.format("{:+.0f}"))

with sens_tab:
    st.caption("One assumption at a time from the sidebar's pricing assumptions: m x {0.5, 0.75, 1.25, 1.5}, "
               "h in {0, 0.05, 0.1, 0.2}, nu in {0.8, 0.9, 1.1, 1.2}, roll offset in {"
               + ", ".join(map(str, OFFSETS)) + "}; each re-runs the sleeve and its frontier (about a minute on a "
               "1990 start). Bars: the lowest and the highest change over each parameter's settings, named in the "
               "label (lowest .. highest).")
    sens_key = (key, args, common.fingerprint(core))  # a sweep belongs to this candidate, its arguments and sample
    if c.spec is None or not c.model_priced:
        st.info(f"{c.name} is not model-priced: no pricing assumption moves it.")
    elif st.button("Run the sensitivity sweep", key="run_sensitivity"):
        m, nu, h, off = args.pricing
        variants = ([("m", f"m x{k:g}", (m * k, nu, h, off)) for k in (0.5, 0.75, 1.25, 1.5)]
                    + [("h", f"h {v:g}", (m, nu, v, off)) for v in (0.0, 0.05, 0.1, 0.2)]
                    + [("nu", f"nu {v:g}", (m, v, h, off)) for v in (0.8, 0.9, 1.1, 1.2)]
                    + [("offset", f"offset {v}", (m, nu, h, v)) for v in OFFSETS])
        grid, out, dcol = tuple(table.index), [], f"dCAGR at w = {w_s:.2%} (bp/yr)"
        bar = st.progress(0.0)
        # ponytail: serial, ~16 sleeve + frontier rebuilds (about a minute on a 1990 start); a process pool if it drags
        for i, (name, label, pt) in enumerate(variants):
            bar.progress(i / len(variants), text=f"{label} ({i + 1} of {len(variants)})...")
            sl, cl = common.sleeve_cached(core, args.params, pt, args.chain_path, args.cover)
            tab, _ = common.frontier_cached(core, sl, grid, cl, args.policy, args.funding)
            out.append({"parameter": name, "setting": label, "w*": backtest.best_allocation(tab, None),
                        dcol: (tab.at[w_s, "cagr"] - tab.at[0.0, "cagr"]) * 1e4})
        bar.empty()
        st.session_state.setdefault("sensitivity", {})[sens_key] = pd.DataFrame(out)
    res = st.session_state.get("sensitivity", {}).get(sens_key)
    if res is not None:
        base_d = (table.at[w_s, "cagr"] - table.at[0.0, "cagr"]) * 1e4  # w_s: w*, or the smallest weight at w* = 0
        for col, base, label, xfmt in ((res.columns[3], base_d, res.columns[3], None),
                                       ("w*", w_star, "w* (% of portfolio)", ".2%")):
            st.plotly_chart(charts.tornado(backtest.tornado_rows(res, col, base), label, xfmt=xfmt),
                            key=f"tornado:{col}")
        st.dataframe(res, hide_index=True)

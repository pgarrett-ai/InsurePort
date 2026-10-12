"""The case study: does portfolio insurance raise geometric return, for which sleeves, at what size, and how
robust is that? Regenerates every figure and table CASE_STUDY.md uses (network required on the first run).

Usage, from the repo root: .venv/Scripts/python scripts/case_study.py
Runs on the live panel (data.load_market from 1990-01-01, term structure back-filled; every series cached
under data/), seed 0, drawdown cap 30%, and writes figures/cs_*.png (figures/cs_*.html when no PNG export
works: the run says so) and figures/case_study_tables.md; every table is also printed. CASE_STUDY.md copies
its numbers from that file.

Experiments:
1. cores: the S&P 500 TR; 60/40 (S&P 500 TR / the 10-year par-bond proxy on FRED DGS10, monthly rebalance);
   samples/tech_concentrated.csv proxy-filled with the S&P 500 TR before each ticker's first price;
2. candidates: every catalog.CANDIDATES entry on every core (catalog.evaluate: under 10 years is
   "insufficient history"), sample windows reported;
3. frontiers (CAGR and max drawdown vs w, the 30% cap, the T-bill frontier, w*) and the cost-effectiveness
   plane at w*, plus (an addition to the brief) the plane at w_eval with its volatility-tax split: at w* = 0
   every candidate sits at the origin;
4. the AQR twins (drawdown-matched cash gap, CAGR-matched divested twin, beats_deleverage) and the critic's
   stand-alone view;
5. hold more equity: funding "relever" (L = 1 / (1 - w)) vs carve-out for the option sleeves;
6. robustness: rolling 10-year win rate, the block bootstrap's p_beat at 5 / 10 / 20 years, decades, the
   roll-offset sweep {0, 5, 10, 15} and the book bootstrap (joint / reshuffle / no-crash) for the insurance
   archetype, the book's insurance cartoon on the 1928-latest annual table as a reference row;
7. dice: the regime table, the Part One cartoons (insurance / alpha / store of value at 97/3 and 90/10, on
   1997-2016 next to the book's reported +67 / +18 / -17 bp, and on 1928-latest), estimated-mode dice for
   the model-priced sleeves with their breakeven crash frequency, and the crash-frequency sweep;
8. pricing sensitivity of dao_ch9 and universa_model on the S&P 500 core: m x {0.5 .. 1.5}, nu {0.8 .. 1.2},
   h {0, 0.05, 0.1, 0.2}, z_cap {2.5, 3, 4}, and beta {0, 0.5, 1} (DEVIATION from the brief, which lists the
   first four: pricing.SKEW_BETA's docstring and scripts/calibrate.py promise the beta sweep here);
9. validation: replicate.validate (PPUT / PPUT3M / CLL) with the replica charts, and the Universa known-answer
   checks (self-reported, informational);
10. crisis windows: the two option presets hedged vs unhedged vs 60/40.
Post-processing experiments (6, 7, 8, 10) read each candidate at w_eval: w* when it is positive, else the
preset's own weight (PRESET_W: 3.33% Universa, 0.5% Dao), else evaluate's w_scores (the smallest grid
weight): at w* = 0 every hedged-minus-unhedged statistic is 0 by construction.

The table builders are plain functions of the panel, the haven prices and the cores (tests/test_case_study.py
runs them on the fixture panel); main() fetches, runs and writes. Edits no source file.
"""

import re
import time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

from insureport import backtest, catalog, charts, data, metrics, portfolio, pricing, replicate, scenarios, sleeves
from insureport.backtest import Policy
from insureport.catalog import CANDIDATES, Candidate

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
HOLDINGS = ROOT / "samples" / "tech_concentrated.csv"
START, SEED, DD_CAP = "1990-01-01", 0, 0.30
PATHS, BOOK_PATHS = 2000, 10_000  # stationary block bootstrap paths (as the app) / the book's annual bootstrap
CORES = {"spx": "S&P 500 TR", "6040": "60/40 (S&P 500 TR / 10-year Treasury proxy, monthly)",
         "tech": "concentrated tech sample (proxy-filled)"}
HAVEN_SOURCES = tuple(sorted({c.source for c in CANDIDATES.values() if c.kind == "etf" or c.key == "sixty_forty"}))
PRESET_W = {"universa_model": catalog.W_UNIVERSA, "universa_2023": 0.02, "dao_ch9": 0.005}  # each preset's own weight
PRESETS = ("dao_ch9", "universa_model")  # experiments 8 and 10
OFFSET_SWEEP = (0, *catalog.OFFSETS)
DECADES = (1990, 2000, 2010, 2020)
HORIZONS = (5, 10, 20)
MODES = ("joint", "reshuffle", "no_crash")
CARTOON_CORE = "S&P 500 1928-latest (book cartoon)"
CARTOON_W = 0.03
# Universa, "Safe Haven Investing" Part One (2017), 1997-2016, annual reset, as reported (bp of CAGR over the S&P 500)
PART_ONE = {("insurance", 0.03): 67, ("alpha", 0.10): 18, ("store of value (CPI + 2%)", 0.10): -17}
VARIANTS = [  # (parameter, setting, pricing_kw): one assumption at a time from the base (catalog / pricing defaults)
    *(("m", f"m x{k:g}", {"m": pricing.SKEW_MULT * k}) for k in (0.5, 0.75, 1.0, 1.25, 1.5)),
    *(("nu", f"nu {v:g}", {"nu": v}) for v in (0.8, 0.9, 1.0, 1.1, 1.2)),
    *(("h", f"h {v:g}", {"h": v}) for v in (0.0, 0.05, 0.1, 0.2)),
    *(("z_cap", f"z_cap {v:g}", {"z_cap": v}) for v in (2.5, 3.0, 4.0)),
    *(("beta", f"beta {v:g}", {"beta": v}) for v in (0.0, 0.5, 1.0)),
]
SCENARIOS = (  # (window, start = the close it is measured from, end); peak-to-trough metrics.CRASHES dates for 2011/1998/1990
    ("2008", "2007-12-31", "2008-12-31"),
    ("Q1 2020", "2019-12-31", "2020-03-31"),
    ("March 2020", "2020-02-28", "2020-03-31"),
    ("2022", "2021-12-31", "2022-12-30"),
    ("2018 Q4", "2018-09-28", "2018-12-31"),
    ("2011 debt ceiling", *metrics.CRASHES["2011 Debt ceiling"]),
    ("1998 LTCM", *metrics.CRASHES["1998 LTCM"]),
    ("1990 Gulf War", *metrics.CRASHES["1990 Gulf War"]),
)
PI_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
PLAIN = {"years", "gross_exposure", "critic_sleeve_sharpe", "critic_beta", "corr", "corr_tol", "downside_beta"}


# ---------------------------------------------------------------------------------------------------- engine runs


def build_cores(mkt: pd.DataFrame, px: pd.DataFrame, holdings: str) -> tuple[dict[str, pd.Series], list[str]]:
    """The three cores as daily returns on mkt's span, each with the 0 placeholder first row, and the proxied
    tickers: "spx" the S&P 500 TR; "6040" 60% of it and 40% of the par-bond proxy on px["DGS10"] (carried onto
    mkt's dates), rebalanced at each month's last close (backtest.simulate, default costs); "tech" the holdings
    CSV text (shares valued at their last prices in px) through portfolio.core_returns with proxy_fill, so each
    ticker holds the S&P 500 TR before its first price."""
    spx = mkt["spxtr"].pct_change().fillna(0.0)
    bond = sleeves.bond_proxy(px["DGS10"].dropna().reindex(mkt.index, method="ffill"))
    nav, _ = backtest.simulate(spx, bond, 0.4, catalog.rebalance_calendar(mkt.index, "monthly"))
    h = portfolio.parse_holdings(holdings)
    tickers = [t for t in h["ticker"] if t != "CASH"]
    p = portfolio.to_portfolio(h, px[tickers].ffill().iloc[-1])
    tech, proxied = portfolio.core_returns(p, px[tickers], mkt["rf"], spxtr=mkt["spxtr"], proxy_fill=True)
    tech = tech.loc[mkt.index[0] : mkt.index[-1]].copy()
    tech.iloc[0] = 0.0
    return {"spx": spx, "6040": nav[0.4].pct_change().fillna(0.0), "tech": tech}, proxied


def evaluate_all(
    cores: dict[str, pd.Series], mkt: pd.DataFrame, px: pd.DataFrame, candidates: dict[str, Candidate], *,
    funding: str = "carve_out", paths: int = PATHS,
) -> dict[tuple[str, str], dict]:
    """catalog.evaluate of every candidate on every core, keyed (core, candidate): build_sleeve with the core's
    downside beta to the S&P 500 TR (the overlays' coverage), the 30% cap, seed 0. Each row also carries core,
    _cal (the sleeve's rebalance calendar) and _beta (that downside beta)."""
    spx_r = mkt["spxtr"].pct_change()
    rows = {}
    for core, r in cores.items():
        beta = metrics.downside_beta(r.iloc[1:], spx_r)
        for key, c in candidates.items():
            sleeve, cal = catalog.build_sleeve(c, mkt, px, r, beta)
            row = catalog.evaluate(c, r, sleeve, cal, mkt, dd_cap=DD_CAP, funding=funding, bootstrap_paths=paths,
                                   seed=SEED, px=px, beta_down=beta)
            rows[core, key] = row | {"core": core, "_cal": cal, "_beta": beta}
    return rows


def w_eval(row: dict) -> float:
    """The weight the post-processing experiments read a candidate at: w* when positive, else the preset's own
    weight (PRESET_W) when it is on the frontier grid, else evaluate's w_scores (the smallest grid weight)."""
    if row["w_star"] > 0:
        return row["w_star"]
    w = PRESET_W.get(row["candidate"])
    return w if w is not None and w in row["_nav"] else row["w_scores"]


def _frontier(c: Candidate, core_r: pd.Series, mkt: pd.DataFrame, px: pd.DataFrame, beta_down: float,
              pricing_kw: dict, grid: np.ndarray) -> pd.DataFrame:
    """The carve-out frontier table of c's sleeve rebuilt with pricing_kw, over grid."""
    s, cal = catalog.build_sleeve(c, mkt, px, core_r, beta_down, pricing_kw=pricing_kw)
    s.attrs = {}
    return catalog.funded_frontier(core_r, s, np.asarray(grid, dtype=float), cal, Policy(), "carve_out", mkt["rf"])[0]


def _d_cagr(t: pd.DataFrame, w: float) -> float:
    """CAGR at w minus at 0 of a frontier table, in bp."""
    return float(t.at[w, "cagr"] - t.at[0.0, "cagr"]) * 1e4


# ---------------------------------------------------------------------------------------------------- tables


def core_table(cores: dict[str, pd.Series], mkt: pd.DataFrame) -> pd.DataFrame:
    """Experiment 1: one row per core: sample, CAGR, the full-calendar-year arithmetic mean and volatility tax
    (metrics.vol_tax full_years, with its sigma^2 / 2 approximation), ann_vol, max_dd, worst full year, time under
    water and the downside beta to the S&P 500 TR."""
    spx_r, recs = mkt["spxtr"].pct_change(), []
    for k, r in cores.items():
        nav = metrics.nav_from_returns(r)
        vt = metrics.vol_tax(nav, full_years=True)
        recs.append({"core": k, "start": nav.index[0], "end": nav.index[-1],
                     "years": (nav.index[-1] - nav.index[0]).days / 365.25, "cagr": float(metrics.cagr(nav)),
                     "arith_mean": vt["arith"], "vol_tax": vt["tax"], "vol_tax_approx": vt["approx"],
                     "ann_vol": float(metrics.ann_vol(r.iloc[1:])), "max_dd": float(metrics.max_drawdown(nav)),
                     "worst_year": float(metrics.full_year_returns(nav).min()),
                     "tuw": float(metrics.time_under_water(nav)), "downside_beta": metrics.downside_beta(r.iloc[1:], spx_r)})
    return pd.DataFrame(recs)


def core_crash_table(cores: dict[str, pd.Series]) -> pd.DataFrame:
    """Experiment 1: each core's return over every metrics.CRASHES window inside its sample ("crash:<name>")."""
    return pd.DataFrame([{"core": k, **{f"crash:{n}": v for n, v in
                                        metrics.crash_returns(metrics.nav_from_returns(r))["ret"].items()}}
                         for k, r in cores.items()])


def _pick(rows: dict, cols: list[str], **extra) -> pd.DataFrame:
    """One record per row: core, candidate, the row's cols, then extra name -> f(row) columns."""
    return pd.DataFrame([{"core": r["core"], "candidate": r["candidate"], **{c: r[c] for c in cols},
                          **{k: f(r) for k, f in extra.items()}} for r in rows.values()])


def candidate_table(rows: dict) -> pd.DataFrame:
    """Experiment 2: each (core, candidate)'s archetype, kind, pricing and sample window."""
    return _pick(rows, ["archetype", "kind", "model_priced", "sample_start", "sample_end", "years",
                        "n_crashes_in_sample"])


def frontier_table(rows: dict) -> pd.DataFrame:
    """Experiment 3: w* (uncapped CAGR maximum) and w_cap (under the 30% cap), CAGR unhedged / at w*, dCAGR, the
    arithmetic cost (change in mean calendar-year return) and the volatility tax saved (the frontier's vol_tax at
    0 minus at w*: dCAGR ~ arithmetic change + tax saved), max drawdowns, the book bootstrap's median 25-year
    dCAGR, verdict and robustness checks."""
    def tax_saved(r: dict) -> float:
        t = r["_table"]
        return float(t.at[0.0, "vol_tax"] - t.at[r["w_star"], "vol_tax"]) * 1e4

    t = _pick(rows, ["archetype", "years", "w_star", "w_cap", "cagr_unhedged", "cagr_star", "d_cagr_bp",
                     "arith_cost_bp"], vol_tax_saved_bp=tax_saved)
    rest = _pick(rows, ["maxdd_unhedged", "maxdd_star", "d_cagr_p50_bp", "verdict", "robustness"])
    return pd.concat([t, rest.drop(columns=["core", "candidate"])], axis=1)


def sleeve_score_table(rows: dict) -> pd.DataFrame:
    """Experiment 3: evaluate's sleeve scores at w_scores (w*, or the smallest grid weight), in portfolio points:
    calm-year bleed, crash payoffs (all / fast / slow windows), premium a year and long notional (% of NAV), the
    crash / non-crash attribution of the mean year difference and the -20% one-month shock P&L (put presets)."""
    return _pick(rows, ["w_scores", "calm_year_bleed", "crash_payoff_mean", "crash_payoff_fast", "crash_payoff_slow",
                        "avg_premium_yr", "avg_coverage", "attr_crash_bp", "attr_noncrash_bp", "shock20_pnl"])


def ce_table(rows: dict) -> pd.DataFrame:
    """Experiment 3: the cost-effectiveness plane at w_eval, where the hedge is actually held (at w* = 0 every
    coordinate is 0): the arithmetic cost (change in the mean calendar-year return), the volatility tax saved and
    dCAGR, in bp (dCAGR ~ cost + tax saved; partial first and last years make the two sides differ a little)."""
    recs = []
    for r in rows.values():
        w, t = w_eval(r), r["_table"]
        recs.append({"core": r["core"], "candidate": r["candidate"], "verdict": r["verdict"], "w_eval": w,
                     "arith_cost_bp": float(t.at[w, "arith_mean"] - t.at[0.0, "arith_mean"]) * 1e4,
                     "vol_tax_saved_bp": float(t.at[0.0, "vol_tax"] - t.at[w, "vol_tax"]) * 1e4,
                     "d_cagr_bp": _d_cagr(t, w)})
    return pd.DataFrame(recs)


def verdict_matrix(frontier: pd.DataFrame) -> pd.DataFrame:
    """The summary: one row per candidate (catalog order), its archetype, then w*, dCAGR and the verdict per core."""
    cores = list(dict.fromkeys(frontier["core"]))
    keys = list(dict.fromkeys(frontier["candidate"]))
    f = frontier.set_index(["candidate", "core"])
    out = pd.DataFrame({"candidate": keys, "archetype": f["archetype"].groupby(level=0).first().reindex(keys).to_numpy()})
    for c in cores:
        for x in ("w_star", "d_cagr_bp", "verdict"):
            out[f"{c}: {x}"] = [f.at[(k, c), x] for k in keys]
    return out


def aqr_table(rows: dict) -> pd.DataFrame:
    """Experiment 4: the AQR twins at w_twin (w_cap, or w* when no weight meets the cap): the CAGR gap to the T-bill
    frontier at equal max drawdown, the CAGR-matched T-bill weight, its max drawdown and the drawdown gap (> 0: the
    candidate falls less for the same growth), beats_deleverage; then the critic's stand-alone sleeve view."""
    recs = []
    for r in rows.values():
        w = r["w_star"] if np.isnan(r["w_cap"]) else r["w_cap"]
        twin = backtest.cagr_matched_cash_dd(r["_table"], r["_cash"], w)
        recs.append({"core": r["core"], "candidate": r["candidate"], "w_star": r["w_star"], "w_cap": r["w_cap"],
                     "w_twin": w, "d_cagr_vs_dd_matched_cash_bp": r["d_cagr_vs_dd_matched_cash_bp"],
                     "w_cash_twin": twin["w_cash"], "maxdd_cash_twin": twin["maxdd_cash"],
                     "dd_gap_vs_cagr_matched_cash": r["dd_gap_vs_cagr_matched_cash"],
                     "beats_deleverage": r["beats_deleverage"], "critic_sleeve_cagr": r["critic_sleeve_cagr"],
                     "critic_sleeve_sharpe": r["critic_sleeve_sharpe"], "critic_alpha": r["critic_alpha"],
                     "critic_beta": r["critic_beta"]})
    return pd.DataFrame(recs)


def relever_table(carve: dict, relever: dict) -> pd.DataFrame:
    """Experiment 5: for every (core, candidate) of `relever`, its carve-out row and its relever row (the core
    levered to 1 / (1 - w), financed at rf + 50 bp) side by side."""
    cols = ["funding", "w_star", "gross_exposure", "cagr_star", "d_cagr_bp", "maxdd_star",
            "d_cagr_vs_dd_matched_cash_bp", "beats_deleverage", "verdict"]
    return _pick({(k, f): r for k in relever for f, r in (("carve_out", carve[k]), ("relever", relever[k]))}, cols)


def robustness_table(rows: dict, paths: int = PATHS) -> pd.DataFrame:
    """Experiment 6: at w_eval, the share of rolling 10-year windows where the hedged portfolio beats the unhedged
    one (NaN without a full window) and the stationary block bootstrap's P(CAGR hedged > unhedged) at 5, 10 and 20
    years (the two portfolio NAVs' rebalance-period returns, paired, mean block 6 periods, seed 0)."""
    recs = []
    for r in rows.values():
        w, nav = w_eval(r), r["_nav"]
        ex = backtest.rolling_excess(nav[[0.0, w]], 10)[w].dropna()
        cal = r["_cal"] if len(r["_cal"]) >= 2 else catalog.rebalance_calendar(nav.index, "annual")
        per = backtest.period_returns(nav[[0.0, w]].pct_change().iloc[1:], cal).dropna()
        boot = scenarios.bootstrap(per[0.0], per[w], np.array([0.0, 1.0]), max(1, round(len(per) / r["years"])),
                                   HORIZONS, n_paths=paths, seed=SEED, dd_cap=DD_CAP)
        recs.append({"core": r["core"], "candidate": r["candidate"], "w_eval": w,
                     "rolling10_win_rate": float((ex > 0).mean()) if len(ex) else np.nan,
                     **{f"p_beat_{h}y": float(boot.at[(h, 1.0), "p_beat"]) for h in HORIZONS}})
    return pd.DataFrame(recs)


def decade_table(rows: dict) -> pd.DataFrame:
    """Experiment 6: dCAGR at w_eval per decade (backtest.subperiods, each decade from the previous one's last
    close), in bp; NaN for a decade outside the sample."""
    recs = []
    for r in rows.values():
        w = w_eval(r)
        d = backtest.subperiods(r["_nav"], w)["d_cagr"].reindex(DECADES) * 1e4
        recs.append({"core": r["core"], "candidate": r["candidate"], "w_eval": w,
                     **{f"d_{y}s_bp": d[y] for y in DECADES}})
    return pd.DataFrame(recs)


def offset_table(rows: dict, cores: dict[str, pd.Series], mkt: pd.DataFrame, px: pd.DataFrame,
                 candidates: dict[str, Candidate]) -> pd.DataFrame:
    """Experiment 6: the roll-offset sweep for every candidate with option rolls (a spec): dCAGR at w_eval with
    every roll moved 0, 5, 10, 15 business days later (option_sleeve's offset_bdays; offset 0 is the evaluated
    sleeve), in bp, and the spread (max - min) across offsets: path dependence on the roll calendar."""
    recs = []
    for (core, key), r in rows.items():
        c = candidates[key]
        if c.spec is None:
            continue
        w = w_eval(r)
        d = {k: _d_cagr(r["_table"] if k == 0 else _frontier(c, cores[core], mkt, px, r["_beta"], {"offset_bdays": k},
                                                             [0.0, w]), w) for k in OFFSET_SWEEP}
        recs.append({"core": core, "candidate": key, "w_eval": w, **{f"d_cagr_off{k}_bp": v for k, v in d.items()},
                     "spread_bp": max(d.values()) - min(d.values())})
    return pd.DataFrame(recs)


def _book(core: pd.Series, haven: pd.Series, w: float, paths: int) -> dict:
    """Median 25-year dCAGR (bp) and p_beat of the book bootstrap of (1 - w) core + w haven (annual returns, reset
    every year) over the core, per mode (NaN when the mode has no year to draw)."""
    out = {}
    for mode in MODES:
        try:
            b = scenarios.book_bootstrap(core, haven, np.array([0.0, w]), n_paths=paths, seed=SEED,
                                         mode=mode).loc[round(w, 12)]
            out |= {f"{mode}_d_cagr_p50_bp": b["d_cagr_p50"] * 1e4, f"{mode}_p_beat": b["p_beat"]}
        except ValueError:
            out |= {f"{mode}_d_cagr_p50_bp": np.nan, f"{mode}_p_beat": np.nan}
    return out


def book_bootstrap_table(rows: dict, paths: int = BOOK_PATHS) -> pd.DataFrame:
    """Experiment 6: the book bootstrap (25-year paths of whole calendar years) for every insurance-archetype
    candidate at w_eval, on the full years of its unhedged NAV and of the haven as the portfolio held it
    (scenarios.implied_haven of the two NAVs' years, mixed back at w_eval), in the three modes: joint (the crash
    link kept), reshuffle (the haven's year drawn apart from the core's: a true insurance gain must vanish) and
    no_crash (years above -15% only: the cost when no crash comes); plus the book's insurance cartoon at 3% on the packaged
    1928-latest S&P 500 table (zero-mean crash payoff (1 - p) / p) as the reference row."""
    recs = []
    for r in rows.values():
        if r["archetype"] != "insurance":
            continue
        w = w_eval(r)
        a = metrics.full_year_returns(r["_nav"][[0.0, w]])
        recs.append({"core": r["core"], "candidate": r["candidate"], "w_eval": w, "years": len(a),
                     **_book(a[0.0], scenarios.implied_haven(a[0.0], a[w], w), w, paths)})
    spx = scenarios.load_sp500_annual()
    h = scenarios.book_haven("insurance", scenarios.regime_table(spx).at[0, "freq"])[scenarios.regime_of(spx.to_numpy())]
    recs.append({"core": CARTOON_CORE, "candidate": "insurance cartoon", "w_eval": CARTOON_W, "years": len(spx),
                 **_book(spx, pd.Series(h, spx.index), CARTOON_W, paths)})
    return pd.DataFrame(recs)


def cartoon_table() -> pd.DataFrame:
    """Experiment 7: the book's cartoon havens mixed into the S&P 500 at 3% and 10%, reset every year: dCAGR over
    the S&P 500 in bp, on 1997-2016 (Universa's Part One window, with the reported figures) and 1928-latest.
    Insurance pays (1 - p) / p in crash years (p = the window's crash-year share: +900% at 1997-2016's 2 of 20)
    and -100% otherwise; alpha +20 / +10 / +5 / +5% by regime; store of value CPI + 2% year by year (Part One's
    rule) and scenarios' per-bin cartoon (each bin's 1928-latest mean of CPI + 2%)."""
    a = pd.read_csv(scenarios.SP500_CSV, comment="#", index_col="year")
    recs = []
    for lo, hi in ((1997, 2016), (int(a.index[0]), int(a.index[-1]))):
        y = a.loc[lo:hi]
        r = y["total_return"].to_numpy(float)
        j = scenarios.regime_of(r)
        p0 = float(np.mean(j == 0))
        havens = {"insurance": scenarios.book_haven("insurance", p0)[j], "alpha": scenarios.book_haven("alpha", p0)[j],
                  "store of value (CPI + 2%)": y["cpi"].to_numpy(float) + scenarios.REAL_RATE,
                  "store of value (per-bin cartoon)": scenarios.book_haven("store_of_value", p0)[j]}
        for name, h in havens.items():
            for w in (0.03, 0.10):
                g = (np.prod((1 - w) * (1 + r) + w * (1 + h)) ** (1 / len(r)) - np.prod(1 + r) ** (1 / len(r))) * 1e4
                recs.append({"haven": name, "w": w, "window": f"{lo}-{hi}", "years": len(r), "crash_years": int((j == 0).sum()),
                             "d_cagr_bp": g, "book_reported_bp": PART_ONE.get((name, w), np.nan) if lo == 1997 else np.nan})
    return pd.DataFrame(recs)


def _estimated(r: dict, spx_annual: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Estimated-mode dice of a row at w_eval: per-regime mean full-year return of the core and of the haven as the
    portfolio held it (scenarios.implied_haven), the years per regime, and whether every regime has a year."""
    w = w_eval(r)
    try:
        _, P, n = scenarios.regime_portfolio_returns(r["_nav"][[0.0, w]], spx_annual)
    except ValueError:  # no full calendar year
        P, n = np.full((len(scenarios.REGIME_LABELS), 2), np.nan), np.zeros(len(scenarios.REGIME_LABELS), dtype=int)
    cj, hj = P[:, 0], scenarios.implied_haven(P[:, 0], P[:, 1], w)
    return cj, hj, n, not np.isnan(np.r_[cj, hj]).any()


def dice_table(rows: dict) -> pd.DataFrame:
    """Experiment 7: estimated-mode dice for every model-priced sleeve (the options and the TAIL proxy) at w_eval,
    regime frequencies from the 1928-latest S&P 500 table: years per regime in the sample, the haven's mean
    return per regime, G'(0) > 0 (cost_effective), the Kelly weight, the breakeven crash frequency (hist_pi: the
    historical crash share) and the dice dCAGR at w_eval; n/a when a regime has no year in the sample. The last
    row is the reference: the book's insurance cartoon at 3% in book mode on the 1928-latest regime means."""
    spx_annual = scenarios.load_sp500_annual()
    reg = scenarios.regime_table(spx_annual)
    p = reg["freq"].to_numpy(float)
    recs = []
    for r in rows.values():
        if not r["model_priced"]:
            continue
        w = w_eval(r)
        cj, hj, n, ok = _estimated(r, spx_annual)
        g = scenarios.dice_growth(p, scenarios.book_portfolio_returns(cj, hj, [0.0, w])) if ok else [np.nan, np.nan]
        recs.append({"core": r["core"], "candidate": r["candidate"], "w_eval": w,
                     "years_per_regime": "/".join(map(str, n)),
                     **{f"haven {lbl}": v for lbl, v in zip(scenarios.REGIME_LABELS, hj)},
                     "cost_effective": scenarios.cost_effective(p, cj, hj) if ok else np.nan,
                     "kelly_w": scenarios.kelly_w(p, cj, hj) if ok else np.nan,
                     "breakeven_pi": scenarios.breakeven_crash_freq(p, cj, hj) if ok else np.nan, "hist_pi": p[0],
                     "dice_d_cagr_bp": (g[1] - g[0]) * 1e4})
    cj, hj = reg["mean_ret"].to_numpy(float), scenarios.book_haven("insurance", p[0])
    g = scenarios.dice_growth(p, scenarios.book_portfolio_returns(cj, hj, [0.0, CARTOON_W]))
    recs.append({"core": CARTOON_CORE, "candidate": "insurance cartoon", "w_eval": CARTOON_W,
                 "years_per_regime": "/".join(map(str, reg["count"])),
                 **{f"haven {lbl}": v for lbl, v in zip(scenarios.REGIME_LABELS, hj)},
                 "cost_effective": scenarios.cost_effective(p, cj, hj), "kelly_w": scenarios.kelly_w(p, cj, hj),
                 "breakeven_pi": scenarios.breakeven_crash_freq(p, cj, hj), "hist_pi": p[0],
                 "dice_d_cagr_bp": (g[1] - g[0]) * 1e4})
    return pd.DataFrame(recs)


def crash_sweep_table(rows: dict) -> pd.DataFrame:
    """Experiment 7: dice dCAGR (growth at w minus at 0, decimal) by crash-year frequency pi (PI_GRID, the crash
    mass reassigned as scenarios.crash_frequency_sweep does): the book's insurance cartoon at 3% on the 1928-latest
    regime means, and each model-priced sleeve on the S&P 500 core at w_eval (estimated mode; NaN when a regime
    has no year)."""
    spx_annual = scenarios.load_sp500_annual()
    reg = scenarios.regime_table(spx_annual)
    p, grid = reg["freq"].to_numpy(float), np.array(PI_GRID)

    def gain(cj: np.ndarray, hj: np.ndarray, w: float) -> np.ndarray:
        s = scenarios.crash_frequency_sweep(p, cj, hj, np.array([0.0, w]), grid=grid)
        return (s.iloc[:, 1] - s.iloc[:, 0]).to_numpy()

    cart = scenarios.book_haven("insurance", p[0])
    out = {f"insurance cartoon at {CARTOON_W:.0%}": gain(reg["mean_ret"].to_numpy(float), cart, CARTOON_W)}
    for r in rows.values():
        if r["core"] == "spx" and r["model_priced"]:
            cj, hj, _, ok = _estimated(r, spx_annual)
            out[f"{r['candidate']} at {w_eval(r):.2%}"] = gain(cj, hj, w_eval(r)) if ok else np.full(len(grid), np.nan)
    return pd.DataFrame(out, index=pd.Index(PI_GRID, name="pi"))


def sensitivity_table(c: Candidate, core_r: pd.Series, mkt: pd.DataFrame, px: pd.DataFrame, beta_down: float,
                      w: float) -> pd.DataFrame:
    """Experiment 8: the base pricing, then each VARIANTS setting alone: the rebuilt sleeve's uncapped w* over c's
    grid and its dCAGR at w (bp)."""
    grid = np.array(sorted({0.0, *c.w_grid}))
    recs = []
    for param, setting, kw in [("base", "base", {}), *VARIANTS]:
        t = _frontier(c, core_r, mkt, px, beta_down, kw, grid)
        recs.append({"parameter": param, "setting": setting, "w_star": backtest.best_allocation(t, None),
                     "d_cagr_bp": _d_cagr(t, w)})
    return pd.DataFrame(recs)


def validation_table(mkt: pd.DataFrame, pput3m: pd.Series | None) -> pd.DataFrame:
    """Experiment 9: replicate.validate (Cboe conventions and fill, h = 0) next to the pre-registered tolerances."""
    v = replicate.validate(mkt, pput3m=pput3m)
    v = v.join(pd.DataFrame(replicate.TOLERANCE).T.add_suffix("_tol")).rename_axis("target").reset_index()
    return v[["target", "window_start", "window_end", "n_months", "te", "te_tol", "te_ex_bad", "dcagr", "dcagr_tol",
              "corr", "corr_tol", "dmaxdd", "te_roll", "te_nonroll", "passed"]]


def scenario_table(navs: dict[str, pd.Series]) -> pd.DataFrame:
    """Experiment 10: each named NAV's return over every SCENARIOS window (metrics.window_return, from the close at
    start to the close at end); NaN outside its sample."""
    recs = []
    for name, a, b in SCENARIOS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        recs.append({"window": name, "start": a, "end": b, **{k: metrics.window_return(v, a, b) for k, v in navs.items()}})
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------------------------------- output


def _cell(col: str, v: object) -> str:
    """One markdown cell: yes / no, dates as days, n/a for NaN / None, "_bp" columns signed whole bp, PLAIN columns
    3 significant digits, every other float a percent (2 decimals), anything else str()."""
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, pd.Timestamp):
        return f"{v:%Y-%m-%d}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "n/a"
        if col.endswith("_bp"):
            return "+0" if round(v) == 0 else f"{v:+.0f}"
        return f"{v:.3g}" if col in PLAIN else f"{v:.2%}"
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    """df as a GitHub table (the index is not written: reset it first), cells by _cell, | escaped as \\|."""
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", r"\|").replace("\n", " ") for c in cells) + " |"

    body = [line([_cell(str(c), v) for c, v in zip(df.columns, row)]) for row in df.itertuples(index=False)]
    return "\n".join([line([str(c) for c in df.columns]), "|" + "---|" * len(df.columns), *body])


def from_markdown(text: str) -> pd.DataFrame:
    """The inverse of to_markdown: a GitHub table (header, separator, rows) as a DataFrame of strings."""
    rows = [[c.strip().replace(r"\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            for line in text.strip().splitlines()]
    return pd.DataFrame(rows[2:], columns=rows[0])


@cache
def _scope():  # noqa: ANN202 - kaleido's PlotlyScope, imported only when plotly's own export refuses
    """One kaleido 0.2.1 PlotlyScope for the run (each starts a Chromium), fed plotly's bundled plotly.js."""
    from kaleido.scopes.plotly import PlotlyScope

    return PlotlyScope(plotlyjs=str(Path(plotly.__file__).parent / "package_data" / "plotly.min.js"))


def save(fig: go.Figure, name: str, scale: float = 1.5) -> str:
    """Write figures/<name>.png on a white background (plotly's write_image, else kaleido 0.2.1's PlotlyScope, as
    scripts/calibrate.py does), or figures/<name>.html if both raise; the other format's file is removed. Returns
    the file name."""
    # ponytail: duplicated from scripts/calibrate.py (a module-level script, not importable); share it if a third
    # script needs it
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white")
    FIGURES.mkdir(exist_ok=True)
    path, size = FIGURES / f"{name}.png", {"width": 1100, "height": 600, "scale": scale}
    try:
        try:
            fig.write_image(path, **size)
        except RuntimeError:  # plotly >= 6 wants kaleido >= 1; 0.2.1 still renders through its own API
            path.write_bytes(_scope().transform(fig, format="png", **size))
    except Exception as e:  # noqa: BLE001 - any kaleido / Chromium failure falls back to HTML
        print(f"PNG export failed for {name} ({e}); writing HTML")
        path = FIGURES / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
    path.with_suffix(".html" if path.suffix == ".png" else ".png").unlink(missing_ok=True)
    return path.name


def bars(df: pd.DataFrame, title: str, x: str, y: str, yfmt: str) -> go.Figure:
    """Grouped bars: one group per row (the index), one series per column (at most four, charts.SERIES)."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_bar(x=[str(v) for v in df.index], y=df[col], name=str(col), marker_color=charts.SERIES[i],
                    hovertemplate=f"{col}<br>%{{x}}: %{{y:{yfmt}}}<extra></extra>")
    fig.update_layout(barmode="group")
    return charts._style(fig, title, x, y, yfmt=yfmt)  # ponytail: charts' private styler, one look for every figure


def main() -> None:
    """Fetch the panel and prices, run the ten experiments, print every table and write figures/."""
    t0 = time.time()
    mkt = pricing.fill_term_structure(data.load_market(START))
    holdings = HOLDINGS.read_text(encoding="utf-8")
    tickers = [t for t in portfolio.parse_holdings(holdings)["ticker"] if t != "CASH"]
    pput3m = data.cboe("PPUT3M")
    px = pd.concat([data.prices([*HAVEN_SOURCES, *tickers]), data.fred("DGS10").rename("DGS10"),
                    pput3m.rename("pput3m")], axis=1, sort=True)
    cores, proxied = build_cores(mkt, px, holdings)
    first = {k: f"{px[k].first_valid_index():%Y-%m-%d}" for k in [*HAVEN_SOURCES, *tickers, "DGS10", "pput3m"]}
    out = [f"# Case study tables\n\nGenerated by `python scripts/case_study.py` on {pd.Timestamp.now():%Y-%m-%d}: market "
           f"panel {mkt.index[0]:%Y-%m-%d} .. {mkt.index[-1]:%Y-%m-%d} ({len(mkt)} trading days), seed {SEED}, drawdown "
           f"cap {DD_CAP:.0%}, {Policy().tc_core * 1e4:g} bp per trade on each leg, {PATHS} block-bootstrap and "
           f"{BOOK_PATHS} book-bootstrap paths, pricing at the catalog "
           f"defaults (m {pricing.SKEW_MULT:g}, beta {pricing.SKEW_BETA:g}, nu 1, h {catalog.H_DEFAULT:g}, z_cap "
           f"{pricing.Z_CAP:g}). Cores: " + "; ".join(f"{k} = {v}" for k, v in CORES.items())
           + f". Proxy-filled with the S&P 500 TR before their first price: {', '.join(proxied) or 'none'}. First "
           "prices: " + ", ".join(f"{k} {v}" for k, v in first.items()) + ". w_eval = w* when positive, else the "
           "preset's own weight (universa_model 3.33%, universa_2023 2%, dao_ch9 0.5%), else the smallest grid "
           "weight. Rates in %, bp = basis points a year. Panel coverage notes (data.load_market; the VIX9D, VIX3M, "
           "VIX6M and VIX1Y pillars are regression-filled before their first date): "
           + "; ".join(mkt.attrs.get("warnings", [])) + ".\n"]
    for w in mkt.attrs.get("warnings", []):
        print(f"panel warning: {w}")
    figs: list[str] = []

    def table(tid: str, title: str, df: pd.DataFrame, note: str = "") -> None:
        md = f"## {tid}: {title}\n\n" + (f"{note}\n\n" if note else "") + to_markdown(df) + "\n"
        print("\n" + md, flush=True)
        out.append(md)

    # 1. cores
    table("t01_cores", "the three cores", core_table(cores, mkt),
          "vol_tax = arithmetic mean - geometric mean of the full calendar years; vol_tax_approx = their variance / 2.")
    table("t01b_core_crashes", "the cores over each crash window", core_crash_table(cores))
    navs = {CORES[k]: metrics.nav_from_returns(r) for k, r in cores.items()}
    fig = charts.growth_chart(navs)
    fig.update_layout(title=f"Growth of $1 in the three cores, {mkt.index[0]:%Y-%m} to {mkt.index[-1]:%Y-%m}")
    figs.append(save(fig, "cs_01_cores_growth"))
    for k, nav in zip(cores, navs.values()):
        fig = charts.underwater_chart(nav)
        fig.update_layout(title=f"{CORES[k]}: drawdown from the running peak (max {metrics.max_drawdown(nav):.1%})")
        figs.append(save(fig, f"cs_02_underwater_{k}"))

    # 2-4. every candidate on every core: sample windows, frontiers, the cost-effectiveness plane, the AQR twins
    rows = evaluate_all(cores, mkt, px, CANDIDATES)
    print(f"evaluated {len(rows)} (core, candidate) pairs in {time.time() - t0:.0f} s", flush=True)
    frontier = frontier_table(rows)
    table("t00_summary", "w*, dCAGR at w* and the verdict per candidate and core", verdict_matrix(frontier),
          "w* = the CAGR-maximizing sleeve weight (no cap); dCAGR = CAGR at w* minus the core's; the verdict is "
          "catalog.evaluate's.")
    table("t02_candidates", "candidates and their sample windows", candidate_table(rows))
    table("t03_frontiers", "frontiers at w* (and w_cap under the 30% drawdown cap)", frontier)
    table("t03b_sleeve_scores", "sleeve scores at w_scores (w*, or the smallest grid weight), portfolio points",
          sleeve_score_table(rows))
    ce = ce_table(rows)
    table("t03c_ce_plane", "the cost-effectiveness plane at w_eval: arithmetic cost, volatility tax saved, dCAGR", ce)
    table("t04_aqr", "the AQR tests: hold less equity at equal drawdown or equal CAGR; the critic's view", aqr_table(rows))
    for (core, key), r in rows.items():
        fig = charts.frontier_chart(r["_table"], DD_CAP, r["_cash"], r["w_star"])
        fig.update_layout(title=f"{CANDIDATES[key].name} on the {CORES[core]} core, {r['sample_start']:%Y-%m} to "
                                f"{r['sample_end']:%Y-%m}: CAGR and max drawdown by sleeve weight")
        figs.append(save(fig, f"cs_03_frontier_{core}_{key}", scale=1.0))
    for core in cores:
        fig = charts.ce_plane(catalog.rank([r for (c, _), r in rows.items() if c == core]))
        fig.update_layout(title=f"Cost-effectiveness plane at w*, {CORES[core]} core")
        figs.append(save(fig, f"cs_04_ce_plane_{core}"))
        fig = charts.ce_plane(ce[ce["core"] == core])
        fig.update_layout(title=f"Cost-effectiveness plane at w_eval (w*, else the preset's or the smallest grid weight), "
                                f"{CORES[core]} core")
        figs.append(save(fig, f"cs_04_ce_plane_weval_{core}"))

    # 5. hold more equity
    opts = {k: c for k, c in CANDIDATES.items() if c.kind == "option"}
    lev = evaluate_all(cores, mkt, px, opts, funding="relever")
    table("t05_relever", "hold more equity: relever (L = 1 / (1 - w)) vs carve-out, option sleeves",
          relever_table(rows, lev), "relever: the core levered to 1 / (1 - w) at rf + 50 bp, so the core stays 100% "
                                    "and the sleeve is financed on top (the overlays prot_put and collar hold the core "
                                    "themselves: 1 + w of equity); grid weights above 1/3 dropped.")
    for key in opts:
        r = lev["spx", key]
        fig = charts.frontier_chart(r["_table"], DD_CAP, r["_cash"], r["w_star"])
        fig.update_layout(title=f"{CANDIDATES[key].name}, relever funding on the S&P 500 TR core: CAGR and max "
                                "drawdown by sleeve weight (the core levered to 1 / (1 - w))")
        figs.append(save(fig, f"cs_05_relever_spx_{key}", scale=1.0))

    # 6. robustness
    rob = robustness_table(rows)
    table("t06_robustness", "rolling 10-year win rate and block-bootstrap P(beat) at w_eval", rob)
    table("t06b_decades", "dCAGR at w_eval by decade (bp/yr)", decade_table(rows))
    off = offset_table(rows, cores, mkt, px, CANDIDATES)
    table("t06c_roll_offsets", "roll-offset sweep: dCAGR at w_eval with rolls k business days later (bp/yr)", off)
    table("t06d_book_bootstrap", "book bootstrap, insurance archetype: 25-year median dCAGR and P(beat) by mode",
          book_bootstrap_table(rows))
    for col, name, fmt, title in (("rolling10_win_rate", "rolling10", ".0%", "Share of rolling 10-year windows the "
                                   "hedged portfolio wins, at w_eval"),
                                  ("p_beat_10y", "pbeat10", ".0%", "Block bootstrap P(10-year CAGR hedged > unhedged), "
                                   "at w_eval")):
        wide = rob.pivot(index="candidate", columns="core", values=col).reindex(index=list(CANDIDATES), columns=list(cores))
        figs.append(save(bars(wide.rename(columns=CORES), title, "candidate", "share (%)", fmt), f"cs_06_{name}"))
    spx_off = off[off["core"] == "spx"].set_index("candidate")[[f"d_cagr_off{k}_bp" for k in OFFSET_SWEEP]]
    figs.append(save(bars(spx_off.rename(columns=lambda c: f"offset {c[10:-3]} bdays"), "Roll-offset sweep on the S&P "
                          "500 TR core: dCAGR at w_eval by roll offset", "candidate", "dCAGR (bp/yr)", "+.0f"),
                     "cs_06_roll_offsets_spx"))

    # 7. dice
    spx_annual = scenarios.load_sp500_annual()
    reg = scenarios.regime_table(spx_annual)
    table("t07_regimes", f"S&P 500 calendar-year regimes {spx_annual.index[0]}-{spx_annual.index[-1]}",
          reg.drop(columns=["lo", "hi"]))
    table("t07b_cartoons", "the book's cartoon havens, reset annually: dCAGR over the S&P 500 (bp/yr)", cartoon_table(),
          "book_reported_bp: Universa, Safe Haven Investing Part One (2017), 1997-2016, as reported.")
    table("t07c_dice", "estimated-mode dice of the model-priced sleeves at w_eval (1928-latest regime frequencies)",
          dice_table(rows))
    table("t07d_crash_sweep", "dice dCAGR by crash-year frequency pi", crash_sweep_table(rows).reset_index())
    figs.append(save(charts.regime_table_fig(reg), "cs_07_regimes"))
    p, cj = reg["freq"].to_numpy(float), reg["mean_ret"].to_numpy(float)
    cart = scenarios.book_haven("insurance", p[0])
    w = np.linspace(0, 0.2, 201)
    P = scenarios.book_portfolio_returns(cj, cart, w)
    fig = charts.dice_chart(w, scenarios.dice_growth(p, P), scenarios.dice_arith(p, P), scenarios.kelly_w(p, cj, cart))
    fig.update_layout(title=f"Dice, the insurance cartoon (+{cart[0]:.0%} in crash years, -100% otherwise), "
                            f"{spx_annual.index[0]}-{spx_annual.index[-1]} regimes")
    figs.append(save(fig, "cs_07_dice_cartoon"))
    sweep = scenarios.crash_frequency_sweep(p, cj, cart, np.array([0.0, 0.03, 0.10]))
    fig = charts.crash_freq_chart(sweep)
    fig.update_layout(title="Growth by crash-year frequency, the book's insurance cartoon at 0%, 3% and 10%")
    figs.append(save(fig, "cs_07_crash_freq_cartoon"))

    # 8. pricing sensitivity
    for key in PRESETS:
        r = rows["spx", key]
        w = w_eval(r)
        sens = sensitivity_table(CANDIDATES[key], cores["spx"], mkt, px, r["_beta"], w)
        table(f"t08_sensitivity_{key}", f"{key} on the S&P 500 TR core: w* and dCAGR at w = {w:.2%} by pricing "
              "assumption", sens, f"base: m {pricing.SKEW_MULT:g}, nu 1, h {catalog.H_DEFAULT:g}, z_cap "
              f"{pricing.Z_CAP:g}, beta {pricing.SKEW_BETA:g}.")
        bars_ = backtest.tornado_rows(sens.iloc[1:], "d_cagr_bp", sens.at[0, "d_cagr_bp"])
        fig = charts.tornado(bars_, f"dCAGR at w = {w:.2%} (bp/yr)")
        fig.update_layout(title=f"{CANDIDATES[key].name}, S&P 500 TR core: sensitivity of dCAGR at w = {w:.2%} to the "
                                "pricing assumptions")
        figs.append(save(fig, f"cs_08_tornado_{key}"))

    # 9. validation and the Universa known answers
    table("t09_validation", "pricer validation against Cboe PPUT, PPUT3M and CLL (out of sample)",
          validation_table(mkt, pput3m))
    table("t09b_universa", "Universa known-answer checks: the 96.67/3.33 model portfolio (informational)",
          catalog.universa_checks(rows["spx", "universa_model"]["_nav"][catalog.W_UNIVERSA], mkt["spxtr"]),
          f"universa_reported: {catalog.UNIVERSA_LABEL}. ours: universa_model at 3.33% on the S&P 500 TR core.")
    with_3m = mkt.assign(pput3m=pput3m.reindex(mkt.index))
    for which in ("PPUT", "PPUT3M", "CLL"):
        synth = replicate.replicate(with_3m, which)
        actual = replicate.unsplice(with_3m[which.lower()], synth, which)
        figs.append(save(charts.replication_chart(synth, actual, replicate.WINDOWS[which][0], which),
                         f"cs_09_replication_{which.lower()}"))

    # 10. crisis windows
    sc_navs = {"S&P 500 TR (unhedged)": rows["spx", PRESETS[0]]["_nav"][0.0], "60/40": navs[CORES["6040"]]}
    for key in PRESETS:
        r = rows["spx", key]
        sc_navs[f"{key} at {w_eval(r):.2%}"] = r["_nav"][w_eval(r)]
    sc = scenario_table(sc_navs)
    table("t10_scenarios", "crisis windows: hedged vs unhedged vs 60/40 (S&P 500 TR core)", sc)
    figs.append(save(bars(sc.set_index("window")[list(sc_navs)], "Crisis windows: the option presets on the S&P 500 "
                          "TR vs unhedged and 60/40", "window", "return over the window (%)", ".0%"), "cs_10_scenarios"))

    runtime = time.time() - t0
    out.append(f"## Run\n\nRuntime {runtime:.0f} s. Figures ({len(figs)}): {', '.join(figs)}.\n")
    (FIGURES / "case_study_tables.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote figures/case_study_tables.md and {len(figs)} figures "
          f"({sum(f.endswith('.html') for f in figs)} as HTML); runtime {runtime:.0f} s")


if __name__ == "__main__":
    main()

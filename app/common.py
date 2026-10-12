"""Shared plumbing for the Streamlit pages: cached loaders and engine wrappers, the settings sidebar, the
portfolio guard, the upload check and provenance captions. No engine math here: every number comes from
insureport.

Cached wrappers take hashable arguments: primitives and tuples (an edited Candidate or a Policy travels
as dataclasses.astuple and is rebuilt inside), plus the core / sleeve return Series and rebalance dates,
which st.cache_data hashes by content (a DatetimeIndex through hash_funcs). One candidate's arguments
travel as an EvalArgs (eval_args), stored with its Evaluate result. The pages call everything through
`common.`, so tests monkeypatch market, prices_cached, pput3m and validation_cached here.
"""

import dataclasses
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from insureport import catalog, chains, data, pricing, replicate, scenarios
from insureport.backtest import Policy
from insureport.catalog import CANDIDATES, Candidate
from insureport.portfolio import Portfolio
from insureport.sleeves import Leg, OptionSpec

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
MAX_UPLOAD = 1_000_000  # bytes: the holdings CSV is a trust boundary
# build_sleeve's px: the catalog's ETF sources and the 60/40 reference's AGG
HAVEN_ETFS = tuple(sorted({c.source for c in CANDIDATES.values() if c.kind == "etf" or c.key == "sixty_forty"}))
FUNDING = ("carve_out", "relever")
# DEVIATION from the brief's h 0.05: catalog.H_DEFAULT (0.04, the addendum's 8%-of-mid deep-OTM spread), so the app
# matches catalog.evaluate's default
DEFAULTS = {"start": date(1990, 1, 1), "end": None, "dd_cap": 0.30, "rebalance": "preset", "band": None,
            "tc": Policy().tc_core, "funding": "carve_out", "leverage": False, "m": pricing.SKEW_MULT, "nu": 1.0,
            "h": catalog.H_DEFAULT, "offset": 0}
PROVENANCE = {
    "self-reported": f"Spitznagel / Universa figures are {catalog.UNIVERSA_LABEL}: shown for comparison, never fitted.",
    "model-priced": "Model-priced options (Black-Scholes-Merton on a VIX / SKEW surface), not traded quotes: see the "
                    "Methodology page for the validation against Cboe PPUT, PPUT3M and CLL.",
}


@st.cache_data(ttl="6h", show_spinner="Loading the market panel (Yahoo Finance, Cboe, FRED)...")
def market() -> pd.DataFrame:
    """The 1990-> market panel (data.load_market) with the VIX term structure back-filled
    (pricing.fill_term_structure). attrs: warnings, first_dates (each series' first date in this 1990-> panel,
    before any back-fill)."""
    raw = data.load_market("1990-01-01")
    mkt = pricing.fill_term_structure(raw)
    mkt.attrs["first_dates"] = {c: raw[c].first_valid_index() for c in raw}
    return mkt


@st.cache_data(ttl="6h", show_spinner="Fetching prices...")
def prices_cached(tickers: tuple[str, ...]) -> pd.DataFrame:
    """data.prices for these tickers (adjusted closes, one column per ticker with data)."""
    return data.prices(list(tickers))


@st.cache_data(ttl="6h", show_spinner=False)
def pput3m() -> pd.Series:
    """Cboe PPUT3M levels (not in the market panel)."""
    return data.cboe("PPUT3M")


@st.cache_data(ttl="6h", show_spinner="Fetching safe-haven ETFs and FRED DGS10...")
def haven_px() -> pd.DataFrame:
    """catalog.build_sleeve's px: HAVEN_ETFS closes, FRED DGS10 (percent) and pput3m; all-NaN columns dropped, so
    a failed fetch reads as a missing source (build_sleeve's ValueError) rather than a NaN sleeve."""
    px = pd.concat([prices_cached(HAVEN_ETFS), data.fred("DGS10").rename("DGS10"), pput3m().rename("pput3m")], axis=1)
    return px.dropna(axis=1, how="all")


# ponytail: one chain in memory, keyed on the path only (a file replaced in place is not reloaded until the
# process restarts); raise max_entries if users compare chains
@st.cache_resource(show_spinner="Loading the option chain...", max_entries=1)
def chain_cached(path: str | None) -> pd.DataFrame | None:
    """chains.load_chain of a file under data/chains (spot from the panel), once per process; None = model-priced.
    A resource, not data: a multi-GB chain is neither hashed nor copied."""
    return None if path is None else chains.load_chain(path, spot=market()["spx"])


class EvalArgs(NamedTuple):
    """One candidate's evaluate_cached arguments besides the core (hashable: the sensitivity sweep's key)."""

    params: tuple  # dataclasses.astuple of the (edited) Candidate
    pricing: tuple  # (m, nu, h, offset_bdays)
    chain_path: str | None  # a file under data/chains, None = model-priced
    cover: float  # overlay coverage, x the core's value (the downside beta by default)
    dd_cap: float
    policy: tuple  # dataclasses.astuple of the Policy (band, tc on both legs)
    funding: str


def candidate_from_tuple(t: tuple) -> Candidate:
    """Inverse of dataclasses.astuple(Candidate): an edited candidate's cache-key form back to the dataclass."""
    spec = None if t[4] is None else OptionSpec(tuple(Leg(*leg) for leg in t[4][0]), *t[4][1:])
    return Candidate(*t[:4], spec, *t[5:])


def eval_args(c: Candidate, s: dict, cover: float, chain_path: str | None = None) -> EvalArgs:
    """The candidate's EvalArgs under the settings s."""
    return EvalArgs(dataclasses.astuple(c), (float(s["m"]), float(s["nu"]), float(s["h"]), int(s["offset"])),
                    chain_path, cover, s["dd_cap"],
                    dataclasses.astuple(Policy(band=s["band"], tc_core=s["tc"], tc_sleeve=s["tc"])), s["funding"])


def coverage(beta_down: float) -> float:
    """Default overlay coverage: the downside beta rounded to 0.01, inside the coverage input's [0.01, 3]."""
    return min(max(round(beta_down, 2), 0.01), 3.0)


def _pricing_kw(t: tuple) -> dict:
    m, nu, h, offset = t
    return {"m": m, "nu": nu, "h": h, "offset_bdays": offset}


@st.cache_data(show_spinner=False, max_entries=128)
def sleeve_cached(
    core_r: pd.Series, params_tuple: tuple, pricing_args: tuple, chain_path: str | None, beta_down: float
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """catalog.build_sleeve of the candidate astuple'd in params_tuple on market() and haven_px(): the core
    core_r, coverage beta_down (overlays), pricing_args (m, nu, h, offset_bdays), the chain at chain_path."""
    return catalog.build_sleeve(candidate_from_tuple(params_tuple), market(), haven_px(), core_r, beta_down,
                                pricing_kw=_pricing_kw(pricing_args), chain=chain_cached(chain_path))


@st.cache_data(show_spinner=False, max_entries=128, hash_funcs={pd.DatetimeIndex: lambda i: i.asi8.tobytes()})
def frontier_cached(
    core_r: pd.Series, sleeve_r: pd.Series, grid: tuple[float, ...], rebalance: pd.DatetimeIndex, policy_args: tuple,
    funding: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """catalog.funded_frontier: the frontier table and NAV frame over grid, carve-out or relever funding."""
    sleeve_r = sleeve_r.copy()
    sleeve_r.attrs = {}  # pandas deep-copies attrs into every derived object
    return catalog.funded_frontier(core_r, sleeve_r, np.array(grid), rebalance, Policy(*policy_args), funding,
                                   market()["rf"])


@st.cache_data(show_spinner=False, max_entries=64)
def evaluate_cached(
    core_r: pd.Series, params_tuple: tuple, pricing_args: tuple, chain_path: str | None, beta_down: float,
    dd_cap: float, policy_args: tuple, funding: str,
) -> dict:
    """catalog.evaluate of the candidate on its sleeve_cached sleeve (row keys as evaluate's, _table, _nav and
    _cash included); the arguments after core_r are an EvalArgs'."""
    sleeve, cal = sleeve_cached(core_r, params_tuple, pricing_args, chain_path, beta_down)
    return catalog.evaluate(candidate_from_tuple(params_tuple), core_r, sleeve, cal, market(), dd_cap=dd_cap,
                            policy=Policy(*policy_args), funding=funding, px=haven_px(), beta_down=beta_down,
                            pricing_kw=_pricing_kw(pricing_args), chain=chain_cached(chain_path))


@st.cache_data(show_spinner=False, max_entries=32)
def bootstrap_cached(
    rc: pd.Series, rs: pd.Series, grid: tuple[float, ...], periods_per_year: int, horizons: tuple[int, ...],
    dd_cap: float,
) -> pd.DataFrame:
    """scenarios.bootstrap (stationary blocks, 2,000 paths, seed 0) over grid and horizons."""
    return scenarios.bootstrap(rc, rs, np.array(grid), periods_per_year, horizons, dd_cap=dd_cap)


@st.cache_data(ttl="6h", show_spinner=False)  # refreshed with market()
def validation_cached(m: float, nu: float, h: float) -> pd.DataFrame:
    """replicate.validate on market() with PPUT3M: the pricer's PASS / FAIL table at these assumptions, on
    Cboe's fill convention (replicate's default: closing mid + FILL_ADD per option per roll through 2015)."""
    return replicate.validate(market(), m=m, nu=nu, h=h, pput3m=pput3m())


@st.cache_data(ttl="6h", show_spinner=False)
def replica_cached(which: str, m: float, nu: float) -> pd.Series:
    """replicate.replicate level of PPUT / PPUT3M / CLL on market() (pput3m added), on Cboe's fill convention
    (closing mid + FILL_ADD per option per roll through 2015, the mid after), h = 0."""
    return replicate.replicate(market().assign(pput3m=pput3m().reindex(market().index)), which, m=m, nu=nu)


def settings_sidebar() -> dict:
    """The sidebar settings, stored in st.session_state["settings"] and returned: start / end dates,
    drawdown cap, rebalance ("preset" keeps each candidate's own calendar), band, transaction cost,
    leverage toggle (off forces carve-out funding), funding, and the pricing assumptions m, nu, h and
    roll offset; then the validation banner for model-priced options. A market panel that cannot load (no network
    and no cache) stops the run with an error rather than a traceback; st.cache_data caches no exception, so the
    next run retries."""
    try:
        mkt = market()
    except RuntimeError as e:
        st.error(f"Market data unavailable ({e}): check the network, or point INSUREPORT_DATA_DIR at a filled cache.")
        st.stop()
    first, last = mkt.index[0].date(), mkt.index[-1].date()
    sb = st.sidebar
    sb.header("Settings")
    start = sb.date_input("Start", value=max(DEFAULTS["start"], first), min_value=first, max_value=last, key="set_start")
    end = sb.date_input("End", value=last, min_value=first, max_value=last, key="set_end")
    dd_cap = sb.slider("Drawdown cap (%)", 5, 60, int(DEFAULTS["dd_cap"] * 100), key="set_dd_cap",
                       help="Sizes w_cap and the AQR deleverage twins; w* is the uncapped CAGR maximum.") / 100
    rebalance = sb.selectbox("Rebalance", ("preset", *catalog.REBALANCE), key="set_rebalance",
                             help="'preset' keeps each candidate's own calendar; anything else overrides them all.")
    band = sb.number_input("Rebalance band (relative; 0 = calendar only)", 0.0, 1.0, float(DEFAULTS["band"] or 0.0),
                           0.05, key="set_band")
    tc = sb.number_input("Transaction cost (bp per trade)", 0.0, 100.0, DEFAULTS["tc"] * 1e4, 0.5, key="set_tc") / 1e4
    leverage = sb.toggle("Allow leverage", value=DEFAULTS["leverage"], key="set_leverage",
                         help="Unlocks relever funding (equity kept at 100%, the sleeve financed at rf + 50 bp) and the "
                              "'hold more equity' lines (1.25x, 1.5x) on Diagnostics.")
    funding = sb.radio("Funding", FUNDING, FUNDING.index(DEFAULTS["funding"]), key="set_funding", disabled=not leverage,
                       horizontal=True,
                       help="carve_out: (1 - w) core + w sleeve; relever: the core levered to 1 / (1 - w).")
    with sb.expander("Pricing assumptions"):
        m = st.number_input("SKEW multiplier m", 0.0, 3.0, float(DEFAULTS["m"]), 0.05, key="set_m")
        nu = st.number_input("Vol multiplier nu", 0.5, 2.0, float(DEFAULTS["nu"]), 0.05, key="set_nu")
        h = st.number_input("Half-spread h (x mid)", 0.0, 0.5, float(DEFAULTS["h"]), 0.01, key="set_h")
        offset = st.number_input("Roll offset (business days)", 0, 20, int(DEFAULTS["offset"]), 1, key="set_offset")
    with st.spinner("Validating the option pricer against Cboe PPUT / PPUT3M / CLL..."):
        try:  # the replicas fill on Cboe's convention (mid + FILL_ADD through 2015), h = 0: h is the strategies' spread
            v = validation_cached(float(m), float(nu), 0.0)
            sb.markdown(f"model-priced options: validation **{'PASS' if v['passed'].all() else 'FAIL'}**")
            sb.caption(", ".join(f"{k} {'pass' if ok else 'fail'}" for k, ok in v["passed"].items()))
        except ValueError as e:
            sb.markdown(f"model-priced options: validation **n/a** ({e})")
    s = {"start": start, "end": end, "dd_cap": dd_cap, "rebalance": rebalance, "band": band or None, "tc": tc,
         "funding": funding if leverage else "carve_out", "leverage": leverage, "m": float(m), "nu": float(nu),
         "h": float(h), "offset": int(offset)}
    st.session_state["settings"] = s
    return s


def settings() -> dict:
    """The sidebar's settings, or DEFAULTS before the sidebar has run (a page run on its own)."""
    return st.session_state.get("settings", DEFAULTS)


def window(x: pd.Series) -> pd.Series:
    """x cut to the sidebar's start / end dates."""
    s = settings()
    return x.loc[pd.Timestamp(s["start"]) : None if s["end"] is None else pd.Timestamp(s["end"])]


def require_portfolio() -> tuple[Portfolio, pd.Series]:
    """The loaded Portfolio and its core returns inside the sidebar's dates; st.stop() with a pointer to the
    Portfolio page when none is loaded (or the dates leave fewer than two days)."""
    if "portfolio" not in st.session_state:
        st.warning("No portfolio loaded yet: build one on the **Portfolio** page first.")
        st.stop()
    core = window(st.session_state["core_r"])
    if len(core) < 2:
        st.warning("The sidebar's dates leave fewer than two days of the portfolio's history (see the **Portfolio** "
                   "page for its sample).")
        st.stop()
    return st.session_state["portfolio"], core


def read_upload(f: UploadedFile) -> str:
    """Text of an uploaded holdings CSV (st.file_uploader's UploadedFile); ValueError over MAX_UPLOAD bytes
    or when it is not UTF-8 (a byte-order mark is dropped)."""
    if f.size > MAX_UPLOAD:
        raise ValueError(f"the file is {f.size:,} bytes; the limit is 1 MB")
    try:
        return f.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(f"not UTF-8 text ({e})") from None


def note(text: str) -> None:
    """Provenance caption: PROVENANCE's wording for "self-reported" / "model-priced", else text as given
    (e.g. "Cboe index levels back-tested before 2015-08-03")."""
    st.caption("Provenance: " + PROVENANCE.get(text, text))


def fingerprint(core: pd.Series) -> tuple:
    """The core's first and last date and a content hash: what a stored result was computed on."""
    return core.index[0], core.index[-1], int(pd.util.hash_pandas_object(core).sum())


def evaluated(core: pd.Series) -> dict:
    """The last Evaluate's results (candidate key -> {"args": EvalArgs, "row": evaluate's dict}) when they were run
    on this core and the current settings; else {} with a pointer to re-run Evaluate (a rebuilt portfolio, other
    dates or another setting since)."""
    ev = st.session_state.get("evaluated")
    if ev and st.session_state.get("evaluated_on") != (fingerprint(core), dict(settings())):
        st.info("The portfolio, its dates or a sidebar setting changed since the last Evaluate: run **Evaluate** on "
                "the Candidates page again for its results.")
        return {}
    return ev or {}


def launch_note(c: Candidate, start: pd.Timestamp) -> None:
    """The "back-tested before <launch>" caption for a Cboe index candidate whose sample starts before the launch."""
    launch = catalog.LAUNCH.get(c.source) if c.kind == "index" else None
    if launch and start < pd.Timestamp(launch):
        note(f"{c.name}: Cboe index levels back-tested before {launch}")

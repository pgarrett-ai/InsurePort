"""Real end-of-day SPX option data: vendor CSV files -> one canonical chain, plus contract and quote lookup.

load_chain reads a purchased end-of-day file (historicaloptiondata.com Level 2 / Level 3 / 2010+ /
ALLSPX, Cboe DataShop, ORATS, OptionMetrics IvyDB; a zipped CSV is fine) and normalize maps it onto
the canonical schema through one header-alias table (ALIASES), not one parser per vendor
(docs/research/data-and-implementation.md, Q8 and design implication 10). sleeves.option_sleeve(chain=)
prices its legs from the result (select_contract, quotes).

Canonical chain (COLUMNS): one row per option per quote date: quote_date, expiration (datetime64,
midnight), strike (float, index points), cp ("C" / "P"), bid, ask, mid = (bid + ask) / 2 (floats; mid,
never Last), underlying (float, NaN allowed), root ("SPX", "SPXW" or the vendor's root string), oi
(open interest, float, NaN allowed); sorted by (quote_date, expiration, strike, cp).
The file is user-supplied data: an unknown layout raises ValueError, unusable rows and rows of another
underlying are dropped and counted (normalize). Snapshot times differ by vendor (historicaloptiondata
~16:00 ET, DataShop 15:45 and EOD, ORATS 14 minutes before the close) and are not recorded.
"""

import os
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd

COLUMNS = ["quote_date", "expiration", "strike", "cp", "bid", "ask", "mid", "underlying", "root", "oi"]
REQUIRED = ("quote_date", "expiration", "strike", "cp", "bid", "ask")
VENDORS = ("hod", "datashop", "orats", "ivydb")
# canonical -> raw names after normalize_headers, first match wins (the research's field map; the canonical
# names themselves come last so a saved canonical chain reads back). ORATS' wide c / p columns are melted
# into bid / ask / openinterest / type first (_MELT). root: exchange / flags are historicaloptiondata's
# W (SPXW) / * (consolidated, SPX) marks.
ALIASES: dict[str, tuple[str, ...]] = {
    "quote_date": ("datadate", "quotedate", "tradedate", "date"),
    "expiration": ("expiration", "expirdate", "exdate"),
    "strike": ("strike", "strikeprice"),
    "cp": ("type", "optiontype", "cpflag", "cp"),
    "bid": ("bid", "bideod", "bid1545", "bestbid"),
    "ask": ("ask", "askeod", "ask1545", "bestoffer"),
    "underlying": ("underlyingprice", "underlyinglast", "activeunderlyingprice1545", "stkpx", "spotpx", "underlying"),
    "root": ("optionroot", "root", "exchange", "flags"),
    "oi": ("openinterest", "oi"),
}
# the underlying's symbol (historicaloptiondata UnderlyingSymbol / 2010+ underlying, DataShop ^SPX, ORATS ticker)
_SYMBOL = ("underlyingsymbol", "ticker", "underlying")
# ORATS wide columns per row type (the near-EOD CSV, research Q8)
_MELT = {"C": {"cbidpx": "bid", "caskpx": "ask", "coi": "openinterest"},
         "P": {"pbidpx": "bid", "paskpx": "ask", "poi": "openinterest"}}


def normalize_headers(cols: Iterable) -> list[str]:
    """Header names lower-cased with every space and underscore removed (" DataDate" -> "datadate",
    "last " -> "last", "open interest" -> "openinterest", "bid_eod" -> "bideod")."""
    return ["".join(str(c).lower().split()).replace("_", "") for c in cols]


def detect_vendor(cols: Iterable) -> str:
    """The layout of a chain file from its header (raw or normalized names).

    "orats" (wide cBidPx / cAskPx / pBidPx / pAskPx), "ivydb" (best_bid and best_offer), "datashop"
    (bid_eod / ask_eod / bid_1545 / ask_1545), else "hod": historicaloptiondata Level 2 / Level 3 /
    2010+ / ALLSPX, or any long layout naming the REQUIRED fields by ALIASES (a saved normalize result
    too). Raises ValueError naming the canonical columns needed and the vendors recognised when none fits.
    """
    cols = list(cols)  # read twice (the error names them)
    c = set(normalize_headers(cols))
    if {"cbidpx", "caskpx", "pbidpx", "paskpx"} <= c:
        return "orats"
    if {"bestbid", "bestoffer"} <= c:  # not strike_price alone: a common name, and IvyDB's is x 1000
        return "ivydb"
    if c & {"bideod", "askeod", "bid1545", "ask1545"}:
        return "datashop"
    if all(c & set(ALIASES[k]) for k in REQUIRED):
        return "hod"
    raise ValueError(
        f"unrecognised option-chain header {list(cols)}: need columns for {', '.join(REQUIRED)} "
        f"(names accepted: chains.ALIASES) in a layout of one of the vendors {', '.join(VENDORS)}"
    )


def normalize(df: pd.DataFrame, *, spot: pd.Series | None = None) -> pd.DataFrame:
    """Any vendor frame (raw headers, as read from the file) -> the canonical chain (module docstring).

    Headers via normalize_headers, the vendor via detect_vendor, each canonical column from the first
    ALIASES name present. ORATS' wide rows become one call and one put row each (_MELT).
    Vendor rules: cp is the first letter of the type upper-cased ("call" / "put" / "C" / "P"); IvyDB's
    strike_price is divided by 1000; DataShop's bid_eod / ask_eod are used where present, else
    bid_1545 / ask_1545; root: historicaloptiondata's Exchange / Flags containing "W" -> "SPXW", else
    "SPX" (an ALLSPX optionroot or DataShop root column wins), "SPX" when the file names none;
    underlying: a price <= 0 (DataShop's active_underlying_price_1545 without an index licence) or
    missing is NaN, then filled from `spot` (the panel's spx) on the quote date when spot is given.
    Dates are parsed with the format pandas infers from the first value, then floored to midnight.
    SPX rows only: where the file names the underlying's symbol (_SYMBOL; the 2010+ `underlying` column
    when it is not numeric), rows of any other symbol (RUT, NDX, XSP ... in a multi-symbol file; caret
    stripped, "^SPX" is SPX) are dropped, so they cannot pose as SPX contracts. Also dropped: rows without
    a valid quote_date, expiration, strike or C / P, or with ask <= 0, bid < 0, bid > ask or a missing
    bid / ask; attrs: vendor, dropped (the count of both).
    Raises ValueError for an unknown layout or a vendor layout missing a REQUIRED column.
    """
    # ponytail: ORATS and IvyDB rows carry no SPX / SPXW split (all "SPX"), so their weeklies count as monthlies
    # and on a third Friday the AM monthly and a PM SPXW weekly share one (quote_date, expiration, strike, cp)
    # key, the one every sleeve opens; quotes() keeps whichever row sorts first. Split the root on IvyDB's
    # symbol / am_settlement fields (unverified, research Q8) and count duplicate keys before relying on these
    vendor = detect_vendor(df.columns)
    df = df.set_axis(normalize_headers(df.columns), axis=1)
    if vendor == "orats":
        df = pd.concat([df.rename(columns=m).assign(type=cp) for cp, m in _MELT.items()], ignore_index=True)
    raw = {k: next((df[a] for a in names if a in df), None) for k, names in ALIASES.items()}
    missing = [k for k in REQUIRED if raw[k] is None]
    if missing:
        raise ValueError(f"{vendor} chain has no column for {missing} (names accepted: chains.ALIASES)")

    def num(k: str) -> pd.Series:
        return pd.Series(np.nan, df.index) if raw[k] is None else pd.to_numeric(raw[k], errors="coerce").astype(float)

    def date(k: str) -> pd.Series:
        return pd.to_datetime(raw[k].astype(str), errors="coerce").dt.floor("D")

    qd, bid, ask, u = date("quote_date"), num("bid"), num("ask"), num("underlying")
    u = u.where(u > 0)
    if spot is not None:
        u = u.fillna(pd.Series(spot.reindex(qd).to_numpy(float), df.index))
    sym = next((df[a] for a in _SYMBOL if a in df and not pd.api.types.is_numeric_dtype(df[a])), None)
    spx = True if sym is None else sym.astype(str).str.strip().str.upper().str.lstrip("^").eq("SPX")
    src = next((a for a in ALIASES["root"] if a in df), None)
    if src is None:
        root = pd.Series("SPX", df.index)
    elif src in ("exchange", "flags"):
        root = pd.Series(np.where(df[src].astype(str).str.upper().str.contains("W"), "SPXW", "SPX"), df.index)
    else:
        root = df[src].astype(str).str.strip()
    out = pd.DataFrame({
        "quote_date": qd, "expiration": date("expiration"),
        "strike": num("strike") / (1000 if vendor == "ivydb" else 1),
        "cp": raw["cp"].astype(str).str.strip().str.upper().str[0],
        "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "underlying": u, "root": root, "oi": num("oi"),
    })
    ok = (out[["quote_date", "expiration", "strike"]].notna().all(axis=1) & out["cp"].isin(["C", "P"])
          & (ask > 0) & (bid >= 0) & (bid <= ask) & spx)
    out = out[ok].sort_values(["quote_date", "expiration", "strike", "cp"], ignore_index=True)
    out.attrs = {"vendor": vendor, "dropped": int((~ok).sum())}
    return out


def load_chain(path: str | os.PathLike, *, spot: pd.Series | None = None) -> pd.DataFrame:
    """normalize(read_csv(path), spot=spot): a vendor CSV, or a zip holding one (compression inferred from
    the extension; read_csv skips a UTF-8 byte-order mark). attrs["vendor"], attrs["dropped"] as normalize."""
    # ponytail: reads every column into memory; pass usecols from the header if multi-GB ALLSPX years strain RAM
    return normalize(pd.read_csv(path), spot=spot)


def select_contract(
    chain: pd.DataFrame, date: pd.Timestamp | str, target_expiry: pd.Timestamp | str, target_strike: float, cp: str,
    root: str = "SPX",
) -> tuple[pd.Timestamp, float] | None:
    """The listed contract closest to a target, among the chain's rows on `date` with this root and cp
    that expire after `date`. DEVIATION from the brief (every expiration listed that day): an option
    expiring on `date` is not one to open.

    Expiration: the nearest listed one >= target_expiry, or the latest one before it when none is >=.
    Strike on that expiration: puts the largest listed <= target_strike, calls the smallest listed >=
    it. Returns (expiration, strike), or None when nothing qualifies (no rows, or no strike on the
    right side of the target).
    """
    # ponytail: an O(rows) date scan per call, one per leg open (~25 ms on 5M rows, most of a chain-priced
    # sleeve's run time); slice the sorted quote_date with searchsorted if full-history chains make sleeves slow
    date = pd.Timestamp(date)
    day = chain[chain["quote_date"] == date]
    day = day[(day["root"] == root) & (day["cp"] == cp) & (day["expiration"] > date)]
    if day.empty:
        return None
    exp, target = day["expiration"], pd.Timestamp(target_expiry)
    e = exp[exp >= target].min() if (exp >= target).any() else exp.max()
    k = day.loc[exp == e, "strike"]
    k = k[k <= target_strike].max() if cp == "P" else k[k >= target_strike].min()
    return None if np.isnan(k) else (pd.Timestamp(e), float(k))


def quotes(
    chain: pd.DataFrame, date: npt.ArrayLike, expiration: npt.ArrayLike, strike: npt.ArrayLike, cp: npt.ArrayLike,
    root: str = "SPX",
) -> np.ndarray:
    """Elementwise quotes: a (3, n) array of (mid, bid, ask) for each (date, expiration, strike, cp) listed
    under `root`, NaN where it is not; a key listed twice takes its first row. Dates of any datetime unit."""

    def key(q: npt.ArrayLike, e: npt.ArrayLike, k: npt.ArrayLike, p: npt.ArrayLike) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([pd.DatetimeIndex(np.atleast_1d(q)).as_unit("ns"),
                                          pd.DatetimeIndex(np.atleast_1d(e)).as_unit("ns"),
                                          np.atleast_1d(k).astype(float), np.atleast_1d(p).astype(object)])

    want = key(date, expiration, strike, cp)
    # only the rows of the wanted expirations and strikes are indexed: a sleeve's legs touch few contracts
    c = chain[(chain["root"] == root) & chain["expiration"].isin(want.levels[1]) & chain["strike"].isin(want.levels[2])]
    c = c.drop_duplicates(["quote_date", "expiration", "strike", "cp"])
    pos = key(c["quote_date"], c["expiration"], c["strike"], c["cp"]).get_indexer(want)
    vals = np.vstack([c[["mid", "bid", "ask"]].to_numpy(float), np.full(3, np.nan)])  # row -1 (not listed): NaN
    return vals[pos].T


def quote_at(
    chain: pd.DataFrame, date: pd.Timestamp | str, expiration: pd.Timestamp | str, strike: float, cp: str,
    root: str = "SPX",
) -> tuple[float, float, float] | None:
    """(mid, bid, ask) of the option listed under `root` on `date`, or None when the chain has no such row.
    DEVIATION from the brief (no root argument): SPX and SPXW can list the same (expiration, strike, cp),
    so the lookup names the root, as select_contract does."""
    mid, bid, ask = quotes(chain, date, expiration, strike, cp, root)[:, 0]
    return None if np.isnan(mid) else (float(mid), float(bid), float(ask))

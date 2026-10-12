"""Market data: fetch, parquet cache and the daily market panel.

Only two functions touch the network, `_yf_download` and `_http_text`; tests monkeypatch
them, and both raise when INSUREPORT_OFFLINE=1. Fetchers are fail-soft: a failed refresh
returns the stale cache or an empty Series with a logged warning, never an exception.
"""

import io
import logging
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
MAX_AGE_S = 20 * 3600
YF_START = "1927-01-01"  # always fetch full history so the cache never depends on `start`
COLUMNS = ["spx", "spx_open", "spxtr", "vix", "vix9d", "vix3m", "vix6m", "vix1y", "skew", "rf", "q", "pput", "cll", "vxth"]
REQUIRED = {"spx", "spxtr", "vix", "skew", "rf"}
_EMPTY = pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def data_dir() -> Path:
    """Cache root: $INSUREPORT_DATA_DIR (read at call time), default ./data."""
    return Path(os.environ.get("INSUREPORT_DATA_DIR", "data"))


def _yf_download(tickers: list[str], start: str, field: str = "Close") -> pd.DataFrame:
    """SEAM. Adjusted daily `field` ("Close", or "Open") from yfinance; columns = tickers, tz-naive DatetimeIndex."""
    if os.environ.get("INSUREPORT_OFFLINE") == "1":
        raise RuntimeError(f"INSUREPORT_OFFLINE=1: refusing yfinance download of {tickers}")
    import yfinance as yf  # lazy: heavy import, only needed live

    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)[field]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    # ponytail: drops today's (possibly in-progress) bar even after the close; it arrives at the next refresh
    px = px[px.index < pd.Timestamp.now("America/New_York").normalize().tz_localize(None)]
    return px.reindex(columns=tickers)


def _http_text(url: str) -> str:
    """SEAM. GET `url` (stdlib urllib, follows redirects) with a User-Agent and a 30 s timeout."""
    if os.environ.get("INSUREPORT_OFFLINE") == "1":
        raise RuntimeError(f"INSUREPORT_OFFLINE=1: refusing GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (InsurePort research tool)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8-sig")


def parse_cboe_csv(text: str, symbol: str) -> pd.Series:
    """Parse a Cboe daily-prices CSV: the CLOSE column if present, else the single value column.

    Header case is ignored; dates may be %m/%d/%Y or ISO. Returns a float Series sorted by date.
    """
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "CLOSE" in df.columns:
        col = "CLOSE"
    elif len(df.columns) == 2:
        col = df.columns[1]
    else:  # a renamed close column must not silently become OPEN
        raise ValueError(f"{symbol}: no CLOSE column in {list(df.columns)}")
    idx = pd.DatetimeIndex(pd.to_datetime(df["DATE"], format="mixed"))
    return pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(float), idx, name=symbol).sort_index()


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a FRED fredgraph CSV (date, value); '.' (missing) becomes NaN."""
    df = pd.read_csv(io.StringIO(text), na_values=".", index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float).rename(series_id)


def _cached(source: str, key: str, fetch: Callable[[], pd.Series]) -> pd.Series:
    """Series `key` from data_dir()/source/key.parquet (`^` stripped), refetched when older than 20 h.

    Duplicate dates keep the last value. An unreadable file counts as stale. An empty or failed
    refresh never overwrites the cache: the stale data is returned with a warning, or an empty
    float Series if there is none. If the cache cannot be written, the fresh data is returned.
    """
    # ponytail: mtime staleness; per-series last-date checks if Yahoo backfills bite
    path = data_dir() / source / f"{key.replace('^', '')}.parquet"
    old = None
    if path.exists():
        try:
            old = pd.read_parquet(path)["value"].rename(key)
        except Exception as e:  # corrupt file: refetch rather than mask the series for 20 h
            log.warning("%s/%s: unreadable cache (%s); refetching", source, key, e)
        else:
            if time.time() - path.stat().st_mtime <= MAX_AGE_S:
                return old
    try:
        s = fetch().dropna().astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            raise ValueError("empty response")
    except Exception as e:  # fail-soft: network, HTTP, parse errors alike
        if old is None:
            log.warning("%s/%s: fetch failed (%s); no cache, returning empty series", source, key, e)
            return _EMPTY.rename(key)
        log.warning("%s/%s: refresh failed (%s); using stale cache", source, key, e)
        return old
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)  # unique: concurrent writers can't interleave
        os.close(fd)
        pd.DataFrame({"value": s}).rename_axis("date").to_parquet(tmp)
        os.replace(tmp, path)  # atomic: readers never see a half-written file
    except OSError as e:
        log.warning("%s/%s: cache write failed (%s); using fresh data uncached", source, key, e)
        return s.rename_axis("date").rename(key)
    return pd.read_parquet(path)["value"].rename(key)


def prices(tickers: list[str], start: str = YF_START) -> pd.DataFrame:
    """Adjusted daily close per ticker from `start` (yfinance, one cache file per ticker).

    Tickers are stripped and upper-cased (yfinance upper-cases them; the cache key must match).
    Tickers with no data from `start` are dropped with a warning and listed in df.attrs["missing"].
    """
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers))
    # ponytail: one download per stale ticker; batch through _yf_download if portfolios get long
    got = {t: _cached("yfinance", t, lambda t=t: _yf_download([t], YF_START)[t]).loc[start:] for t in tickers}
    missing = [t for t, s in got.items() if s.empty]
    if missing:
        log.warning("no price data from %s for %s; dropped", start, missing)
    df = pd.DataFrame({t: s for t, s in got.items() if not s.empty})
    df.index = pd.DatetimeIndex(df.index)  # all-missing gives a RangeIndex otherwise
    df.attrs["missing"] = missing
    return df


def cboe(symbol: str) -> pd.Series:
    """Daily close of a Cboe index (VIX, VIX9D, VIX6M, VIX1Y, SKEW, PPUT, CLL, VXTH) from the Cboe CDN, cached."""
    return _cached("cboe", symbol, lambda: parse_cboe_csv(_http_text(CBOE_URL.format(sym=symbol)), symbol))


def fred(series_id: str) -> pd.Series:
    """FRED series (e.g. DTB3, DGS10) in its native units, cached."""
    return _cached("fred", series_id, lambda: parse_fred_csv(_http_text(FRED_URL.format(sid=series_id)), series_id))


def load_market(start: str = "1990-01-01") -> pd.DataFrame:
    """Daily market panel on ^GSPC trading days from `start`.

    Columns: spx / spx_open / spxtr (S&P 500 close / open / total-return index), vix, vix9d, vix3m,
    vix6m, vix1y (Cboe VIX1Y, from 2007-01-03), skew, rf (DTB3 / 100, decimal), q (dividend yield),
    pput, cll, vxth. spx_open (^GSPC Open, the replicas' proxy for Cboe's AM settlement and strike
    reference) is not filled: a missing open stays NaN and sleeves use that row's close. Every
    non-SPX series is forward-filled at most 5 rows. q = ln(TR_t/TR_{t-252}) - ln(PX_t/PX_{t-252}) clipped to
    [0, 0.06], computed on full history so the lookback reaches before `start`; gaps in TR carry
    the last q forward. ASSUMPTION: q = 0.03 until TR has 252 rows of history (TR starts 1988,
    so this only matters before 1989).

    Raises RuntimeError if spx, spxtr, vix, skew or rf has no data from `start`. Every other
    column with missing rows is described in df.attrs["warnings"].
    """
    px = prices(["^GSPC", "^SP500TR", "^VIX3M"])
    # ponytail: ^GSPC Open is the 9:30 index print, a stale SOQ proxy: it equals the previous close on 60-96% of
    # days before 2005 (31% 2005-09, 10% 2010-15, none from 2016), and from 2016 its overnight gap is 0.71x SPY's
    # traded one (2016-01-15: -0.27% vs -2.69%). Upgrade: SPY's open scaled to SPX (1993->), or Cboe SET values
    spx_open = _cached("yfinance", "^GSPC_open", lambda: _yf_download(["^GSPC"], YF_START, "Open")["^GSPC"])
    raw = {
        "spx": px.get("^GSPC", _EMPTY),
        "spxtr": px.get("^SP500TR", _EMPTY),
        "vix": cboe("VIX"),
        "vix9d": cboe("VIX9D"),
        "vix3m": px.get("^VIX3M", _EMPTY),
        "vix6m": cboe("VIX6M"),
        "vix1y": cboe("VIX1Y"),
        "skew": cboe("SKEW"),
        "rf": fred("DTB3") / 100,
        "pput": cboe("PPUT"),
        "cll": cboe("CLL"),
        "vxth": cboe("VXTH"),
    }
    idx = raw["spx"].dropna().index
    df = pd.DataFrame({k: s.reindex(idx).ffill(limit=5) for k, s in raw.items()}, index=idx)
    df["spx_open"] = spx_open.reindex(idx)  # unfilled: a stale open must not settle or strike a roll
    tr = raw["spxtr"].reindex(idx)  # unfilled: a missing TR print must not pair with a real PX move
    excess = np.log(tr / tr.shift(252)) - np.log(df["spx"] / df["spx"].shift(252))
    # ponytail: q is a one-day 252-row estimate, so one-day ln(TR/PX) glitches show through (Yahoo's
    # 2019-07-03 ^SP500TR print gives q = 0.0144 vs ~0.020 around it); take a rolling median or
    # filter reversals if pricing turns out sensitive.
    df["q"] = excess.clip(0, 0.06).ffill().fillna(0.03)
    df = df[COLUMNS].loc[start:]

    warnings = []
    for c in COLUMNS:
        n = int(df[c].isna().sum())
        if n == len(df):
            if c in REQUIRED:
                raise RuntimeError(f"load_market: required series {c!r} has no data from {start}")
            warnings.append(f"{c}: no data")
        elif n:
            warnings.append(f"{c}: {n} of {len(df)} rows missing (data from {df[c].first_valid_index():%Y-%m-%d})")
    df.attrs["warnings"] = warnings
    return df

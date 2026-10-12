"""Tests for insureport.data: the Cboe and FRED CSV parsers, the parquet cache (fresh hit,
20 h refresh, stale fallback, empty fallback, duplicate dates, corrupt file, unwritable dir),
the network seams (offline guard, request headers, yfinance call and post-processing, the Open
field), prices() normalising and dropping tickers, load_market() panel assembly (calendar, source
mapping incl. the VIX1Y pillar and the S&P 500 open, ffill limit and the unfilled open, rf scaling,
the dividend yield q with its clip and TR gaps, required-series errors, a failed Open fetch reported)
and the committed fixture's columns.

Why: every later module trusts this panel. All network I/O goes through the two seams
(_yf_download, _http_text), which are monkeypatched here with synthetic series, so the
suite is offline and deterministic.
"""

import logging
import os
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insureport import data


def test_parse_cboe_ohlc_uses_close_sorted_float():
    text = (
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "01/03/1990,1,2,0.5,18.19\n"
        "01/02/1990,3,4,2.5,17.24\n"
        "01/04/1990,5,6,4.5,19.22\n"
    )
    s = data.parse_cboe_csv(text, "VIX")
    assert list(s.index) == list(pd.to_datetime(["1990-01-02", "1990-01-03", "1990-01-04"]))
    assert s.tolist() == [17.24, 18.19, 19.22]
    assert s.dtype == np.float64 and s.name == "VIX"
    with pytest.raises(ValueError, match="CLOSE"):  # renamed close column: never fall back to OPEN
        data.parse_cboe_csv("DATE,OPEN,HIGH,LOW,LAST\n01/02/1990,1,2,0.5,18.19\n", "VIX")


def test_parse_cboe_single_column_any_case_and_iso_dates():
    s = data.parse_cboe_csv("DATE,SKEW\n01/02/1990,126.09\n01/03/1990,123.34\n", "SKEW")
    assert s.tolist() == [126.09, 123.34] and s.name == "SKEW"
    assert s.index[0] == pd.Timestamp("1990-01-02")
    iso = data.parse_cboe_csv("date,pput\n1986-07-01,101.5\n1986-06-30,100\n", "PPUT")
    assert iso.tolist() == [100.0, 101.5] and iso.index[-1] == pd.Timestamp("1986-07-01")


def test_parse_fred_dot_is_nan():
    s = data.parse_fred_csv("observation_date,DTB3\n2020-01-01,.\n2020-01-02,1.52\n", "DTB3")
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 1.52
    assert s.dtype == np.float64 and s.name == "DTB3"
    assert s.index[1] == pd.Timestamp("2020-01-02")


def _series(*values: float) -> pd.Series:
    return pd.Series(values, pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_cached_writes_parquet_and_second_call_skips_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    calls = []

    def fetch():
        calls.append(1)
        return _series(1.0, 2.0, 3.0)

    first = data._cached("yfinance", "^ABC", fetch)
    second = data._cached("yfinance", "^ABC", fetch)
    assert len(calls) == 1
    assert (tmp_path / "yfinance" / "ABC.parquet").exists()
    assert list(pd.read_parquet(tmp_path / "yfinance" / "ABC.parquet").columns) == ["value"]
    pd.testing.assert_series_equal(first, second)
    assert second.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("hours,expected", [(19, [20.0, 21.0]), (21, [30.0])])
def test_cache_refreshes_only_after_20h(tmp_path, monkeypatch, hours, expected):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    data._cached("cboe", "VIX", lambda: _series(20.0, 21.0))
    path = tmp_path / "cboe" / "VIX.parquet"
    old = time.time() - hours * 3600
    os.utime(path, (old, old))
    assert data._cached("cboe", "VIX", lambda: _series(30.0)).tolist() == expected
    assert pd.read_parquet(path)["value"].tolist() == expected
    assert (time.time() - path.stat().st_mtime < 3600) == (hours > 20)  # rewritten only when stale


def _boom() -> pd.Series:
    raise OSError("network down")


@pytest.mark.parametrize("bad_fetch", [_boom, lambda: _series()], ids=["raises", "empty"])
def test_stale_cache_returned_when_refresh_fails(tmp_path, monkeypatch, caplog, bad_fetch):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    data._cached("cboe", "VIX", lambda: _series(20.0, 21.0))
    path = tmp_path / "cboe" / "VIX.parquet"
    old = time.time() - 21 * 3600
    os.utime(path, (old, old))
    tried = []

    def fetch():
        tried.append(1)
        return bad_fetch()

    with caplog.at_level(logging.WARNING, logger="insureport.data"):
        s = data._cached("cboe", "VIX", fetch)
    assert tried, "a file older than 20 h must trigger a refetch"
    assert s.tolist() == [20.0, 21.0]
    assert "stale" in caplog.text
    assert pd.read_parquet(path)["value"].tolist() == [20.0, 21.0]  # never clobbered


def test_no_cache_and_fetch_fails_returns_empty(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="insureport.data"):
        s = data._cached("fred", "DTB3", _boom)
    assert s.empty and s.dtype == np.float64
    assert isinstance(s.index, pd.DatetimeIndex)
    assert "DTB3" in caplog.text


def test_duplicate_dates_keep_last_and_sort(tmp_path, monkeypatch):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data, "_http_text", lambda url: "DATE,VIX\n01/03/2020,1\n01/02/2020,2\n01/03/2020,3\n")
    s = data.cboe("VIX")
    assert s.tolist() == [2.0, 3.0] and s.index.is_unique
    assert s.reindex(pd.bdate_range("2020-01-02", periods=3)).notna().sum() == 2  # reindex no longer raises


def test_corrupt_fresh_cache_is_refetched(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    path = tmp_path / "cboe" / "VIX.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"PAR1\x00\x00trunc")  # fresh mtime, unreadable
    calls = []

    def fetch():
        calls.append(1)
        return _series(20.0, 21.0)

    with caplog.at_level(logging.WARNING, logger="insureport.data"):
        s = data._cached("cboe", "VIX", fetch)
    assert len(calls) == 1 and s.tolist() == [20.0, 21.0]
    assert pd.read_parquet(path)["value"].tolist() == [20.0, 21.0]
    assert "unreadable" in caplog.text


def test_cache_write_failure_returns_fresh_data(tmp_path, monkeypatch, caplog):
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(not_a_dir))
    with caplog.at_level(logging.WARNING, logger="insureport.data"):
        s = data._cached("cboe", "VIX", lambda: _series(20.0, 21.0))
    assert s.tolist() == [20.0, 21.0] and s.name == "VIX"
    assert "cache write failed" in caplog.text and "fetch failed" not in caplog.text


def test_offline_seams_raise_before_touching_network(monkeypatch):
    monkeypatch.setenv("INSUREPORT_OFFLINE", "1")

    def no_network(*a, **k):
        raise AssertionError("network touched")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setitem(sys.modules, "yfinance", None)  # importing it would raise ImportError
    with pytest.raises(RuntimeError):
        data._http_text("https://example.invalid/x.csv")
    with pytest.raises(RuntimeError):
        data._yf_download(["^GSPC"], "1990-01-01")


def test_seam_bodies_with_stubbed_transport(monkeypatch):
    """Online seam code paths, with urlopen and the yfinance module replaced (no network)."""
    monkeypatch.delenv("INSUREPORT_OFFLINE")
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"\xef\xbb\xbfDATE,X\n"

    def urlopen(req, timeout):
        seen.update(ua=req.get_header("User-agent"), timeout=timeout)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert data._http_text("https://example.invalid/x.csv") == "DATE,X\n"  # BOM stripped
    assert seen["ua"] and seen["timeout"] == 30

    today = pd.Timestamp.now("America/New_York").normalize()  # in-progress bar: must be dropped
    idx = pd.date_range("2020-01-02", periods=2, tz="America/New_York").append(pd.DatetimeIndex([today]))
    cols = pd.MultiIndex.from_tuples([("Close", "^GSPC"), ("Open", "^GSPC")])
    frame = pd.DataFrame([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]], idx, cols)
    yf = types.SimpleNamespace(download=lambda tickers, **k: (seen.update(yf=(tickers, k)), frame)[1])
    monkeypatch.setitem(sys.modules, "yfinance", yf)
    df = data._yf_download(["^GSPC", "ZZZ"], "2020-01-01")
    assert seen["yf"][1]["auto_adjust"] is True and seen["yf"][1]["start"] == "2020-01-01"
    assert df.index.tz is None and list(df.columns) == ["^GSPC", "ZZZ"]
    assert df["^GSPC"].tolist() == [1.0, 2.0] and df["ZZZ"].isna().all()
    assert data._yf_download(["^GSPC"], "2020-01-01", "Open")["^GSPC"].tolist() == [9.0, 9.0]  # the spx_open field


def test_prices_drops_missing_tickers(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    good = _series(1.0, 2.0, 3.0)
    old = pd.Series([5.0], [pd.Timestamp("2019-06-03")])  # data only before `start`
    seen = []

    def fake(tickers, start):
        seen.append(start)
        return pd.DataFrame({"AAA": good, "OLD": old}).reindex(columns=tickers)

    monkeypatch.setattr(data, "_yf_download", fake)
    with caplog.at_level(logging.WARNING, logger="insureport.data"):
        df = data.prices([" aaa", "ZZZ", "OLD"], start="2020-01-02")  # yfinance upper-cases tickers
    assert list(df.columns) == ["AAA"]
    assert df.attrs["missing"] == ["ZZZ", "OLD"]
    assert df["AAA"].tolist() == [2.0, 3.0]
    assert "dropped" in caplog.text
    assert seen == ["1927-01-01"] * 3  # cache always holds full history, whatever `start`
    cache = tmp_path / "yfinance"
    assert (cache / "AAA.parquet").exists() and (cache / "OLD.parquet").exists()
    assert not (cache / "ZZZ.parquet").exists()
    none = data.prices(["ZZZ"])
    assert none.empty and isinstance(none.index, pd.DatetimeIndex)


# --- load_market on synthetic sources --------------------------------------------------

N = 600
IDX = pd.bdate_range("2000-01-03", periods=N)
T = np.arange(N)
PX = pd.Series(100 * np.exp(0.0003 * T), IDX)
TR = (PX * np.exp(0.02 * T / 252)).iloc[100:]  # log TR outgrows log PX by exactly 0.02 per 252 rows; starts at row 100
VIX = pd.Series(20 + T / 100, IDX).drop(IDX[200:206])  # 6-row gap at rows 200..205
SAT = pd.Series([1.0], [pd.Timestamp("2000-01-15")])  # a Saturday: off-calendar prints must add no panel row
DTB3 = pd.Series(1 + T / 100, IDX)
DTB3.iloc[10] = np.nan  # FRED '.'
DTB3 = pd.concat([DTB3, pd.Series([9.99], [pd.Timestamp("2000-01-08")])]).sort_index()  # a Saturday, off the SPX calendar
CBOE = {
    "VIX": pd.concat([VIX, SAT]),
    "VIX9D": pd.Series(19.0, IDX[300:]),  # starts late, like the real VIX9D
    "VIX6M": pd.Series(23.0, IDX),
    "VIX1Y": pd.Series(24.0, IDX[250:]),  # starts late, like the real VIX1Y (2007)
    "SKEW": pd.concat([pd.Series(120.0, IDX), SAT]),
    "PPUT": pd.Series(500.0, IDX),
    "CLL": pd.Series(400.0, IDX),
}  # VXTH deliberately absent: its fetch fails
YF = {"^GSPC": PX, "^SP500TR": pd.concat([TR, SAT]), "^VIX3M": pd.Series(22.0, IDX)}
OPEN = (0.999 * PX).drop(IDX[50])  # ^GSPC Open; no print on row 50


def _fake_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drop: frozenset[str] | set[str] = frozenset()
) -> None:
    """Point both seams at the synthetic series above; any key in `drop` fails to fetch ("^GSPC_open": the
    ^GSPC Open field alone)."""
    monkeypatch.setenv("INSUREPORT_DATA_DIR", str(tmp_path))
    texts = {  # URL literals from the brief, so a typo in data.CBOE_URL / FRED_URL fails here
        f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{k}_History.csv": f"DATE,{k}\n"
        + "".join(f"{d:%m/%d/%Y},{v}\n" for d, v in s.items())
        for k, s in CBOE.items()
        if k not in drop
    }
    if "DTB3" not in drop:
        rows = "".join(f"{d:%Y-%m-%d},{'.' if np.isnan(v) else v}\n" for d, v in DTB3.items())
        texts["https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"] = "observation_date,DTB3\n" + rows
    monkeypatch.setattr(data, "_http_text", lambda url: texts[url])  # KeyError = failed fetch
    monkeypatch.setattr(
        data, "_yf_download",
        lambda tickers, start, field="Close": pd.DataFrame(  # any other field: KeyError, a failed fetch
            {t: {"Close": YF, "Open": {"^GSPC": OPEN}}[field][t] for t in tickers
             if (t if field == "Close" else t + "_open") not in drop})[tickers],
    )


def test_load_market_panel(tmp_path, monkeypatch):
    _fake_sources(monkeypatch, tmp_path)
    m = data.load_market("1999-01-01")
    assert list(m.columns) == [
        "spx", "spx_open", "spxtr", "vix", "vix9d", "vix3m", "vix6m", "vix1y", "skew", "rf", "q", "pput", "cll", "vxth"
    ]
    assert m.index.equals(IDX)  # ^GSPC days only: Saturday VIX, SKEW, TR and DTB3 prints add no row
    assert np.allclose(m["spx"], PX)
    # the ^GSPC Open, not forward-filled: a stale open must not settle or strike a roll (sleeves use the close)
    assert np.allclose(m["spx_open"].drop(IDX[50]), OPEN) and np.isnan(m["spx_open"].iloc[50])
    # each column comes from its own source (m.skew would be the DataFrame method, hence item access)
    assert (m["vix3m"] == 22).all() and (m["vix6m"] == 23).all() and (m["skew"] == 120).all()
    assert (m["pput"] == 500).all() and (m["cll"] == 400).all()
    assert m["vix9d"].iloc[:300].isna().all() and (m["vix9d"].iloc[300:] == 19).all()
    assert m["vix1y"].iloc[:250].isna().all() and (m["vix1y"].iloc[250:] == 24).all()  # Cboe VIX1Y, its own source
    # ffill(limit=5): rows 200..204 carry row 199, row 205 stays NaN
    assert (m["vix"].iloc[200:205] == VIX.iloc[199]).all()
    assert np.isnan(m["vix"].iloc[205]) and m["vix"].iloc[206] == VIX[IDX[206]]
    # rf = DTB3 / 100, '.' forward-filled
    assert m["rf"].iloc[9] == pytest.approx(0.0109) and m["rf"].iloc[10] == pytest.approx(0.0109)
    assert m["rf"].iloc[11] == pytest.approx(0.0111)
    # q: 0.03 until TR has 252 rows of history (row 100 + 252), then exactly 0.02
    assert (m["q"].iloc[:352] == 0.03).all()
    assert np.allclose(m["q"].iloc[352:], 0.02, atol=1e-12, rtol=0)
    # optional series that failed is NaN and reported; partial ones are reported too
    assert m["vxth"].isna().all()
    assert m.attrs["warnings"] == [
        "spx_open: 1 of 600 rows missing (data from 2000-01-03)",
        "spxtr: 100 of 600 rows missing (data from 2000-05-22)",  # TR starts at IDX[100]
        "vix: 1 of 600 rows missing (data from 2000-01-03)",
        "vix9d: 300 of 600 rows missing (data from 2001-02-26)",  # IDX[300]
        "vix1y: 250 of 600 rows missing (data from 2000-12-18)",  # IDX[250]
        "vxth: no data",
    ]


def test_load_market_start_slices_after_lookback(tmp_path, monkeypatch):
    _fake_sources(monkeypatch, tmp_path)
    m = data.load_market(str(IDX[400].date()))
    assert m.index[0] == IDX[400] and len(m) == N - 400
    assert m["q"].iloc[0] == pytest.approx(0.02, abs=1e-12)  # lookback reaches before `start`


_PX_DROP = PX.where(T < N - 1, PX * 0.96)  # -4% close on the last day


@pytest.mark.parametrize(
    "px,tr,q",
    [
        (PX, PX * np.exp(0.10 * T / 252), 0.06),
        (PX, PX * np.exp(-0.01 * T / 252), 0.0),
        # no TR print on rows 400..409 (longer than the ffill limit) nor on the last day
        (_PX_DROP, (_PX_DROP * np.exp(0.02 * T / 252)).drop(IDX[[*range(400, 410), N - 1]]), 0.02),
    ],
    ids=["clip-high", "clip-low", "tr-gaps"],
)
def test_load_market_q_clip_and_tr_gaps(tmp_path, monkeypatch, px, tr, q):
    monkeypatch.setitem(YF, "^GSPC", px)
    monkeypatch.setitem(YF, "^SP500TR", tr)
    _fake_sources(monkeypatch, tmp_path)
    m = data.load_market("1999-01-01")
    assert np.allclose(m["q"].iloc[252:], q, atol=1e-12, rtol=0)  # TR gaps carry the last real q


@pytest.mark.parametrize(
    "drop,name", [("SKEW", "skew"), ("VIX", "vix"), ("^GSPC", "spx"), ("DTB3", "rf"), ("^SP500TR", "spxtr")]
)
def test_load_market_missing_required_raises(tmp_path, monkeypatch, drop, name):
    _fake_sources(monkeypatch, tmp_path, drop={drop})
    with pytest.raises(RuntimeError, match=f"'{name}'"):
        data.load_market("1999-01-01")


def test_load_market_without_the_open(tmp_path, monkeypatch):
    # the ^GSPC Open fetch fails, the close does not: spx_open is all NaN and reported (sleeves then refuse to
    # strike or settle at the open rather than silently use the close)
    _fake_sources(monkeypatch, tmp_path, drop={"^GSPC_open"})
    m = data.load_market("1999-01-01")
    assert m["spx_open"].isna().all() and np.allclose(m["spx"], PX)
    assert "spx_open: no data" in m.attrs["warnings"]


def test_fixture_has_the_panel_columns():
    # the committed 2018-2019 fixture is a load_market slice: the 14 columns, VIX1Y and the open included and complete
    fx = pd.read_csv(Path(__file__).parent / "fixtures" / "market_2018_2019.csv", index_col="date", parse_dates=True)
    assert list(fx.columns) == data.COLUMNS and len(fx.columns) == 14
    assert fx[["vix1y", "spx_open"]].notna().all().all() and len(fx) == 503
    assert (fx["spx_open"] != fx["spx"]).mean() > 0.95  # the Open field, not the close again

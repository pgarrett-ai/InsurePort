"""Tests for insureport.chains and the real-quote branch of sleeves.option_sleeve: every vendor layout
(historicaloptiondata Level 2 with its leading-space ` DataDate` stamped 16:00, the 2010+ layout with
`last ` and `open interest`, ALLSPX's optionroot, Level 3's Flags, Cboe DataShop with `^SPX`, end-of-day
quotes preferred over the 15:45 ones and a zero underlying filled from the spot by date, ORATS' wide rows
melted into calls and puts, OptionMetrics IvyDB's strike x 1000) normalised to the canonical schema, other
underlyings of a multi-symbol file dropped, header spellings (DataShop's "Bid EOD"), a zipped file, an
unknown header refused, crossed and zero-ask quotes dropped and counted (a locked one kept), contract
selection (expiry and strike snapping, root and cp filters, nothing listed), the exact quote lookup (a
duplicate key takes its first row), the canonical CSV round trip (with a byte-order mark), and
option_sleeve pricing legs off a chain: opened at the chain's ask (a short leg at its bid), marked at its
mid, closed at its bid (a short leg at its ask), the model filling the days without a quote and a close
without one (counted in fallback_quotes; a stale input under a quote is not), intrinsic at expiry, the leg
ledger's source, monetize re-opens on the chain, the listed monthly kept under a holiday third Friday
(Thursday listing, inside or past the panel), a pre-2015 Saturday expiration (also on a panel ending on
its Friday) and offset rolls, a chain short of the target expiry left to the model unless its listing
outlives the leg, and the no-chain fallback count of a close on carried inputs.

Why: real end-of-day quotes replace the model wherever a user has bought them, so every vendor header
must land on the same columns and every sleeve price must come from the quote it claims. Offline: the
vendor samples are the two hand-written files in tests/fixtures/chains/ and inline strings.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insureport import chains, pricing, sleeves
from insureport.sleeves import Leg, OptionSpec

FIX = Path(__file__).parent / "fixtures" / "chains"
D0, D1 = pd.Timestamp("2019-01-18"), pd.Timestamp("2019-01-22")
FEB, MAR, JAN25 = pd.Timestamp("2019-02-15"), pd.Timestamp("2019-03-15"), pd.Timestamp("2019-01-25")
APR = pd.Timestamp("2019-04-19")
SPOT = pd.Series([2670.71, 2632.90], pd.DatetimeIndex([D0, D1]))


def _read(text: str, **kw) -> pd.DataFrame:
    """normalize() of an inline vendor CSV."""
    return chains.normalize(pd.read_csv(io.StringIO(text)), **kw)


def _canonical(df: pd.DataFrame, vendor: str) -> None:
    """Assert the canonical schema: columns, dtypes, mid, C/P, sort order and the vendor label."""
    assert list(df.columns) == chains.COLUMNS and df.attrs["vendor"] == vendor
    assert all(pd.api.types.is_datetime64_dtype(df[c]) for c in ("quote_date", "expiration"))
    assert all(df[c].dtype == np.float64 for c in ("strike", "bid", "ask", "mid", "underlying", "oi"))
    assert set(df["cp"]) <= {"C", "P"} and df["root"].map(type).eq(str).all()
    np.testing.assert_array_equal(df["mid"], (df["bid"] + df["ask"]) / 2)
    assert df.equals(df.sort_values(["quote_date", "expiration", "strike", "cp"]))


def test_hod_level2_leading_space_datadate():
    df = chains.load_chain(FIX / "hod_level2.csv")  # DataDate "01/18/2019 16:00", the vendor's snapshot stamp
    _canonical(df, "hod")
    assert df.attrs["dropped"] == 2  # the 1500 put's zero ask and the 2015 put's crossed quote
    assert df["quote_date"].eq(D0).all() and df["underlying"].eq(2670.71).all()
    # Exchange "W" marks SPXW, "*" (consolidated) is SPX; "put" / "call" / "P" all map to C / P
    assert df[["expiration", "strike", "cp", "root"]].values.tolist() == [
        [JAN25, 2000.0, "P", "SPXW"], [FEB, 2000.0, "P", "SPX"], [FEB, 2005.0, "P", "SPX"],
        [FEB, 2010.0, "P", "SPX"], [FEB, 2950.0, "C", "SPX"], [MAR, 2000.0, "P", "SPX"]]
    assert df["mid"].tolist() == pytest.approx([0.025, 0.85, 0.9, 0.95, 0.5, 4.1], abs=1e-15)
    assert df["oi"].tolist() == [1520.0, 15230.0, 210.0, 4120.0, 20871.0, 8001.0]


def test_hod_2010_and_allspx_lowercase_layouts():
    tail = "02/15/2019,01/18/2019,2000,0.85,0.75,0.95,120,15230,0.34,-0.008,0.0001,-0.04,0.31,SPX190215P2000"
    weekly = "01/25/2019,01/18/2019,2000,0.05,0.00,0.05,0,1520,0.61,-0.0003,0,-0.01,0.01,SPXW190125P2000"
    # a Russell 2000 put on the same expiry and strike, first in the file (a multi-symbol daily file)
    rut = "02/15/2019,01/18/2019,2000,70.0,69.0,71.0,5,100,0.2,-0.5,0.001,-0.3,1.0,RUT190215P2000"
    v2010 = ("underlying,underlying_last,exchange,optionsymbol,blank,optiontype,expiration,quotedate,strike,last ,"
             "bid,ask,volume,open interest,implied volatility,delta,gamma,theta,vega,alias\n"
             f"RUT,1500.25,*,RUT190215P02000000,,put,{rut}\n"
             f"SPX,2670.71,*,SPX190215P02000000,,put,{tail}\nSPX,2670.71,W,SPXW190125P02000000,,put,{weekly}\n")
    allspx = ("underlying,underlying_last,exchange,optionroot,optionext,type,expiration,quotedate,strike,last,bid,ask,"
              "volume,openinterest,impliedvol,delta,gamma,theta,vega,optionalias,IVBid,IVAsk\n"
              f"RUT,1500.25,*,RUT,,put,{rut},0.19,0.21\n"
              f"SPX,2670.71,*,SPX,,put,{tail},0.33,0.35\nSPX,2670.71,*,SPXW,,put,{weekly},0.60,0.62\n")
    level3 = ("UnderlyingSymbol,UnderlyingPrice,Flags,OptionSymbol,Type,Expiration,DataDate,Strike,Last,Bid,Ask,"
              "Volume,OpenInterest,T1OpenInterest,IVMean,IVBid,IVAsk,Delta,Gamma,Theta,Vega,AKA\n"
              "RUT,1500.25,*,RUT190215P02000000,put,02/15/2019,01/18/2019,2000,70.0,69.0,71.0,5,100,90,0.2,"
              "0.19,0.21,-0.5,0.001,-0.3,1.0,RUT190215P2000\n"
              "SPX,2670.71,*,SPX190215P02000000,put,02/15/2019,01/18/2019,2000,0.85,0.75,0.95,120,15230,15100,0.34,"
              "0.33,0.35,-0.008,0.0001,-0.04,0.31,SPX190215P2000\n"
              "SPX,2670.71,W,SPXW190125P02000000,put,01/25/2019,01/18/2019,2000,0.05,0.00,0.05,0,1520,1500,0.61,"
              "0.60,0.62,-0.0003,0,-0.01,0.01,SPXW190125P2000\n")
    # ALLSPX: optionroot names the root even where exchange says "*"; Level 3: Flags "W" marks SPXW
    for text in (v2010, allspx, level3):
        df = _read(text)
        _canonical(df, "hod")
        assert df[["expiration", "root", "oi"]].values.tolist() == [[JAN25, "SPXW", 1520.0], [FEB, "SPX", 15230.0]]
        assert df["underlying"].eq(2670.71).all() and df["mid"].tolist() == pytest.approx([0.025, 0.85], abs=1e-15)
        # the RUT row is dropped (counted), not merged into SPX where it would shadow the SPX quote
        assert df.attrs["dropped"] == 1
        assert chains.quote_at(df, D0, FEB, 2000.0, "P") == pytest.approx((0.85, 0.75, 0.95), abs=1e-15)


def test_datashop_eod_quotes_caret_and_spot_fill():
    df = chains.load_chain(FIX / "datashop_eod.csv", spot=SPOT)
    _canonical(df, "datashop")
    assert df.attrs["dropped"] == 2  # crossed 2010 put, zero-ask 2950 call on 01-22
    assert df[["quote_date", "expiration", "strike", "cp", "root"]].values.tolist() == [
        [D0, JAN25, 2000.0, "P", "SPXW"], [D0, FEB, 2000.0, "P", "SPX"], [D0, FEB, 2010.0, "P", "SPX"],
        [D0, FEB, 2950.0, "C", "SPX"], [D1, FEB, 2000.0, "P", "SPX"]]
    # bid_eod / ask_eod, not the 15:45 snapshot (0.80 / 1.00 on the first SPX row)
    assert df[["bid", "ask"]].values.tolist() == [[0.0, 0.05], [0.75, 0.95], [0.85, 1.05], [0.45, 0.55], [1.0, 1.2]]
    no_eod = chains.normalize(pd.read_csv(FIX / "datashop_eod.csv").drop(columns=["bid_eod", "ask_eod"]))
    assert no_eod.loc[1, ["root", "strike", "bid", "ask"]].tolist() == ["SPX", 2000.0, 0.80, 1.00]  # the 15:45 quote
    # active_underlying_price_1545 = 0 (no index licence) -> the spot that day; non-zero -> kept
    assert df["underlying"].tolist() == [2670.71] * 4 + [2633.12]
    assert np.isnan(chains.load_chain(FIX / "datashop_eod.csv")["underlying"].iloc[0])  # no spot: NaN
    # without a root column the root is the underlying symbol, caret stripped
    raw = pd.read_csv(FIX / "datashop_eod.csv").drop(columns="root")
    assert set(chains.normalize(raw)["root"]) == {"SPX"}


def test_orats_wide_rows_melt_into_calls_and_puts():
    text = ("ticker,stkPx,expirDate,yte,strike,cVolu,cOi,pVolu,pOi,cBidPx,cValue,cAskPx,pBidPx,pValue,pAskPx,"
            "spot_px,trade_date\n"
            "SPX,2670.50,2/15/2019,0.0767,2000,0,10,120,15230,668.9,670.6,672.3,0.75,0.84,0.95,2670.71,1/18/2019\n"
            "SPX,2670.50,2/15/2019,0.0767,2950,512,20871,0,30,0.45,0.5,0.55,280.0,279.8,281.0,2670.71,1/18/2019\n")
    df = _read(text)
    _canonical(df, "orats")
    assert len(df) == 4 and df.attrs["dropped"] == 0  # two rows out per input row
    assert df[["strike", "cp", "bid", "ask", "oi"]].values.tolist() == [
        [2000.0, "C", 668.9, 672.3, 10.0], [2000.0, "P", 0.75, 0.95, 15230.0],
        [2950.0, "C", 0.45, 0.55, 20871.0], [2950.0, "P", 280.0, 281.0, 30.0]]
    assert df["quote_date"].eq(D0).all() and df["expiration"].eq(FEB).all()
    assert df["underlying"].eq(2670.50).all() and df["root"].eq("SPX").all()  # stkPx before spot_px


def test_ivydb_strike_price_over_1000():
    text = ("secid,date,exdate,cp_flag,strike_price,best_bid,best_offer,volume,open_interest,impl_volatility,delta,"
            "optionid\n"
            "108105,2019-01-18,2019-02-15,P,2000000,0.75,0.95,120,15230,0.3412,-0.0081,123456789\n"
            "108105,2019-01-18,2019-02-15,C,2950000,0.45,0.55,512,20871,0.1201,0.0412,123456790\n"
            "108105,2019-01-22,2019-02-15,P,2000000,1.00,1.20,210,15301,0.3502,-0.0101,123456789\n")
    df = _read(text, spot=SPOT)
    _canonical(df, "ivydb")
    assert df[["strike", "cp"]].values.tolist() == [[2000.0, "P"], [2950.0, "C"], [2000.0, "P"]]
    assert df["mid"].tolist() == pytest.approx([0.85, 0.5, 1.1], abs=1e-15)
    # no root or price columns: SPX, and the spot of each row's own quote date
    assert df["root"].eq("SPX").all() and df["underlying"].tolist() == [2670.71, 2670.71, 2632.90]


def test_zipped_file_and_unknown_header(tmp_path):
    z = tmp_path / "hod.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(FIX / "hod_level2.csv", "hod_level2.csv")
    pd.testing.assert_frame_equal(chains.load_chain(z), chains.load_chain(FIX / "hod_level2.csv"))
    for cols in (["date", "symbol", "close"], ["quote_date", "strike", "bid", "ask"]):
        with pytest.raises(ValueError, match="quote_date, expiration, strike, cp, bid, ask"):
            chains.detect_vendor(cols)
    with pytest.raises(ValueError, match="hod, datashop, orats, ivydb"):
        chains.normalize(pd.DataFrame({"Date": ["2019-01-18"], "Close": [2670.71]}))
    with pytest.raises(ValueError, match="expiration"):  # a recognised vendor with a needed column missing
        _read("secid,date,cp_flag,strike_price,best_bid,best_offer\n108105,2019-01-18,P,2000000,0.75,0.95\n")


def test_header_normalisation_and_detection():
    assert chains.normalize_headers([" DataDate", "last ", "open interest", "Bid_EOD", "cBidPx"]) == [
        "datadate", "last", "openinterest", "bideod", "cbidpx"]
    # DataShop's product pages spell the fields "Bid EOD", "Active Underlying Price 1545": the same names
    assert chains.normalize_headers(["Bid EOD", "Active Underlying Price 1545", "Quote Date"]) == [
        "bideod", "activeunderlyingprice1545", "quotedate"]
    hod = ["UnderlyingSymbol", " DataDate", "Expiration", "Strike", "Type", "Bid", "Ask"]
    datashop = ["quote_date", "expiration", "strike", "option_type", "bid_1545", "ask_1545"]
    assert chains.detect_vendor(hod) == "hod" and chains.detect_vendor(datashop) == "datashop"
    assert chains.detect_vendor(chains.COLUMNS) == "hod"  # a saved canonical chain is a long layout
    # strike_price is a common name: without best_bid / best_offer it is no IvyDB file (strike x 1000)
    assert chains.detect_vendor(["quote_date", "expiration", "strike_price", "type", "bid", "ask"]) == "hod"


def test_crossed_and_zero_ask_rows_are_dropped_and_counted():
    raw = pd.DataFrame({"quote_date": [D0] * 6, "expiration": [FEB] * 6,
                        "strike": [2000.0, 2005, 2010, 2015, 2020, 2025], "cp": "P",
                        "bid": [0.75, 1.1, 0.0, 0.0, -0.1, 1.0], "ask": [0.95, 1.0, 0.0, 0.05, 0.05, 1.0]})
    df = chains.normalize(raw)
    # bid > ask, ask <= 0, bid < 0 dropped; a locked quote (bid == ask) kept
    assert df["strike"].tolist() == [2000.0, 2015.0, 2025.0] and df.attrs["dropped"] == 3
    assert df["mid"].tolist() == pytest.approx([0.85, 0.025, 1.0], abs=1e-15)
    assert df["root"].eq("SPX").all() and df["oi"].isna().all()


def _chain(rows: list[tuple]) -> pd.DataFrame:
    """Canonical chain from (quote_date, expiration, strike, cp, bid, ask[, root]) rows (root SPX)."""
    cols = ["quote_date", "expiration", "strike", "cp", "bid", "ask", "root"]
    return chains.normalize(pd.DataFrame([(*r, "SPX")[:7] for r in rows], columns=cols))


def test_select_contract_snaps_expiry_then_strike():
    ch = _chain([(D0, JAN25, 2000.0, "P", 0.0, 0.05), (D0, FEB, 2990.0, "P", 1.0, 1.2),
                 (D0, FEB, 3010.0, "P", 1.2, 1.4), (D0, FEB, 2950.0, "P", 0.6, 0.8),
                 (D0, FEB, 3000.0, "P", 0.9, 1.1, "SPXW"), (D0, FEB, 3000.0, "C", 300.0, 302.0),
                 (D0, FEB, 3020.0, "C", 290.0, 292.0), (D0, MAR, 3000.0, "P", 4.0, 4.4),
                 (D1, FEB, 3000.0, "P", 1.0, 1.2), (FEB, FEB, 3000.0, "P", 0.0, 0.05)])
    sel = chains.select_contract
    # nearest expiry >= the target, then the put strike at or below / the call strike at or above it (roots apart)
    assert sel(ch, D0, FEB, 3000.0, "P") == (FEB, 2990.0)
    assert sel(ch, D0, FEB, 3000.0, "P", root="SPXW") == (FEB, 3000.0)
    assert sel(ch, D0, FEB, 3005.0, "C") == (FEB, 3020.0) and sel(ch, D0, FEB, 3000.0, "C") == (FEB, 3000.0)
    assert sel(ch, D0, pd.Timestamp("2019-02-01"), 3000.0, "P") == (FEB, 2990.0)  # 02-01 not listed: next is 02-15
    assert sel(ch, D0, pd.Timestamp("2019-04-19"), 3000.0, "P") == (MAR, 3000.0)  # nothing >= target: the latest
    assert isinstance(sel(ch, D0, FEB, 3000.0, "P")[0], pd.Timestamp)
    # None: nothing listed that day / for that root / below a put target; an expiry on the quote date is not
    # a contract to open
    assert sel(ch, pd.Timestamp("2019-01-23"), FEB, 3000.0, "P") is None
    assert sel(ch, D0, FEB, 3000.0, "P", root="SPXQ") is None and sel(ch, D0, MAR, 2000.0, "P") is None
    assert sel(ch, FEB, MAR, 3000.0, "P") is None


def test_quote_at_exact_and_missing():
    ch = _chain([(D0, FEB, 2990.0, "P", 1.0, 1.2), (D0, FEB, 2990.0, "P", 0.5, 0.7, "SPXW")])
    assert chains.quote_at(ch, D0, FEB, 2990.0, "P") == pytest.approx((1.1, 1.0, 1.2), abs=1e-15)
    assert chains.quote_at(ch, D0, FEB, 2990.0, "P", root="SPXW") == pytest.approx((0.6, 0.5, 0.7), abs=1e-15)
    for args in ((D1, FEB, 2990.0, "P"), (D0, MAR, 2990.0, "P"), (D0, FEB, 2995.0, "P"), (D0, FEB, 2990.0, "C")):
        assert chains.quote_at(ch, *args) is None
    # one key twice under one root (ORATS / IvyDB on a third Friday: the AM monthly and the PM weekly): the first row
    dup = _chain([(D0, FEB, 2990.0, "P", 1.0, 1.2), (D0, FEB, 2990.0, "P", 0.5, 0.7)])
    assert chains.quote_at(dup, D0, FEB, 2990.0, "P") == pytest.approx((1.1, 1.0, 1.2), abs=1e-15)


def test_canonical_csv_round_trip(tmp_path):
    for src in ("hod_level2.csv", "datashop_eod.csv"):
        df = chains.load_chain(FIX / src, spot=SPOT)
        df.to_csv(tmp_path / "c.csv", index=False, encoding="utf-8-sig")  # with a byte-order mark, as Excel saves
        back = chains.load_chain(tmp_path / "c.csv")
        pd.testing.assert_frame_equal(back, df)
        assert back.attrs["dropped"] == 0


S0 = 4000.0
PUT25 = Leg("P", otm=0.25)  # 1-month 25% OTM put, K target 3000 on a flat 4000 panel


def _panel(periods: int = 40) -> pd.DataFrame:
    """Flat synthetic panel (test_sleeves' _panel): S 4000, VIX pillars 20, SKEW 130, rf = q = 0."""
    df = pd.DataFrame({"spx": S0, "vix": 20.0, "skew": 130.0, "rf": 0.0, "q": 0.0},
                      index=pd.bdate_range("2019-01-01", periods=periods))
    for c in ("vix9d", "vix3m", "vix6m", "vix1y", "spxtr"):
        df[c] = df["spx"] if c == "spxtr" else df["vix"]
    return df


def _model_mid(K: float, expiry: pd.Timestamp, t: pd.Timestamp) -> float:
    """The model mid of a put on the flat panel (pricing.quote at the option_sleeve defaults)."""
    T = (expiry - t).days / 365
    return float(pricing.quote(S0, K, T, pricing.var_swap_vol(T, *(20.0,) * 5), 130.0, 0.0, 0.0, "P")[0])


def test_option_sleeve_prices_legs_off_the_chain():
    mkt = _panel()
    d = {k: pd.Timestamp(f"2019-01-{k}") for k in (21, 22, 23, 24, 25, 28)}
    mkt.loc[[d[23], d[24]], "skew"] = np.nan  # stale model inputs: no fallback where the chain quotes (01-23)
    marks = [(d[21], 1.0, 1.1), (d[22], 0.9, 1.0), (d[23], 0.8, 0.9), (d[25], 0.7, 0.8)]  # no quote on 01-24
    ch = _chain([(D0, FEB, 2990.0, "P", 1.0, 1.2), (D0, FEB, 3010.0, "P", 1.2, 1.4), (D0, FEB, 2950.0, "P", 0.6, 0.8),
                 (D0, FEB, 3000.0, "P", 0.9, 1.1, "SPXW"), (D0, FEB, 3000.0, "C", 990.0, 1000.0),
                 (D0, pd.Timestamp("2019-02-01"), 3000.0, "P", 0.2, 0.3), (D0, MAR, 3000.0, "P", 4.0, 4.4),
                 *((t, FEB, 2990.0, "P", b, a) for t, b, a in marks),
                 (FEB, FEB, 2990.0, "P", 5.0, 6.0)])  # an expiry-day quote: settlement ignores it
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), chain=ch)
    g, nxt = df.attrs["legs"][:2]
    # strike and expiry snap to the listed SPX put (max strike <= 3000 on the nearest expiry >= 02-15)
    assert (g["open_date"], g["expiry"], g["K"], g["source"]) == (D0, FEB, 2990.0, "chain")
    assert g["unit_cost"] == 1.2 and g["n"] == pytest.approx(1 / 1.2, rel=1e-12)  # f = 1: the cash buys 1 / ask
    nav = (1 + df["ret"]).cumprod()
    assert nav[d[21]] == pytest.approx(1.05 / 1.2, rel=1e-12) and nav[d[25]] == pytest.approx(0.75 / 1.2, rel=1e-12)
    assert nav[d[23]] == pytest.approx(0.85 / 1.2, rel=1e-12)
    # 01-24 has no quote: the model's mid for the same strike and expiry (inputs carried from 01-22, the same on
    # the flat panel), counted as a fallback that day
    assert nav[d[24]] == pytest.approx(_model_mid(2990.0, FEB, d[24]) / 1.2, rel=1e-12)
    # from 01-28 the chain is silent until expiry (fallbacks); on 02-15 the put settles at intrinsic (0, not the
    # 5.5 quote) and the next leg finds no contract after 02-15 listed: model strike and expiry, every day a fallback
    assert df.at[FEB, "ret"] == -1.0
    assert (nxt["open_date"], nxt["K"], nxt["expiry"], nxt["source"]) == (FEB, 3000.0, MAR, "model")
    assert df["fallback_quotes"].tolist() == [int(t == d[24] or t >= d[28]) for t in mkt.index]
    # overlay: the premium paid is the chain ask's share of the hedged unit, u / (1 + u) with u = ask / S
    ov = sleeves.option_sleeve(mkt, OptionSpec((PUT25,), mode="overlay"), chain=ch)
    assert ov.at[D0, "premium_paid"] == pytest.approx((1.2 / S0) / (1 + 1.2 / S0), rel=1e-12)


def test_option_sleeve_closes_at_the_chain_bid_and_ask():
    mkt = _panel()
    put = Leg("P", otm=0.25, tenor_months=2, hold_months=1)  # closed at the 02-15 roll, a month before expiry
    rows = [(D0, MAR, 3000.0, "P", 4.0, 4.4), (FEB, MAR, 3000.0, "P", 2.0, 2.6), (FEB, APR, 3000.0, "P", 3.0, 3.4)]
    df = sleeves.option_sleeve(mkt, OptionSpec((put,)), chain=_chain(rows))
    g = df.attrs["legs"][0]
    assert (g["K"], g["expiry"], g["source"], g["closed_on"]) == (3000.0, MAR, "chain", FEB)
    assert (1 + df["ret"])[:FEB].prod() == pytest.approx(2.0 / 4.4, rel=1e-12)  # f = 1, C = 0: n x the 02-15 bid
    assert df.at[FEB, "fallback_quotes"] == 0  # the next leg opens on the listed APR put's quote
    # no 02-15 quote for the MAR put: it is sold at the model's bid, a fallback that day
    assert sleeves.option_sleeve(mkt, OptionSpec((put,)), chain=_chain(rows[::2])).at[FEB, "fallback_quotes"] == 1
    # a collar (overlay): the short call opens at the chain bid and is bought back at its ask
    call = Leg("C", otm=0.1, qty=-1.0, tenor_months=2, hold_months=1)
    ch = _chain([*rows, (D0, MAR, 4400.0, "C", 2.0, 2.4), (FEB, MAR, 4400.0, "C", 1.0, 1.3)])
    ov = sleeves.option_sleeve(mkt, OptionSpec((put, call), mode="overlay"), chain=ch)
    p, c = ov.attrs["legs"][:2]
    assert (p["unit_cost"], c["K"], c["source"], c["unit_cost"]) == (4.4, 4400.0, "chain", 2.0)
    # A = 1 / (1 + u / S), u = 4.4 - 2.0, N = A / S; on 02-15 C = N (2.0 - 1.3): the put's bid less the call's ask
    assert (1 + ov["ret"])[:FEB].prod() == pytest.approx((1 + 0.7 / S0) / (1 + 2.4 / S0), rel=1e-12)


def test_option_sleeve_opens_the_thursday_listing_of_a_holiday_third_friday():
    # Good Friday 2019-04-19: no trading, so the April AM monthly is listed with (and settles on) Thursday 04-18
    mar15, apr18 = pd.Timestamp("2019-03-15"), pd.Timestamp("2019-04-18")
    ch = _chain([(mar15, apr18, 3000.0, "P", 1.0, 1.2), (mar15, pd.Timestamp("2019-05-17"), 3000.0, "P", 4.0, 4.4)])
    # the Friday missing from the panel, and past a panel ending 2019-04-05 (which cannot show the holiday)
    inside, past = (next(g for g in sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), chain=ch).attrs["legs"]
                         if g["open_date"] == mar15)
                    for mkt in (_panel(85).drop(pd.Timestamp("2019-04-19")), _panel(69)))
    assert (inside["expiry"], inside["source"], inside["unit_cost"], inside["closed_on"], inside["close_reason"]) == (
        apr18, "chain", 1.2, apr18, "expiry")
    assert (past["expiry"], past["source"], past["unit_cost"], past["closed_on"]) == (apr18, "chain", 1.2, None)


def test_option_sleeve_snaps_to_a_saturday_expiration():
    # before 2015 the SPX monthly expires on the Saturday after the third Friday and settles on the Friday
    mkt = _panel(60).set_axis(pd.bdate_range("2014-01-01", periods=60))
    jan17, fri, sat = pd.Timestamp("2014-01-17"), pd.Timestamp("2014-02-21"), pd.Timestamp("2014-02-22")
    ch = _chain([(t, sat, 3000.0, "P", 1.0, 1.2) for t in mkt.index[(mkt.index >= jan17) & (mkt.index < fri)]])
    df = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), chain=ch)
    g = df.attrs["legs"][0]
    assert (g["open_date"], g["expiry"], g["source"], g["unit_cost"]) == (jan17, sat, "chain", 1.2)
    assert (g["closed_on"], g["close_reason"]) == (fri, "expiry")
    assert (df.loc[jan17:"2014-02-20", "fallback_quotes"] == 0).all()  # every leg-day on the Saturday listing
    # a panel ending on that Friday: still settled there at intrinsic, not sold at the bid as a live roll
    g = sleeves.option_sleeve(mkt.loc[:fri], OptionSpec((PUT25,)), chain=ch).attrs["legs"][0]
    assert (g["closed_on"], g["close_reason"], g["settle_px"]) == (fri, "expiry", S0)


def test_option_sleeve_chain_keeps_the_listed_monthly_under_offsets_and_short_chains():
    mkt = _panel()
    feb19 = pd.Timestamp("2019-02-19")
    # offset rolls 01-22, 02-19: the model's expiry moves with the roll; the chain's leg stays the listed 02-15
    # monthly (same tenor), settled at intrinsic two rows before its roll
    ch = _chain([(D1, FEB, 3000.0, "P", 1.0, 1.2), (D1, MAR, 3000.0, "P", 4.0, 4.4)])
    g = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), offset_bdays=2, chain=ch).attrs["legs"][0]
    assert (g["open_date"], g["expiry"], g["source"], g["unit_cost"]) == (D1, FEB, "chain", 1.2)
    assert (g["closed_on"], g["close_reason"], g["settle_px"]) == (feb19, "expiry", S0)
    model = sleeves.option_sleeve(mkt, OptionSpec((PUT25,)), offset_bdays=2).attrs["legs"][0]
    assert (model["expiry"], model["closed_on"], model["close_reason"]) == (feb19, feb19, "expiry")
    # nothing listed out to a 2-month leg's March expiry: the latest listing (February) would die a month before
    # the leg's roll, so the leg keeps the model's contract
    feb_only = _chain([(D0, FEB, 3000.0, "P", 1.0, 1.2)])
    two = OptionSpec((Leg("P", otm=0.25, tenor_months=2),))
    g = sleeves.option_sleeve(mkt, two, chain=feb_only).attrs["legs"][0]
    assert (g["expiry"], g["K"], g["source"]) == (MAR, 3000.0, "model")
    # held one month instead, the leg closes at the 02-15 roll, February's settle row: that listing is opened
    held = OptionSpec((Leg("P", otm=0.25, tenor_months=2, hold_months=1),))
    g = sleeves.option_sleeve(mkt, held, chain=feb_only).attrs["legs"][0]
    assert (g["expiry"], g["source"], g["unit_cost"]) == (FEB, "chain", 1.2)


def test_option_sleeve_monetize_reopens_on_the_chain():
    mkt = _panel()
    crash = pd.Timestamp("2019-01-28")
    mkt.loc[crash:, "spx"] = 0.6 * S0  # the model-priced 3000 put is then worth ~600, over 2 x its cost
    # the re-struck put (K target 0.75 x 2400 = 1800, February) snaps to the listed 1790 at its ask
    ch = _chain([(crash, FEB, 1790.0, "P", 9.0, 9.7), (crash, FEB, 1810.0, "P", 11.0, 11.8)])
    legs = sleeves.option_sleeve(mkt, OptionSpec((PUT25,), monetize_mult=2.0), chain=ch).attrs["legs"]
    assert (legs[0]["source"], legs[0]["closed_on"], legs[0]["close_reason"]) == ("model", crash, "monetize")
    g = next(g for g in legs if g["open_date"] == crash)
    assert (g["K"], g["expiry"], g["source"], g["unit_cost"]) == (1790.0, FEB, "chain", 9.7)


def test_fallback_quotes_count_a_close_on_carried_inputs_without_a_chain():
    # slice 9 counts a close priced off carried inputs in the model path too (slice 4 counted held leg-days only)
    mkt = _panel(60)
    mkt.loc[FEB, "skew"] = np.nan
    df = sleeves.option_sleeve(mkt, OptionSpec((Leg("P", otm=0.25, tenor_months=2, hold_months=1),)))
    assert df.at[FEB, "fallback_quotes"] == 2  # the March put sold at the 02-15 roll plus the April put bought there
    assert df["fallback_quotes"].sum() == 2

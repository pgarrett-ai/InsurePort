"""Tests for insureport.portfolio: holdings CSV parsing (weights, shares, CASH rows, header and
ticker case, percent weights, the 50-row limit, every rejection including malformed rows and
repeated headers), Portfolio construction in weight and shares mode, cash_returns
(rf_{t-1} * calendar days / 360), core_returns (constant-mix weighted sum, cash leg on a varying
rf carried forward and lagged, all-CASH, sample start, price gaps, proxy fill, missing tickers)
and the leverage transform (financing on the previous day's rate, L < 1 and short rf rejected).

Why: every diagnostic and backtest starts from the core return series built here, and the
holdings CSV is a trust boundary (user upload). Synthetic inline inputs, no I/O.
"""

import numpy as np
import pandas as pd
import pytest

from insureport import portfolio
from insureport.portfolio import Portfolio

IDX = pd.bdate_range("2020-01-01", periods=12)  # Wed 01-01, Thu, Fri 01-03, Mon 01-06, ...
_RNG = np.random.default_rng(0)
PX = pd.DataFrame(100 * np.cumprod(1 + _RNG.normal(0, 0.01, (12, 4)), axis=0), IDX, columns=["A", "B", "C", "D"])
PX.iloc[:2, 1] = np.nan  # B prices from row 2
PX.iloc[:4, 2] = np.nan  # C prices from row 4: the latest first-valid date
SPXTR = pd.Series(3000 * np.cumprod(1 + _RNG.normal(0, 0.01, 12)), IDX).iloc[1:]  # the proxy starts at row 1
RF = pd.Series(0.05, IDX)
RF_UP = pd.Series(0.01 + np.arange(12) / 100, IDX)  # a different rate every day: rf timing is visible


def test_parse_weights_csv():
    df = portfolio.parse_holdings("ticker,weight\nVOO,0.6\nQQQ,0.3\nCASH,0.1\n")
    assert list(df.columns) == ["ticker", "weight"]
    assert df["ticker"].tolist() == ["VOO", "QQQ", "CASH"] and df["weight"].tolist() == [0.6, 0.3, 0.1]
    p = portfolio.to_portfolio(df)
    assert p.weights == pytest.approx({"VOO": 0.6, "QQQ": 0.3, "CASH": 0.1}) and p.value is None
    assert sum(p.weights.values()) == pytest.approx(1, abs=1e-9)
    fifty = "ticker,weight\n" + "".join(f"T{i},1\n" for i in range(50))
    assert len(portfolio.parse_holdings(fifty)) == 50  # the row limit itself is legal


def test_parse_case_whitespace_and_percent_weights():
    df = portfolio.parse_holdings(" Ticker , WEIGHT \n voo , 60\nbrk-b,30\n^gspc,10\n")
    assert df["ticker"].tolist() == ["VOO", "BRK-B", "^GSPC"]
    assert df["weight"].tolist() == pytest.approx([0.6, 0.3, 0.1])  # sum 100 > 1.5: percents
    exact = portfolio.parse_holdings("ticker,weight\nA,1\nB,0.5\n")  # sum 1.5 is not > 1.5: fractions
    assert exact["weight"].tolist() == [1.0, 0.5]
    assert portfolio.to_portfolio(exact).weights == pytest.approx({"A": 2 / 3, "B": 1 / 3})


def test_shares_csv_cash_row_and_cash_dollars():
    df = portfolio.parse_holdings("Ticker,Shares\nvoo,10\nqqq,5\nCASH,500\n")
    assert list(df.columns) == ["ticker", "shares"]
    p = portfolio.to_portfolio(df, pd.Series({"VOO": 500.0, "QQQ": 400.0, "ZZZ": 1.0}), cash=500.0)
    assert p.value == 8000  # 5000 + 2000 + (500 CASH row + 500 cash), CASH shares are dollars
    assert p.weights == pytest.approx({"VOO": 0.625, "QQQ": 0.25, "CASH": 0.125})


def test_weight_mode_cash_is_an_extra_weight_and_zeros_drop():
    df = portfolio.parse_holdings("ticker,weight\nVOO,0.5\nQQQ,0\nCASH,0.25\n")
    p = portfolio.to_portfolio(df, cash=0.25)
    assert p.weights == pytest.approx({"VOO": 0.5, "CASH": 0.5}) and "QQQ" not in p.weights


@pytest.mark.parametrize(
    "text,match",
    [
        ("ticker,weight\nVOO,-0.1\n", "negative weight"),
        ("ticker,shares\nVOO,abc\n", "non-numeric shares"),
        ("ticker,weight\nVOO,\n", "non-numeric weight"),
        ("ticker,weight\nVOO,0.5\nvoo,0.5\n", "duplicate ticker"),
        ("ticker,weight\n" + "".join(f"T{i},1\n" for i in range(51)), "more than 50 rows"),
        ("ticker,weight\nVOO;X,1\n", "bad ticker"),
        ("ticker,weight\nABCDEFGHIJKLM,1\n", "bad ticker"),  # 13 characters
        ("symbol,weight\nVOO,1\n", "column 'ticker'"),
        ("ticker,ticker,weight\nVOO,QQQ,1\n", "column 'ticker'"),
        ("ticker,amount\nVOO,1\n", "'weight' or 'shares'"),
        ("ticker,weight,shares\nVOO,1,1\n", "'weight' or 'shares'"),
        ("ticker,weight,weight\nVOO,1,2\n", "'weight' or 'shares'"),
        # every row longer than the header: pandas would make column 1 the index and shift the rest
        ("ticker,weight\nVOO,60,100\nQQQ,40,50\n", "Expected 2 fields in line 2, saw 3"),
        ("ticker,weight\nVOO,0.6,\nQQQ,0.4,\n", "Expected 2 fields"),  # trailing commas
        ("ticker,shares\nVOO,1,000\n", "Expected 2 fields"),  # unquoted thousands separator
        ("  \n", "empty"),
        ("ticker,weight\n", "empty"),
    ],
)
def test_parse_rejects(text, match):
    with pytest.raises(ValueError, match=match):
        portfolio.parse_holdings(text)


def test_to_portfolio_rejects():
    df = portfolio.parse_holdings("ticker,shares\nVOO,10\nQQQ,5\n")
    with pytest.raises(ValueError, match=r"last_prices.*QQQ"):
        portfolio.to_portfolio(df, pd.Series({"VOO": 500.0}))
    with pytest.raises(ValueError, match="last_prices"):
        portfolio.to_portfolio(df)
    with pytest.raises(ValueError, match="sum to zero"):
        portfolio.to_portfolio(portfolio.parse_holdings("ticker,weight\nVOO,0\n"))


def test_cash_returns_previous_rate_and_calendar_days():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])  # Thu, Fri, Mon
    c = portfolio.cash_returns(pd.Series([0.05, 0.06, 0.07], idx))
    np.testing.assert_allclose(c, [0, 0.05 / 360, 0.06 * 3 / 360], rtol=0, atol=1e-16)


def test_core_returns_weighted_sum_from_latest_start():
    w = {"A": 0.5, "B": 0.3, "C": 0.2}
    core, proxied = portfolio.core_returns(Portfolio(w), PX, RF)
    r = PX.loc[IDX[4]:, list(w)].pct_change()
    expected = (0.5 * r["A"] + 0.3 * r["B"] + 0.2 * r["C"]).fillna(0.0)  # first row is a 0 placeholder
    assert core.index[0] == IDX[4] and core.index.equals(IDX[4:])
    np.testing.assert_allclose(core, expected, rtol=0, atol=1e-12)
    assert proxied == []
    # a missing print drops that date; the next return spans both days instead of NaN or 0
    gap = PX.copy()
    gap.iloc[7, 0] = np.nan
    core_gap, _ = portfolio.core_returns(Portfolio(w), gap, RF)
    assert IDX[7] not in core_gap.index
    span = PX.loc[[IDX[6], IDX[8]], list(w)].pct_change().iloc[1]
    assert core_gap[IDX[8]] == pytest.approx(0.5 * span["A"] + 0.3 * span["B"] + 0.2 * span["C"], abs=1e-12)


def test_core_returns_cash_leg():
    p = Portfolio({"A": 0.6, "CASH": 0.4})
    core, _ = portfolio.core_returns(p, PX, RF_UP.drop(IDX[5]))  # rf on a different calendar
    cash = core - 0.6 * PX["A"].pct_change().fillna(0.0)
    assert cash[IDX[3]] == pytest.approx(0.4 * RF_UP[IDX[2]] * 3 / 360, abs=1e-15)  # Fri's rate, Fri -> Mon
    assert cash[IDX[6]] == pytest.approx(0.4 * RF_UP[IDX[4]] / 360, abs=1e-15)  # IDX[5] missing: IDX[4]'s rate
    assert cash[IDX[7]] == pytest.approx(0.4 * RF_UP[IDX[6]] / 360, abs=1e-15)  # the previous day's rate
    used = RF_UP.copy()
    used[IDX[5]] = RF_UP[IDX[4]]  # carried forward, never back-filled from IDX[6]
    np.testing.assert_allclose(cash, 0.4 * portfolio.cash_returns(used), rtol=0, atol=1e-12)
    late, _ = portfolio.core_returns(p, PX, RF_UP.iloc[3:])
    assert late.index[0] == IDX[3]  # with CASH the sample starts no earlier than rf
    with pytest.raises(ValueError, match="rf data"):
        portfolio.core_returns(p, PX, pd.Series(np.nan, IDX))


def test_core_returns_all_cash():
    core, proxied = portfolio.core_returns(Portfolio({"CASH": 1.0}), PX, RF_UP)
    pd.testing.assert_series_equal(core, portfolio.cash_returns(RF_UP).rename("core"))
    assert proxied == []


def test_core_returns_missing_ticker_raises():
    with pytest.raises(KeyError, match="ZZZ"):
        portfolio.core_returns(Portfolio({"A": 0.5, "ZZZ": 0.5}), PX, RF)


def test_core_returns_proxy_fill():
    rs, ra, rc = SPXTR.pct_change(), PX["A"].pct_change(), PX["C"].pct_change()
    p = Portfolio({"A": 0.5, "C": 0.5})
    core, proxied = portfolio.core_returns(p, PX, RF, spxtr=SPXTR, proxy_fill=True)
    assert proxied == ["C"]  # A starts before the proxy: kept as is
    assert core.index[0] == IDX[1]  # the proxy's first date
    np.testing.assert_allclose(core[IDX[2] : IDX[4]], 0.5 * ra[IDX[2] : IDX[4]] + 0.5 * rs[IDX[2] : IDX[4]], atol=1e-12)
    np.testing.assert_allclose(core[IDX[5] :], 0.5 * ra[IDX[5] :] + 0.5 * rc[IDX[5] :], atol=1e-12)
    # prices that only hold the short ticker still extend back to the proxy's history
    alone, proxied = portfolio.core_returns(Portfolio({"C": 1.0}), PX[["C"]].dropna(), RF, SPXTR, proxy_fill=True)
    assert proxied == ["C"] and alone.index.equals(IDX[1:])
    np.testing.assert_allclose(alone[IDX[2] : IDX[4]], rs[IDX[2] : IDX[4]], atol=1e-12)
    # without the flag the proxy is ignored
    off, proxied = portfolio.core_returns(p, PX, RF, spxtr=SPXTR)
    assert proxied == [] and off.index[0] == IDX[4]
    with pytest.raises(ValueError, match="spxtr"):
        portfolio.core_returns(p, PX, RF, proxy_fill=True)


def test_lever():
    r = PX["A"].pct_change().fillna(0.0)
    np.testing.assert_allclose(portfolio.lever(r, pd.Series(0.0, IDX), 2, spread=0), 2 * r, rtol=0, atol=1e-16)
    pd.testing.assert_series_equal(portfolio.lever(r, RF, 1), r)
    lev = portfolio.lever(r, RF_UP, 1.5)  # default spread 50 bp, on Friday's rate over Fri -> Mon
    assert lev[IDX[3]] == pytest.approx(1.5 * r[IDX[3]] - 0.5 * (RF_UP[IDX[2]] + 0.005) * 3 / 360, abs=1e-16)
    assert lev.iloc[0] == 0.0
    gap = portfolio.lever(r, RF_UP.drop(IDX[5]), 1.5)  # the missing rate is carried forward, then lagged
    assert gap[IDX[6]] == pytest.approx(1.5 * r[IDX[6]] - 0.5 * (RF_UP[IDX[4]] + 0.005) / 360, abs=1e-16)
    with pytest.raises(ValueError, match="L >= 1"):
        portfolio.lever(r, RF_UP, 0.5)
    with pytest.raises(ValueError, match="rf starts"):
        portfolio.lever(r, RF_UP.iloc[2:], 1.5)

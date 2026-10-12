"""The user's portfolio: holdings CSV parsing and validation, weights, the daily core return
series and the leverage transform.

Pure pandas/numpy, no I/O. The holdings CSV is a trust boundary (user upload), so
parse_holdings validates everything and raises ValueError with a message naming the problem.
Returns are daily simple returns on a DatetimeIndex; rf is the panel's annualized T-bill rate
(decimal), accrued actual/360 like a money-market account.
"""

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

TICKER_RE = r"[A-Z0-9.\-^=]{1,12}"
MAX_ROWS = 50


@dataclass(frozen=True)
class Portfolio:
    """Target weights and, when built from shares, the total dollar value.

    weights: ticker -> weight, including "CASH" when > 0; no zero weights; sums to 1 (+-1e-9).
    """

    weights: dict[str, float]
    value: float | None = None


def parse_holdings(text: str) -> pd.DataFrame:
    """Parse holdings CSV text into a DataFrame with columns `ticker` and `weight` or `shares`.

    The header needs `ticker` plus exactly one of `weight` / `shares` (case and surrounding
    spaces ignored; other columns are ignored). Tickers are stripped and upper-cased and must
    match TICKER_RE ("CASH" is a ticker). Weights summing to more than 1.5 are percents and are
    divided by 100. Raises ValueError on empty input, a missing or repeated ticker / weight /
    shares column, a row with
    more fields than the header (pandas' "Expected 2 fields in line 3, saw 3"), more than 50
    rows, a bad or duplicate ticker, or a negative / non-numeric value.
    """
    if not text.strip():
        raise ValueError("empty input: expected a CSV with a header row")
    # header=None: pandas would otherwise turn the first column into the index when every row has
    # an extra field (trailing comma, unquoted 1,000) and rename duplicate headers to `x.1`
    df = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)
    cols = [c.strip().lower() for c in df.iloc[0]]
    df = df.iloc[1:]
    if cols.count("ticker") != 1:
        raise ValueError(f"need exactly one column 'ticker', got {cols}")
    kinds = [c for c in cols if c in ("weight", "shares")]
    if len(kinds) != 1:
        raise ValueError(f"need exactly one of the columns 'weight' or 'shares', got {cols}")
    kind = kinds[0]
    df.columns = cols
    if df.empty:
        raise ValueError("empty input: no holdings rows under the header")
    if len(df) > MAX_ROWS:
        raise ValueError(f"more than {MAX_ROWS} rows ({len(df)})")
    tickers = df["ticker"].str.strip().str.upper()
    bad = tickers[~tickers.str.fullmatch(TICKER_RE)].tolist()
    if bad:
        raise ValueError(f"bad ticker {bad}: use 1-12 characters from A-Z 0-9 . - ^ =")
    dup = tickers[tickers.duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"duplicate ticker {dup}")
    values = pd.to_numeric(df[kind].str.strip(), errors="coerce").astype(float)
    nonnum = tickers[~np.isfinite(values)].tolist()
    if nonnum:
        raise ValueError(f"non-numeric {kind} for {nonnum}")
    neg = tickers[values < 0].tolist()
    if neg:
        raise ValueError(f"negative {kind} for {neg}")
    if kind == "weight" and values.sum() > 1.5:
        values = values / 100
    return pd.DataFrame({"ticker": tickers.tolist(), kind: values.tolist()})


def to_portfolio(holdings: pd.DataFrame, last_prices: pd.Series | None = None, cash: float = 0.0) -> Portfolio:
    """Portfolio from parse_holdings output.

    Shares mode: value_i = shares_i * last_prices[ticker] (a CASH row's shares are dollars),
    `cash` is extra dollars, Portfolio.value = total dollars, w_i = value_i / total. Every
    non-CASH ticker needs a last price (ValueError naming the missing ones otherwise).
    Weight mode: `cash` is an extra CASH weight; all weights are normalized to sum to 1 and
    value is None. Zero weights are dropped. Raises ValueError if cash < 0 or the total is 0.
    """
    if cash < 0:
        raise ValueError(f"cash must be >= 0, got {cash}")
    tickers = holdings["ticker"].tolist()
    shares = "shares" in holdings.columns
    if shares:
        px = (last_prices if last_prices is not None else pd.Series(dtype=float)).reindex(tickers)
        px[[t == "CASH" for t in tickers]] = 1.0
        missing = px.index[px.isna()].tolist()
        if missing:
            raise ValueError(f"shares mode needs last_prices for {missing}")
        amounts = holdings["shares"].to_numpy(float) * px.to_numpy(float)
    else:
        amounts = holdings["weight"].to_numpy(float)
    w = dict(zip(tickers, amounts.tolist()))
    w["CASH"] = w.get("CASH", 0.0) + cash
    total = sum(w.values())
    if not total > 0:
        raise ValueError("holdings sum to zero")
    return Portfolio({t: v / total for t, v in w.items() if v > 0}, total if shares else None)


def cash_returns(rf: pd.Series) -> pd.Series:
    """Daily T-bill return rf_{t-1} * (calendar days from t-1 to t) / 360; the first row is 0."""
    out = rf.shift(1) * rf.index.to_series().diff().dt.days / 360
    out.iloc[0] = 0.0
    return out


def core_returns(
    p: Portfolio, prices: pd.DataFrame, rf: pd.Series, spxtr: pd.Series | None = None, proxy_fill: bool = False
) -> tuple[pd.Series, list[str]]:
    """Daily constant-mix core return sum_i w_i r_i + w_CASH * cash_returns(rf), and the proxied tickers.

    r_i are adjusted-close pct changes over the dates where every ticker has a price, so the
    sample starts at the latest first price and a missing print drops that date (the next
    return spans both days). The first row is 0 (no prior price), as in cash_returns. rf is
    carried forward onto the sample dates; with a CASH weight the sample also starts no earlier
    than rf. proxy_fill=True: each ticker whose first price comes after the proxy's first date
    gets spxtr's returns before (and on) its first price date, so the sample starts at the
    proxy's first date; such tickers are listed in the returned list.

    An all-CASH portfolio is w_CASH * cash_returns(rf) on rf's own dates.

    Raises KeyError naming tickers missing from prices.columns, ValueError if proxy_fill is set
    without spxtr, a CASH weight comes with no rf data, or no date has a price for every ticker.
    """
    # ponytail: daily constant-mix; monthly drift is negligible
    tickers = [t for t in p.weights if t != "CASH"]
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise KeyError(f"no prices for {missing}")
    if proxy_fill and spxtr is None:
        raise ValueError("proxy_fill needs spxtr")
    w_cash = p.weights.get("CASH", 0.0)
    if w_cash > 0 and rf.first_valid_index() is None:
        raise ValueError("a CASH weight needs rf data")
    if not tickers:
        return (w_cash * cash_returns(rf.dropna())).rename("core"), []
    px, proxied = prices[tickers], []
    if proxy_fill:
        px = px.reindex(px.index.union(spxtr.dropna().index))
        sp = spxtr.dropna().reindex(px.index).ffill()
        sp0 = sp.first_valid_index()
        for t in tickers:
            first = px[t].first_valid_index()
            if first is not None and sp0 is not None and first > sp0:
                before = px.index < first  # splice spxtr levels onto the first price: returns before it are spxtr's
                px.loc[before, t] = sp[before] * (px.at[first, t] / sp[first])
                proxied.append(t)
    if w_cash > 0:
        px = px.loc[rf.first_valid_index() :]
    px = px.dropna()
    if px.empty:
        raise ValueError(f"no date with a price for every ticker in {tickers}" + (" and rf" if w_cash > 0 else ""))
    r = px.pct_change()
    r.iloc[0] = 0.0
    core = r @ pd.Series({t: p.weights[t] for t in tickers}, dtype=float)
    if w_cash > 0:
        core = core + w_cash * cash_returns(rf.dropna().reindex(px.index, method="ffill"))
    return core.rename("core"), proxied


def lever(r: pd.Series, rf: pd.Series, L: float, spread: float = 0.005) -> pd.Series:
    """Levered return r_L = L r - (L - 1)(rf_{t-1} + spread) * calendar days / 360.

    Borrowing costs the T-bill rate plus `spread` (default 50 bp), accrued like cash_returns on
    r's dates (rf carried forward onto them). L = 1 returns r unchanged. Raises ValueError for
    L < 1 (deleverage with the T-bill sleeve: cash earns rf, not rf + spread) or when rf has no
    value on or before r's first date.
    """
    if L < 1:
        raise ValueError(f"lever needs L >= 1, got {L}; deleverage with the T-bill sleeve")
    if L == 1:
        return r
    first = rf.first_valid_index()
    if first is None or first > r.index[0]:
        raise ValueError(f"rf starts {first}, after r's first date {r.index[0]}")
    return L * r - (L - 1) * cash_returns(rf.dropna().reindex(r.index, method="ffill") + spread)

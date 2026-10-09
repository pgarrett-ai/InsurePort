"""Rebuild src/insureport/sp500_annual.csv: S&P 500 calendar-year total returns, 1928 to the last
complete year (network required).

Usage, from the repo root: .venv/Scripts/python scripts/make_sp500_annual.py
Primary source: Damodaran, "Historical Returns on Stocks, Bonds and Bills" (histretSP), the
"S&P 500 (includes dividends)" column, read from the HTML rendering of the same workbook
(histretSP.xls needs xlrd, which is not a dependency); it prints percents to 2 decimals, so
values carry 4 decimals. Fallback: Shiller's monthly price and dividend series (ie_data.xls),
read from the datasets/s-and-p-500 CSV mirror for the same reason. The CSV's first line names the
source used; nothing is written unless the 2008 / 2013 / 1931 spot checks pass. The test suite's
golden fixtures are Damodaran's numbers, so a fallback build is written with a warning.
"""

import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd

from insureport.data import _http_text

OUT = Path(__file__).resolve().parents[1] / "src" / "insureport" / "sp500_annual.csv"
DAMODARAN = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html"
SHILLER = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
CHECKS = {2008: -0.37, 2013: 0.32, 1931: -0.44}  # +-0.03: catches parse errors, passes the Shiller approximation


def damodaran() -> pd.Series:
    """Year -> total return from the first "S&P 500" column of Damodaran's table."""
    t = pd.read_html(io.StringIO(_http_text(DAMODARAN)))[0].astype(str)
    col = t.columns[t.apply(lambda x: x.str.contains("S&P 500").any()).argmax()]  # returns; the 2nd is $ value
    year = pd.to_numeric(t.iloc[:, 0], errors="coerce")
    ok = year >= 1928
    return pd.Series(t.loc[ok, col].str.rstrip("%").astype(float).to_numpy() / 100, year[ok].astype(int).to_numpy())


def shiller() -> pd.Series:
    """Year -> total return compounded from Shiller's monthly (P_t + D_t / 12) / P_{t-1} - 1."""
    m = pd.read_csv(io.StringIO(_http_text(SHILLER)), parse_dates=["Date"], index_col="Date")
    # ponytail: P is a monthly average of closes, not the year-end close, and recent D prints are 0
    # (missing) in the mirror, carried forward: vs Damodaran 1928-2025 the yearly error is 2.2 pp on
    # average, 8 pp at worst (1938), enough to move a year near a regime edge. Fallback only.
    d = m["Dividend"].replace(0.0, np.nan).ffill()
    r = (m["SP500"] + d / 12) / m["SP500"].shift() - 1
    by_year = (1 + r).groupby(r.index.year)
    out = by_year.prod().where(by_year.count() == 12) - 1
    return out.loc[1928:].dropna()


def check(s: pd.Series) -> None:
    """Raise unless years run gap-free from 1928 to the last complete year, or to the year before
    (a source can lag a year; the script then prints a warning), and the spot values hold."""
    last = dt.date.today().year - 1
    if s.index[0] != 1928 or s.index[-1] < last - 1 or (np.diff(s.index) != 1).any() or s.isna().any():
        raise ValueError(f"years {s.index[0]}..{s.index[-1]} (n={len(s)}) are not a gap-free 1928..{last}")
    bad = {y: round(s.get(y, np.nan), 4) for y, v in CHECKS.items() if not abs(s.get(y, np.nan) - v) <= 0.03}
    if bad:
        raise ValueError(f"spot checks failed: {bad} vs {CHECKS}")


SOURCES = (("Damodaran histretSP", DAMODARAN, damodaran), ("Shiller ie_data (CSV mirror)", SHILLER, shiller))
for name, url, fetch in SOURCES:
    try:
        s = fetch()
        s = s[s.index < dt.date.today().year]  # complete years only
        check(s)
        break
    except Exception as e:  # network, layout change, failed check: try the next source
        print(f"{name} failed: {e}")
else:
    raise SystemExit("no source passed the checks; sp500_annual.csv not written")

s = s.round(6).rename("total_return").rename_axis("year")
header = f"# source: {name}, S&P 500 total return incl. dividends, {url} (fetched {dt.date.today():%Y-%m-%d})\n"
OUT.write_text(header + s.to_csv(lineterminator="\n"), encoding="utf-8", newline="\n")
print(f"{OUT.name}: {len(s)} years {s.index[0]}-{s.index[-1]} from {name}")
print("checks:", {y: float(s[y]) for y in CHECKS})
if s.index[-1] < dt.date.today().year - 1:
    print(f"WARNING: source ends {s.index[-1]}, last complete year is {dt.date.today().year - 1}")
if fetch is not damodaran:
    print(
        "WARNING: fallback data. tests/test_scenarios.py::test_part_one_cartoon_on_1997_2016 pins"
        " Damodaran's values (the 1997-2016 bins and spreads), so it may fail until a Damodaran rebuild."
    )

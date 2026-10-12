"""Regenerate tests/fixtures/market_2018_2019.csv from live data (network required).

Usage, from the repo root: .venv/Scripts/python scripts/make_fixture.py
The 2018-2019 window holds the Feb-2018 volatility spike and the Q4-2018 drawdown; the panel
starts 2017-06-01 so the window's first rows have real values, not start-of-data fills.
"""

from pathlib import Path

from insureport.data import load_market

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "market_2018_2019.csv"

m = load_market("2017-06-01").loc["2018-01-01":"2019-12-31"].round(6)
# every series exists in 2018-2019, so a NaN means a fail-soft fetch failed: don't write it
if m.isna().any().any():
    raise SystemExit(f"NaN in panel, not writing fixture: {m.attrs['warnings']}")
# ponytail: known source glitches kept as-is in the fixture: ^SP500TR 2019-07-03 (bad q that day,
# see the note in data.load_market) and Cboe CLL stale prints 2019-01-03..01-18 (801.64 for six
# days, then 834.07 for six, a +4.05% step on 2019-01-11 with SPX flat). Not a replication error.
OUT.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT, index_label="date")
print(m.shape, m.index[0].date(), m.index[-1].date())
if m.attrs["warnings"]:
    print("warnings:", m.attrs["warnings"])

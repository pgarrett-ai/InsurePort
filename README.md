# InsurePort

Does portfolio insurance raise this portfolio's compound growth rate? InsurePort takes a portfolio of US-listed holdings, measures its exposure to crashes, proposes safe-haven sleeves and tests each one. The test comes from Mark Spitznagel of Universa Investments (*Safe Haven: Investing for Financial Storms*, 2021): a hedge is worth holding only if adding it raises the geometric (compound) return of the whole portfolio, not just if it cuts losses. The critics' tests run next to his, so both scorecards are on the screen. The tool takes no side.

> **Validation status: FAIL.** Option prices in InsurePort are model-priced, not traded quotes. Out of sample, the pricer tracks the path of Cboe's PPUT and PPUT3M put-protection indices closely (tracking error 1.19% and 0.92% a year, hedge-P&L correlation 0.981), but its replicas trail their level by 1.68 and 1.33 points a year, and the CLL collar replica misses its tracking-error limit. All three targets fail at least one pre-registered tolerance ([details](#how-the-option-prices-are-made-and-validated)). Every option result in the app and the case study is conditional on this pricer.

## What it does

- **Diagnostics.** CAGR against the arithmetic mean and the volatility tax between them, drawdowns and time under water, beta and downside beta to the S&P 500, returns over the named crash windows inside the sample (11 are defined, 1987 to 2025; the 1990 panel holds 10), concentration (HHI, effective number of holdings) and the SPX / XSP put notional that would hedge the downside beta.
- **Candidate sleeves.** A catalog of 20 editable presets: 19 safe havens sorted into the book's archetypes (insurance, alpha, store of value) and a 60/40 reference row. The havens are far out-of-the-money SPX put programs (a Universa-style 96.67/3.33 model portfolio, the *Dao of Capital* ch. 9 prototype), a put spread, a protective put, a collar, the real Cboe PPUT, PPUT3M, CLL and VXTH index levels, Cambria's TAIL ETF and a TAIL proxy rebuilt from its filings, T-bills, Treasuries, gold, S&P 500 trend following and a managed-futures ETF. A shortlist is suggested from the diagnostics.
- **The three tests.** The historical frontier: the portfolio holds (1 - w) of the core and w of the sleeve for every w on a grid, and the CAGR-maximizing weight w* is the empirical Kelly choice. The bootstraps: a stationary block bootstrap of the paired portfolios and the book's 25-year annual bootstrap. The dice: Spitznagel's regime engine on calendar-year S&P 500 returns since 1928, with the Kelly weight and the crash frequency at which a hedge breaks even.
- **The critics' tests.** Hold less equity instead: the sleeve against the T-bill frontier at the same maximum drawdown and at the same CAGR (AQR's test). The sleeve on its own: CAGR, Sharpe ratio, alpha and beta. Roll timing: the same strategy rolled 5, 10 and 15 business days later. Pricing sensitivity: the wing slope, the volatility risk premium, the spread and the roll offset, one at a time (the case study adds the far-wing shape and the slope's SKEW sensitivity). Hold more equity: the hedge financed on top of a fully invested core.
- **Real option data adapter.** A purchased end-of-day SPX chain (historicaloptiondata.com, Cboe DataShop, ORATS or OptionMetrics IvyDB layouts) is read into one schema. Option sleeves then open listed contracts and price from its quotes, and fall back to the model only where a quote is missing.

Each candidate ends with a verdict: cost-effective (robust), cost-effective (fragile), not cost-effective, or insufficient history.

## Case study

[`CASE_STUDY.md`](CASE_STUDY.md) runs every candidate on three cores (the S&P 500 TR, a 60/40 and a concentrated tech sample) from 1990-01-02 to 2026-10-09. Under model prices none of the six model-priced option presets raised any core's CAGR. [METHODOLOGY.md, section 9](METHODOLOGY.md#9-case-study-summary) summarizes the rest.

## Screens

The app's charts and the case study's figures come from the same chart builders (`src/insureport/charts.py`). These are case-study figures, one per page that draws charts.

**Diagnostics.** Growth of 1 dollar in the three case-study cores (log scale).

![Growth of 1 dollar in the three cores](figures/cs_01_cores_growth.png)

**Candidates.** The cost-effectiveness plane: arithmetic cost against the change in CAGR, one point per candidate.

![Cost-effectiveness plane, S&P 500 core](figures/cs_04_ce_plane_weval_spx.png)

**Stress lab.** A frontier: CAGR and maximum drawdown by sleeve weight, against the T-bill frontier and the 30% drawdown cap.

![Universa model frontier, S&P 500 core](figures/cs_03_frontier_spx_universa_model.png)

**Methodology.** The PPUT replica built from model prices against the Cboe index.

![PPUT replica against the Cboe index](figures/cs_09_replication_pput.png)

The Portfolio page is a holdings editor and has no chart.

## Stack

- Python 3.13 (`pyproject.toml` allows 3.12 and later); numpy, pandas, scipy.
- Streamlit for the app, plotly for the charts; kaleido (optional) for PNG export in the scripts.
- Data: Yahoo Finance through yfinance, Cboe's public index CSVs and FRED, cached as parquet files under `data/` (pyarrow).
- No database, no API keys, no paid data. A purchased option chain is optional.

## Quickstart

```bash
git clone https://github.com/pgarrett-ai/InsurePort.git
cd InsurePort
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

The app opens on the Portfolio page. Pick a sample from `samples/` or upload a CSV with a `ticker` column and either `weight` or `shares` (a `CASH` row is cash; at most 50 rows and 1 MB), click **Build**, then work through Diagnostics, Candidates and Stress lab. The settings sidebar holds the dates, the drawdown cap, costs, funding and the pricing knobs.

The first run downloads the market panel (13 series from Yahoo Finance, Cboe and FRED) into `data/`, which is gitignored; the haven ETFs and your tickers follow as the pages need them. Each series is refreshed after 20 hours. When a download fails the app uses the cached copy and logs a warning.

To use real option prices, put a vendor CSV or zip in `data/chains/` and pick it on the Candidates page.

The two scripts regenerate the published numbers (network required):

```bash
pip install -e ".[case-study]"      # kaleido, for PNG figures
python scripts/calibrate.py         # pricer validation: figures/validation_table.md
python scripts/case_study.py        # CASE_STUDY.md's tables and figures: figures/case_study_tables.md, figures/cs_*.png
```

`scripts/case_study.py` took 337 s on the run behind the committed case study. Without a working kaleido both scripts write HTML figures instead.

## Tests

```bash
pytest -q
```

372 tests, offline by construction. `tests/conftest.py` sets `INSUREPORT_OFFLINE=1`, which makes both network seams in `src/insureport/data.py` raise, and points the cache at a temporary directory. I/O is monkeypatched, the market tests run on a committed 2018-2019 fixture panel, and each Streamlit page has a smoke test through `streamlit.testing`. The suite runs in about a minute.

## Data sources and coverage

| Series | Source | First date | Note |
|---|---|---|---|
| S&P 500 price `^GSPC` (close and open) | Yahoo Finance | 1927 | The market panel runs on its trading days from 1990. The open stands in for Cboe's AM settlement value. |
| S&P 500 total return `^SP500TR` | Yahoo Finance | 1988 | The equity leg of every core and replica. |
| VIX, SKEW | Cboe | 1990 | |
| VIX9D | Cboe | 2011-01-04 | Regression-filled from VIX before. |
| VIX3M | Yahoo Finance `^VIX3M` | 2006-07-17 | Cboe's own CSV starts in 2009. Regression-filled from VIX before. |
| VIX6M | Cboe | 2008-01-02 | Regression-filled from VIX3M before. |
| VIX1Y | Cboe | 2007-01-03 | Regression-filled from VIX6M before. |
| PPUT, PPUT3M, CLL, VXTH | Cboe | 1986, 2004-03-19, 2008-08-26, 2006-03-31 | Cboe back-test levels before each launch (2015-08-03, 2021-08-02, 2008-09-04, 2011-07-29). |
| 3-month T-bill `DTB3`, 10-year yield `DGS10` | FRED | 1954, 1962-01-02 | |
| TLT, IEF, GLD, AGG, TAIL, DBMF, your tickers | Yahoo Finance (adjusted close) | 2002-07-30, 2002-07-30, 2004-11-18, 2003-09-29, 2017-04-06, 2019-05-08 | |
| S&P 500 calendar-year total return, CPI-U | Damodaran (NYU Stern); FRED `CPIAUCNS` | 1928 | Packaged as `src/insureport/sp500_annual.csv`. |

No synthetic option is priced before 1990. Before the Treasury ETFs start in 2002, Treasuries are covered only by a 10-year par-bond proxy on DGS10, and there is no gold before 2004-11-18. [METHODOLOGY.md, section 2](METHODOLOGY.md#2-data) gives the endpoints, the panel construction and the known glitches.

## How the option prices are made and validated

Each SPX option is priced with Black-Scholes-Merton at an implied volatility read from a surface rebuilt every day from the Cboe VIX term structure (VIX9D, VIX, VIX3M, VIX6M and VIX1Y as variance-swap pillars) and the Cboe SKEW index. The put wing is linear in standardized moneyness, with a slope anchored to one observed SPX smile (the 2026-10-08 Cboe XSP chain) and damped in SKEW; the smile is normalized so its mean variance equals the variance-swap level and capped so put prices never fall as the strike rises. The pricer is validated, not fitted: replicas of Cboe's PPUT, PPUT3M and CLL strategy indices, built from model prices under Cboe's conventions, are compared out of sample with the real index levels against tolerances written down before the run. They follow the indices' path but trail their level, so the model's near-money puts are richer than Cboe's fills, and no free index validates the far wing where the Universa-style presets buy.

| Target | Out-of-sample window | Tracking error (limit) | dCAGR (absolute limit) | Hedge-P&L correlation (limit) | Result |
|---|---|---|---|---|---|
| PPUT | 2016-01 to 2026-10 | 1.19% (1.50%) | -1.68% (0.50%) | 0.981 (0.90) | FAIL (dCAGR) |
| PPUT3M | 2016-01 to 2026-10 | 0.92% (2.00%) | -1.33% (0.75%) | 0.981 (0.85) | FAIL (dCAGR) |
| CLL | 2008-09 to 2026-10 | 3.59% (2.50%) | -0.71% (1.00%) | 0.947 (0.85) | FAIL (tracking error) |

Monthly returns, replica minus index; from `figures/validation_table.md`, written by `scripts/calibrate.py`. The method, the conventions and the full table are in [METHODOLOGY.md, section 3](METHODOLOGY.md#3-option-pricing) and [section 5](METHODOLOGY.md#5-validation).

## Known limitations

- **Model-priced options.** Option results come from a model, not from traded quotes, and the model fails its validation on level (above). The protective put and collar are probably conservative, because the model's near-money puts cost more than Cboe's index fills.
- **Far-wing uncertainty.** The far wing rests on one calm day of real quotes and an assumed SKEW sensitivity. On that day the model matched 30% out-of-the-money put mids within about 5% at 29 to 84 days (market / model 0.96 to 1.05). One assumption at a time, the *Dao* preset's dCAGR at a 0.50% weight moves from -85 bp (base) to +258 bp with today's slope and the wing flat past three standard deviations (z_cap 3, also the verdict's wing robustness check), and runs from -331 bp to +246 bp across SKEW sensitivities of 0 to 1.
- **AM settlement and fills.** Cboe settles at the AM Special Opening Quotation and fills at an intraday VWAP; the replicas use the `^GSPC` open and closing marks. On 175 of the 312 PPUT roll days from 1990 to 2015 Yahoo's open equals the previous close.
- **Volatility risk premium.** Options are priced at the variance-swap level (nu = 1). Pricing at 0.8 times it moves the *Dao* preset from -85 bp to +415 bp.
- **Path dependence.** Across roll dates 0, 5, 10 and 15 business days late, the Universa-style sleeve's dCAGR spans 210 bp and the *Dao* preset's 372 bp (case study t06c).
- **Few crashes.** The options era since 1990 holds three crash years (S&P 500 down 15% or more). Bootstraps resample these events; they cannot add new ones.
- **In-sample w\*.** The weight is chosen on the history it is scored on. The bootstraps, rolling windows, the post-2015 sample and the roll-offset sweep are guardrails, not proof.
- **Survivorship and hindsight.** A portfolio of today's holdings carries survivorship, and the presets come from descriptions published after the fact.
- **Short histories.** TAIL (from 2017) and DBMF (from 2019) have fewer than 10 years and are reported as insufficient history.
- **Not modelled.** Taxes, margin, contract granularity, capacity, financing beyond T-bills plus 50 bp, and Cboe's CLL cross-roll. Universa's figures are self-reported and hypothetical, and serve as consistency checks only.

The full list is in [METHODOLOGY.md, section 10](METHODOLOGY.md#10-limitations-and-disclosures).

## Methodology

[`METHODOLOGY.md`](METHODOLOGY.md) gives every formula, data source and modelling choice, section by section:

1. [Philosophy](METHODOLOGY.md#1-philosophy): compound growth, the volatility tax, the three kinds of haven and the case against.
2. [Data](METHODOLOGY.md#2-data): endpoints, coverage, the market panel, rates, back-fill and caching.
3. [Option pricing](METHODOLOGY.md#3-option-pricing): the volatility surface, the smile, its caps and its limits.
4. [Sleeves](METHODOLOGY.md#4-sleeves): premium and overlay accounting, rolls, settlement and the 20 presets.
5. [Validation](METHODOLOGY.md#5-validation): the Cboe replicas, the tolerances and the results.
6. [Backtest engine](METHODOLOGY.md#6-backtest-engine): rebalancing, costs, funding, the frontier and the AQR twins.
7. [Scenario engines](METHODOLOGY.md#7-scenario-engines): regimes, Kelly, the breakeven crash frequency and the bootstraps.
8. [Suggestions, verdicts and the two scorecards](METHODOLOGY.md#8-suggestions-verdicts-and-the-two-scorecards).
9. [Case-study summary](METHODOLOGY.md#9-case-study-summary).
10. [Limitations and disclosures](METHODOLOGY.md#10-limitations-and-disclosures).
11. [References](METHODOLOGY.md#11-references): books, papers, index methodology documents and data sources.

## Disclaimer

InsurePort is a research and education tool. Nothing in it is investment advice or a recommendation to buy or sell any security, option or fund, and its results are hypothetical back-tests.

InsurePort is not affiliated with, endorsed by or sponsored by Universa Investments, Cboe Global Markets, Cambria Investment Management, S&P Dow Jones Indices, Yahoo or the Federal Reserve Bank of St. Louis (FRED). S&P 500 is a trademark of S&P Dow Jones Indices LLC. Cboe, VIX, SKEW and the Cboe index names are trademarks of Cboe Exchange, Inc. Other names are the trademarks of their owners.

Market data is fetched at run time and is not redistributed in this repository, apart from small test fixtures and the annual S&P 500 and CPI table with its source line. yfinance is not affiliated with Yahoo, and Yahoo Finance data is intended for personal use: read Yahoo's terms before using it otherwise. Cboe and FRED data are subject to their providers' terms.

Licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

# InsurePort methodology

This document describes every formula, data source, modelling choice, validation result and known limitation in InsurePort. It is written against the code in `src/insureport/` (module and function names are given as `module.function`) and the outputs committed with it.

**Where the numbers come from.** Validation numbers come from `scripts/calibrate.py`: the table it writes to `figures/validation_table.md`, and the sensitivity rows it prints (sections 5.4 and 5.7). Case-study numbers come from `figures/case_study_tables.md`, written by `scripts/case_study.py` (run of 2026-10-11, market data to 2026-10-09) and copied into [`CASE_STUDY.md`](https://github.com/pgarrett-ai/InsurePort/blob/main/CASE_STUDY.md) with table ids such as "t08". Formula checks come from the test suite (`tests/`), cited by test name. Figures quoted from books and papers are marked with their reference in section 11. Numbers move when the data refresh.

**Conventions.** Returns are simple decimal returns. CAGR uses calendar days: `(NAV_last / NAV_first) ^ (365.25 / days) - 1`. "bp" means basis points a year. "dCAGR" is the CAGR of the hedged portfolio minus that of the unhedged one. The core is the user's portfolio; a sleeve is the safe haven added to it; w is the sleeve's share of the portfolio.

## 1. Philosophy

InsurePort tests one idea from Mark Spitznagel's *Safe Haven* (2021) and the Universa research notes before it: a hedge is worth holding only if adding it raises the compound growth rate of the whole portfolio.

### 1.1 The objective is compound growth

Wealth compounds as the product of (1 + r_t), so the rate an investor earns is the geometric mean, `CAGR = exp(mean ln(1 + r_t)) - 1`. The arithmetic mean of r_t overstates it. A year of -50% and a year of +100% average +25% but compound to 0% (`test_two_outcome_volatility_tax`). Recovering from a loss L takes a gain of L / (1 - L): 43% after a 30% loss, 100% after a 50% loss.

Spitznagel calls the gap between the arithmetic and the geometric mean the *volatility tax* [Spitznagel 2018a]. For roughly lognormal returns it is about sigma^2 / 2: two points a year at 20% volatility. His example: over 2008-2017 the S&P 500 averaged 10.4% with 19.3% volatility but compounded at about 8.5% [Spitznagel 2018b]. He stresses that large losses, not volatility as such, drive it. InsurePort measures the tax on realized calendar-year returns (`metrics.vol_tax`: arithmetic mean minus geometric mean) and shows var / 2 only as a labelled approximation. Two checks: +50% then -50% compounds at -13.3975% a year (`test_vol_tax_plus_minus_fifty`), and for two-point lognormal years the tax is exactly `e^mu (cosh sigma - 1)`, 0.021522 at mu 0.07, sigma 0.2 (`test_vol_tax_two_point_lognormal`).

### 1.2 One path, not the average of many

The arithmetic mean averages across many parallel copies of the market. The geometric mean is what one investor earns on the single path that happens. Under compounding these differ, so the process is *non-ergodic*. Peters [2011] shows that for geometric Brownian motion the ensemble growth rate is mu but the time-average rate is mu - sigma^2 / 2. Maximizing the time average gives the Kelly leverage (mu - r) / sigma^2 [Kelly 1956] with no utility assumption. Spitznagel makes the same point with Nietzsche's eternal return: if you had to relive your realized returns forever, only that path's compound rate would matter [Spitznagel 2019; 2021, ch. 3]. InsurePort scores candidates on historical CAGR and on the median, not the mean, of bootstrapped growth. `scenarios.bootstrap` also returns the mean-wealth ("ensemble") growth next to the median (section 7.7), and the Stress lab's dice chart draws the arithmetic (ensemble) and geometric (time-average) growth side by side.

### 1.3 Cost-effective risk mitigation

Spitznagel defines risk as exposure to loss, not volatility or correlation. A mitigation strategy is cost-effective if adding it raises the portfolio's CAGR over time. One that lowers risk only by lowering growth is a cure worse than the disease [Spitznagel 2021, ch. 1]. The test only works one way. Failing to raise CAGR disqualifies a strategy, but raising it proves nothing on its own, since the gain could come from added risk. InsurePort therefore splits every gain into crash-year payoff and calm-year cost, and reports the change in equity beta (section 8.6).

A two-state market shows the mechanics. The market returns +10% in nine years of ten and -30% in the tenth: arithmetic mean 6.0%, CAGR 5.14%. An insurance contract loses its whole cost in normal years and pays +900% in the crash year, so its expected return is zero. With 2% in the contract and 98% in the market, the arithmetic mean falls to 5.88% and the CAGR rises to 5.71%. The hedge gives up 0.12 points of arithmetic return and saves 0.69 points of volatility tax, a net gain of 0.57. CAGR peaks at a 3.2% allocation (the Kelly weight, 0.30 / 9.3 in closed form). With these payoffs the hedge adds growth whenever crashes come in more than 7% of years. (`test_part_three_cartoon_insurance` pins the two CAGRs, 5.14% and 5.71%, and the 7% breakeven; the other figures are hand arithmetic on the same inputs.)

The book's *safe haven frontier* rests on the same split:

```text
change in CAGR = volatility tax saved - arithmetic return given up
```

The book first sizes every haven to the same downside protection (in its bootstrap, the same 5th-percentile 25-year CAGR) and then places each on a plane of arithmetic cost against geometric effect [Spitznagel 2021, ch. 5-6]. A haven is cost-effective when it plots above the line where the two are equal. InsurePort draws this plane from its own backtests, but at each candidate's CAGR-maximizing weight (or a preset's own weight), not at equal protection: the book's equal-tail sizing is not implemented yet (section 8.2).

### 1.4 Three kinds of haven

Spitznagel sorts havens by how they pay across market states [Spitznagel 2017a; 2021, ch. 4]. The Universa notes split calendar-year S&P 500 returns at -15%, 0% and +15%, and call the bin at or below -15% a crash. The cartoon payoffs are:

- **Store of value.** The same return in every state, for example cash at 2% above inflation. It cuts losses only by holding less of the market.
- **Alpha.** Positive in every state and higher in a crash: +20%, +10%, +5%, +5% across the bins.
- **Insurance.** -100% outside the crash bin, +900% inside it. Its stand-alone mean is zero by construction; at crash frequency p the zero-mean crash payoff is (1 - p) / p.

Over the 20 years 1997-2016, with two crash years, 3% in insurance added 0.67 points to the S&P 500's CAGR, 10% in alpha added 0.18, and 10% in store of value cost 0.17 [Spitznagel 2017a]. InsurePort reproduces them on Damodaran's annual returns as +66, +18 and -18 bp (case study t07b; `test_part_one_cartoon_on_1997_2016`). The book's own bootstrap reports the same order: a median 25-year CAGR of 9.5% for the S&P 500, 10.0% with insurance and 9.1% with store of value [Spitznagel 2021, ch. 5].

### 1.5 Why far out-of-the-money puts

An index put is the closest real instrument to the insurance cartoon:

1. **The crash payoff is contractual.** Bonds, gold and trend pay in crashes only by statistical regularity [Spitznagel 2021, ch. 6], which can fail, as long Treasuries did in 2022 [Lagnado 2024].
2. **The payoff per dollar is largest in the far wing.** A crash lifts a put through both the price move and the jump in implied volatility, and the second effect is strongest far from the money [Taleb 2013; Hoffstein 2020].
3. **A large multiple needs only a small allocation**, so the drag in normal years stays small. A rough rule: allocation equals the crash-year loss divided by the crash payoff, for example 30% / 1000% = 3% [Spitznagel 2017b].

To offset a market loss L, a sleeve of weight w must return L (1 - w) / w, about 800% for a 3% sleeve against a 25% fall [Ilmanen et al. 2020]. Published prototypes buy puts about 30% out of the money with two to three months to expiry and roll them monthly, sized at 0.5% of the portfolio a month [Spitznagel 2013, ch. 9] or to break even on a one-month 20% fall [Mann, Spitznagel and Yarckin 2015]. Universa's live rules are not public.

### 1.6 The case against

The critics make three serious arguments.

**The volatility risk premium.** Option buyers pay for protection on average. Over 1990-2014, implied volatility exceeded the realized volatility that followed by about 3.4 points [Israelov and Nielsen 2015]; variance swaps show the same premium [Carr and Wu 2009], and the put wing carries an extra one [Kozhan, Neuberger and Schneider 2013]. Rolling one-month 5% out-of-the-money puts lost money in every decade of AQR's 35-year test [Ilmanen et al. 2020], and delta-hedged puts lost in every VIX decile, the calmest included [Israelov and Nielsen 2015].

**Path dependence.** A put protects only from purchase to expiry. Slow drawdowns and those that cross roll dates are poorly hedged, and moving the roll date can change the result [Israelov 2019]. Cboe's PPUT index, which always holds a 5% out-of-the-money S&P 500 put, was flat in the first quarter of 2020 while the S&P 500 fell 19.6% [Cboe, PPUT quarterly returns].

**Hold less equity.** Over 1986-2016, a fixed 36.5% equity / 63.5% cash mix matched PPUT's return with half its beta and shallower drawdowns in nearly every window [Israelov 2019]. Selling equity cut downside beta as much as a protective put's long-volatility component did, for about 0.6% a year against about 2.0% [Israelov and Nielsen 2015].

The two sides agree on more than it seems. AQR accepts the volatility-tax arithmetic, and that a hedge's cost could be offset if it lets the investor hold more equity [Ilmanen et al. 2020]. Taleb agrees that at-the-money options are expensive but argues that no one has shown the far wing to be [Taleb 2013; Ilmanen 2013]. Most critical evidence uses puts at most 10% out of the money at full notional, while Spitznagel's case concerns puts near 30% out of the money on a small budget. Still in dispute: whether the far wing is overpriced, how many crashes it takes to trust a payoff estimate, which scoreboard to use (CAGR, Sharpe ratio, or drawdown at equal return), and how long investors will stay with a strategy that lags. Universa's published results cannot settle this: they are self-reported hypothetical pairings with the S&P 500 [Universa 2020].

### 1.7 What InsurePort does with this

The tool does not take a side. The headline is the change in the portfolio's CAGR, shown next to the critics' tests: holding less equity at equal maximum drawdown or equal CAGR, the sleeve's stand-alone record, and a variant in which the hedge funds extra equity. The pricing model's volatility-premium, slope and wing settings can be swept. Results are reported across roll dates, rolling windows and bootstrap paths, so readers can see which conclusions rest on assumptions the data cannot pin down.

## 2. Data

### 2.1 Sources and endpoints

| Source | Endpoint | Series |
|---|---|---|
| Yahoo Finance, through yfinance | `yf.download(..., auto_adjust=True)`, full history from 1927 | `^GSPC` close and open, `^SP500TR`, `^VIX3M`, ETFs and user tickers (adjusted close) |
| Cboe | `https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv` | VIX, VIX9D, VIX6M, VIX1Y, SKEW, PPUT, PPUT3M, CLL, VXTH |
| FRED | `https://fred.stlouisfed.org/graph/fredgraph.csv?id={ID}` (no key) | DTB3, DGS10, CPIAUCNS |
| Damodaran, NYU Stern | `histretSP` HTML table | S&P 500 calendar-year total return, 1928 on |

Only two functions touch the network, `data._yf_download` and `data._http_text`. Cboe's VIX-family files are `DATE,OPEN,HIGH,LOW,CLOSE` and the strategy-index files `DATE,<SYMBOL>`; `data.parse_cboe_csv` reads the close (or the single value column) and refuses a file without one. FRED's missing value `.` becomes NaN.

### 2.2 Coverage

| Series | Source | First date | Before it |
|---|---|---|---|
| `^GSPC` close and open | Yahoo | 1927 | |
| `^SP500TR` | Yahoo | 1988 | |
| VIX, SKEW | Cboe | 1990 | |
| VIX9D | Cboe | 2011-01-04 | regression-filled from VIX |
| VIX3M | Yahoo `^VIX3M` (Cboe's own file starts in 2009) | 2006-07-17 | regression-filled from VIX |
| VIX6M | Cboe | 2008-01-02 | regression-filled from VIX3M |
| VIX1Y | Cboe | 2007-01-03 | regression-filled from VIX6M |
| PPUT, PPUT3M, CLL, VXTH | Cboe | 1986, 2004-03-19, 2008-08-26, 2006-03-31 | none; Cboe back-test levels before each launch (section 5.8) |
| DTB3, DGS10 | FRED | 1954, 1962-01-02 | |
| TLT, IEF, GLD, AGG, TAIL, DBMF | Yahoo | 2002-07-30, 2002-07-30, 2004-11-18, 2003-09-29, 2017-04-06, 2019-05-08 | DGS10 par-bond proxy for AGG in the 60/40 reference; nothing for the others |

The exact first dates are those printed in the case-study header (`figures/case_study_tables.md`); the year-only ones were checked live when the data layer was built. No synthetic option is priced before 1990, when VIX and SKEW start, and `data.load_market` cuts the panel at 1990, so 1987 enters only through the calendar-year S&P 500 table of the scenario engines (sections 2.7 and 7). Of the 11 crash windows (section 6.5), the 1990 panel holds 10.

### 2.3 The market panel

`data.load_market(start="1990-01-01")` builds one daily panel on `^GSPC` trading days with the columns `spx, spx_open, spxtr, vix, vix9d, vix3m, vix6m, vix1y, skew, rf, q, pput, cll, vxth`. Every series other than the S&P 500 price is reindexed to those days and forward-filled at most 5 rows. `spx_open` is not filled: a missing open stays missing and the sleeves use that row's close instead. The panel raises if `spx`, `spxtr`, `vix`, `skew` or `rf` has no data, and lists every column with missing rows in its warnings, a required one too when it is only partly missing (SKEW lacked 3 of 9,261 rows on the case-study run); the app shows them. On the case-study run the panel held 9,261 trading days from 1990-01-02 to 2026-10-09.

### 2.4 Rates and the dividend yield

```text
rf_t      = DTB3_t / 100                                   (annualized T-bill rate, decimal)
cash_t    = rf_{t-1} * (calendar days from t-1 to t) / 360  (daily T-bill return, actual/360)
r_t       = ln(1 + rf_t)                                   (continuously compounded rate for option pricing)
q_t       = ln(TR_t / TR_{t-252}) - ln(PX_t / PX_{t-252}), clipped to [0, 0.06]
```

q is the trailing-year gap between the S&P 500 total-return and price indices, computed on the full history so the lookback reaches before the panel's start. A missing total-return print carries the last q forward, and q = 0.03 is assumed until the total-return index has 252 rows (only before 1989).

### 2.5 Term-structure back-fill

`pricing.fill_term_structure` fills each VIX pillar before its first date by an ordinary least-squares fit on the rows where both series are positive in the data:

```text
ln target = a + b ln regressor
order: vix3m from vix; vix6m from vix3m; vix9d from vix; vix1y from vix6m
```

A back-filled regressor can fill but never fits (2007 VIX1Y is not fitted on regression-filled VIX6M). A target with fewer than 250 overlapping rows is left unfilled with a warning. Filled rows are flagged `ts_filled`, and the app's Methodology page marks each back-filled series. Before 2006 every pillar except VIX is a regression estimate.

### 2.6 Caching and failure handling

Each series is cached as one parquet file at `{INSUREPORT_DATA_DIR}/{source}/{key}.parquet`, where `INSUREPORT_DATA_DIR` is an environment variable (default `./data`, gitignored). A file older than 20 hours is refetched, always as full history. A failed or empty refresh never overwrites the cache: the stale series is returned with a logged warning, or an empty series when there is no cache. `INSUREPORT_OFFLINE=1` makes both network functions raise; the test suite sets it.

### 2.7 The annual table

`src/insureport/sp500_annual.csv` holds the S&P 500 calendar-year total return from 1928 to 2025 (Damodaran's "S&P 500 (includes dividends)" column) and December-to-December CPI-U inflation (FRED CPIAUCNS). `scripts/make_sp500_annual.py` rebuilds it and writes nothing unless spot checks pass (2008, 2013 and 1931 returns within 3 points; 1932, 1980 and 2022 inflation within 0.5 points). Shiller's monthly data is a fallback source, with a warning. The table and Yahoo's index differ slightly year by year: 2008 is -36.55% in the table and -37.00% from `^SP500TR` (case study t01).

### 2.8 Known data issues

- **The `^GSPC` open.** It is the 9:30 index print, a stale proxy for Cboe's settlement value: it equals the previous close on 60-96% of days before 2005, 31% in 2005-09, 10% in 2010-15 and none from 2016 (`data.load_market`). On 175 of the 312 PPUT roll days from 1990 to 2015 it equals the previous close, and on 0 of the 129 later ones (`figures/validation_table.md`).
- **One-day glitches.** Yahoo's 2019-07-03 `^SP500TR` print gives q = 0.0144 against about 0.020 around it; q is a one-day estimate and shows such glitches.
- **Cboe file breaks.** Cboe's CLL file steps -19.8% on 2009-10-16 with the S&P 500 down 0.8%, and holds likely bad put marks on 2020-02-28 and 2020-03-02 (section 5.2). Forward-filled Cboe gaps would read as unchanged prints; the validation drops them.
- **SKEW methodology.** Cboe announced in 2025 that it would change the SKEW calculation [Cboe 2025]. The tool uses the published series as it is.

### 2.9 Real option chains

`chains.load_chain` reads a purchased end-of-day SPX file (CSV or zipped CSV) in the layout of historicaloptiondata.com (Level 2, Level 3, the 2010+ layout, ALLSPX), Cboe DataShop, ORATS or OptionMetrics IvyDB. One header-alias table maps each layout to a canonical chain with one row per option per quote date: `quote_date, expiration, strike, cp, bid, ask, mid, underlying, root, oi`, with `mid = (bid + ask) / 2` (never the last trade). Headers are lower-cased with spaces and underscores removed; the option type becomes `C` or `P`; IvyDB strikes are divided by 1,000; DataShop's end-of-day quotes are used where present, else the 15:45 ones; historicaloptiondata's `W` marks become the `SPXW` root. Rows of another underlying, and rows with a missing date, strike or quote, `ask <= 0`, `bid < 0` or `bid > ask`, are dropped and counted. Vendors snapshot at different times (historicaloptiondata about 16:00 ET, DataShop 15:45 and the close, ORATS 14 minutes before the close); the time is not recorded. Section 4.6 describes how sleeves use a chain.

## 3. Option pricing

Every option in InsurePort is a European SPX option priced by `pricing.quote` from the market panel of its day. Units: S, K and prices in index points; T in years; volatilities, rates and yields as decimals; the VIX family and SKEW in index points.

### 3.1 Black-Scholes-Merton

```text
d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),   d2 = d1 - sigma sqrt T
call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
delta: call e^(-qT) N(d1), put -e^(-qT) N(-d1)
```

At T <= 0 or sigma <= 0 the price is intrinsic value [Black and Scholes 1973; Merton 1973]. Checked against Hull's textbook values: S = K = 100, T = 1, r = 5%, sigma = 20% gives call 10.4506, put 5.5735, call delta 0.6368 (`test_bs_hull_values`).

### 3.2 Variance-swap term structure

The VIX indices are 30-day (and 9-, 93-, 184- and 365-day) variance-swap volatilities. They are the pillars of a term structure:

```text
pillars T_i = (9, 30, 93, 184, 365) / 365 years: VIX9D, VIX, VIX3M, VIX6M, VIX1Y
total variance w_i = (VIX_i / 100)^2 T_i
vs(T) = sqrt(w(T) / T), w linear in T between the nearest available pillars
outside the pillars: the nearest pillar's volatility (flat)
```

A missing pillar is skipped. `test_var_swap_term_structure` and `test_var_swap_one_year_pillar` check hand values.

### 3.3 From SKEW to the wing slope

Cboe defines SKEW = 100 - 10 S, with S the risk-neutral skewness of 30-day S&P 500 returns [Cboe 2011]. InsurePort turns it into the slope of the put wing:

```text
gamma = max(SKEW - 100, 0) / 10                    (= -S when skew is negative)
c     = m * 0.263 * (gamma / 4.919) ^ beta         (c = 0 when gamma = 0)
defaults: m = SKEW_MULT = 1, beta = SKEW_BETA = 0.5
```

The slope is anchored to an observed smile. On the 2026-10-08 close, the Cboe delayed XSP chain fitted IV / vs = 0.968 + 0.263 z across four expiries (R^2 0.998, linear out to z of about 12), with SKEW at 149.19 (gamma 4.919) [Cboe XSP chain]. At m = 1 the model reproduces that slope exactly on that day (`test_skew_slope_anchored_and_damped`). The live endpoint serves only the current session, so the committed traces of that day are `pricing.SLOPE_REF` (0.263), `pricing.GAMMA_REF` (4.919), and the pillar volatilities and four 30% out-of-the-money put mids in `test_xsp_quotes_anchor_loosely`.

beta sets how the slope moves with SKEW. beta = 1 is the first-order Gram-Charlier rule c proportional to gamma [Backus, Foresi and Wu 2004], which makes the whole smile slope proportional to SKEW - 100. beta = 0 is a constant wing. The default 0.5 is in between: at the 1990-2026 average SKEW of about 122 (gamma about 2.2) it gives c of about 0.176. The proportional rule made the deep wing collapse in moderate-SKEW years: at SKEW 126, with the 2026-10-08 volatilities, the 84-day 30% out-of-the-money put costs 0.035% of spot at beta 0.5, 0.104% at beta 0 and 0.009% under the earlier rule c = 0.3 gamma / 6 (measured on the inputs of `test_xsp_quotes_anchor_loosely` and recorded in its docstring; the test asserts only that the beta 0.5 price lies between 0.02% and 0.08%). One chain cannot estimate beta, so it is a disclosed assumption; the case study sweeps it over {0, 0.5, 1}.

### 3.4 The smile

```text
F  = S e^((r - q) T),   k = ln(K / F),   z = -k / (sigma0 sqrt T)     (z > 0 for out-of-the-money puts)
sigma(K, T) = nu * sigma0 * max(1 + c * min(z+, z_cap) - (c/2) * z-, 0.5)
```

z+ and z- are the positive and negative parts of z. The put wing is linear in z with slope c; the call wing has half that slope (an assumption; the CLL replica's short calls are its only check) and a floor at half the at-the-money level. `z_cap` (default infinity) flattens the put wing beyond z_cap standard deviations; it is a sensitivity knob. `nu` (default 1) scales the whole smile: 1 prices at the variance-swap level, below 1 prices the premium out. At fixed K/F the wing's implied volatility falls with tenor in contango, as on the real chain: at K/S 0.70 the market's IV fell from 0.478 to 0.366 over 29 to 84 days and the model's from 0.478 to 0.367 (recorded in the docstring of `test_xsp_quotes_anchor_loosely`).

### 3.5 Normalization

The at-the-money level sigma0 is set so that the smile's mean variance, for z drawn from N(0, 1), equals the variance-swap variance:

```text
sigma0 = vs / sqrt(A_f(c, z_cap))
A(c)   = E[(1 + c z+ - (c/2) z-)^2] = 1 + c / sqrt(2 pi) + (5/8) c^2      (A(0.5) = 1.355721)
A_f    = A(c) + floor + cap, with truncated-normal moments, a = -1/c, b = z_cap, Q = 1 - Phi(b):
floor  = 0.25 Phi(a) - [Phi(a) - c phi(a) + (c^2/4)(Phi(a) - a phi(a))]
cap    = 2c (b Q - phi(b)) + c^2 ((b^2 - 1) Q - b phi(b))                  (0 when z_cap is infinite)
```

The closed form matches numerical integration to a relative 1e-9 (`test_floored_smile_factor_closed_form`), and the full smile satisfies E[sigma(z)^2] = (nu vs)^2 wherever Lee's bound is slack (`test_smile_is_variance_swap_consistent`).

### 3.6 The slope cap

A linear wing that is too steep makes a lower-strike put worth more than a higher-strike one, an arbitrage. On the put wing:

```text
dP/dK = e^(-rT) [N(-d2) - nu c phi(d2)] >= 0   iff   R(d2) = N(-d2) / phi(d2) >= nu c
```

R is the Mills ratio. `pricing._max_slope` finds, for s = sigma0 sqrt T, the steepest slope c_max(nu, s) that keeps this true at every z on the wing (for example c_max(1, s) = 0.186, 0.517 and 1.255 at s = 0.003, 0.05 and 0.4). Because sigma0 itself depends on c, the cap is the fixed point c* = c_max(nu, sigma0(c*) sqrt T). A second cap, `(1 - 1e-6)^2 / (2 nu^2 vs sqrt T)`, keeps Lee's far root real (section 3.7). The smile uses `min(c, caps)`. The cap tightens as T falls, so a far out-of-the-money put still decays to zero at expiry. It does not depend on SKEW, so the cap never makes a far out-of-the-money put cheaper as SKEW rises (checked for the 25% out-of-the-money 61-day put at m = 1 and 2, `test_far_put_non_decreasing_in_skew`). Near-money puts do get cheaper as SKEW rises, because a steeper slope lowers sigma0 through the normalization (section 3.5). Where it binds, the whole smile flattens and nu moves the slope as well as the level. Put mids are tested non-decreasing in strike over a grid of 6 tenors, 3 volatilities, 4 SKEW levels, 3 slope settings, 3 values of nu and two wing settings (`test_put_mids_non_decreasing_in_strike`).

### 3.7 Lee's bound on the far wing

Lee's moment formula bounds total implied variance in the far wing: sigma^2 T <= 2 |k| as |k| grows [Lee 2004]. The linear wing's total volatility nu (s + c |k|) meets (1 - 1e-6) sqrt(2 |k|) at the roots in x = sqrt|k| of `nu c x^2 - (1 - 1e-6) sqrt(2) x + nu s = 0`; past the far root sigma is clipped to (1 - 1e-6) sqrt(2 |k| / T). For ordinary inputs the bound is far from binding out to z = 12 and only guards the extreme tail (`test_lee_bound_guards_the_extreme_tail`). The call wing needs no clip.

### 3.8 Spreads and fills

```text
half = max(h * mid, 0.05 index points)   for h > 0;   half = 0 for h = 0 (mid fills)
bid = max(mid - half, 0),   ask = mid + half
```

Options open at the ask (long) or the bid (short), close at the bid or the ask before expiry, and are marked at mid in between. h = 0.05 is `pricing.quote`'s default. The catalog and the app use h = 0.04, half of an 8%-of-mid spread, the deep out-of-the-money spread level reported by Andersen, Fusari and Todorov [2015]. The Cboe replicas use h = 0 and Cboe's own fill convention (section 5.1).

### 3.9 Strikes by delta and rates

`pricing.strike_for_delta` solves |delta(K, sigma(K))| = target for the strike by root-finding on k = ln(K/F) over [-8, 1]. A put's |delta| is at most e^(-qT) N(-sqrt(2|k|)) at any volatility, so every target above 3.2e-5 is bracketed. Options are priced with r = ln(1 + rf) and the panel's q.

### 3.10 Limits of the pricer

- **No butterfly (convexity) check.** The deep linear wing implies a negative density once c z is large. At vs 0.15, SKEW 183 and 30 days, the wing turns concave in strike from z of about 33 (K/S below 0.28), and the 0.15 / 0.20 / 0.25 x spot put butterfly costs -0.0026 per 1,000 of spot (`test_deep_linear_wing_breaks_convexity`). The slope of put prices in strike also drops where the smile's slope jumps: at Lee's far root and at a finite z_cap, where the flat wing meets the linear one.
- **A first-order smile.** One slope parameter, linear wings and a floor, in the spirit of a first-order Gram-Charlier expansion. No curvature, no second far-wing slope, no SVI fit. The call-wing slope is assumed to be half the put slope.
- **One anchor day.** The slope comes from one calm day of real quotes (VIX near 15). There are no real far-wing quotes from high-volatility regimes in the record. On that day the model's 30% out-of-the-money put mids were within about 5% of the market's at 29 to 84 days (market / model 0.96, 1.05, 1.04, 1.00, printed by `test_xsp_quotes_anchor_loosely`, which asserts only that each is within 2x). The earlier pricer (slope c = 0.3 gamma / 6, wing flat past three standard deviations) had priced the same puts about 64,000, 314, 11.6 and 5.8 times too cheap (the test's docstring).
- **Far-wing dependence.** The near-money validation cannot see the far wing: moving z_cap from infinity to 3 changes the PPUT replica's out-of-sample dCAGR from -1.683% to -1.687% (`figures/validation_table.md`). Far out-of-the-money results move a lot. One assumption at a time, the *Dao* preset's dCAGR at a 0.50% weight on the S&P 500 core is -85 bp at the defaults, +404, +258 and +36 bp with the wing flat past 2.5, 3 and 4 standard deviations, and -331 and +246 bp at beta 0 and 1; the Universa-style sleeve at 3.33% moves from -172 bp to -19, -86 and -158 bp with the flat wings and to -216 and -66 bp at beta 0 and 1 (case study t08). The wing flat past three standard deviations (z_cap = 3, on today's slope) is still a sensitivity knob and the verdict's wing robustness check (section 8.3).
- **The SKEW sensitivity (beta)** is an assumption (section 3.3).
- **Back-filled pillars.** Before 2006 the three-month to one-year volatilities are regression estimates from VIX (section 2.5).
- **No premium adjustment.** nu = 1 prices at the market's own variance-swap level, which already contains the volatility risk premium; nu below 1 is the counterfactual of cheaper insurance.

## 4. Sleeves

### 4.1 One convention

Every safe haven is a daily simple-return series per unit of sleeve capital, on the market panel's dates. The backtest engine (section 6) combines it with the core as (1 - w) core + w sleeve. Option sleeves differ from the others only in how they size their options.

### 4.2 Option sleeves: state and accounting

`sleeves.option_sleeve` holds cash C, stock A (overlay mode only) and option legs with signed quantities n_j in index units:

```text
NAV = A + C + sum_j n_j mid_j
cash grows by rf_{t-1} * days / 360; stock grows with the underlying (the S&P 500 TR, or the core in the catalog's overlays)
```

At each roll the package is opened with unit cost u = sum_j qty_j fill_j (ask for long legs, bid for short ones):

```text
premium mode:  B = premium_frac * C                       (budget base "cash")
               B = min(premium_frac * base, C)            (budget base "anchor": base = the sleeve's NAV at the last anchor date)
               N = B / u,   n_j = qty_j N,   C = C - B
overlay mode:  A = (A + C) / (1 + coverage * u / S),   C = 0
               N = coverage * A / S,   n_j = qty_j N        (the premium is financed by selling stock)
```

Premium mode is a cash sleeve that spends a budget on options each roll (Universa-style tail hedges, put spreads). Overlay mode holds the underlying plus options on coverage times its value (protective put, collar, the Cboe replicas). New legs are booked at cost, so the spread shows up in the next day's return. If a sleeve's NAV falls to 1e-12 of its value at the previous event, the day's return is -100% exactly and the sleeve restarts from unit capital; the portfolio engine refills it at its next rebalance. Output columns: `ret`, `event`, `premium_paid` (budget over NAV on event days), `coverage` (long option notional over NAV) and `fallback_quotes`.

### 4.3 Events: rolls, monetization, harvests

- **Rolls.** The third Friday of each month, snapped back to the last trading day on or before it (Good Friday 2019-04-19 rolls on 2019-04-18, `test_third_friday_and_roll_dates`), or the first trading day of each month (`roll_when="month_start"`, Universa's rule). `roll_every_months` keeps every N-th roll; `offset_bdays` moves every roll k trading days later (the roll-offset sweep). At a roll the legs due close, expiring legs settle, and the new package opens at the close.
- **Monetization.** With `monetize_mult = M` set, on the first day the long legs are worth at least M times what the package cost (its net debit, or the long legs' cost when the package was a net credit, as a collar is), every leg is sold at the bid / bought back at the ask and the same legs reopen that day at new strikes. It is off in every preset; the app can switch it on.
- **Harvests.** With `harvest_at_anchor`, every anchor date (the portfolio's rebalance date) sells all held legs into cash, so the next year's budget is set on T-bills, not on put marks that decay before they are sold.

### 4.4 Strikes, expiries and settlement

A leg is set either by moneyness, K = S_ref (1 - otm) for puts and S_ref (1 + otm) for calls, or by |delta| (section 3.9). S_ref is the roll day's close or open. A strike rule then applies: `continuous` keeps the strike as computed, `floor` takes the largest 5-point grid strike strictly below it (PPUT's rule), `ceil` the smallest grid strike at or above it (the CLL family, PPUT3M). The expiry is the third Friday of the opening month plus `tenor_months`. A leg is closed `ceil(hold_months / roll_every_months)` rolls after it opens, and never after its expiry. From the last trading day on or before the expiry it is worth intrinsic value against that day's close (PM settlement) or open (`settle="open"`, the proxy for Cboe's AM Special Opening Quotation).

### 4.5 Ladders

A spec can hold several legs with different tenors, holding periods and opening frequencies (`every_months`). The Universa-style preset buys one tranche a month and holds each for two rolls, so about two tranches are live; the collar holds a quarterly put and a monthly short call; the TAIL proxy splits each month's purchase into a 75% and a 25% leg with different exits.

### 4.6 Pricing from a real chain

Given a canonical chain (section 2.9), every opening leg snaps to a listed contract (`chains.select_contract`): the nearest listed SPX expiration on or after the target third Friday, then for puts the largest listed strike at or below the target (for calls the smallest at or above). If nothing is listed, or the listing found would expire before the leg's scheduled close, the leg keeps the model's strike and expiry. Opens, marks and closes take the chain's quotes; a leg-day with no quote is priced by the model and counted in `fallback_quotes`. Settlement stays intrinsic. Only the `SPX` root is used. Where the vendor marks weeklies (historicaloptiondata's `W` flag, an ALLSPX or DataShop root), `SPXW` weeklies are ignored. ORATS and IvyDB files carry no root, so their weeklies, the PM-settled third-Friday `SPXW` included, are read as `SPX` and can be opened or quoted in place of the AM monthly.

### 4.7 Other sleeves

```text
T-bills:     ret_t = rf_{t-1} * days / 360
bond proxy:  hold a par bond with coupon c = y_{t-1} (percent, semiannual) and maturity M = 10 years,
             repriced at today's yield, i = y_t / 200:
             P_t = (c/2) (1 - (1 + i)^(-2M)) / i + 100 (1 + i)^(-2M)
             ret_t = P_t / 100 - 1 + (c / 100) * days / 365
trend:       at each month end, signal = TR_t / TR_{t-252} - CB_t / CB_{t-252}
             (TR the S&P 500 total-return index, CB the compounded T-bill index, 252 rows);
             next month's position = 1 if the signal > 0, else 0 (long / flat);
             ret_t = cash_t + position * (r_spx,t - cash_t)
```

A 5% to 6% yield move reprices the 10-year proxy to 92.5613 (`test_bond_proxy_carry_and_repricing`). The constant-maturity proxy ignores an aging bond's roll-down and convexity. Index and ETF sleeves are the daily change of the Cboe level or the adjusted close; the CLL index is used only from its 2009-10-16 file break. The 60/40 reference holds 60% core and 40% US Aggregate bonds (AGG, with the DGS10 proxy before AGG's first price), rebalanced annually.

### 4.8 The TAIL proxy

Cambria's TAIL ETF has traded only since 2017, so the catalog also rebuilds its structure from its SEC filings. Each month (third-Friday rolls) the sleeve spends 1% of its NAV on SPX puts 8% out of the money expiring 14 months out (TAIL buys the quarterly expiry 13 to 15 months out), 75% of each purchase sold 10 rolls later (about 4 months left) and 25% at 13 rolls (about one month left). The rest of the capital earns 90% of the 10-year par-bond proxy and 10% T-bills, and a 0.60% a year fee accrues daily. The 13- to 15-month puts are priced off the VIX1Y pillar. The proxy is duration-contaminated by design, as TAIL is. It is a structural rebuild from filings and is not validated against TAIL, which is not one of the validation targets of section 5.

### 4.9 The presets

`catalog.CANDIDATES` holds 20 presets, each editable (`dataclasses.replace`, or the app's parameter expanders). Weights tried are fractions of the total portfolio; every evaluation also includes w = 0.

| Key | Archetype | Structure | Weights tried | Portfolio rebalance |
|---|---|---|---|---|
| `universa_model` | insurance | On the first trading day of each month, a tranche of 30% out-of-the-money SPX puts on the third monthly expiry (about 11 weeks), sold two rolls later with 14 to 18 days left (11 when the month opens on a Friday market holiday). Budget: 6% of the sleeve's year-end NAV a month, 0.20% of portfolio NAV at w = 3.33%. Year-end harvest into T-bills. No profit-taking trigger. | 0.25% to 6% in 0.25% steps, and 3.33% | year-end |
| `universa_2023` | insurance | The same sleeve at 98/2 (display preset) | 2% | quarter-end |
| `dao_ch9` | insurance | 2-month SPX puts at \|delta\| 0.005, sold after a month; each roll spends the sleeve's cash and the portfolio resets at each roll (0.5% of NAV a month at w = 0.5%) | 0.25% to 2% | each roll |
| `put_spread` | insurance | Long 10% / short 25% out-of-the-money puts on `universa_model`'s calendar and budget | as `universa_model` | year-end |
| `prot_put` | insurance | Overlay: 5% out-of-the-money one-month puts held to expiry, on coverage = the core's downside beta | 5% to 100% | each roll |
| `collar` | insurance | Overlay: a 3-month 95% put every third month and a one-month 110% call sold monthly, coverage = downside beta | 5% to 100% | each roll |
| `pput_idx`, `pput3m_idx`, `cll_idx`, `vxth_idx` | insurance | Cboe index levels | 5% to 100% | year-end |
| `tail` | insurance | Cambria TAIL ETF | 5% to 100% | year-end |
| `tail_proxy` | insurance | Section 4.8 | 5% to 100% | year-end |
| `tbills` | store of value | 3-month T-bills (AQR's "hold less equity" baseline) | 5% to 100% | year-end |
| `ust10`, `tlt`, `ief` | store of value | 10-year par-bond proxy; 20+ year and 7-10 year Treasury ETFs | 5% to 100% | year-end |
| `gold` | insurance (candidate) | GLD | 5% to 100% | year-end |
| `trend`, `dbmf` | alpha | 12-month S&P 500 trend (section 4.7); a managed-futures ETF | 5% to 100% | year-end |
| `sixty_forty` | baseline | 60% core / 40% AGG | 40% | year-end |

Model-priced presets fill at mid plus or minus a half-spread of max(0.04 x mid, 0.05 index points). The archetype labels follow the book: gold is filed under insurance, T-bills and Treasuries under store of value, trend and managed futures under alpha. The option presets are mechanical havens; bonds, gold and trend are statistical ones, whose crash behaviour can change (bonds fell with stocks in 2022).

### 4.10 Limits of the sleeves

- Strikes snap to a fixed 5-point grid; real listings step 5, 10 or 25 points by level and tenor.
- A leg past settlement sits at its intrinsic value, without interest, until its roll, when a month is missing from the data or a chain-priced leg is rolled with a positive offset.
- Sleeve returns are stored as doubles. Under a flat or nearly flat far wing, a far out-of-the-money put held to expiry by a sleeve that spends almost all its cash can mark below 1e-16 of the previous day, so the return rounds to -100% and an expiry-week payoff is lost (`test_expiry_week_payoff_is_lost_to_return_rounding`). The default linear wing keeps such marks alive, and no premium preset holds its puts to expiry.
- On a monetization day only the option legs pay a spread.
- A day with unusable pricing inputs prices every leg at its own spot with the last usable volatilities, SKEW and rates, and is counted in `fallback_quotes`.

## 5. Validation

The option pricer is validated, not fitted, against three Cboe strategy indices built from real SPX option prices. The tolerances were written down before the first out-of-sample run and have not been changed since.

### 5.1 Targets and replicas

| Target | Strategy (Cboe methodology) | Replica conventions (`replicate.SPECS["cboe"]`) |
|---|---|---|
| PPUT | S&P 500 plus a monthly put 5% out of the money | Strike: the largest 5-point strike strictly below 0.95 x the roll day's open; settled at the open |
| PPUT3M | S&P 500 plus a quarterly put 10% out of the money (March, June, September, December) | Strike: the smallest 5-point strike at or above 0.90 x the open |
| CLL | S&P 500 plus a quarterly 95% put, minus a monthly 110% call | Put at or above 0.95 x the open, call at or above 1.10 x the open (live CLL rounds up despite its "at or below" text); Cboe's cross-roll not modelled |

Each replica is an overlay sleeve with coverage 1 on the S&P 500 total-return index, rolled on Cboe's calendar (the third Friday, or the trading day before it), marked at mid with h = 0. The open stands in for Cboe's pre-11:00 strike reference and its AM settlement value (SOQ). Cboe fills new options at an 11:30 ET VWAP, which closing data cannot see. The fill convention: the closing mid plus 0.30 index points per option per roll on rolls from 1990 to 2015, and the closing mid after 2015 (`replicate.FILL_ADD`). The 0.30 is an InsurePort assumption: PPUT's roll-day returns in Cboe's back-history point to 1990-2015 fills somewhat above the closing mid but below the ask, and closing data cannot pin the convention down. The replica is scaled to equal the real index on their first common date [Cboe PPUT methodology; Cboe CLL methodology].

### 5.2 Metrics

On the dates where the replica, the index and the S&P 500 total return all have a level (`replicate.tracking`):

```text
monthly returns between month-ends of replica (s), index (a) and S&P 500 TR (b)
TE     = std(s - a) * sqrt(12)
dCAGR  = CAGR(replica) - CAGR(index)
corr   = corr(s - b, a - b)                    (the hedge P&L of the two)
dMaxDD = max drawdown(replica) - max drawdown(index)
TE roll / non-roll = std of daily (s - a) on days whose span holds a roll / on the others, * sqrt(252)
```

An index level equal to the one before it is a stale print and is dropped. Cboe's CLL file break on 2009-10-16 is neutralised by rescaling the index from that date so its return that day equals the replica's. The likely bad CLL marks of 2020-02-28 and 2020-03-02 are kept; "TE ex bad months" drops the months that hold them, and pass / fail uses TE with them in.

### 5.3 Pre-registered tolerances and windows

| Target | Out-of-sample window | TE | abs(dCAGR) | Hedge-P&L corr |
|---|---|---|---|---|
| PPUT | 2016-01-01 on | <= 1.5% a year | <= 0.50% a year | >= 0.90 |
| PPUT3M | 2016-01-01 on | <= 2.0% | <= 0.75% | >= 0.85 |
| CLL | 2008-08-26 on (all of Cboe's file) | <= 2.5% | <= 1.00% | >= 0.85 |

A target passes only if all three hold (`replicate.TOLERANCE`, `replicate.WINDOWS`). The training window is 1990 to 2015 (`replicate.TRAIN_END`). A failure triggers a prominent note in the README, and the app's sidebar shows the validation result (PASS or FAIL per target) on every page.

### 5.4 What is fitted, and what is not

Nothing is fitted on the out-of-sample windows. The slope scale m is not fitted at all: it is anchored to the observed chain (section 3.3). Earlier versions fitted the old rule's m (c = m gamma / 6, a different scale from the m below) to PPUT's 1990-2015 level. It moved from 0.3 to 0.2 with the Cboe conventions and to 0.05 under a CAGR-matching rule, so the level cannot identify it from closing data, and the Cboe replicas now validate the pricer and do not calibrate it. For information, `scripts/calibrate.py` prints how PPUT 1990-2015 responds to m (beta 0.5, Cboe conventions and fill):

| m | 0.5 | 0.75 | 1.0 (committed) | 1.25 | 1.5 |
|---|---|---|---|---|---|
| Training TE | 2.08% | 2.03% | 1.97% | 1.92% | 1.87% |
| Training dCAGR | -0.66% | -1.12% | -1.59% | -2.06% | -2.53% |

TE barely moves while the level moves a lot, and PPUT's level would pull m to the low edge of the grid. For a 30-day option the slope cap binds on none of the training days at any of these m (`replicate.calibrate_skew`'s `cap_share`, printed by `scripts/calibrate.py`). The cap tightens as expiry nears, so it binds on some days close to expiry, mostly at the larger m. Masking the 175 stale roll-day opens of 1990-2015 (section 2.8), so those rolls strike and settle at their own close, changes the training TE at m = 1 from 1.97% to 1.74% and the dCAGR from -1.59% to -1.22%; the stale opens are kept and disclosed (`figures/validation_table.md`).

### 5.5 Out-of-sample results

At m = 1, beta = 0.5, nu = 1, h = 0, z_cap infinite, Cboe conventions and fill (`figures/validation_table.md`, which also holds the z_cap = 3 rows summarized in section 3.10):

| Target | Window (monthly returns) | Months | TE | TE ex bad months | dCAGR | Corr | dMaxDD | TE roll / non-roll days | Result |
|---|---|---|---|---|---|---|---|---|---|
| PPUT | 2016-01 to 2026-10 | 130 | 1.191% | 1.191% | -1.683% | 0.981 | +0.95% | 3.99% / 1.29% | FAIL (dCAGR) |
| PPUT3M | 2016-01 to 2026-10 | 130 | 0.916% | 0.916% | -1.329% | 0.981 | -0.48% | 3.00% / 2.33% | FAIL (dCAGR) |
| CLL | 2008-09 to 2026-10 | 218 | 3.593% | 2.769% | -0.706% | 0.947 | +5.29% | 5.79% / 3.90% | FAIL (TE) |

Informational rows, not part of the gate:

| Target | Window | Months | TE | TE ex bad months | dCAGR | Corr | Result |
|---|---|---|---|---|---|---|---|
| PPUT3M 2004-2015 (inside the training period) | 2004-04 to 2015-12 | 141 | 0.859% | 0.859% | -0.333% | 0.988 | PASS |
| CLL after the 2009-10-16 file break | 2009-11 to 2026-10 | 204 | 2.829% | 1.540% | -1.634% | 0.915 | FAIL (TE, dCAGR) |

### 5.6 Roll days against other days

The replicas open their options at the close, while Cboe fills at an 11:30 VWAP and settles at the morning's SOQ, so the timing error lands on roll days. PPUT's tracking error is 3.99% a year on days whose return spans a roll against 1.29% on the others; PPUT3M's 3.00% against 2.33%; CLL's 5.79% against 3.90% (table above).

### 5.7 What the conventions changed

`scripts/calibrate.py` also prints the validation under naive conventions (strikes at exactly 0.95 / 0.90 / 1.10 x the close, settlement at the close, mid fills) and under the earlier slope rule c = 0.3 gamma / 6. With the committed slope and z_cap infinite, the naive conventions give PPUT a TE of 1.78% (6.10% on roll days) and a dCAGR of -0.90%. Cboe's published conventions cut PPUT's TE by about a third, to 1.19% (3.99% on roll days), and widen the level gap to -1.68%. Under the naive conventions PPUT3M passes under both slopes. PPUT fails on tracking error alone under the old slope (dCAGR -0.42%), and on both tracking error and dCAGR (-0.90%) under the committed one. The naive conventions are not Cboe's rules, so they are not the gate. The conventions were taken from Cboe's methodology documents, not chosen from these results.

### 5.8 Back-tested periods

Cboe's index levels before each launch are back-tests, not live index values: PPUT before 2015-08-03, PPUT3M before 2021-08-02, CLL before 2008-09-04, VXTH before 2011-07-29 (`replicate.LAUNCH`; `catalog.LAUNCH` adds VXTH). Nearly all of PPUT's 1990-2015 training window, PPUT3M's out-of-sample window up to August 2021 and the first days of CLL's window are back-test levels. The app hatches these periods in its replication charts, and the evaluation and the Candidates and Stress lab pages add a back-test note to the index candidates.

### 5.9 What passed, what did not, and what it means

- **Tracking passes.** PPUT and PPUT3M track their indices within tolerance (TE 1.19% and 0.92%, correlation 0.981 for both). The model reproduces how protective puts move with the market.
- **Level fails.** The replicas' CAGR trails PPUT by 1.68 and PPUT3M by 1.33 points a year at 100% notional coverage. The model's 5-10% out-of-the-money one- to three-month puts cost more than Cboe's VWAP fills. Fitting m to that level (the grid's low edge) would contradict the observed smile, so the gap is attributed to fill and settlement conventions closing data cannot reproduce. Consequence: protective-put and collar backtests are conservative, because their puts cost more in the model than in the indices.
- **CLL fails on tracking error** (3.59%, 2.77% without the two bad-mark months), while its level gap (-0.71%) and correlation (0.947) are inside tolerance. After the 2009 file break it still fails (TE 2.83%, dCAGR -1.63%).
- **The far wing is not validated by any free index.** PPUT and PPUT3M sit 5% and 10% out of the money; the Universa-style and *Dao* presets buy puts 25-35% out. Their prices rest on the one-day chain anchor and on the beta and wing-shape assumptions, and section 3.10 shows the *Dao* verdict changing sign with them. Every far out-of-the-money result is therefore reported as "under model prices", next to its sensitivity sweep, and a real end-of-day chain (section 2.9) is the way to settle it.

### 5.10 Known-answer checks (informational)

Universa has published figures for its 96.67 / 3.33 model portfolio [Universa 2020]. They are self-reported, hypothetical and window-dependent, so they are never fitted; `catalog.universa_checks` compares them with the `universa_model` sleeve at 3.33% on the S&P 500 TR core (case study t09b):

| Window | Universa reported | InsurePort | S&P 500 |
|---|---|---|---|
| March 2020 | +0.40% | -7.31% | -12.35% |
| Q1 2020 | -6.30% | -11.62% | -19.60% |
| 2019 sleeve drag (portfolio points) | -1.69% | -2.26% | +31.49% |
| March to December 2008 | +9.90% | -19.06% | -30.73% |

The model sleeve trails the reported figures in every window.

## 6. Backtest engine

### 6.1 Portfolio inputs and the core

`portfolio.parse_holdings` reads a CSV with a `ticker` column and exactly one of `weight` or `shares`. It is a trust boundary: at most 50 rows, tickers of 1 to 12 characters from `A-Z 0-9 . - ^ =`, no duplicates, no negative or non-numeric values; weights summing to more than 1.5 are read as percent. Shares are valued at the last adjusted close; a `CASH` row is cash. `portfolio.core_returns` builds the daily constant-mix return:

```text
core_t = sum_i w_i r_i,t + w_CASH * cash_t
```

over the dates where every ticker has a price, so the sample starts at the latest first price. With proxy fill on, a ticker listed after the S&P 500 TR's first date takes the S&P 500 TR's returns before its first price.

### 6.2 Accounting

`backtest.simulate` runs the whole grid of weights w at once. On the dates both series have:

```text
start: c = 1 - w, s = w
each day: c = c (1 + r_core), s = s (1 + r_sleeve), v = c + s
rebalance when the date is a rebalance date, or when |s / v - w| > band * w:
    cost = tc_core |(1 - w) v - c| + tc_sleeve |w v - s|
    v = v - cost,   c = (1 - w) v,   s = w v
```

A return on a date only one series has compounds into its next common date. A sleeve return of -100% empties the sleeve until the next rebalance refills it (`test_wiped_out_sleeve_is_refilled_at_the_next_rebalance`). The engine also reports turnover (sleeve flows over NAV, per year) and cost drag. A Kelly check: a sleeve alternating +100% and -50% on a flat core has its best weight at 0.5, growing by 1.125 per pair of days, sqrt(1.125) a day (`test_kelly_alternating_sleeve`).

### 6.3 Rebalancing and costs

Rebalance calendars are the last trading day of each year, quarter or month (a trailing partial period has none), or an option sleeve's own event dates (its rolls, monetizations and harvests). An optional relative band also rebalances when the sleeve's share leaves w by more than band x w. The default cost is 2 bp of each leg's flow (`Policy`).

### 6.4 Funding modes

- **Carve-out** (default): (1 - w) core + w sleeve. The hedge is paid for by holding less of the core.
- **Relever**: the core is levered to L = 1 / (1 - w), so (1 - w) L = 1 keeps the core at 100% and the sleeve is financed on top, for w up to 1/3 (L up to 1.5):

```text
r_L,t = L r_t - (L - 1) (rf_{t-1} + 0.005) * days / 360
```

Rows run under relever funding are labelled levered, with gross exposure 1 + w*. An overlay sleeve holds the core itself, so under relever funding it adds equity on top of the full core.

### 6.5 The frontier table

`backtest.frontier` reports, for each w: `cagr`, `ann_vol` (daily, x sqrt(252)), `arith_mean` (mean calendar-year return), `vol_tax` (section 1.1), `max_dd`, `calmar` (cagr over |max_dd|), `tuw` (share of days below the running peak), `longest_dd_days` (longest peak-to-recovery span in calendar days), `worst_year`, `turnover`, `cost_drag`, and one column per crash window inside the sample. The 11 crash windows (`metrics.CRASHES`) are Harvey et al.'s eight S&P 500 drawdowns of more than 15% from 1985 to 2018 [Harvey et al. 2019], COVID (2020-02-19 to 2020-03-23), the 2022 rate grind and the 2025 tariff shock. Windows of at most six months count as fast crashes, the others as slow.

### 6.6 Best allocation (empirical Kelly) and the drawdown cap

```text
best_allocation(table, cap) = argmax_w cagr(w) subject to max_dd(w) >= -cap   (ties to the smaller w)
```

With no cap this is the empirical Kelly weight w*, the historical CAGR maximum, which is the one sizing rule every candidate is judged at. With the drawdown cap (default 30%) it gives w_cap, where the AQR twins are compared. The 30% cap is an InsurePort choice, not a rule from the book; it is close to a 60/40 portfolio's 2007-09 loss (-32.28% for the case study's 60/40 core in that window, t01b).

### 6.7 The AQR twins

Both compare the sleeve with simply holding less equity, using the T-bill frontier (the core mixed with T-bills over 41 weights from 0 to 1, same calendar and costs):

- **Drawdown-matched gap** (`dd_matched_cash_gap`): CAGR at w minus the T-bill frontier's CAGR at the same maximum drawdown (linear interpolation). Positive means the sleeve beats deleveraging.
- **CAGR-matched divested twin** (`cagr_matched_cash_dd`, after Israelov [2019]): the T-bill mix with the same CAGR, and the gap between its maximum drawdown and the sleeve portfolio's. When the cash frontier's CAGR is humped, the twin is taken on the deleverage side of the peak.

`beats_deleverage` is true when the comparison weight is above 0 and the drawdown-matched gap is positive. The gaps are blank (n/a) when the sleeve's drawdown or CAGR lies outside the cash frontier's range.

### 6.8 Robustness views

Rolling windows (`rolling_excess`: trailing 5- or 10-year CAGR of w minus that of w = 0), decades (`subperiods`, each chained from the previous decade's last NAV), the roll-offset sweep (rolls 5, 10 and 15 business days later, where the legs are sold before expiry) and one-at-a-time pricing sweeps reduced to tornado bars (`tornado_rows`).

### 6.9 Diagnostics

`metrics.diagnose` reports, over the window the S&P 500 TR covers: CAGR; arithmetic mean, volatility tax and its var / 2 approximation over full calendar years; annual volatility; maximum drawdown, time under water and the longest recovery; beta and downside beta (the OLS slope on days the S&P 500 falls); concentration of the non-cash weights (HHI = sum w_i^2, effective N = 1 / HHI, top-1 and top-5 shares); returns over each crash window next to the S&P 500's; and the index-put hedge:

```text
notional      = portfolio value * max(downside beta, 0)
SPX contracts = notional / (100 * SPX)
XSP contracts = notional / (10 * SPX)
```

USD 1,000,000 at downside beta 1.2 with SPX at 5,000 needs 2.4 SPX or 24 XSP contracts (`test_hedge_notional`).

## 7. Scenario engines

### 7.1 Regimes

Calendar-year S&P 500 total returns fall into four right-closed bins, crash bin first: at or below -15%, -15% to 0, 0 to 15%, above 15% (`scenarios.REGIME_EDGES`). On the packaged 1928-2025 table (98 years, Damodaran's data):

| Regime | Years | Frequency | Mean S&P 500 return |
|---|---|---|---|
| at or below -15% (crash) | 7 | 7.14% | -29.54% |
| -15% to 0 | 19 | 19.39% | -7.58% |
| 0 to 15% | 23 | 23.47% | 7.55% |
| above 15% | 49 | 50.00% | 27.32% |

The crash years are 1930, 1931, 1937, 1974, 2002, 2008 and 2022 (`src/insureport/sp500_annual.csv`; case study t07 gives the count). The sample since 1990 holds three of them (2002, 2008, 2022). 1973 (-14.31%) sits just above the crash edge, and 2020, the COVID crash, was a +18.02% year.

### 7.2 Book mode

Each regime j has a probability p_j, a core return c_j and a haven return h_j. With an annual reset to w:

```text
P_j(w) = (1 - w) c_j + w h_j
g(w)   = exp(sum_j p_j ln(1 + P_j(w))) - 1          (geometric growth per year, the objective)
a(w)   = sum_j p_j P_j(w)                            (arithmetic mean, shown for contrast)
```

The cartoon havens (`scenarios.BOOK_HAVENS`): insurance (1 - p) / p in the crash bin and -100% elsewhere (+900% at p = 0.1; +1,300% at the 1928-2025 frequency of 7/98); alpha +20 / +10 / +5 / +5%; store of value CPI + 2% real (the mean of CPI_y + 2% over each bin's years) or the book's 7% nominal. The book's die (Safe Haven ch. 2) averages +3.333% a roll but compounds at -1.530% (`test_book_die_and_its_kelly_bet`).

### 7.3 Estimated mode

The same engine with each regime's returns taken from a backtest: the mean calendar-year return of the portfolio at w over the full years whose S&P 500 return fell in that regime, and p_j from the count of those years. A regime with no year gets no mass in the growth sums; the case study reads the regime frequencies from the 1928-2025 table instead.

### 7.4 Kelly and cost-effectiveness

```text
G(w)  = sum_j p_j ln(1 + c_j + w d_j),   d_j = h_j - c_j
G'(w) = sum_j p_j d_j / (1 + c_j + w d_j)
cost-effective  iff  G'(0) = sum_j p_j (h_j - c_j) / (1 + c_j) > 0
```

G is concave, so `kelly_w` returns 0 when G'(0) <= 0, else the root of G' (or the upper limit when G' stays positive). w is kept where 1 + c_j + w d_j > 0 in every regime, a zero-probability one included, because a crash that the sample lacks can still come. For two outcomes this is Kelly's w* = p / a - (1 - p) / b [Kelly 1956]: 0.2 for a +100% / -100% bet at p = 0.6 (`test_kelly_two_outcome_closed_form`).

### 7.5 Breakeven crash frequency

With x_j = (h_j - c_j) / (1 + c_j), and A and B the probability-weighted means of x over the crash regimes and over the others:

```text
G'(0; pi) = pi A + (1 - pi) B   =>   pi* = B / (B - A)
```

The haven raises growth when crashes come more often than pi* (if it pays in crashes, A > B). In the two-state example of section 1.3, pi* = 0.07 (`test_part_three_cartoon_insurance`). On the 1928-2025 table the zero-mean insurance cartoon at 3% has a Kelly weight of 2.22% and pi* = 5.03%, against a historical crash frequency of 7.14% (case study t07c). `crash_frequency_sweep` shows g(w) as the crash frequency varies.

### 7.6 The book bootstrap

`scenarios.book_bootstrap` follows the book: paths of 25 calendar years drawn with replacement from history, the same draws for every w.

- **joint**: a year's core and haven returns are drawn together, so the crash link is kept;
- **reshuffle**: the haven's year is drawn separately, so a true insurance payoff's gain must vanish;
- **no_crash**: only years with a core return above -15%, the cost when no crash comes.

Wealth is reset to w every year (or not rebalanced, as an option). Outputs per w: CAGR percentiles, median terminal wealth, P(CAGR_w > CAGR_0) and the median dCAGR. The case study's insurance cartoon at 3% gains +58 bp with the crash link and loses 75 bp when its years are reshuffled (case study t06d).

### 7.7 The stationary block bootstrap

`scenarios.bootstrap` resamples the paired core and sleeve returns per rebalance period with the Politis-Romano stationary bootstrap [Politis and Romano 1994]: each step starts a new block with probability 1 / 6 (mean block length 6 periods), else continues the previous one. One set of 2,000 paths (seed 0) serves every w and every horizon (5, 10 and 20 years). Per horizon and w: CAGR percentiles, P(beat) against w = 0 on the same paths, ensemble growth (from mean terminal wealth) against median growth, maximum-drawdown percentiles and P(drawdown worse than the cap). Drawdowns are measured per period, so drawdowns inside a period are missed.

### 7.8 Book fixtures

Pinned by tests: the die at 3.333% / -1.530% with Kelly 0.379 against cash; the book's 91/9 insurance blend at 3.033% arithmetic / 2.089% geometric with Kelly near 1/11 (`test_book_91_9_insurance_blend`); Universa's Part One results within 2 bp (`test_part_one_cartoon_on_1997_2016`).

## 8. Suggestions, verdicts and the two scorecards

### 8.1 The shortlist

`catalog.suggest` reads the diagnostics: always T-bills (the baseline every haven must beat); downside beta of 0.7 or more adds the index-put presets (`universa_model`, `dao_ch9`, `put_spread`, `pput_idx`, `collar`); downside beta below 0.4 adds trend and the 10-year proxy, which need no index beta, and attaches a basis-risk warning to every index-put candidate on the list; then always the 10-year proxy (with 2022 shown), gold, trend, TAIL and DBMF (short histories) and the TAIL proxy. An effective N below 5 or a top holding above 25% attaches "index puts hedge beta, not single-name risk".

### 8.2 Sizing

Every candidate is judged at its CAGR-maximizing weight w* on its grid (section 6.6). A haven that cuts drawdowns enough to meet the cap is therefore not forced to a larger weight than one that cannot. The book's comparison rule, sizing every haven to the same 5th-percentile 25-year CAGR, is not implemented yet. At w* = 0 the scores that need a hedged portfolio are read at the smallest grid weight (or, in the case study, at the preset's own weight) and say so.

### 8.3 Verdict tiers

`catalog.evaluate` assigns one of four tiers:

1. **insufficient history**: fewer than 10 years, or no crash window inside the sample;
2. otherwise the **base case**: the median 25-year dCAGR of the book bootstrap (joint mode, 2,000 paths) is positive and the historical dCAGR at w* is positive. If not: **not cost-effective**;
3. if it passes, the robustness checks run: the rolling 10-year win rate is at least 0.5; dCAGR stays positive since 2015; the geometric dCAGR stays positive without the best of 2008 and 2020; the estimated-mode dice still favour the haven with the crash edge at -10% and at -20%; and for option sleeves (and the TAIL proxy), dCAGR stays positive on rebuilt sleeves with rolls 5, 10 and 15 business days later (only when legs are sold before expiry) and with mid fills (h = 0), and for model-priced ones also with the far wing flat past three standard deviations (or linear, when the base is flat; section 3.10 shows how far out-of-the-money results move with it). A check that cannot run (for example no full 10-year window) is n/a;
4. **cost-effective (robust)** when every check that ran passes; **cost-effective (fragile)** when any fails.

The tiers are InsurePort's own construction from the book's base case and the critics' objections, not a published standard. A pass is never called proven.

### 8.4 Both scorecards

Each evaluation shows two views side by side.

- **Portfolio view (Spitznagel):** w*, CAGR unhedged and at w*, dCAGR, maximum drawdowns, the block bootstrap's P(beat) at 10 years, the book bootstrap's median dCAGR, the rolling 10-year win rate, the calm-year bleed and the mean crash payoff.
- **Critic view (AQR):** the sleeve on its own (`catalog.critic_view`: its CAGR, Sharpe ratio on excess returns over T-bills, and the alpha and beta of its daily excess return on the S&P 500's), the divested twin's drawdown gap and `beats_deleverage`.

Each `catalog.evaluate` row also carries the CAGR and maximum drawdown of two benchmarks over the same sample, the S&P 500 TR and a 60/40 (S&P 500 TR / AGG, annual), in its `bench_*` columns; the app's ranking tables do not show them. When the views disagree, both are shown and neither wins.

### 8.5 Three denominators

The same option P&L can be quoted on three bases. Only the first is the objective.

1. **Portfolio points**: hedged minus unhedged portfolio return or CAGR. Every headline uses it.
2. **Sleeve capital**: the sleeve's own CAGR (the critic's view) and the calm-year bleed per unit of option notional (`bleed_notional`).
3. **Premium**: net option P&L per dollar of premium (`catalog.return_on_premium`), shown only next to its base. A small premium base turns modest portfolio effects into very large percentages.

### 8.6 Other scores

At w_scores (w*, or the smallest grid weight when w* = 0), in portfolio points: the mean crash-window payoff, split into fast and slow crashes; the calm-year bleed (mean yearly return difference in years the S&P 500 was above -15%); the average premium spent a year and the average option coverage; the attribution of the mean yearly difference to crash-bin and other years (the two add up to it); and the change in daily beta to the S&P 500 (d_beta), which shows when a gain comes from added equity exposure. For the put presets: the portfolio P&L over a one-month -20% S&P 500 fall with every VIX pillar ending at 40, 55 or 70 (`catalog.shock_pnl`, from the median market state of the panel's last year), and for premium presets a warning when the calm-year bleed passes 2.6% of NAV a year. These two are computed at 3.33% for the presets whose grid holds it (the Universa model and the put spread), else at w_scores.

## 9. Case-study summary

[`CASE_STUDY.md`](https://github.com/pgarrett-ai/InsurePort/blob/main/CASE_STUDY.md) runs all 20 candidates on three cores (the S&P 500 TR, a monthly-rebalanced 60/40 of the S&P 500 TR and the 10-year Treasury proxy, and a concentrated five-stock tech sample) from 1990-01-02 to 2026-10-09, with seed 0, a 30% drawdown cap and 2 bp per trade. Under model prices, none of the six model-priced option presets raised any core's CAGR, and neither did the TAIL proxy or the real PPUT and CLL index levels: all have w* = 0 and are not cost-effective on every core. On the 60/40 core, trend, gold, VXTH and the real PPUT3M index reach w* = 100%: they beat the 60/40 by replacing it, and their volatility tax rises rather than falls, so that gain is equity-like return, not insurance. At 3.33% on the S&P 500 core the Universa-style sleeve gives up 196 bp of arithmetic return to save 25 bp of volatility tax, a dCAGR of -172 bp, and it beats the unhedged index in 0.56% of rolling 10-year windows. The only positive weights on the S&P 500 core are gold (35%, +35 bp, fragile), trend (75%, +31 bp, not cost-effective because its bootstrap median is negative) and VXTH (10%, +1 bp, fragile).

The critics' tests point the same way on this pricer: no option preset beats the drawdown-matched T-bill mix on the S&P 500 core, while Treasuries, gold and trend do. Financed on top of a fully invested core, the protective put and the collar turn positive (+40 and +48 bp at 30%) with deeper drawdowns, and the premium sleeves stay at w* = 0. The cartoons reproduce Universa's Part One within 1 bp, but the priced sleeves do not pay like the cartoon: in estimated mode the Universa-style sleeve needs crashes in 35.19% of years to break even. The far out-of-the-money results are the least certain: the *Dao* preset's dCAGR runs from -331 to +498 bp across one-at-a-time pricing assumptions, and the roll calendar alone spreads it over 372 bp. The verdict is "not cost-effective under model prices", and the model's far wing is what decides it.

## 10. Limitations and disclosures

**Option prices**
- All option results are model-priced, not traded quotes, and the pricer fails its pre-registered validation on level (PPUT, PPUT3M) and on tracking error (CLL). Section 5.
- The near-money puts are richer in the model than in Cboe's index fills, so protective-put and collar results are conservative.
- The far wing is anchored to one calm day of real quotes and rests on an assumed SKEW sensitivity (beta) and wing shape; no free index validates it, and far out-of-the-money verdicts change sign across the sweeps (section 3.10).
- The smile has one slope and linear wings; convexity (no butterfly arbitrage) is not enforced in the deep wing; the call-wing slope is assumed.
- The volatility risk premium is taken as priced by the market (nu = 1); the nu sweep shows the counterfactual.
- Before 2006 the longer volatility pillars are regression estimates from VIX; Cboe announced a SKEW methodology change in 2025.

**Execution and conventions**
- Cboe settles at the AM Special Opening Quotation and fills at an intraday VWAP; InsurePort uses the `^GSPC` open and closing marks, and the open is a stale print on most roll days before 2005.
- Strikes snap to a fixed 5-point grid; Cboe's CLL cross-roll is not modelled.
- Real chains are read on the SPX root only, but ORATS and IvyDB chains carry no SPX / SPXW root, so their weeklies, the PM-settled third-Friday `SPXW` included, are read as SPX and can be opened or quoted in place of the AM monthly (section 4.6).
- Costs are 2 bp per trade on the portfolio legs plus the option spread (h); there are no taxes, no margin, no contract granularity and no capacity limits, and leverage is financed at T-bills plus 50 bp.

**Data and samples**
- Few crashes: the options era since 1990 has three crash years, and fast crashes are fewer still. Bootstraps resample the same events and cannot add new ones.
- Calendar-year regimes miss crashes inside up years (2020 was a +18.02% year).
- Index levels before each Cboe launch are back-tests; Cboe's CLL file has a break and likely bad marks.
- TAIL (from 2017) and DBMF (from 2019) are too short for a verdict; the TAIL proxy is inferred from filings and is duration-contaminated; VXTH's VIX-call schedule is not verified.
- Yahoo, Cboe and FRED endpoints can change; fetches fail soft to a cache, and numbers move when the data refresh.

**Method**
- w* is chosen on the history it is scored on; the bootstraps, rolling windows, the post-2015 check and the roll-offset sweep are guardrails, not out-of-sample proof.
- The presets come from descriptions published after the fact, and a portfolio of today's holdings carries survivorship (the case study's tech sample is chosen with hindsight).
- The equal-tail sizing of the book is not implemented; the 30% drawdown cap and the verdict tiers are InsurePort choices.
- Path dependence: roll timing alone moves option results by hundreds of basis points.
- The block bootstrap measures drawdowns per period; the book bootstrap draws years independently.
- The diagnostics use a daily constant-mix core; the bond proxy is constant-maturity; trend is long or flat only.

**Disclosures**
- Universa and Spitznagel figures are self-reported, hypothetical and window-dependent; they are labelled and used as consistency checks only.
- InsurePort is research software, not investment advice. It is not affiliated with Universa Investments, Cboe Global Markets, Cambria Investment Management, S&P Dow Jones Indices, Yahoo or FRED. S&P 500, Cboe, VIX, SKEW and the Cboe index names are trademarks of their owners. Yahoo Finance data through yfinance is intended for personal use. Licensed under GPL-3.0-or-later.

## 11. References

**Spitznagel and Universa**
- Spitznagel, M. (2021). *Safe Haven: Investing for Financial Storms*. Wiley. Chapter 1 excerpt: https://catalogimages.wiley.com/images/db/pdf/9781119401797.excerpt.pdf
- Spitznagel, M. (2013). *The Dao of Capital: Austrian Investing in a Distorted World*. Wiley. https://www.wiley.com/en-us/The+Dao+of+Capital:+Austrian+Investing+in+a+Distorted+World-p-9781394377954
- Spitznagel, M. (2017a). Safe Haven Investing, Part One: Not All Risk Mitigation Is Created Equal. Universa Investments. https://lionscrestcapital.weebly.com/uploads/3/4/1/8/34182934/spitznagel_-_safe_haven_investing_-_p1_-_risk_mitigation.pdf
- Spitznagel, M. (2017b). Safe Haven Investing, Part Three: Those Wonderful Tenbaggers. Universa Investments. https://lionscrestcapital.weebly.com/uploads/3/4/1/8/34182934/spitznagel_-_safe_haven_investing_-_p3_-_those_wonderful_tenbaaggers.pdf
- Spitznagel, M. (2018a). Safe Haven Investing, Part Four: The Volatility Tax. Universa Investments. https://www.scribd.com/document/370967187/Universa-Mark-Spitznagel-Volatility-Tax
- Spitznagel, M. (2018b). Thanks to volatility, you can't always get what you want in investing. *Pensions & Investments* (reprint). https://hedgefundalpha.com/strategies/mark-spitznagel-volatility-investing/
- Spitznagel, M. (2019). Safe Haven Investing: Amor Fati. Universa Investments. https://www.universa.net/UniversaResearch_SafeHaven_AmorFati.pdf
- Mann, C., Spitznagel, M. and Yarckin, B. (2015). Capital Asset Pricing Mistakes. Universa white paper (reprint). https://hedgefundalpha.com/strategies/universa-capital-asset-pricing/
- Lagnado, R. (2024). The 100% Stock / 60% Bond Portfolio: Why Not? Universa Investments. https://www.universa.net/UniversaResearch_60-40withLeverage.pdf
- Universa Investments (2020). Memo of 27 April 2020 (the 96.67 / 3.33 model portfolio, with methodology disclosures). https://www.nakedcapitalism.com/wp-content/uploads/2020/05/00-UniversaLetter-4272020.pdf

**Critics and the option-premium literature**
- Israelov, R. (2019). Pathetic Protection: The Elusive Benefits of Protective Puts. *Journal of Alternative Investments*. https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/Pathetic-Protection-JAI-Wint19.pdf
- Israelov, R. and Nielsen, L. N. (2015). Still Not Cheap: Portfolio Protection in Calm Markets. *Journal of Portfolio Management* 41(4). https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-Still-Not-Cheap.pdf
- Ilmanen, A., Thapar, A., Tummala, H. and Villalon, D. (2020). Tail Risk Hedging: Contrasting Put and Trend Strategies. AQR. https://images.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Tail-Risk-Hedging-Contrasting-Put-and-Trend-Strategies.pdf
- Taleb, N. N. (2013). "Do Financial Markets Reward Buying or Selling Insurance and Lottery Tickets?": A Comment. *Financial Analysts Journal*. https://rpc.cfainstitute.org/research/financial-analysts-journal/2013/do-financial-markets-reward-buying-or-selling-insurance-and-lottery-tickets-a-comment
- Ilmanen, A. (2013). Author Response. *Financial Analysts Journal*. https://www.aqr.com/Insights/Research/Journal-Article/Do-Financial-Markets-Reward-Buying-or-Selling-Insurance-and-Lottery-Tickets-Author-Response
- Harvey, C. R., Hoyle, E., Rattray, S., Sargaison, M., Taylor, D. and Van Hemert, O. (2019). The Best of Strategies for the Worst of Times: Can Portfolios Be Crisis Proofed? *Journal of Portfolio Management*. https://people.duke.edu/~charvey/Research/Published_Papers/P140_The_best_of.pdf
- Carr, P. and Wu, L. (2009). Variance Risk Premiums. *Review of Financial Studies* 22(3), 1311-1341. https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf
- Kozhan, R., Neuberger, A. and Schneider, P. (2013). The Skew Risk Premium in the Equity Index Market. *Review of Financial Studies* 26(9). https://openaccess.city.ac.uk/id/eprint/15209/
- Andersen, T. G., Fusari, N. and Todorov, V. (2015). The Risk Premia Embedded in Index Options. *Journal of Financial Economics*. https://www.kellogg.northwestern.edu/faculty/todorov/htm/papers/opa.pdf
- Hoffstein, C. (2020). Tail Hedging. Newfound Research. https://blog.thinknewfound.com/2020/06/tail-hedging/

**Growth, sizing and statistics**
- Kelly, J. L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal* 35(4), 917-926. https://doi.org/10.1002/j.1538-7305.1956.tb03809.x
- Peters, O. (2011). Optimal leverage from non-ergodicity. *Quantitative Finance* 11(11). https://arxiv.org/abs/0902.2965
- Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428), 1303-1313. https://doi.org/10.1080/01621459.1994.10476870

**Option pricing**
- Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy* 81(3), 637-654. https://doi.org/10.1086/260062
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of Economics and Management Science* 4(1), 141-183. https://doi.org/10.2307/3003143
- Lee, R. W. (2004). The Moment Formula for Implied Volatility at Extreme Strikes. *Mathematical Finance* 14(3), 469-480. https://doi.org/10.1111/j.0960-1627.2004.00200.x
- Backus, D., Foresi, S. and Wu, L. (2004). Accounting for Biases in Black-Scholes. Working paper. https://www.ssrn.com/abstract=585623

**Index methodology**
- Cboe. S&P 500 Put Protection Indices Methodology (PPUT, PPUT3M). https://cdn.cboe.com/api/global/us_indices/governance/PPUT_Methodology.pdf
- Cboe. Collar Indices Methodology (CLL, CLL1M, CLL3M). https://cdn.cboe.com/api/global/us_indices/governance/CLL_Methodology.pdf
- Cboe. Volatility Index Mathematics Methodology (the VIX family). https://cdn.cboe.com/resources/indices/Cboe_Volatility_Index_Mathematics_Methodology.pdf
- Cboe (2011). The CBOE SKEW Index. https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf
- Cboe (2025). Consultation Results Regarding Proposed Changes to the Cboe SKEW Index. https://cdn.cboe.com/resources/release_notes/2025/Consultation-Results-Regarding-Proposed-Changes-to-the-Cboe-SKEW-Index-SKEW-.pdf
- Cboe. Portfolio protection indices (PPUT quarterly returns). https://www.cboe.com/us/indices/index-protection/
- Cambria Investments. Tail Risk ETF (TAIL) summary prospectus. https://www.cambriafunds.com/assets/docs/Cambria_TAIL_Summary.pdf

**Data**
- Cboe daily index history files (one per symbol). https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
- Cboe delayed option quotes, XSP (the 2026-10-08 close used as the smile anchor; live endpoint, current session only; the 2026-10-08 snapshot is not redistributed). https://cdn-api.cboe.com/api/global/delayed_quotes/options/_XSP.json
- Federal Reserve Bank of St. Louis, FRED: DTB3, DGS10, CPIAUCNS. https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3
- Yahoo Finance, S&P 500 Total Return Index (^SP500TR). https://finance.yahoo.com/quote/%5ESP500TR/
- yfinance (Python package). https://github.com/ranaroussi/yfinance
- Damodaran, A. Historical Returns on Stocks, Bonds and Bills. NYU Stern. https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histret.html

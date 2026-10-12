"""Plotly figure builders shared by the Streamlit app and the case-study script.

Pure: each function takes engine outputs (pandas / numpy) and returns a plotly.graph_objects.Figure
with a title, axis titles with units and hover text on every trace; no Streamlit, no I/O.
Theme-neutral: transparent backgrounds, hairline grey grids and the font left to the host
(Streamlit's light or dark theme, or the case-study template), so one figure reads on both.

Colour is defined once here. SERIES holds four categorical slots in fixed order (never cycled: a
fifth series raises), checked with the dataviz skill's validate_palette.js for lightness band,
chroma, colour-blind separation (worst adjacent dE 8.4) and 3:1 contrast on Streamlit's light
(#ffffff) and dark (#0e1117) backgrounds; NEG is the red pole for losses against the blue gain
pole, NEUTRAL the grey of reference lines and shading, STATUS the verdict colours (always shown
with the verdict text). Rates come in as decimals and are formatted as % on the axes; bp inputs are
already x 1e4. One y scale per panel: the frontier stacks its max-drawdown panel under the CAGR
panel instead of overlaying a second scale.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from insureport import metrics, replicate, scenarios

# ponytail: four validated slots, indexed and never cycled (SERIES[4] raises IndexError: the app draws at most
# four series per chart); validate a fifth with the dataviz palette script, or fold / facet, if a chart needs more
SERIES = ("#2a78d6", "#d95926", "#199e70", "#c98500")  # blue, orange, aqua, ochre
NEG = "#e34948"  # losses (the diverging pole opposite SERIES[0])
NEUTRAL = "#898781"  # reference lines, caps, zero lines, shading
GRID = "rgba(137, 135, 129, 0.25)"
BAND = ("rgba(42, 120, 214, 0.15)", "rgba(42, 120, 214, 0.30)")  # SERIES[0] at two opacities (fan chart)
STATUS = {"cost-effective (robust)": "#0ca30c", "cost-effective (fragile)": "#fab219",
          "not cost-effective": "#d03b3b", "insufficient history": NEUTRAL}


def _style(fig: go.Figure, title: str, x: str, y: str, *, xfmt: str | None = None, yfmt: str | None = None) -> go.Figure:
    """Title, axis titles and tick formats, transparent background, hairline grid, legend below."""
    fig.update_layout(title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend={"orientation": "h", "yanchor": "top", "y": -0.2}, margin={"t": 60, "l": 10, "r": 10},
                      hoverlabel={"namelength": -1})
    fig.update_xaxes(title=x, gridcolor=GRID, zerolinecolor=GRID, tickformat=xfmt)
    fig.update_yaxes(title=y, gridcolor=GRID, zerolinecolor=GRID, tickformat=yfmt)
    return fig


def _zero(fig: go.Figure) -> None:
    """A solid hairline at y = 0."""
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)


def growth_chart(navs: dict[str, pd.Series], log: bool = True) -> go.Figure:
    """Growth of $1 per named NAV (each rebased to 1 on its first value), log scale by default."""
    fig = go.Figure()
    for i, (name, nav) in enumerate(navs.items()):
        nav = nav.dropna()
        fig.add_scatter(x=nav.index, y=nav / nav.iloc[0], name=name, mode="lines", line={"color": SERIES[i], "width": 2},
                        hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}x<extra></extra>")
    fig.update_yaxes(type="log" if log else "linear")
    return _style(fig, "Growth of $1", "date", "growth of $1 (x" + (", log scale)" if log else ")"))


def underwater_chart(nav: pd.Series) -> go.Figure:
    """Drawdown from the running peak (metrics.drawdown), filled to zero."""
    dd = metrics.drawdown(nav.dropna())
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, name="drawdown", mode="lines", fill="tozeroy",
                               line={"color": NEG, "width": 1}, hovertemplate="%{x|%Y-%m-%d}: %{y:.1%}<extra></extra>"))
    fig.update_layout(showlegend=False)
    return _style(fig, f"Underwater: drawdown from the running peak (max {dd.min():.1%})", "date", "drawdown (%)",
                  yfmt=".0%")


def annual_bars(annual: pd.Series, arith: float, geo: float) -> go.Figure:
    """Calendar-year returns (bars, losses red) with the arithmetic mean and the geometric mean (CAGR) as
    two horizontal lines: the gap between them is the volatility tax. Pass full calendar years and their
    means (metrics.vol_tax(nav, full_years=True)), so the geometric line is the CAGR over those bars."""
    x = np.asarray(annual.index)
    fig = go.Figure(go.Bar(x=x, y=annual, name="calendar-year return", showlegend=False,
                           marker_color=[SERIES[0] if v >= 0 else NEG for v in annual],
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    span = [x.min() - 0.5, x.max() + 0.5] if len(x) else [0, 1]
    for v, name, colour, dash in ((arith, "arithmetic mean", SERIES[1], "dash"),
                                  (geo, "geometric (CAGR)", SERIES[2], "solid")):
        fig.add_scatter(x=span, y=[v, v], mode="lines", name=name, line={"color": colour, "width": 2, "dash": dash},
                        hovertemplate=f"{name}: {v:.2%}<extra></extra>")
    return _style(fig, f"Calendar-year returns: arithmetic {arith:.1%} vs geometric {geo:.1%} "
                       f"(volatility tax {arith - geo:.1%}/yr)", "year", "calendar-year return (%)", yfmt=".0%")


def crash_table_fig(df: pd.DataFrame) -> go.Figure:
    """metrics.crash_returns as grouped bars: the portfolio's return over each crash window next to the
    S&P 500 TR's (bench_ret), the participation ratio in the hover."""
    with np.errstate(divide="ignore", invalid="ignore"):
        part = (df["ret"] / df["bench_ret"]).to_numpy(float)
    fig = go.Figure()
    fig.add_bar(x=df.index, y=df["ret"], name="portfolio", marker_color=SERIES[0], customdata=part,
                hovertemplate="%{x}<br>portfolio %{y:.1%} (%{customdata:.0%} of the S&P 500's move)<extra></extra>")
    fig.add_bar(x=df.index, y=df["bench_ret"], name="S&P 500 TR", marker_color=SERIES[1],
                hovertemplate="%{x}<br>S&P 500 TR %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="group")
    return _style(fig, "Crash participation: return over each crash window", "crash window",
                  "return over the window (%)", yfmt=".0%")


def frontier_chart(
    table: pd.DataFrame, dd_cap: float, cash_table: pd.DataFrame | None = None, w_star: float | None = None
) -> go.Figure:
    """The safe-haven frontier: CAGR vs sleeve weight w (top panel) and max drawdown vs w (bottom panel,
    with the -dd_cap line), the T-bill frontier (cash_table, AQR's "hold less equity") overlaid in both
    and w* marked on the CAGR curve. Tables are backtest.frontier tables indexed by w (ascending). The
    overlay is clipped to the sleeve's weights and linearly interpolated at the sleeve grid's last weight,
    so both lines end together: a 0-100% T-bill grid would squeeze a 0-6% option sleeve into the left
    edge, and plotly autoranges y on points outside a set x range too."""
    # DEVIATION from the brief's "max-DD on a secondary axis": a stacked panel under the CAGR panel (one y scale
    # per panel, the dataviz rule), not a second scale over it
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    w_max = float(table.index.max())
    lines = [(table, "sleeve", SERIES[0])]
    if cash_table is not None:
        w = np.union1d(cash_table.index[cash_table.index <= w_max], [w_max])
        cash = pd.DataFrame({k: np.interp(w, cash_table.index, cash_table[k]) for k in ("cagr", "max_dd")}, index=w)
        lines.append((cash, "T-bills (hold less equity)", SERIES[1]))
    for t, name, colour in lines:
        fig.add_scatter(x=t.index, y=t["cagr"], name=name, legendgroup=name, mode="lines+markers",
                        line={"color": colour, "width": 2}, marker={"size": 5},
                        hovertemplate=f"{name}<br>w %{{x:.2%}}: CAGR %{{y:.2%}}<extra></extra>", row=1, col=1)
        fig.add_scatter(x=t.index, y=t["max_dd"], name=name, legendgroup=name, showlegend=False, mode="lines+markers",
                        line={"color": colour, "width": 2}, marker={"size": 5},
                        hovertemplate=f"{name}<br>w %{{x:.2%}}: max drawdown %{{y:.1%}}<extra></extra>", row=2, col=1)
    fig.add_scatter(x=[float(table.index.min()), w_max], y=[-dd_cap, -dd_cap], mode="lines",
                    name=f"drawdown cap -{dd_cap:.0%}",
                    line={"color": NEUTRAL, "width": 1, "dash": "dash"},
                    hovertemplate=f"drawdown cap -{dd_cap:.0%}<extra></extra>", row=2, col=1)
    if w_star is not None and np.isfinite(w_star):
        g = float(np.interp(w_star, table.index.to_numpy(float), table["cagr"].to_numpy(float)))
        fig.add_scatter(x=[w_star], y=[g], mode="markers", name=f"w* = {w_star:.2%} (max CAGR)",
                        marker={"symbol": "star", "size": 14, "color": SERIES[0]},
                        hovertemplate=f"w* {w_star:.2%}: CAGR {g:.2%}<extra></extra>", row=1, col=1)
    _style(fig, "Safe-haven frontier: CAGR and max drawdown by sleeve weight", "sleeve weight w (% of portfolio)",
           "CAGR (%)", xfmt=".1%")
    fig.update_xaxes(title=None, row=1, col=1)  # one x title, under the bottom panel
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title="max drawdown (%)", tickformat=".0%", row=2, col=1)
    return fig


def ce_plane(rows: pd.DataFrame | list[dict]) -> go.Figure:
    """The cost-effectiveness plane, one labelled point per candidate (catalog.rank / evaluate rows):
    x = arith_cost_bp (the change in the mean calendar-year return, < 0 = an arithmetic cost), y =
    d_cagr_bp (the change in CAGR); the rejection region y <= 0 shaded."""
    df = pd.DataFrame(rows)
    y = df["d_cagr_bp"].to_numpy(float)
    fig = go.Figure(go.Scatter(
        x=df["arith_cost_bp"], y=y, text=df["candidate"], customdata=df["verdict"], mode="markers+text",
        textposition="top center", marker={"size": 10, "color": SERIES[0]}, name="candidates",
        hovertemplate="%{text}<br>arithmetic cost %{x:+.0f} bp/yr<br>dCAGR %{y:+.0f} bp/yr<br>%{customdata}"
                      "<extra></extra>"))
    finite = y[np.isfinite(y)]
    lo = min([0.0, *finite]) if len(finite) else 0.0
    fig.add_hrect(y0=lo - 0.1 * (np.ptp(finite) if len(finite) else 0) - 10, y1=0, fillcolor=NEG, opacity=0.08,
                  line_width=0, annotation_text="rejection region: dCAGR <= 0", annotation_position="bottom left")
    fig.update_layout(showlegend=False)
    return _style(fig, "Cost-effectiveness plane: does the haven raise CAGR?",
                  "arithmetic cost: change in mean annual return (bp/yr)", "dCAGR, hedged minus unhedged (bp/yr)")


def payoff_scatter(sleeve_annual: pd.Series, spx_annual: pd.Series, label: str) -> go.Figure:
    """The haven's calendar-year return against the S&P 500's (years both have), one point per year, the
    four regime bins of scenarios.REGIME_EDGES shaded (the crash bin red)."""
    both = pd.concat([sleeve_annual, spx_annual], axis=1, join="inner").dropna()
    x, y = both.iloc[:, 1], both.iloc[:, 0]
    fig = go.Figure(go.Scatter(x=x, y=y, text=both.index.astype(str), mode="markers", name=label,
                               marker={"size": 9, "color": SERIES[0]},
                               hovertemplate=f"%{{text}}: S&P 500 %{{x:.1%}}, {label} %{{y:.1%}}<extra></extra>"))
    lo, hi = min([-0.30, *x]) - 0.05, max([0.30, *x]) + 0.05
    edges = scenarios.REGIME_EDGES
    for j, (a, b) in enumerate(zip(edges, edges[1:])):
        fig.add_vrect(x0=max(a, lo), x1=min(b, hi), fillcolor=NEG if j == 0 else NEUTRAL,
                      opacity=0.12 if j == 0 else 0.05 * (1 + j % 2), line_width=0,
                      annotation_text=scenarios.REGIME_LABELS[j], annotation_position="top left")
    _zero(fig)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[lo, hi])
    return _style(fig, f"{label} vs the S&P 500 by calendar year (regime bins shaded)", "S&P 500 total return (%)",
                  f"{label} return (%)", xfmt=".0%", yfmt=".0%")


def fan_chart(boot_df: pd.DataFrame, horizon: int) -> go.Figure:
    """Block-bootstrap CAGR by sleeve weight at one horizon (scenarios.bootstrap rows (horizon, w)): the
    p5-p95 and p25-p75 bands, the median, and P(beat w = 0) in the median's hover and the title."""
    b = boot_df.xs(horizon, level="horizon")
    w = b.index.to_numpy(float)
    fig = go.Figure()
    for (hi, lo), fill, name in ((("cagr_p95", "cagr_p5"), BAND[0], "p5-p95"), (("cagr_p75", "cagr_p25"), BAND[1],
                                                                               "p25-p75")):
        fig.add_scatter(x=w, y=b[hi], mode="lines", line={"width": 0}, showlegend=False, name=hi,
                        hovertemplate=f"w %{{x:.1%}}: {hi[5:]} %{{y:.2%}}<extra></extra>")
        fig.add_scatter(x=w, y=b[lo], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=fill, name=name,
                        hovertemplate=f"w %{{x:.1%}}: {lo[5:]} %{{y:.2%}}<extra></extra>")
    fig.add_scatter(x=w, y=b["cagr_p50"], mode="lines+markers", name="median", customdata=b["p_beat"],
                    line={"color": SERIES[0], "width": 2}, marker={"size": 5},
                    hovertemplate="w %{x:.1%}: median CAGR %{y:.2%}<br>P(beat w = 0) %{customdata:.0%}<extra></extra>")
    best = b["p_beat"].idxmax()
    return _style(fig, f"{horizon}-year CAGR by sleeve weight, block bootstrap (P(beat w = 0) peaks at "
                       f"{b.at[best, 'p_beat']:.0%} at w = {best:.1%})", "sleeve weight w (% of portfolio)",
                  f"{horizon}-year CAGR (%)", xfmt=".1%", yfmt=".1%")


def dice_chart(w: np.ndarray, g: np.ndarray, a: np.ndarray, w_star: float | None) -> go.Figure:
    """Geometric (time-average) vs arithmetic (ensemble-average) growth per year by haven weight
    (scenarios.dice_growth / dice_arith), w* marked on the geometric curve, a zero line."""
    fig = go.Figure()
    fig.add_scatter(x=w, y=g, mode="lines", name="geometric (what one portfolio compounds)",
                    line={"color": SERIES[0], "width": 2}, hovertemplate="w %{x:.1%}: geometric %{y:.2%}<extra></extra>")
    fig.add_scatter(x=w, y=a, mode="lines", name="arithmetic (ensemble average)",
                    line={"color": SERIES[1], "width": 2, "dash": "dash"},
                    hovertemplate="w %{x:.1%}: arithmetic %{y:.2%}<extra></extra>")
    if w_star is not None and np.isfinite(w_star):
        gs = float(np.interp(w_star, w, g))
        fig.add_scatter(x=[w_star], y=[gs], mode="markers", name=f"w* = {w_star:.1%}",
                        marker={"symbol": "star", "size": 14, "color": SERIES[0]},
                        hovertemplate=f"w* {w_star:.1%}: geometric {gs:.2%}<extra></extra>")
    _zero(fig)
    return _style(fig, "Dice: geometric vs arithmetic growth by haven weight", "haven weight w (% of portfolio)",
                  "growth per year (%)", xfmt=".0%", yfmt=".1%")


def crash_freq_chart(sweep_df: pd.DataFrame) -> go.Figure:
    """scenarios.crash_frequency_sweep: geometric growth vs the crash-year frequency, one line per w column."""
    fig = go.Figure()
    for i, w in enumerate(sweep_df.columns):
        fig.add_scatter(x=sweep_df.index, y=sweep_df[w], mode="lines", name=f"w = {w:.1%}",
                        line={"color": SERIES[i], "width": 2},
                        hovertemplate=f"w {w:.1%}<br>crash frequency %{{x:.0%}}: growth %{{y:.2%}}<extra></extra>")
    _zero(fig)
    return _style(fig, "Growth by crash frequency: where does the haven stop paying?",
                  "crash-year frequency (% of years)", "geometric growth per year (%)", xfmt=".0%", yfmt=".1%")


def rolling_chart(rolling_excess_df: pd.DataFrame, w: float, years: int | None = None) -> go.Figure:
    """Column w of backtest.rolling_excess: trailing CAGR at w minus the unhedged portfolio's, by window end;
    `years` names the window in the title."""
    s = rolling_excess_df[w].dropna()
    fig = go.Figure(go.Scatter(x=s.index, y=s, mode="lines", name=f"w = {w:.1%}", line={"color": SERIES[0], "width": 2},
                               hovertemplate="window ending %{x|%Y-%m-%d}: %{y:+.2%}/yr<extra></extra>"))
    _zero(fig)
    fig.update_layout(showlegend=False)
    span = f"{years}-year " if years else ""
    return _style(fig, f"Rolling {span}CAGR gain at w = {w:.1%} over the unhedged portfolio", "window end date",
                  "dCAGR (%/yr)", yfmt=".1%")


def tornado(sens: pd.DataFrame, metric: str = "dCAGR at w* (bp/yr)", xfmt: str | None = None) -> go.Figure:
    """One-at-a-time sensitivity: per parameter (the index), the lowest and highest change in `metric`
    from the base case over its settings (columns low, high), side by side so same-sign changes both
    show, widest swing on top. xfmt: the d3 format of the changes (e.g. ".2%"; 3 significant digits
    when None, for a metric already in bp)."""
    s = sens.assign(span=(sens["high"] - sens["low"]).abs()).sort_values("span")
    fig = go.Figure()
    for col, name, colour in (("low", "lowest", SERIES[0]), ("high", "highest", SERIES[1])):
        fig.add_bar(y=s.index, x=s[col], orientation="h", name=name, marker_color=colour,
                    hovertemplate=f"%{{y}}, {name}: %{{x:+{xfmt or '.3g'}}}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    return _style(fig, f"Sensitivity of {metric} to the pricing assumptions", f"change in {metric} from the base case",
                  "parameter", xfmt=xfmt)


def replication_chart(synth: pd.Series, actual: pd.Series, split_date: str | pd.Timestamp, label: str) -> go.Figure:
    """Synthetic replica vs the real Cboe index (label: "PPUT" / "PPUT3M" / "CLL") on their common dates,
    log scale; the dates before split_date (the start of the target's validation window, replicate.WINDOWS)
    shaded, the levels before the index's launch (replicate.LAUNCH) hatched as back-tested."""
    both = pd.concat([actual, synth], axis=1, keys=["a", "s"]).dropna()
    fig = go.Figure()
    for col, name, colour in (("a", f"Cboe {label}", SERIES[0]), ("s", "synthetic replica", SERIES[1])):
        fig.add_scatter(x=both.index, y=both[col], mode="lines", name=name, line={"color": colour, "width": 2},
                        hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}: %{{y:,.1f}}<extra></extra>")
    if len(both):
        x0, x1, split = both.index[0], both.index[-1], pd.Timestamp(split_date)
        if x0 < split:
            fig.add_vrect(x0=x0, x1=min(split, x1), fillcolor=NEUTRAL, opacity=0.08, line_width=0,
                          annotation_text="before the validation window", annotation_position="top left")
        launch = pd.Timestamp(replicate.LAUNCH[label]) if label in replicate.LAUNCH else None
        if launch is not None and x0 < launch:
            b, lo, hi = min(launch, x1), float(both.min().min()), float(both.max().max())
            fig.add_scatter(x=[x0, b, b, x0, x0], y=[lo, lo, hi, hi, lo], fill="toself", mode="lines",
                            line={"width": 0}, fillcolor="rgba(0,0,0,0)",
                            fillpattern={"shape": "/", "fgcolor": NEUTRAL, "fgopacity": 0.35},
                            name=f"back-tested before {launch:%Y-%m-%d}",
                            hovertemplate=f"Cboe back-test levels before the {launch:%Y-%m-%d} launch<extra></extra>")
    fig.update_yaxes(type="log")
    return _style(fig, f"{label}: synthetic replica vs the Cboe index", "date", "index level (log scale)",
                  yfmt=",.0f")


def regime_table_fig(df: pd.DataFrame) -> go.Figure:
    """scenarios.regime_table as bars: the share of years in each regime (crash bin red), with the year
    count and the mean return of those years in the hover."""
    fig = go.Figure(go.Bar(
        x=df["label"], y=df["freq"], name="share of years", customdata=np.c_[df["count"], df["mean_ret"]],
        marker_color=[NEG, *[SERIES[0]] * (len(df) - 1)],
        hovertemplate="%{x}: %{y:.1%} of years (%{customdata[0]:.0f} years, mean %{customdata[1]:.1%})<extra></extra>"))
    fig.update_layout(showlegend=False)
    return _style(fig, "S&P 500 calendar-year regimes (crash bin first)", "S&P 500 calendar-year total return",
                  "share of years (%)", yfmt=".0%")

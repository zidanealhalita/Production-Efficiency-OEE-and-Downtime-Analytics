"""Shared Plotly chart helpers: one theme, colour-blind-safe palette, consistent line colours."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import LINE_COLORS, LOSS_COLORS, NEUTRAL, OEE_TYPICAL_LOW, OEE_WORLD_CLASS, SHIFT_COLORS

FONT = "Inter, Segoe UI, Arial, sans-serif"


def style(fig: go.Figure, title: str, xlab: Optional[str] = None, ylab: Optional[str] = None, height: int = 420, legend: bool = True) -> go.Figure:
    """Apply the shared layout. Titles should state the insight, axes carry units."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)),
        template="plotly_white", font=dict(family=FONT, size=12), height=height,
        margin=dict(l=10, r=10, t=60, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, title=dict(text="")),
        hovermode="closest",
    )
    if xlab is not None:
        fig.update_xaxes(title_text=xlab)
    if ylab is not None:
        fig.update_yaxes(title_text=ylab)
    return fig


def pct_axis(fig: go.Figure, axis: str = "y", lo: Optional[float] = None, hi: Optional[float] = None) -> go.Figure:
    upd = fig.update_yaxes if axis == "y" else fig.update_xaxes
    upd(tickformat=".0%", range=[lo, hi] if lo is not None else None)
    return fig


def add_oee_bands(fig: go.Figure) -> go.Figure:
    """Rule-of-thumb benchmark bands (world-class >=85%, typical 60-85%, low <60%)."""
    fig.add_hline(y=OEE_WORLD_CLASS, line_dash="dot", line_color="#009E73", annotation_text="85% world-class (rule of thumb)", annotation_position="top left")
    fig.add_hline(y=OEE_TYPICAL_LOW, line_dash="dot", line_color="#D55E00", annotation_text="60% low/typical boundary", annotation_position="bottom left")
    return fig


def line_trend(df: pd.DataFrame, x: str, y: str, color: str, title: str, ylab: str, roll_col: Optional[str] = None, bands: bool = False, height: int = 420) -> go.Figure:
    """Multi-line trend using the fixed colour per line / shift."""
    cmap = {**LINE_COLORS, **SHIFT_COLORS}
    fig = px.line(df, x=x, y=y, color=color, color_discrete_map=cmap, markers=False)
    fig.update_traces(line=dict(width=1.2), opacity=0.45 if roll_col else 1.0)
    if roll_col:
        for key, g in df.groupby(color):
            fig.add_trace(go.Scatter(x=g[x], y=g[roll_col], mode="lines", name=f"{key} (4-wk avg)", line=dict(color=cmap.get(key), width=3), legendgroup=str(key)))
    if bands:
        add_oee_bands(fig)
    style(fig, title, "Week", ylab, height)
    return pct_axis(fig) if ylab and "%" in ylab else fig


def heatmap(pivot: pd.DataFrame, title: str, xlab: str, ylab: str, fmt: str = ".0%", colorscale="RdYlGn", zmid: Optional[float] = None, height: int = 380, cbar: str = "") -> go.Figure:
    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorscale=colorscale, zmid=zmid,
                               text=np.vectorize(lambda v: "" if np.isnan(v) else format(v, fmt))(z), texttemplate="%{text}",
                               colorbar=dict(title=cbar), hoverongaps=False))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title, xlab, ylab, height, legend=False)


def pareto_chart(table: pd.DataFrame, label_col: str, title: str, xlab: str, value_label: str = "Downtime (min)", colors: Optional[dict] = None, height: int = 430, unit: str = "min") -> go.Figure:
    """Bars of minutes + cumulative % line + 80% mark."""
    colors = colors or {}
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table[label_col], y=table["minutes"], name=value_label,
                         marker_color=[colors.get(v, "#0072B2") for v in table[label_col]],
                         hovertemplate="%{x}<br>%{y:,.0f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=table[label_col], y=table["cum_share"], name="Cumulative %", yaxis="y2", mode="lines+markers", line=dict(color="#222222", width=2),
                             hovertemplate="%{x}<br>%{y:.0%} cumulative<extra></extra>"))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=0.8, y1=0.8, line=dict(color=NEUTRAL, dash="dash", width=1.5))
    fig.add_annotation(xref="paper", x=1, yref="y2", y=0.8, text="80% mark", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=NEUTRAL))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], tickformat=".0%", title=dict(text="Cumulative share")))
    return style(fig, title, xlab, value_label, height)


def waterfall(tree: pd.DataFrame, title: str, unit_label: str = "Minutes", as_pct: bool = False, height: int = 460) -> go.Figure:
    """OEE loss waterfall from the loss tree."""
    planned = tree.loc[tree["kind"] == "total", "minutes"].iloc[0]
    vals = tree["minutes"] / planned * 100 if as_pct else tree["minutes"]
    measure = ["absolute" if k == "total" and i == 0 else ("total" if k == "total" else "relative") for i, k in enumerate(tree["kind"])]
    text = [f"{v:,.1f}%" if as_pct else f"{v:,.0f}" for v in vals]
    fig = go.Figure(go.Waterfall(
        x=tree["label"], y=vals, measure=measure, text=text, textposition="outside",
        connector=dict(line=dict(color="#999999")),
        increasing=dict(marker=dict(color="#009E73")), decreasing=dict(marker=dict(color="#D55E00")), totals=dict(marker=dict(color="#0072B2")),
    ))
    fig.update_xaxes(tickangle=-35)
    return style(fig, title, None, "% of planned time" if as_pct else unit_label, height, legend=False)


def bar_group(df: pd.DataFrame, x: str, y: str, color: Optional[str], title: str, xlab: str, ylab: str, barmode: str = "group", cmap: Optional[dict] = None, height: int = 400, fmt: Optional[str] = None) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, color_discrete_map=cmap or {**LINE_COLORS, **SHIFT_COLORS, **LOSS_COLORS})
    if fmt:
        fig.update_traces(texttemplate="%{y:" + fmt + "}", textposition="outside")
    return style(fig, title, xlab, ylab, height, legend=color is not None)

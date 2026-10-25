"""Page 1 - Plant Overview: where do we stand, and which line / period needs attention first?"""
import numpy as np
import pandas as pd
import streamlit as st

from src.config import LINE_COLORS, OEE_TYPICAL_LOW, OEE_WORLD_CLASS
from src.oee import benchmark_band, loss_tree, oee_by, oee_summary, weekly_oee
from src.ui import fmt_pct, get_ctx, kpi
from src.viz import bar_group, heatmap, line_trend

ctx = get_ctx()
S = ctx.shifts
st.title("Plant Overview")
st.caption("Decision question: where do we lose the most OEE, and which line or period needs attention first?")

cur, prev = oee_summary(S), oee_summary(ctx.prev_shifts)
good = S["good_units"].sum()
prev_good = ctx.prev_shifts["good_units"].sum()
down_h = (S["planned_production_time_min"].sum() - S["operating_time_min"].sum()) / 60
prev_down_h = (ctx.prev_shifts["planned_production_time_min"].sum() - ctx.prev_shifts["operating_time_min"].sum()) / 60
rej = S["reject_units"].sum() / S["total_units"].sum()
prev_rej = (ctx.prev_shifts["reject_units"].sum() / ctx.prev_shifts["total_units"].sum()) if len(ctx.prev_shifts) else np.nan
has_prev = len(ctx.prev_shifts) > 0

c = st.columns(7)
kpi(c[0], "OEE (weighted)", fmt_pct(cur["oee"]), cur["oee"] - prev["oee"] if has_prev else None, help="Availability x Performance x Quality, computed from summed minutes and units.")
kpi(c[1], "Availability", fmt_pct(cur["availability"]), cur["availability"] - prev["availability"] if has_prev else None)
kpi(c[2], "Performance", fmt_pct(cur["performance"]), cur["performance"] - prev["performance"] if has_prev else None)
kpi(c[3], "Quality", fmt_pct(cur["quality"]), cur["quality"] - prev["quality"] if has_prev else None)
c[4].metric("Good units", f"{good / 1e6:,.1f} M", f"{(good - prev_good) / 1e6:+,.1f} M" if has_prev else None)
c[5].metric("Downtime (h)", f"{down_h:,.0f}", f"{down_h - prev_down_h:+,.0f} h" if has_prev else None, delta_color="inverse", help="Availability loss: planned production time minus operating time.")
kpi(c[6], "Reject %", fmt_pct(rej, 2), (rej - prev_rej) if has_prev else None, invert=True)
st.caption(
    f"Selection: {ctx.start:%d %b %Y} - {ctx.end:%d %b %Y}. Deltas compare with the previous period of equal length"
    f"{'' if has_prev else ' (no earlier data available for this selection)'}. "
    f"Benchmark band for OEE: **{benchmark_band(cur['oee'])}** - rule of thumb: >=85% world-class, 60-85% typical, <60% low "
    "(a common industry convention, not a Danone target; the right target differs by line type, e.g. UHT vs water)."
)

# --- So what? ------------------------------------------------------------------------------------
by_line = oee_by(S, "line_id").sort_values("oee")
worst = by_line.iloc[0]
monthly = oee_by(S, ["line_id", "month"]).sort_values(["line_id", "month"])
monthly["chg"] = monthly.groupby("line_id")["oee"].diff()
tree = loss_tree(S)
losses = tree[tree["kind"] != "total"].assign(pct=lambda d: d["minutes"].abs() / tree["minutes"].iloc[0]).sort_values("pct", ascending=False)
top_loss, second_loss = losses.iloc[0], losses.iloc[1]
bullets = [f"**Worst line:** {worst['line_id']} has the lowest OEE at **{fmt_pct(worst['oee'])}** (A {fmt_pct(worst['availability'])}, P {fmt_pct(worst['performance'])}, Q {fmt_pct(worst['quality'])}); the best line is {by_line.iloc[-1]['line_id']} at {fmt_pct(by_line.iloc[-1]['oee'])}."]
if monthly["chg"].notna().any():
    r = monthly.loc[monthly["chg"].abs().idxmax()]
    prev_m = monthly[(monthly["line_id"] == r["line_id"]) & (monthly["month"] < r["month"])].iloc[-1]
    bullets.append(f"**Biggest trend change:** {r['line_id']} moved {r['chg'] * 100:+.1f} pts between {prev_m['month']:%b} ({fmt_pct(prev_m['oee'])}) and {r['month']:%b} ({fmt_pct(r['oee'])}). Check page 2 and 5 for the cause.")
bullets.append(f"**Biggest loss bucket:** {top_loss['label']} takes **{top_loss['pct'] * 100:.1f}%** of planned time, followed by {second_loss['label']} ({second_loss['pct'] * 100:.1f}%). The largest *downtime* category is shown on page 2.")
st.info("**So what?**\n\n" + "\n\n".join(f"- {b}" for b in bullets))

# --- Trend ---------------------------------------------------------------------------------------
st.subheader("OEE trend")
w = weekly_oee(S, "line_id", 4)
w = w[w["planned"] > 0]
st.plotly_chart(line_trend(w, "week", "oee", "line_id", "Weekly OEE by line - thin = weekly, thick = 4-week rolling average", "OEE (%)", roll_col="oee_roll", bands=True, height=470), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    pv = monthly.pivot(index="line_id", columns="month", values="oee")
    pv.columns = [m.strftime("%b") for m in pv.columns]
    st.plotly_chart(heatmap(pv, "OEE by line and month - dark red = weak months", "Month (2025)", "Line", zmid=0.70, cbar="OEE"), use_container_width=True)
with c2:
    apq = by_line.melt(id_vars="line_id", value_vars=["availability", "performance", "quality"], var_name="factor", value_name="value")
    apq["factor"] = apq["factor"].str.capitalize()
    fig = bar_group(apq, "line_id", "value", "factor", "Performance is the weakest factor on most lines; quality is consistently high", "Line", "Factor value (%)",
                    cmap={"Availability": "#0072B2", "Performance": "#E69F00", "Quality": "#009E73"}, fmt=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Line summary table"):
    t = by_line.sort_values("line_id")[["line_id", "availability", "performance", "quality", "oee", "downtime_h", "good_units", "shifts"]].copy()
    t["band"] = t["oee"].map(benchmark_band)
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "availability": st.column_config.NumberColumn("Availability", format="%.3f"), "performance": st.column_config.NumberColumn("Performance", format="%.3f"),
        "quality": st.column_config.NumberColumn("Quality", format="%.3f"), "oee": st.column_config.NumberColumn("OEE", format="%.3f"),
        "downtime_h": st.column_config.NumberColumn("Downtime (h)", format="%.0f"), "good_units": st.column_config.NumberColumn("Good units", format="%d")})

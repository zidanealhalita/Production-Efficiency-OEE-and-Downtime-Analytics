"""Page 2 - OEE Loss Analysis: where do we lose the most, and on which equipment?"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import LINE_COLORS, LOSS_CATEGORIES, LOSS_COLORS, SHIFT_COLORS
from src.oee import check_loss_tree, loss_tree, oee_by
from src.pareto import drilldown, mtbf_mttr, pareto_table, vital_few
from src.ui import fmt_pct, get_ctx
from src.viz import bar_group, pareto_chart, style, waterfall

ctx = get_ctx()
S, E = ctx.shifts, ctx.events
st.title("OEE Loss Analysis")
st.caption("Decision question: where do we lose the most time, on which equipment, and why?")

# --- Waterfall -----------------------------------------------------------------------------------
tree = loss_tree(S)
view = st.radio("Waterfall unit", ["Minutes", "% of planned time"], horizontal=True)
top = tree[tree["kind"] != "total"].assign(a=lambda d: d["minutes"].abs()).sort_values("a", ascending=False).iloc[0]
planned = tree["minutes"].iloc[0]
title = f"{top['label']} is the largest loss: {top['a'] / planned:.1%} of planned time; only {tree['minutes'].iloc[-1] / planned:.1%} is fully productive"
st.plotly_chart(waterfall(tree, title, as_pct=(view != "Minutes")), use_container_width=True)
diff = check_loss_tree(tree)
st.caption(f"Reconciliation check: planned time - sum of losses - fully productive time = {diff:,.4f} min (should be ~0). "
           "Speed loss is split into minor stoppages (logged events) and reduced speed (the residual: running slower than ideal rate).")

# --- Pareto drill-down ---------------------------------------------------------------------------
st.subheader("Pareto drill-down: category > equipment > reason")
c1, c2 = st.columns(2)
cats = ["(all categories)"] + sorted(E["loss_category"].unique())
cat = c1.selectbox("1) Loss category", cats)
cat_v = None if cat == cats[0] else cat
eq_v = None
if cat_v:
    eqs = ["(all equipment)"] + sorted(E[E["loss_category"] == cat_v]["equipment"].unique())
    eq = c2.selectbox("2) Equipment", eqs)
    eq_v = None if eq == eqs[0] else eq
pt = drilldown(E, cat_v, eq_v)
level = "loss_category" if cat_v is None else ("equipment" if eq_v is None else "reason")
lvl_name = {"loss_category": "Loss category", "equipment": "Equipment", "reason": "Reason"}[level]
n80 = vital_few(pt)
scope = "All downtime" if cat_v is None else (f"{cat_v}" if eq_v is None else f"{cat_v} on {eq_v}")
top_item = pt.iloc[0] if len(pt) else None
if top_item is not None:
    ttl = f"{scope}: {top_item[level]} leads with {top_item['share']:.0%}; {n80} of {len(pt)} items reach 80%"
    st.plotly_chart(pareto_chart(pt.head(15), level, ttl, lvl_name, colors=LOSS_COLORS if level == "loss_category" else None), use_container_width=True)
    st.caption(f"Total in scope: {pt['minutes'].sum() / 60:,.0f} downtime hours across {int(pt['events'].sum()):,} events.")
with st.expander("Treemap view (category > equipment > reason, sized by minutes)"):
    if len(E):
        fig = px.treemap(E, path=["loss_category", "equipment", "reason"], values="duration_min", color="loss_category", color_discrete_map=LOSS_COLORS, maxdepth=3)
        st.plotly_chart(style(fig, "Treemap of downtime minutes: area = minutes", height=520, legend=False), use_container_width=True)

# --- MTBF / MTTR ---------------------------------------------------------------------------------
st.subheader("MTBF / MTTR by equipment")
mm = mtbf_mttr(E, S)
if len(mm):
    worst = mm.iloc[0]
    fig = px.bar(mm.head(12), x="downtime_h", y=mm.head(12)["line_id"] + " - " + mm.head(12)["equipment"], color="line_id", orientation="h", color_discrete_map=LINE_COLORS)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, f"{worst['line_id']} {worst['equipment']} loses the most hours to breakdowns ({worst['downtime_h']:,.0f} h, MTBF {worst['mtbf_h']:.1f} h)", "Breakdown downtime (hours)", "Line - equipment", height=430), use_container_width=True)
    st.dataframe(mm, hide_index=True, use_container_width=True, column_config={
        "downtime_h": st.column_config.NumberColumn("Downtime (h)", format="%.0f"), "mttr_min": st.column_config.NumberColumn("MTTR (min)", format="%.1f"),
        "mtbf_h": st.column_config.NumberColumn("MTBF (h)", format="%.1f")})
    st.caption("MTBF = operating hours of the line / number of breakdowns on that equipment. MTTR = average breakdown duration. "
               "Equipment with fewer than 3 breakdowns are hidden. Ranked by breakdown downtime hours; low MTBF = fails often, high MTTR = slow to repair.")
else:
    st.info("No breakdown events in the current selection.")

# --- Comparison toggles --------------------------------------------------------------------------
st.subheader("Compare: line vs line, shift vs shift, month vs month")
dim = st.radio("Compare by", ["Line", "Shift", "Month"], horizontal=True)
key = {"Line": "line_id", "Shift": "shift_label", "Month": "month_label"}[dim]
cols = [f"min_{s}" for s in LOSS_CATEGORIES]
g = S.groupby(key)[cols + ["planned_production_time_min"]].sum().reset_index()
long = g.melt(id_vars=[key, "planned_production_time_min"], value_vars=cols, var_name="slug", value_name="minutes")
long["Loss category"] = long["slug"].str.replace("min_", "", regex=False).map(LOSS_CATEGORIES)
long["pct"] = long["minutes"] / long["planned_production_time_min"]
fig = px.bar(long, x=key, y="pct", color="Loss category", color_discrete_map=LOSS_COLORS, barmode="stack")
fig.update_yaxes(tickformat=".0%")
tot = long.groupby(key)["pct"].sum()
st.plotly_chart(style(fig, f"Logged stops as % of planned time by {dim.lower()} - highest: {tot.idxmax()} ({tot.max():.1%})", dim, "Stops, % of planned time", height=430), use_container_width=True)
cmp_t = oee_by(S, key).rename(columns={key: dim})
st.dataframe(cmp_t[[dim, "availability", "performance", "quality", "oee", "downtime_h"]], hide_index=True, use_container_width=True, column_config={
    "availability": st.column_config.NumberColumn("Availability", format="%.3f"), "performance": st.column_config.NumberColumn("Performance", format="%.3f"),
    "quality": st.column_config.NumberColumn("Quality", format="%.3f"), "oee": st.column_config.NumberColumn("OEE", format="%.3f"),
    "downtime_h": st.column_config.NumberColumn("Downtime (h)", format="%.0f")})

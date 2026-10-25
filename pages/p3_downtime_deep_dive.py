"""Page 3 - Downtime deep dive: which stops hurt most, when do they happen, and what is controllable?"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import LINE_COLORS, LOSS_COLORS, SHIFT_COLORS
from src.impact import changeover_cost_monthly, changeover_events, changeover_stats, smed_whatif
from src.oee import oee_by, oee_summary
from src.pareto import count_vs_minutes
from src.stats import bootstrap_ci, poisson_rate_ratio, shift_effect_regression, two_proportion_z
from src.ui import fmt_idr, fmt_pct, get_ctx
from src.viz import bar_group, heatmap, style

ctx = get_ctx()
S, E, A = ctx.shifts, ctx.events, ctx.assump
st.title("Downtime Deep Dive & Root-Cause Hints")
st.caption("Decision question: which stops matter most, when do they happen, and which causes can we actually control?")
if len(E) == 0:
    st.info("No downtime events in the current selection.")
    st.stop()

# --- Distribution --------------------------------------------------------------------------------
st.subheader("1. How long are stops?")
edges = np.logspace(0, np.log10(max(E["duration_min"].max(), 10) * 1.05), 36)
centers = np.sqrt(edges[:-1] * edges[1:])
fig = go.Figure()
for cat, g in E.groupby("loss_category"):
    cnt, _ = np.histogram(g["duration_min"], bins=edges)
    fig.add_trace(go.Bar(x=np.log10(centers), y=cnt, width=np.diff(np.log10(edges)) * 0.95, name=cat, marker_color=LOSS_COLORS.get(cat)))
fig.update_layout(barmode="stack")
ticks = [1, 3, 10, 30, 100, 300]
fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[str(t) for t in ticks])
med, p90 = E["duration_min"].median(), E["duration_min"].quantile(0.9)
st.plotly_chart(style(fig, f"Most stops are short (median {med:.0f} min) but the tail is long (P90 {p90:.0f} min, max {E['duration_min'].max():.0f} min)", "Stop duration (minutes, log scale)", "Number of events", height=400), use_container_width=True)

st.subheader("2. Frequency vs severity: top reasons by count AND by minutes")
E2 = E.assign(cat_reason=E["loss_category"] + " / " + E["reason"])
cv = count_vs_minutes(E2, "cat_reason", top_n=1000)
by_count = cv.sort_values("events", ascending=False).head(10)
by_min = cv.sort_values("minutes", ascending=False).head(10)
c1, c2 = st.columns(2)
for col, d, xcol, ttl, xl in [(c1, by_count, "events", "Top 10 by number of stops", "Events"), (c2, by_min, "minutes", "Top 10 by minutes lost", "Minutes lost")]:
    f = px.bar(d, x=xcol, y="cat_reason", orientation="h", color_discrete_sequence=["#0072B2"] if xcol == "events" else ["#D55E00"])
    f.update_yaxes(autorange="reversed")
    col.plotly_chart(style(f, ttl, xl, "", height=420, legend=False), use_container_width=True)
overlap = len(set(by_count["cat_reason"]) & set(by_min["cat_reason"]))
st.info(f"**Why both views?** Only {overlap} of the top-10 reasons appear in both lists. Frequent stops (e.g. minor jams of 2-3 minutes) "
        "annoy operators but cost little time per event, so they call for *standard work / small fixes*. Rare but long stops (a 2-hour motor failure) "
        "dominate lost minutes and call for *root-cause repair and preventive maintenance*. Fixing by count alone would chase the wrong problems.")

# --- Heatmap -------------------------------------------------------------------------------------
st.subheader("3. When do stops happen?")
c1, c2 = st.columns(2)
metric = c1.radio("Measure", ["Minutes lost", "Number of stops"], horizontal=True)
cat_sel = c2.selectbox("Loss category", ["(all)"] + sorted(E["loss_category"].unique()))
Eh = E if cat_sel == "(all)" else E[E["loss_category"] == cat_sel]
Eh = Eh.assign(day=Eh["dow"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))))
val = ("duration_min", "sum") if metric == "Minutes lost" else ("event_id", "count")
pv = Eh.pivot_table(index="day", columns="hour", values=val[0], aggfunc=val[1], fill_value=0).reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
pv = pv.reindex(columns=range(24), fill_value=0)
peak = pv.stack().idxmax()
st.plotly_chart(heatmap(pv, f"{metric} peak on {peak[0]} around {int(peak[1]):02d}:00 (start time of the stop)", "Hour of day", "Day of week", fmt=".0f", colorscale="YlOrRd", height=360, cbar=metric), use_container_width=True)
st.caption("Hour = start of the stop. Night-shift hours include Waiting-operator stops; Sundays have fewer scheduled shifts, so absolute counts are lower.")

# --- Changeover ----------------------------------------------------------------------------------
st.subheader("4. Changeover analysis & SMED what-if")
CO = changeover_events(E)
if len(CO) == 0:
    st.info("No changeovers in this selection (single-SKU lines have none).")
else:
    cs = changeover_stats(E, ("line_id",))
    c1, c2 = st.columns([1, 1])
    f = px.bar(cs, x="line_id", y="mean", color="line_id", color_discrete_map=LINE_COLORS, text=cs["mean"].round(0))
    c1.plotly_chart(style(f, f"{cs.sort_values('mean').iloc[-1]['line_id']} changeovers are the longest ({cs['mean'].max():.0f} min on average)", "Line", "Mean changeover (min)", height=360, legend=False), use_container_width=True)
    c2.dataframe(cs.rename(columns={"n": "changeovers", "mean": "mean_min", "median": "median_min", "p90": "p90_min", "total_min": "total_min"}), hide_index=True, use_container_width=True,
                 column_config={"mean_min": st.column_config.NumberColumn(format="%.1f"), "median_min": st.column_config.NumberColumn(format="%.1f"), "p90_min": st.column_config.NumberColumn(format="%.1f"), "total_min": st.column_config.NumberColumn(format="%d")})
    tr = changeover_stats(E.assign(transition=E["prev_sku"].astype(str) + " > " + E["sku_id"].astype(str)), ("line_id", "transition"))
    tr = tr[tr["n"] >= 5].sort_values("mean", ascending=False)
    with st.expander("SKU transition detail (min 5 changeovers)"):
        st.dataframe(tr, hide_index=True, use_container_width=True, column_config={"mean": st.column_config.NumberColumn("mean_min", format="%.1f"), "median": st.column_config.NumberColumn("median_min", format="%.1f"), "p90": st.column_config.NumberColumn("p90_min", format="%.1f")})
    cc = changeover_cost_monthly(S, E, A)
    cc["month_s"] = cc["month"].dt.strftime("%b")
    f = px.bar(cc, x="month_s", y="idr", color="line_id", color_discrete_map=LINE_COLORS, barmode="stack", category_orders={"month_s": [m.strftime("%b") for m in sorted(cc["month"].unique())]})
    st.plotly_chart(style(f, f"Changeovers cost about {fmt_idr(cc['idr'].sum())} of contribution in the selected period", "Month", "Lost contribution (IDR; assumption-based)", height=360), use_container_width=True)

    st.markdown("**SMED what-if**: reduce changeover time by X% and convert the recovered minutes into extra good units and IDR.")
    red = st.slider("Changeover time reduction", 0, 80, int(A["smed_reduction"] * 100), 5, format="%d%%") / 100
    line_opts = ["(all lines)"] + sorted(CO["line_id"].unique())
    ln = st.selectbox("Line", line_opts, index=line_opts.index("L05") if "L05" in line_opts else 0)
    sw = smed_whatif(S, E, red, A, ctx.start, ctx.end, None if ln == line_opts[0] else ln)
    m = st.columns(4)
    m[0].metric("Recovered time", f"{sw['hours'].sum():,.0f} h", help="In the selected period")
    m[1].metric("Extra good units", f"{sw['units'].sum() / 1e6:,.2f} M")
    m[2].metric("Contribution", fmt_idr(sw["idr"].sum()))
    m[3].metric("Annualised", fmt_idr(sw["idr_annual"].sum()), help="Selected period scaled to 365 days. Uses the margin-per-unit assumptions in the sidebar.")
    st.caption("Assumes recovered minutes run at the line's realised average output and are sold at the assumed contribution margin.")

# --- Start-up after idle -------------------------------------------------------------------------
STATE_COLORS = {"Line was running": "#0072B2", "After an idle shift": "#D55E00"}
st.subheader("5. Start-up after idle: does restarting a line cost us?")
S2 = S.copy()
g = S2.groupby("prev_idle").agg(shifts=("shift_id", "count"), op_h=("operating_time_min", lambda x: x.sum() / 60), bd=("n_breakdowns", "sum"), bd_min=("min_breakdown", "sum"),
                                rej=("reject_units", "sum"), tot=("total_units", "sum")).reset_index()
if g["prev_idle"].nunique() == 2:
    g["bd_per_100h"] = g["bd"] / g["op_h"] * 100
    g["reject_rate"] = g["rej"] / g["tot"]
    g["state"] = g["prev_idle"].map({True: "After an idle shift", False: "Line was running"})
    i, r = g[g["prev_idle"]].iloc[0], g[~g["prev_idle"]].iloc[0]
    rr = poisson_rate_ratio(i["bd"], i["op_h"], r["bd"], r["op_h"])
    z = two_proportion_z(i["rej"], i["tot"], r["rej"], r["tot"])
    c1, c2 = st.columns(2)
    f1 = px.bar(g, x="state", y="bd_per_100h", color="state", color_discrete_map=STATE_COLORS, text=g["bd_per_100h"].round(1))
    c1.plotly_chart(style(f1, f"Breakdowns per 100 operating hours: {rr['ratio']:.2f}x after idle", "", "Breakdowns / 100 operating h", height=340, legend=False), use_container_width=True)
    f2 = px.bar(g, x="state", y="reject_rate", color="state", color_discrete_map=STATE_COLORS, text=g["reject_rate"].map(lambda v: f"{v:.2%}"))
    f2.update_yaxes(tickformat=".1%")
    c2.plotly_chart(style(f2, f"Reject rate: {z['p1']:.2%} after idle vs {z['p2']:.2%} when running", "", "Reject rate (%)", height=340, legend=False), use_container_width=True)
    st.markdown(
        f"- **{int(i['shifts'])}** shifts followed an idle shift (e.g. after Sundays or the Lebaran shutdown). "
        f"Breakdown rate is **{rr['ratio']:.2f}x** higher (95% CI {rr['ci_low']:.2f}-{rr['ci_high']:.2f}); reject rate is **{(z['p1'] - z['p2']) * 100:+.2f} pts** (z-test p = {z['p_value']:.3g}).\n"
        "- Plain words: if the CI for the ratio excludes 1.0, the extra breakdowns are unlikely to be chance. Because shifts hold tens of thousands of units, even small reject-rate gaps test as 'significant' - judge the size of the gap, not only the p-value.\n"
        "- Limitation: not adjusted for line (idle starts happen more on some lines) - see the line-by-line view on page 4.")
else:
    st.info("The current selection has no start-ups after idle (or only start-ups). Widen the filters.")

# --- Shift effect --------------------------------------------------------------------------------
st.subheader("6. Shift effect: is the night shift really different?")
sh = oee_by(S, "shift_label")
extra = S.groupby("shift_label").agg(crew_exp=("avg_crew_experience_years", "mean"), crew=("crew_size", "mean"), wait_min=("min_waiting_operator", "sum"), planned=("planned_production_time_min", "sum")).reset_index()
extra["waiting_operator_min_per_100h"] = extra["wait_min"] / (extra["planned"] / 60) * 100
sh = sh.merge(extra, on="shift_label")
sh["reject_rate"] = sh["reject_units"] / sh["total_units"]
c1, c2 = st.columns([1.2, 1])
long = sh.melt(id_vars="shift_label", value_vars=["availability", "performance", "quality", "oee"], var_name="metric", value_name="value")
long["metric"] = long["metric"].str.capitalize().replace({"Oee": "OEE"})
f = px.bar(long, x="metric", y="value", color="shift_label", barmode="group", color_discrete_map=SHIFT_COLORS, text=long["value"].map(lambda v: f"{v:.1%}"))
f.update_yaxes(tickformat=".0%", range=[0, 1.1])
c1.plotly_chart(style(f, "OEE factors by shift (weighted)", "", "Value (%)", height=380), use_container_width=True)
c2.dataframe(sh[["shift_label", "oee", "reject_rate", "crew_exp", "crew", "waiting_operator_min_per_100h"]], hide_index=True, use_container_width=True, column_config={
    "oee": st.column_config.NumberColumn("OEE", format="%.3f"), "reject_rate": st.column_config.NumberColumn("Reject %", format="%.4f"),
    "crew_exp": st.column_config.NumberColumn("Crew exp. (yrs)", format="%.2f"), "crew": st.column_config.NumberColumn("Crew size", format="%.1f"),
    "waiting_operator_min_per_100h": st.column_config.NumberColumn("Waiting operator min / 100 planned h", format="%.1f")})

if S["shift_label"].nunique() >= 2 and S["line_id"].nunique() >= 1 and (S["shift_label"] == "S3").any() and (S["shift_label"] != "S3").any():
    reg = shift_effect_regression(S)
    reg_c = shift_effect_regression(S, controls=["avg_crew_experience_years"])
    r3 = reg[reg["term"].str.startswith("S3")]
    r3c = reg_c[reg_c["term"].str.startswith("S3")]
    # bootstrap on matched line-week differences (night vs day), weighted OEE inside each cell
    lw = oee_by(S.assign(is_night=np.where(S["shift_label"] == "S3", "night", "day")), ["line_id", "week", "is_night"])
    pw = lw.pivot_table(index=["line_id", "week"], columns="is_night", values="oee").dropna()
    if len(pw):
        bs = bootstrap_ci((pw["night"] - pw["day"]).to_numpy(), np.zeros(1))
    if len(r3):
        r3 = r3.iloc[0]
        st.markdown(
            f"**Controlling for line** (regression with line fixed effects, robust errors): night shift OEE is **{r3['diff'] * 100:+.1f} pts** vs S1 "
            f"(95% CI {r3['ci_low'] * 100:.1f} to {r3['ci_high'] * 100:.1f}; p = {r3['p_value']:.2g}). "
            + (f"With average crew experience also in the model the gap is {r3c.iloc[0]['diff'] * 100:+.1f} pts, so experience explains part of it. " if len(r3c) else "")
            + (f"Cross-check by bootstrap on matched line-weeks (night minus day): {bs['diff'] * 100:+.1f} pts (95% CI {bs['ci_low'] * 100:.1f} to {bs['ci_high'] * 100:.1f}). " if len(pw) else "")
        )
        st.caption("How to read: the CI is the range of plausible true gaps. If it excludes 0 the gap is unlikely to be chance (statistically significant). "
                   "This is synthetic data, so treat the numbers as a demonstration of method, not as a finding about a real plant.")
    st.dataframe(reg, hide_index=True, use_container_width=True, column_config={"diff": st.column_config.NumberColumn("OEE difference", format="%.4f"), "ci_low": st.column_config.NumberColumn("CI low", format="%.4f"),
                                                                                   "ci_high": st.column_config.NumberColumn("CI high", format="%.4f"), "p_value": st.column_config.NumberColumn("p-value", format="%.3g")})
else:
    st.info("Select shift S3 together with at least one day shift to test the shift effect.")
